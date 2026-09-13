"""
Background worker that drains the persistent job queue.

A single daemon thread repeatedly claims the oldest queued job (see
:mod:`job_store`) and runs it to completion, writing progress back to the DB as
it goes.  Because the queue is drained by exactly one worker, jobs are naturally
serialised on the CPU — no semaphore, and therefore none of the
"one hung job wedges everything at pending" failure mode.

The heavy MusicGen models are cached across jobs (``_make_generator``) so the
~1.5 GB melody model loads **once** per process instead of on every request.

Testability
-----------
* :func:`run_one` claims and processes a single job and returns whether it did
  any work — tests drive the queue deterministically without threads.
* :data:`PROCESSORS` maps job mode → handler and can be monkeypatched to avoid
  loading real models in tests.
"""

from __future__ import annotations

import os
import threading
import time
import traceback
from pathlib import Path

import numpy as np

import job_store
from audio_io import write_mp3

OUTPUT_DIR = Path("generated")

_AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".aiff"}

# ──────────────────────────────────────────────────────────────────────────────
# Model cache (load MusicGen once, reuse across jobs)
# ──────────────────────────────────────────────────────────────────────────────

_generators: dict[bool, object] = {}
_gen_lock = threading.Lock()


def _make_generator(use_melody: bool):
    """Return a cached :class:`MusicGenerator`, loading it on first use."""
    with _gen_lock:
        gen = _generators.get(use_melody)
        if gen is None:
            from generator import MusicGenerator
            gen = MusicGenerator(use_melody_model=use_melody)
            _generators[use_melody] = gen
        return gen


# ──────────────────────────────────────────────────────────────────────────────
# Job processors
# ──────────────────────────────────────────────────────────────────────────────

def process_text(job: dict) -> None:
    """Text-prompt → backing track (mirrors the old ``_generate_sync``)."""
    from effects import process_audio
    from prompt_builder import build_prompt, infer_parameters

    job_id = job["id"]
    p = job["params"]

    # ── Detect melody type ────────────────────────────────────────────────────
    melody = p["melody"]
    path = Path(melody)
    is_audio_file = path.suffix.lower() in _AUDIO_EXTS and path.exists()
    melody_path = str(path) if is_audio_file else None
    melody_desc = "melody from audio file" if is_audio_file else melody

    # ── Build prompt ──────────────────────────────────────────────────────────
    inferred = infer_parameters(melody_desc)
    inst_list = (
        [i.strip() for i in p["instruments"].split(",") if i.strip()]
        if p.get("instruments") else None
    )
    prompt = build_prompt(
        melody_description=melody_desc,
        genre=p.get("genre"),
        mood=p.get("mood"),
        instruments=inst_list,
        frequency_range=p.get("frequency_range"),
        inferred=inferred,
    )
    effective_genre = p.get("genre") or inferred["genre"]
    effective_mood = p.get("mood") or inferred["mood"]

    # ── Generate ──────────────────────────────────────────────────────────────
    job_store.update_job(job_id, status=job_store.PROCESSING,
                         message="Loading AI model…", progress=10)
    gen = _make_generator(is_audio_file)
    job_store.update_job(job_id, message="Generating music… (CPU)", progress=30)

    _last = {"pct": -1}

    def _cb(frac: float) -> None:
        pct = 30 + int(50 * frac)
        if pct != _last["pct"]:
            _last["pct"] = pct
            job_store.update_job(
                job_id, status=job_store.PROCESSING, progress=pct,
                message=f"Generating music… {int(frac * 100)}%",
            )

    audio, sr = gen.generate(
        prompt=prompt,
        melody_path=melody_path,
        duration=float(p.get("duration", 15.0)),
        guidance_scale=float(p.get("guidance_scale", 3.5)),
        temperature=float(p.get("temperature", 1.05)),
        progress_cb=_cb,
    )

    # ── Effects ───────────────────────────────────────────────────────────────
    job_store.update_job(job_id, message="Applying audio effects…", progress=80)
    audio = process_audio(
        audio, sr,
        genre=effective_genre,
        mood=effective_mood,
        crescendo_pattern=p.get("crescendo", "rise-fall"),
    )

    # ── Export ────────────────────────────────────────────────────────────────
    job_store.update_job(job_id, message="Exporting to MP3…", progress=92)
    out_file = OUTPUT_DIR / f"{job_id}.mp3"
    write_mp3(audio, sr, out_file)

    job_store.update_job(
        job_id,
        status=job_store.DONE,
        message="Your track is ready!",
        progress=100,
        file=str(out_file),
        result={
            "mode": "text",
            "genre": effective_genre,
            "mood": effective_mood,
            "duration": float(p.get("duration", 15.0)),
        },
    )


def process_vocal(job: dict) -> None:
    """Vocal recording → complementary accompaniment (mirrors ``_generate_vocal_sync``)."""
    from vocal_pipeline import run_vocal_to_music

    job_id = job["id"]
    p = job["params"]
    vocal_path = p["vocal_path"]
    overrides = p.get("overrides", {})

    try:
        def _cb(pct: int, msg: str) -> None:
            job_store.update_job(job_id, status=job_store.PROCESSING,
                                 progress=int(pct), message=msg)

        job_store.update_job(job_id, status=job_store.PROCESSING,
                             message="Starting…", progress=3)

        if not Path(vocal_path).exists():
            raise FileNotFoundError(
                "Uploaded vocal is no longer available (server was restarted "
                "before processing) — please re-upload."
            )

        result = run_vocal_to_music(
            vocal_path,
            overrides=overrides,
            generator_factory=lambda: _make_generator(True),
            progress=_cb,
        )

        # ── Export both the full mix and the accompaniment-only track ─────────
        job_store.update_job(job_id, message="Exporting audio…", progress=94)
        mix_file = OUTPUT_DIR / f"{job_id}.mp3"
        acc_file = OUTPUT_DIR / f"{job_id}_accompaniment.mp3"
        write_mp3(result.mix, result.sample_rate, mix_file)
        write_mp3(result.accompaniment, result.sample_rate, acc_file)

        job_store.update_job(
            job_id,
            status=job_store.DONE,
            message="Your track is ready!",
            progress=100,
            file=str(mix_file),
            file_accompaniment=str(acc_file),
            result={
                "mode": "vocal",
                "analysis": result.analysis.to_dict(),
                "plan": result.plan.to_dict(),
                "warnings": result.warnings,
                "truncated": result.truncated,
                "segments": result.segments,
                "genre": result.plan.genre,
                "mood": result.plan.mood,
                "key": result.analysis.key_name,
                "tempo": result.analysis.tempo_bpm,
                "duration": round(result.mix.shape[-1] / result.sample_rate, 1),
            },
        )
    finally:
        try:
            Path(vocal_path).unlink(missing_ok=True)
        except OSError:
            pass


# Dispatch table — monkeypatchable in tests.
PROCESSORS = {
    "text": process_text,
    "vocal": process_vocal,
}


# ──────────────────────────────────────────────────────────────────────────────
# Queue draining
# ──────────────────────────────────────────────────────────────────────────────

def run_one() -> bool:
    """
    Claim and run a single queued job.

    Returns ``True`` if a job was processed (regardless of success/failure),
    ``False`` if the queue was empty.
    """
    job = job_store.claim_next_job()
    if job is None:
        return False

    proc = PROCESSORS.get(job["mode"])
    try:
        if proc is None:
            raise ValueError(f"Unknown job mode: {job['mode']!r}")
        proc(job)
    except Exception as exc:  # noqa: BLE001 - surface any failure to the user
        job_store.update_job(
            job["id"],
            status=job_store.ERROR,
            progress=0,
            message=f"Generation failed: {exc}",
            error=traceback.format_exc(),
        )
    return True


# ──────────────────────────────────────────────────────────────────────────────
# Background thread lifecycle
# ──────────────────────────────────────────────────────────────────────────────

_stop = threading.Event()
_thread: threading.Thread | None = None
_thread_lock = threading.Lock()

_IDLE_SLEEP = float(os.environ.get("ARK_WORKER_IDLE_SLEEP", "1.0"))


def _loop() -> None:
    while not _stop.is_set():
        try:
            did_work = run_one()
        except Exception:  # pragma: no cover - defensive; run_one already guards
            did_work = False
        if not did_work:
            _stop.wait(_IDLE_SLEEP)


def start() -> None:
    """Start the background worker thread (idempotent; no-op if disabled)."""
    if os.environ.get("ARK_DISABLE_WORKER") == "1":
        return
    global _thread
    with _thread_lock:
        if _thread is not None and _thread.is_alive():
            return
        _stop.clear()
        _thread = threading.Thread(target=_loop, name="job-worker", daemon=True)
        _thread.start()


def stop(timeout: float = 5.0) -> None:
    """Signal the worker to stop and wait for it (used in tests/shutdown)."""
    _stop.set()
    global _thread
    with _thread_lock:
        if _thread is not None:
            _thread.join(timeout=timeout)
            _thread = None


__all__ = ["run_one", "start", "stop", "PROCESSORS", "process_text", "process_vocal"]
