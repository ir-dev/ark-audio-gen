"""
FastAPI web application for the AI Sing-Along Music Generator.

Routes
------
GET  /                          Serve index.html (UI)
POST /api/generate              Submit a generation job → {job_id}
GET  /api/status/{job_id}       Poll job status
GET  /api/download/{job_id}     Stream the finished MP3
GET  /api/health                Azure load-balancer probe
DELETE /api/job/{job_id}        Client-side cleanup (optional)

Generation runs in a thread-pool so no HTTP request ever blocks the event
loop. The client polls /api/status every 3 s until status == "done"|"error".
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import (
    BackgroundTasks, FastAPI, File, Form, HTTPException, Query, UploadFile,
)
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# ──────────────────────────────────────────────────────────────────────────────
# App setup
# ──────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="AI Sing-Along Music Generator",
    description="Generate rhythmic, lively sing-along backing tracks using Meta MusicGen.",
    version="1.0.0",
)

OUTPUT_DIR = Path("generated")
OUTPUT_DIR.mkdir(exist_ok=True)

# Uploaded vocals live here transiently (removed once a job finishes).
UPLOAD_DIR = OUTPUT_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# Accepted vocal upload formats / size cap.
_VOCAL_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".aiff", ".m4a", ".aac"}
MAX_UPLOAD_BYTES = 25 * 1024 * 1024   # 25 MB

# One generation at a time on CPU (prevents OOM on small Azure SKUs)
_gen_semaphore = threading.Semaphore(1)

# Thread pool – keep separate from the asyncio event loop
_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="gen")

# In-memory job store  {job_id: {...}}
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()

# How long (seconds) to keep finished jobs before cleanup
JOB_TTL = 3600  # 1 hour


# ──────────────────────────────────────────────────────────────────────────────
# Request / response schemas
# ──────────────────────────────────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    melody: str = Field(..., min_length=3, max_length=600,
                        description="Melody description or absolute path to an audio file.")
    genre: Optional[str] = Field(None, description="pop | rock | jazz | classical | folk | electronic | hip-hop | r-and-b | ambient | reggae | bossa-nova")
    mood: Optional[str] = Field(None, description="happy | sad | energetic | calm | romantic | uplifting | mysterious | aggressive | nostalgic | playful")
    instruments: Optional[str] = Field(None, description="Comma-separated list, e.g. 'piano,guitar,drums'")
    frequency_range: Optional[str] = Field(None, description="bass | mid | treble | full")
    duration: float = Field(default=15.0, ge=5.0, le=20.0, description="Length in seconds (5–20)")
    crescendo: str = Field(default="rise-fall", description="rise | fall | rise-fall | natural | verse-chorus")
    guidance_scale: float = Field(default=3.5, ge=1.0, le=10.0)
    temperature: float = Field(default=1.05, ge=0.5, le=1.5)


# ──────────────────────────────────────────────────────────────────────────────
# Job utilities
# ──────────────────────────────────────────────────────────────────────────────

def _set_job(job_id: str, **fields) -> None:
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id].update(fields)
        else:
            _jobs[job_id] = {"created_at": time.time(), **fields}


def _get_job(job_id: str) -> dict | None:
    with _jobs_lock:
        return dict(_jobs[job_id]) if job_id in _jobs else None


def _remove_job_files(job: dict) -> None:
    """Delete every on-disk artefact referenced by a job (keys starting '_file')."""
    for key, val in job.items():
        if key.startswith("_file") and val:
            try:
                Path(val).unlink(missing_ok=True)
            except OSError:
                pass


def _cleanup_expired() -> None:
    """Remove jobs (and their output files) that are older than JOB_TTL."""
    cutoff = time.time() - JOB_TTL
    with _jobs_lock:
        expired = [jid for jid, j in _jobs.items() if j.get("created_at", 0) < cutoff]
        for jid in expired:
            _remove_job_files(_jobs[jid])
            del _jobs[jid]


# ──────────────────────────────────────────────────────────────────────────────
# Core generation (runs in thread pool)
# ──────────────────────────────────────────────────────────────────────────────

def _generate_sync(job_id: str, req: GenerateRequest) -> None:
    """Blocking generation pipeline.  Called from ThreadPoolExecutor."""

    _gen_semaphore.acquire()
    try:
        # ── Imports (deferred so startup is instant) ──────────────────────────
        from effects import process_audio
        from generator import MusicGenerator
        from prompt_builder import build_prompt, infer_parameters

        # ── Detect melody type ────────────────────────────────────────────────
        _AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".aiff"}
        p = Path(req.melody)
        is_audio_file = p.suffix.lower() in _AUDIO_EXTS and p.exists()
        melody_path = str(p) if is_audio_file else None
        melody_desc = "melody from audio file" if is_audio_file else req.melody

        # ── Build prompt ──────────────────────────────────────────────────────
        inferred = infer_parameters(melody_desc)
        inst_list = (
            [i.strip() for i in req.instruments.split(",") if i.strip()]
            if req.instruments
            else None
        )
        prompt = build_prompt(
            melody_description=melody_desc,
            genre=req.genre,
            mood=req.mood,
            instruments=inst_list,
            frequency_range=req.frequency_range,
            inferred=inferred,
        )
        effective_genre = req.genre or inferred["genre"]
        effective_mood = req.mood or inferred["mood"]

        # ── Generate ──────────────────────────────────────────────────────────
        _set_job(job_id, status="processing", message="Loading AI model…", progress=10)
        gen = MusicGenerator(use_melody_model=is_audio_file)
        _set_job(job_id, message="Generating music… (2–10 min on CPU)", progress=30)

        # Map MusicGen's per-token decode progress into the 30–80 % band so the
        # polled status bar advances during the long generation call.
        _last_pct = {"v": -1}

        def _music_progress(frac: float) -> None:
            pct = 30 + int(50 * frac)
            if pct != _last_pct["v"]:
                _last_pct["v"] = pct
                _set_job(
                    job_id, status="processing",
                    message=f"Generating music… {int(frac * 100)}% (CPU)",
                    progress=pct,
                )

        audio, sr = gen.generate(
            prompt=prompt,
            melody_path=melody_path,
            duration=req.duration,
            guidance_scale=req.guidance_scale,
            temperature=req.temperature,
            progress_cb=_music_progress,
        )

        # ── Effects ───────────────────────────────────────────────────────────
        _set_job(job_id, message="Applying audio effects…", progress=80)
        audio = process_audio(
            audio, sr,
            genre=effective_genre,
            mood=effective_mood,
            crescendo_pattern=req.crescendo,
        )

        # ── Export MP3 ────────────────────────────────────────────────────────
        _set_job(job_id, message="Exporting to MP3…", progress=92)
        out_file = OUTPUT_DIR / f"{job_id}.mp3"
        _write_mp3(audio, sr, out_file)

        _set_job(
            job_id,
            status="done",
            message="Your track is ready!",
            progress=100,
            _file=str(out_file),
            genre=effective_genre,
            mood=effective_mood,
            duration=req.duration,
        )

    except Exception as exc:
        _set_job(
            job_id,
            status="error",
            message=f"Generation failed: {exc}",
            progress=0,
            _detail=traceback.format_exc(),
        )
    finally:
        _gen_semaphore.release()


def _build_overrides(
    genre: Optional[str],
    mood: Optional[str],
    instruments: Optional[str],
    tempo_bpm: Optional[float],
    guidance_scale: Optional[float],
    temperature: Optional[float],
    crescendo: Optional[str],
) -> dict:
    """Assemble arrangement overrides from optional form fields (blanks ignored)."""
    inst_list = (
        [i.strip() for i in instruments.split(",") if i.strip()]
        if instruments else None
    )
    overrides = {
        "genre": genre,
        "mood": mood,
        "instruments": inst_list,
        "tempo_bpm": tempo_bpm,
        "guidance_scale": guidance_scale,
        "temperature": temperature,
        "crescendo": crescendo,
    }
    return {k: v for k, v in overrides.items() if v not in (None, "", [])}


def _generate_vocal_sync(job_id: str, vocal_path: str, overrides: dict) -> None:
    """Blocking vocal-to-music pipeline.  Called from the ThreadPoolExecutor."""

    _gen_semaphore.acquire()
    try:
        from vocal_pipeline import run_vocal_to_music

        def _cb(pct: int, msg: str) -> None:
            _set_job(job_id, status="processing", message=msg, progress=pct)

        _set_job(job_id, status="processing", message="Starting…", progress=3)

        result = run_vocal_to_music(vocal_path, overrides=overrides, progress=_cb)

        # ── Export both the full mix and the accompaniment-only track ─────────
        _set_job(job_id, message="Exporting audio…", progress=94)
        mix_file = OUTPUT_DIR / f"{job_id}.mp3"
        acc_file = OUTPUT_DIR / f"{job_id}_accompaniment.mp3"
        _write_mp3(result.mix, result.sample_rate, mix_file)
        _write_mp3(result.accompaniment, result.sample_rate, acc_file)

        _set_job(
            job_id,
            status="done",
            message="Your track is ready!",
            progress=100,
            _file=str(mix_file),
            _file_accompaniment=str(acc_file),
            analysis=result.analysis.to_dict(),
            plan=result.plan.to_dict(),
            warnings=result.warnings,
            truncated=result.truncated,
            segments=result.segments,
            genre=result.plan.genre,
            mood=result.plan.mood,
            key=result.analysis.key_name,
            tempo=result.analysis.tempo_bpm,
            duration=round(result.mix.shape[-1] / result.sample_rate, 1),
        )

    except Exception as exc:
        _set_job(
            job_id,
            status="error",
            message=f"Generation failed: {exc}",
            progress=0,
            _detail=traceback.format_exc(),
        )
    finally:
        _gen_semaphore.release()
        try:
            Path(vocal_path).unlink(missing_ok=True)
        except OSError:
            pass


async def _save_vocal_upload(file: UploadFile) -> str:
    """Validate and persist an uploaded vocal file; return its temp path."""
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in _VOCAL_EXTS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format '{suffix or '?'}'. "
                   f"Use one of: {', '.join(sorted(_VOCAL_EXTS))}",
        )
    data = await file.read()
    if len(data) < 1024:
        raise HTTPException(status_code=400, detail="Upload is empty or too small.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large (max {MAX_UPLOAD_BYTES // (1024 * 1024)} MB).",
        )
    fd, path = tempfile.mkstemp(suffix=suffix, dir=str(UPLOAD_DIR))
    with os.fdopen(fd, "wb") as fh:
        fh.write(data)
    return path


def _analyze_vocal_sync(vocal_path: str) -> dict:
    """Run analysis only (no generation); used by /api/vocal/analyze."""
    try:
        from vocal_analysis import analyze_vocal
        return analyze_vocal(vocal_path).to_dict()
    finally:
        try:
            Path(vocal_path).unlink(missing_ok=True)
        except OSError:
            pass


def _find_ffmpeg() -> str:
    """
    Locate the ffmpeg binary.

    Search order:
      1. FFMPEG_PATH env-var  (set by startup.sh on Azure)
      2. PATH via shutil.which
      3. Common Linux paths
      4. Homebrew path on macOS
    """
    import shutil

    candidate = (
        os.environ.get("FFMPEG_PATH")
        or shutil.which("ffmpeg")
        or "/home/bin/ffmpeg"        # Azure cached location
        or "/usr/bin/ffmpeg"         # standard Linux
        or "/usr/local/bin/ffmpeg"   # Homebrew macOS
    )
    if candidate and Path(candidate).exists():
        return candidate
    raise FileNotFoundError(
        "ffmpeg not found. On Azure ensure startup.sh ran; locally run: brew install ffmpeg"
    )


def _write_mp3(audio: np.ndarray, sr: int, out_path: Path, bitrate: str = "192k") -> None:
    """
    Write a numpy audio array to an MP3 file.

    Uses ffmpeg via subprocess directly — this avoids pydub's internal
    ffprobe auto-detection which fails when ffmpeg is not on the system PATH
    (e.g. Azure App Service containers).
    """
    import soundfile as sf
    import subprocess

    ffmpeg = _find_ffmpeg()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_wav = tmp.name
    try:
        data = audio.T if audio.ndim == 2 else audio
        peak = np.max(np.abs(data))
        if peak > 0:
            data = (data / peak * 0.97).astype(np.float32)
        sf.write(tmp_wav, data, sr, format="WAV", subtype="FLOAT")

        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Call ffmpeg directly — no pydub, no ffprobe probe step
        result = subprocess.run(
            [
                ffmpeg, "-y",           # overwrite without asking
                "-i", tmp_wav,          # input WAV
                "-codec:a", "libmp3lame",
                "-b:a", bitrate,
                "-q:a", "2",            # VBR quality hint (1=best, 9=worst)
                str(out_path),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed:\n{result.stderr[-1000:]}")
    finally:
        try:
            os.unlink(tmp_wav)
        except OSError:
            pass


# ──────────────────────────────────────────────────────────────────────────────
# API routes  (register BEFORE static-file mount)
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/api/health", tags=["ops"])
async def health():
    return {"status": "ok", "jobs": len(_jobs)}


@app.post("/api/generate", tags=["generation"])
async def start_generation(req: GenerateRequest, background_tasks: BackgroundTasks):
    _cleanup_expired()

    job_id = str(uuid.uuid4())
    _set_job(job_id, status="pending", message="Queued…", progress=0)

    # Submit to thread pool via async background task
    async def _submit():
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(_executor, _generate_sync, job_id, req)

    background_tasks.add_task(_submit)
    return {"job_id": job_id}


@app.get("/api/status/{job_id}", tags=["generation"])
async def get_status(job_id: str):
    job = _get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    # Strip internal fields before returning to client
    return {k: v for k, v in job.items() if not k.startswith("_")}


@app.post("/api/vocal/analyze", tags=["vocal"])
async def analyze_vocal_endpoint(file: UploadFile = File(...)):
    """
    Analyse an uploaded vocal recording and return its musical characteristics.

    Fast, model-free — intended for the UI to preview detected key/tempo/mood
    before committing to a (slow) generation job.
    """
    vocal_path = await _save_vocal_upload(file)
    loop = asyncio.get_event_loop()
    analysis = await loop.run_in_executor(_executor, _analyze_vocal_sync, vocal_path)
    return {"analysis": analysis}


@app.post("/api/vocal/generate", tags=["vocal"])
async def start_vocal_generation(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    genre: Optional[str] = Form(None),
    mood: Optional[str] = Form(None),
    instruments: Optional[str] = Form(None),
    tempo_bpm: Optional[float] = Form(None),
    guidance_scale: Optional[float] = Form(None),
    temperature: Optional[float] = Form(None),
    crescendo: Optional[str] = Form(None),
):
    """
    Upload a vocal recording and generate a complementary accompaniment.

    All arrangement fields are optional — anything omitted is auto-detected from
    the vocal.  Returns ``{job_id}``; poll ``/api/status/{job_id}`` for progress
    and the detected analysis, then download via ``/api/download/{job_id}``
    (``?variant=mix`` default, or ``?variant=accompaniment``).
    """
    _cleanup_expired()
    vocal_path = await _save_vocal_upload(file)
    overrides = _build_overrides(
        genre, mood, instruments, tempo_bpm, guidance_scale, temperature, crescendo,
    )

    job_id = str(uuid.uuid4())
    _set_job(job_id, status="pending", message="Queued…", progress=0, mode="vocal")

    async def _submit():
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            _executor, _generate_vocal_sync, job_id, vocal_path, overrides
        )

    background_tasks.add_task(_submit)
    return {"job_id": job_id}


@app.get("/api/download/{job_id}", tags=["generation"])
async def download_track(job_id: str, variant: str = Query("mix")):
    job = _get_job(job_id)
    if job is None or job.get("status") != "done":
        raise HTTPException(status_code=404, detail="Track not ready or not found")

    if variant == "accompaniment":
        file_path = job.get("_file_accompaniment")
        filename = "ark_accompaniment.mp3"
    else:
        file_path = job.get("_file")
        filename = "ark_vocal_track.mp3" if job.get("mode") == "vocal" \
            else "ai_singalong_track.mp3"

    if not file_path or not Path(file_path).exists():
        raise HTTPException(status_code=404, detail="File missing on server")
    return FileResponse(
        path=file_path,
        media_type="audio/mpeg",
        filename=filename,
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@app.delete("/api/job/{job_id}", tags=["generation"])
async def delete_job(job_id: str):
    with _jobs_lock:
        job = _jobs.pop(job_id, None)
    if job:
        _remove_job_files(job)
    return {"deleted": job_id}


# ──────────────────────────────────────────────────────────────────────────────
# Static files (SPA fallback) — MUST be last
# ──────────────────────────────────────────────────────────────────────────────

app.mount("/", StaticFiles(directory="static", html=True), name="static")


# ──────────────────────────────────────────────────────────────────────────────
# Run directly:  python api.py
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    print(f"\n  Open your browser at  http://localhost:{port}\n")
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=True)
