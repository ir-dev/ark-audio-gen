"""
Vocal-to-music pipeline orchestrator.

Ties the pieces together into the end-to-end flow described in the feature spec:

    Upload Vocal → Find the sung region → Analyse Vocal → Extract Musical Structure
                 → Generate Arrangement → Synchronise with Vocal → Export

Reuses:
  * :mod:`vocal_analysis`  – detect the vocal's musical characteristics
  * :mod:`arrangement`     – turn the analysis into a generation + mix plan
  * :mod:`generator`       – MusicGen (melody-conditioned) accompaniment
  * :mod:`vocal_mixer`     – vocal-aware processing, level matching, ducking & mix

How the accompaniment is produced
---------------------------------
1. The recording is scanned for where the singing actually starts and ends and
   the leading/trailing silence is dropped, so the model never has to invent
   music for empty air.  (A 13 s count-in of room noise used to eat half of
   the first generation window and the result had nothing to follow.)
2. The sung region — capped at ``ARK_VOCAL_MAX_SECONDS`` (default 30 s) — is
   analysed and turned into a short, MusicGen-friendly arrangement prompt.
3. MusicGen generates the backing in **one pass** whenever the region fits its
   native 30 s window (its melody/chroma conditioning covers exactly 30 s).
   Longer regions are produced window by window, each new window *continuing*
   the previous one from an audio prompt of its last few seconds, so the music
   stays coherent instead of being unrelated clips glued together.
4. The backing is shaped around the vocal, level-matched to it, ducked under it
   and mixed.  The original vocal *performance* is untouched — only its overall
   level is normalised.

The heavy MusicGen dependency is injected via ``generator_factory`` so the
orchestration logic can be unit-tested with a stub.  A stub only needs a
``generate(prompt, melody_path, duration, guidance_scale, temperature)`` method
and a ``sample_rate`` attribute; optional keyword arguments (``progress_cb``,
``continuation``) are passed only when the callable accepts them.
"""

from __future__ import annotations

import inspect
import os
import tempfile
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import soundfile as sf

from arrangement import ArrangementPlan, plan_arrangement
from vocal_analysis import VocalAnalysis, analyze_vocal, detect_active_region
from vocal_mixer import mix_vocal_over_accompaniment, process_accompaniment

# Sample rate everything is mixed at (MusicGen's native rate).
MIX_SR = 32_000

# MusicGen was trained on 30 s clips and its chroma conditioning is exactly
# 30 s long, so that is the natural single-pass generation length.
MUSICGEN_WINDOW_SECONDS = 30.0

# Bounds (env-tunable) that keep CPU generation time sane.
MAX_TOTAL_SECONDS = float(os.environ.get("ARK_VOCAL_MAX_SECONDS", "30"))
SEGMENT_SECONDS = float(os.environ.get("ARK_VOCAL_SEGMENT_SECONDS", "30"))

# How much of a (long) upload is scanned for the sung region.
SCAN_SECONDS = float(os.environ.get("ARK_VOCAL_SCAN_SECONDS", "300"))

# Silence kept around the sung region so the first note is not clipped and the
# last one can ring out.
PRE_ROLL_SECONDS = 0.5
POST_ROLL_SECONDS = 1.0

# Audio prompt handed from one generation window to the next.
CONTINUATION_SECONDS = float(os.environ.get("ARK_VOCAL_CONTINUATION_SECONDS", "3"))

# Overlap between windows: near-invisible when the model continues the previous
# window, generous when it cannot (independent clips need a real crossfade).
CONTINUATION_CROSSFADE_SECONDS = 0.02
INDEPENDENT_CROSSFADE_SECONDS = 0.5

ProgressCB = Callable[[int, str], None]


@dataclass
class VocalPipelineResult:
    analysis: VocalAnalysis
    plan: ArrangementPlan
    sample_rate: int
    mix: np.ndarray                    # (2, samples) vocal + accompaniment
    accompaniment: np.ndarray          # (2, samples) instrumental only
    segments: int = 1
    truncated: bool = False            # True if the sung region exceeded the cap
    warnings: list = field(default_factory=list)
    window_start_sec: float = 0.0      # where in the source recording the output starts
    window_end_sec: float = 0.0        # …and ends
    source_duration_sec: float = 0.0   # full length of the uploaded recording
    continuation_used: bool = False    # windows were chained with audio prompts


# ──────────────────────────────────────────────────────────────────────────────
# Audio helpers
# ──────────────────────────────────────────────────────────────────────────────

def _load_vocal(path: str, sr: int, max_seconds: Optional[float] = None) -> np.ndarray:
    """Load a vocal file as (2, samples) float32 at ``sr``."""
    import librosa

    y, _ = librosa.load(path, sr=sr, mono=False, duration=max_seconds)
    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 1:
        y = np.stack([y, y])
    elif y.shape[0] == 1:
        y = np.repeat(y, 2, axis=0)
    elif y.ndim == 2 and y.shape[0] > 2:      # (samples, ch) safety
        y = y.T if y.shape[1] <= 2 else np.stack([y.mean(0), y.mean(0)])
    return y


def _fit_length(audio: np.ndarray, n: int) -> np.ndarray:
    """Trim or zero-pad a (2, samples) array to exactly ``n`` samples."""
    cur = audio.shape[-1]
    if cur == n:
        return audio
    if cur > n:
        return audio[:, :n]
    pad = np.zeros((audio.shape[0], n - cur), dtype=audio.dtype)
    return np.concatenate([audio, pad], axis=1)


def _to_stereo(audio, sr: int, target_sr: int) -> np.ndarray:
    chunk = np.asarray(audio, dtype=np.float32)
    if chunk.ndim == 1:
        chunk = np.stack([chunk, chunk])
    elif chunk.shape[0] == 1:
        chunk = np.repeat(chunk, 2, axis=0)
    if sr != target_sr:
        chunk = _resample_stereo(chunk, sr, target_sr)
    return chunk


def _resample_stereo(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return audio
    import librosa

    return np.stack([
        librosa.resample(audio[c], orig_sr=orig_sr, target_sr=target_sr)
        for c in range(audio.shape[0])
    ]).astype(np.float32)


def choose_window(
    active_start: float,
    active_end: float,
    total_sec: float,
    max_total_seconds: float,
    pre_roll: float = PRE_ROLL_SECONDS,
    post_roll: float = POST_ROLL_SECONDS,
) -> tuple[float, float, bool]:
    """
    Pick the slice of the recording to build music for.

    Returns ``(start, end, truncated)``: the sung region padded by a short
    pre/post-roll, clamped to the recording and capped at ``max_total_seconds``
    *of singing* — so a long silent count-in no longer eats the budget.
    """
    start = max(0.0, float(active_start) - pre_roll)
    end = min(float(total_sec), float(active_end) + post_roll)
    if end <= start:                                   # nothing detected
        start, end = 0.0, min(float(total_sec), max_total_seconds)
    truncated = False
    if end - start > max_total_seconds + 1e-6:
        end = start + max_total_seconds
        truncated = True
    return start, end, truncated


def _segment_bounds(total_sec: float, seg_sec: float) -> list[tuple[float, float]]:
    """Even-length windows covering [0, total_sec], each <= seg_sec."""
    if total_sec <= seg_sec:
        return [(0.0, total_sec)]
    n = int(np.ceil(total_sec / seg_sec))
    step = total_sec / n
    return [(i * step, min((i + 1) * step, total_sec)) for i in range(n)]


def _assemble(chunks: list[np.ndarray], fade: int) -> np.ndarray:
    """
    Join (2, samples) chunks that overlap by ``fade`` samples with an equal-power
    crossfade.  Every chunk except the last is expected to carry ``fade`` extra
    samples at its end, so the joined length equals the sum of the window
    lengths and stays aligned with the vocal timeline.
    """
    chunks = [c for c in chunks if c.shape[-1] > 0]
    if not chunks:
        return np.zeros((2, 0), dtype=np.float32)
    out = chunks[0]
    for nxt in chunks[1:]:
        f = min(fade, out.shape[-1], nxt.shape[-1])
        if f <= 0:
            out = np.concatenate([out, nxt], axis=1)
            continue
        t = np.linspace(0.0, 1.0, f, dtype=np.float32)
        fade_out = np.cos(t * np.pi / 2.0)      # equal power
        fade_in = np.sin(t * np.pi / 2.0)
        head, tail = out[:, :-f], out[:, -f:]
        blended = tail * fade_out[np.newaxis, :] + nxt[:, :f] * fade_in[np.newaxis, :]
        out = np.concatenate([head, blended, nxt[:, f:]], axis=1)
    return out.astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Generator adapter
# ──────────────────────────────────────────────────────────────────────────────

def _accepts_kwarg(fn, name: str) -> bool:
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    if name in params:
        return True
    return any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


def _call_generate(gen, **kwargs):
    """Call ``gen.generate`` passing only the keyword arguments it accepts."""
    fn = gen.generate
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return fn(**kwargs)
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return fn(**kwargs)
    return fn(**{k: v for k, v in kwargs.items() if k in params})


# ──────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ──────────────────────────────────────────────────────────────────────────────

def run_vocal_to_music(
    vocal_path: str,
    overrides: Optional[dict] = None,
    generator_factory: Optional[Callable[[], object]] = None,
    progress: Optional[ProgressCB] = None,
    max_total_seconds: float = MAX_TOTAL_SECONDS,
    segment_seconds: float = SEGMENT_SECONDS,
) -> VocalPipelineResult:
    """
    Run the full vocal-to-music pipeline and return audio + metadata.

    Parameters
    ----------
    vocal_path        : path to the uploaded vocal recording.
    overrides         : optional user overrides for the arrangement plan
                        (genre/mood/instruments/tempo/… — see
                        :func:`arrangement.plan_arrangement`).
    generator_factory : callable returning an object with a
                        ``generate(prompt, melody_path, duration, guidance_scale,
                        temperature[, progress_cb][, continuation])`` method and
                        a ``sample_rate`` attribute.  Defaults to a
                        melody-conditioned :class:`generator.MusicGenerator`.
    progress          : optional ``callback(percent:int, message:str)`` for UI.
    max_total_seconds : cap on the length of *singing* that is scored.
    segment_seconds   : generation window (≤ 30 s, MusicGen's native length).
    """
    def _p(pct: int, msg: str) -> None:
        if progress:
            progress(pct, msg)

    warnings_out: list[str] = []
    segment_seconds = float(min(max(segment_seconds, 1.0), MUSICGEN_WINDOW_SECONDS))

    # ── 1. Load the recording and find the sung region ────────────────────────
    _p(4, "Loading vocal…")
    vocal_full = _load_vocal(vocal_path, MIX_SR, max_seconds=SCAN_SECONDS)
    source_sec = vocal_full.shape[-1] / MIX_SR

    active_start, active_end = detect_active_region(vocal_full.mean(axis=0), MIX_SR)
    win_start, win_end, truncated = choose_window(
        active_start, active_end, source_sec, max_total_seconds,
    )
    vocal = vocal_full[:, int(round(win_start * MIX_SR)): int(round(win_end * MIX_SR))]
    del vocal_full
    total_sec = vocal.shape[-1] / MIX_SR

    if win_start >= 0.5:
        warnings_out.append(
            f"Skipped {win_start:.1f}s of silence at the start of the recording — "
            f"the track begins where the singing does."
        )
    if truncated:
        warnings_out.append(
            f"Used the first {int(round(max_total_seconds))}s of singing "
            f"({win_start:.1f}s–{win_end:.1f}s of the recording); longer vocals are "
            f"capped to keep CPU generation time sane (ARK_VOCAL_MAX_SECONDS)."
        )

    # ── 2. Analyse exactly the audio the model will hear ─────────────────────
    _p(8, "Analysing vocal performance…")
    analysis = analyze_vocal((vocal.mean(axis=0), MIX_SR), max_seconds=None)
    analysis.duration_sec = round(source_sec, 2)

    # ── 3. Extract musical structure → arrangement plan ───────────────────────
    _p(18, "Extracting musical structure…")
    plan = plan_arrangement(analysis, overrides=overrides)

    # ── 4. Generate accompaniment (single pass, or chained windows) ───────────
    if generator_factory is None:
        def generator_factory():                      # pragma: no cover - heavy
            from generator import MusicGenerator
            return MusicGenerator(use_melody_model=True)

    gen = generator_factory()
    gen_sr = int(getattr(gen, "sample_rate", MIX_SR) or MIX_SR)
    can_continue = _accepts_kwarg(gen.generate, "continuation")

    fade_sec = CONTINUATION_CROSSFADE_SECONDS if can_continue else INDEPENDENT_CROSSFADE_SECONDS
    fade_n = int(fade_sec * MIX_SR)
    window = segment_seconds if total_sec <= segment_seconds else segment_seconds - fade_sec
    bounds = _segment_bounds(total_sec, window)
    n_seg = len(bounds)

    acc_chunks: list[np.ndarray] = []
    prev_tail: Optional[np.ndarray] = None
    continuation_used = False

    # The accompaniment step owns the 30–75 % band; give each window an equal
    # slice of it and drive that slice from MusicGen's per-token progress.
    ACC_BASE, ACC_SPAN = 30, 45
    for i, (start, end) in enumerate(bounds):
        is_last = i == n_seg - 1
        seg_base = ACC_BASE + ACC_SPAN * i / max(n_seg, 1)
        seg_slice = ACC_SPAN / max(n_seg, 1)
        label = "Generating accompaniment" if n_seg == 1 else f"Generating accompaniment {i + 1}/{n_seg}"
        _p(int(seg_base), f"{label}…")

        def _seg_progress(frac: float, _base=seg_base, _slice=seg_slice, _label=label) -> None:
            _p(int(_base + _slice * frac), f"{_label}… {int(frac * 100)}%")

        s0, s1 = int(round(start * MIX_SR)), int(round(end * MIX_SR))
        n_want = s1 - s0
        # Non-final windows carry an extra `fade` so neighbours overlap.
        extra_n = 0 if is_last else fade_n
        n_gen = n_want + extra_n
        seg_len = max(n_gen / MIX_SR, 0.5)

        # The model hears the vocal of this window (plus the overlap) as melody.
        seg_vocal = vocal[:, s0: min(s1 + extra_n, vocal.shape[-1])]
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            seg_path = tmp.name
        try:
            sf.write(seg_path, seg_vocal.T, MIX_SR, format="WAV", subtype="FLOAT")
            kwargs = dict(
                prompt=plan.prompt,
                melody_path=seg_path,
                duration=seg_len,
                guidance_scale=plan.guidance_scale,
                temperature=plan.temperature,
                progress_cb=_seg_progress,
            )
            if can_continue and prev_tail is not None and prev_tail.shape[-1] > 0:
                kwargs["continuation"] = (prev_tail, MIX_SR)
                continuation_used = True
            audio, sr = _call_generate(gen, **kwargs)
        finally:
            try:
                os.unlink(seg_path)
            except OSError:
                pass

        chunk = _fit_length(_to_stereo(audio, sr, MIX_SR), n_gen)
        acc_chunks.append(chunk)

        # Hand the tail of *this* window (up to its boundary) to the next one.
        tail_n = min(int(CONTINUATION_SECONDS * MIX_SR), n_want)
        prev_tail = chunk[:, n_want - tail_n: n_want].copy() if tail_n > 0 else None

    accompaniment_raw = _assemble(acc_chunks, fade_n)

    # ── 5. Vocal-aware processing of the accompaniment ────────────────────────
    _p(80, "Shaping accompaniment around the vocal…")
    accompaniment = process_accompaniment(
        accompaniment_raw, MIX_SR,
        genre=plan.genre, mood=plan.mood,
        crescendo_pattern=plan.crescendo,
        vocal_center_hz=plan.vocal_center_hz,
    )
    accompaniment = _fit_length(accompaniment, vocal.shape[-1])

    # ── 6. Level-match, duck & mix the vocal over the accompaniment ──────────
    _p(88, "Synchronising vocal with the music…")
    mix = mix_vocal_over_accompaniment(
        vocal, accompaniment, MIX_SR,
        accompaniment_gain_db=plan.accompaniment_gain_db,
        vocal_gain_db=plan.vocal_gain_db,
        duck_depth_db=plan.duck_depth_db,
    )

    return VocalPipelineResult(
        analysis=analysis,
        plan=plan,
        sample_rate=MIX_SR,
        mix=mix,
        accompaniment=accompaniment,
        segments=n_seg,
        truncated=truncated,
        warnings=warnings_out,
        window_start_sec=round(win_start, 2),
        window_end_sec=round(win_end, 2),
        source_duration_sec=round(source_sec, 2),
        continuation_used=continuation_used,
    )


__all__ = ["run_vocal_to_music", "VocalPipelineResult", "choose_window",
           "MAX_TOTAL_SECONDS", "SEGMENT_SECONDS", "MUSICGEN_WINDOW_SECONDS"]
