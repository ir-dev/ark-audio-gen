"""
Vocal-to-music pipeline orchestrator.

Ties the pieces together into the end-to-end flow described in the feature spec:

    Upload Vocal → Analyse Vocal → Extract Musical Structure
                 → Generate Arrangement → Synchronise with Vocal → Export

Reuses:
  * :mod:`vocal_analysis`  – detect the vocal's musical characteristics
  * :mod:`arrangement`     – turn the analysis into a generation + mix plan
  * :mod:`generator`       – MusicGen (melody-conditioned) accompaniment
  * :mod:`vocal_mixer`     – vocal-aware processing, ducking, sync & mix

Long vocals are handled by **segmented generation**: the vocal is split into
model-sized windows, each window conditions its own accompaniment, and the
segments are crossfaded back together — then the *original* vocal is mixed over
the whole thing so it stays perfectly aligned and unchanged end-to-end.

The heavy MusicGen dependency is injected via ``generator_factory`` so the
orchestration logic can be unit-tested with a stub.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import soundfile as sf

from arrangement import ArrangementPlan, plan_arrangement
from vocal_analysis import VocalAnalysis, analyze_vocal
from vocal_mixer import mix_vocal_over_accompaniment, process_accompaniment

# Sample rate everything is mixed at (MusicGen's native rate).
MIX_SR = 32_000

# Bounds (env-tunable) that keep CPU generation time sane.
MAX_TOTAL_SECONDS = float(os.environ.get("ARK_VOCAL_MAX_SECONDS", "30"))
SEGMENT_SECONDS = float(os.environ.get("ARK_VOCAL_SEGMENT_SECONDS", "15"))
CROSSFADE_SECONDS = 0.25

ProgressCB = Callable[[int, str], None]


@dataclass
class VocalPipelineResult:
    analysis: VocalAnalysis
    plan: ArrangementPlan
    sample_rate: int
    mix: np.ndarray                    # (2, samples) vocal + accompaniment
    accompaniment: np.ndarray          # (2, samples) instrumental only
    segments: int = 1
    truncated: bool = False            # True if the vocal was longer than the cap
    warnings: list = field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────────────
# Audio helpers
# ──────────────────────────────────────────────────────────────────────────────

def _load_vocal(path: str, sr: int) -> np.ndarray:
    """Load a vocal file as (2, samples) float32 at ``sr``."""
    import librosa

    y, _ = librosa.load(path, sr=sr, mono=False)
    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 1:
        y = np.stack([y, y])
    elif y.shape[0] == 1:
        y = np.repeat(y, 2, axis=0)
    elif y.ndim == 2 and y.shape[0] > 2:      # (samples, ch) safety
        y = y.T if y.shape[1] <= 2 else np.stack([y.mean(0), y.mean(0)])
    return y


def _segment_bounds(total_sec: float, seg_sec: float) -> list[tuple[float, float]]:
    """Even-length windows covering [0, total_sec], each <= seg_sec."""
    if total_sec <= seg_sec:
        return [(0.0, total_sec)]
    n = int(np.ceil(total_sec / seg_sec))
    step = total_sec / n
    return [(i * step, min((i + 1) * step, total_sec)) for i in range(n)]


def _crossfade_concat(chunks: list[np.ndarray], sr: int, fade_sec: float) -> np.ndarray:
    """Equal-power crossfade a list of (2, samples) chunks into one array."""
    chunks = [c for c in chunks if c.shape[-1] > 0]
    if not chunks:
        return np.zeros((2, 0), dtype=np.float32)
    if len(chunks) == 1:
        return chunks[0]

    fade = int(sr * fade_sec)
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


def _resample_stereo(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return audio
    import librosa

    return np.stack([
        librosa.resample(audio[c], orig_sr=orig_sr, target_sr=target_sr)
        for c in range(audio.shape[0])
    ]).astype(np.float32)


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
                        temperature)`` method and a ``sample_rate`` attribute.
                        Defaults to a melody-conditioned :class:`MusicGenerator`.
    progress          : optional ``callback(percent:int, message:str)`` for UI.
    """
    def _p(pct: int, msg: str) -> None:
        if progress:
            progress(pct, msg)

    warnings_out: list[str] = []

    # ── 1. Analyse vocal ──────────────────────────────────────────────────────
    _p(6, "Analysing vocal performance…")
    analysis = analyze_vocal(vocal_path, max_seconds=max_total_seconds)

    # ── 2. Extract musical structure → arrangement plan ───────────────────────
    _p(18, "Extracting musical structure…")
    plan = plan_arrangement(analysis, overrides=overrides)

    # ── 3. Load vocal for mixing & decide segmentation ────────────────────────
    vocal = _load_vocal(vocal_path, MIX_SR)
    total_sec = vocal.shape[-1] / MIX_SR
    truncated = False
    if total_sec > max_total_seconds:
        vocal = vocal[:, : int(max_total_seconds * MIX_SR)]
        total_sec = max_total_seconds
        truncated = True
        warnings_out.append(
            f"Vocal longer than the {int(max_total_seconds)}s limit — using the first "
            f"{int(max_total_seconds)}s."
        )

    bounds = _segment_bounds(total_sec, segment_seconds)

    # ── 4. Generate accompaniment (segmented, melody-conditioned) ─────────────
    if generator_factory is None:
        def generator_factory():                      # pragma: no cover - heavy
            from generator import MusicGenerator
            return MusicGenerator(use_melody_model=True)

    gen = generator_factory()
    gen_sr = int(getattr(gen, "sample_rate", MIX_SR) or MIX_SR)

    acc_chunks: list[np.ndarray] = []
    n_seg = len(bounds)
    # The accompaniment step owns the 30–75 % band; give each segment an equal
    # slice of it and drive that slice from MusicGen's per-token progress.
    ACC_BASE, ACC_SPAN = 30, 45
    for i, (start, end) in enumerate(bounds):
        seg_len = max(end - start, 0.5)
        seg_base = ACC_BASE + ACC_SPAN * i / max(n_seg, 1)
        seg_slice = ACC_SPAN / max(n_seg, 1)
        _p(int(seg_base), f"Generating accompaniment {i + 1}/{n_seg}…")

        def _seg_progress(frac: float, _base=seg_base, _slice=seg_slice,
                          _idx=i) -> None:
            _p(
                int(_base + _slice * frac),
                f"Generating accompaniment {_idx + 1}/{n_seg}… {int(frac * 100)}%",
            )

        seg_vocal = vocal[:, int(start * MIX_SR): int(end * MIX_SR)]
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            seg_path = tmp.name
        try:
            sf.write(seg_path, seg_vocal.T, MIX_SR, format="WAV", subtype="FLOAT")
            audio, sr = gen.generate(
                prompt=plan.prompt,
                melody_path=seg_path,
                duration=seg_len,
                guidance_scale=plan.guidance_scale,
                temperature=plan.temperature,
                progress_cb=_seg_progress,
            )
        finally:
            try:
                os.unlink(seg_path)
            except OSError:
                pass

        chunk = np.asarray(audio, dtype=np.float32)
        if chunk.ndim == 1:
            chunk = np.stack([chunk, chunk])
        elif chunk.shape[0] == 1:
            chunk = np.repeat(chunk, 2, axis=0)
        if sr != MIX_SR:
            chunk = _resample_stereo(chunk, sr, MIX_SR)
        acc_chunks.append(chunk)

    accompaniment_raw = _crossfade_concat(acc_chunks, MIX_SR, CROSSFADE_SECONDS)

    # ── 5. Vocal-aware processing of the accompaniment ────────────────────────
    _p(80, "Shaping accompaniment around the vocal…")
    accompaniment = process_accompaniment(
        accompaniment_raw, MIX_SR,
        genre=plan.genre, mood=plan.mood,
        crescendo_pattern=plan.crescendo,
        vocal_center_hz=plan.vocal_center_hz,
    )

    # ── 6. Synchronise & mix vocal over accompaniment ─────────────────────────
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
    )


__all__ = ["run_vocal_to_music", "VocalPipelineResult",
           "MAX_TOTAL_SECONDS", "SEGMENT_SECONDS"]
