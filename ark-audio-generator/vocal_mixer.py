"""
Vocal-aware mixing & synchronisation for the vocal-to-music pipeline.

Once MusicGen has produced an accompaniment for the analysed vocal, this module:

1. **Processes the accompaniment** with a lighter, vocal-friendly effects chain
   (reusing the primitives in :mod:`effects`) instead of the full sing-along
   chain, so the backing sits *behind* the singer.
2. **Carves space for the vocal** — a gentle EQ dip around the vocal fundamental
   and the presence band, so instruments do not unnecessarily mask the voice.
3. **Ducks the accompaniment** dynamically wherever the vocal is loud
   (sidechain-style), keeping the vocal upfront.
4. **Synchronises & mixes** — trims/pads the accompaniment to the exact length of
   the original vocal so the two stay aligned start-to-finish, then sums them and
   normalises.  The original vocal is preserved unchanged.

All arrays are float32.  Stereo has shape ``(2, samples)``; mono ``(samples,)``.
"""

from __future__ import annotations

import numpy as np
import scipy.signal as sig

import effects


# ──────────────────────────────────────────────────────────────────────────────
# Channel helpers
# ──────────────────────────────────────────────────────────────────────────────

def _to_stereo(audio: np.ndarray) -> np.ndarray:
    """Return a (2, samples) float32 array from mono or stereo input."""
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 1:
        return np.stack([audio, audio])
    if audio.shape[0] == 1:
        return np.repeat(audio, 2, axis=0)
    if audio.shape[0] == 2:
        return audio
    # (samples, 2) → (2, samples)
    if audio.ndim == 2 and audio.shape[1] == 2:
        return audio.T
    return np.stack([audio.mean(axis=0), audio.mean(axis=0)])


def _fit_length(audio: np.ndarray, n: int) -> np.ndarray:
    """Trim or zero-pad a (2, samples) array to exactly ``n`` samples."""
    cur = audio.shape[-1]
    if cur == n:
        return audio
    if cur > n:
        return audio[:, :n]
    pad = np.zeros((audio.shape[0], n - cur), dtype=audio.dtype)
    return np.concatenate([audio, pad], axis=1)


# ──────────────────────────────────────────────────────────────────────────────
# Vocal-space carving
# ──────────────────────────────────────────────────────────────────────────────

def carve_vocal_band(
    accompaniment: np.ndarray,
    sr: int,
    center_hz: float = 220.0,
    fundamental_cut_db: float = -3.0,
    presence_cut_db: float = -3.5,
) -> np.ndarray:
    """
    Apply two gentle EQ dips so the accompaniment leaves room for the vocal:

    * a narrow dip around the vocal *fundamental* (``center_hz``)
    * a wider dip across the vocal *presence / intelligibility* band (2–4 kHz)

    Cuts are intentionally modest — enough to unmask the voice without hollowing
    out the backing track.
    """
    out = np.asarray(accompaniment, dtype=np.float64)
    nyq = sr / 2.0

    def _notch(y: np.ndarray, low_hz: float, high_hz: float, gain_db: float) -> np.ndarray:
        if abs(gain_db) < 0.1:
            return y
        low = max(low_hz, 20.0) / nyq
        high = min(high_hz, nyq - 1.0) / nyq
        if low >= high or high >= 1.0:
            return y
        b, a = sig.butter(2, [low, high], btype="band")
        band = sig.filtfilt(b, a, y)
        g = 10 ** (gain_db / 20.0) - 1.0        # negative → subtractive
        return y + band * g

    center_hz = float(np.clip(center_hz or 220.0, 80.0, 600.0))
    fund_lo, fund_hi = center_hz * 0.8, center_hz * 1.25

    def _process(ch: np.ndarray) -> np.ndarray:
        ch = _notch(ch, fund_lo, fund_hi, fundamental_cut_db)
        ch = _notch(ch, 2000.0, 4000.0, presence_cut_db)
        return ch

    if out.ndim == 2:
        for c in range(out.shape[0]):
            out[c] = _process(out[c])
    else:
        out = _process(out)

    return np.clip(out, -1.5, 1.5).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Sidechain ducking
# ──────────────────────────────────────────────────────────────────────────────

def _vocal_gain_envelope(
    vocal_mono: np.ndarray,
    sr: int,
    duck_depth_db: float,
    attack_ms: float = 40.0,
    release_ms: float = 240.0,
) -> np.ndarray:
    """
    Build a per-sample gain curve for the accompaniment that dips when the vocal
    is loud.  Returns a 1-D array in [10**(-depth/20), 1.0].
    """
    # Short-time RMS of the vocal, smoothed into a control signal.
    frame = max(int(sr * 0.02), 1)
    energy = np.sqrt(
        np.convolve(vocal_mono.astype(np.float64) ** 2,
                    np.ones(frame) / frame, mode="same")
    )
    peak = float(energy.max())
    if peak < 1e-6:
        return np.ones_like(vocal_mono, dtype=np.float32)

    activity = np.clip(energy / peak, 0.0, 1.0)

    # Asymmetric one-pole smoothing (fast attack, slow release) for natural duck.
    atk = np.exp(-1.0 / max(sr * attack_ms / 1000.0, 1.0))
    rel = np.exp(-1.0 / max(sr * release_ms / 1000.0, 1.0))
    smoothed = np.empty_like(activity)
    prev = 0.0
    for i, x in enumerate(activity):
        coef = atk if x > prev else rel
        prev = coef * prev + (1.0 - coef) * x
        smoothed[i] = prev

    depth_lin = 10 ** (-abs(duck_depth_db) / 20.0)
    gain = 1.0 - smoothed * (1.0 - depth_lin)
    return gain.astype(np.float32)


def sidechain_duck(
    accompaniment: np.ndarray,
    vocal_mono: np.ndarray,
    sr: int,
    duck_depth_db: float = 6.0,
) -> np.ndarray:
    """Duck the (2, samples) accompaniment under the vocal energy envelope."""
    gain = _vocal_gain_envelope(vocal_mono, sr, duck_depth_db)
    gain = gain[: accompaniment.shape[-1]]
    if gain.shape[-1] < accompaniment.shape[-1]:
        gain = np.pad(gain, (0, accompaniment.shape[-1] - gain.shape[-1]),
                      constant_values=1.0)
    return (accompaniment * gain[np.newaxis, :]).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Accompaniment post-processing (vocal-friendly, reuses effects primitives)
# ──────────────────────────────────────────────────────────────────────────────

def process_accompaniment(
    audio: np.ndarray,
    sr: int,
    genre: str | None = None,
    mood: str | None = None,
    crescendo_pattern: str = "natural",
    vocal_center_hz: float = 220.0,
) -> np.ndarray:
    """
    Lighter cousin of :func:`effects.process_audio`, tuned for backing that sits
    behind a lead vocal.  Reuses the compressor, EQ, reverb, stereo-widener and
    normaliser from :mod:`effects`, skips the aggressive tremolo, applies a
    gentle crescendo, and carves space for the vocal.
    """
    audio = np.asarray(audio, dtype=np.float32)

    # Gentle glue compression.
    audio = effects.apply_compression(audio, threshold_db=-16.0, ratio=3.0, makeup_db=1.5)

    # Genre/mood EQ (shared table), but softened for a supporting role.
    bass_db, mid_db, treble_db = effects._resolve_eq(genre, mood)
    audio = effects.apply_eq(audio, sr, bass_db=bass_db * 0.6,
                             mid_db=mid_db * 0.5, treble_db=treble_db * 0.6)

    # Carve space for the vocal.
    audio = carve_vocal_band(audio, sr, center_hz=vocal_center_hz)

    # Subtle dynamic arc (no tremolo — it fights a real vocal).
    audio = effects.apply_crescendo(audio, sr, pattern=crescendo_pattern)

    # Light room + width.
    audio = effects.apply_reverb(audio, sr, room_size=0.22, wet=0.10)
    audio = effects.apply_stereo_width(audio, width=1.10)

    audio = effects.normalise(audio, target_db=-3.0)
    return audio.astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Level matching
# ──────────────────────────────────────────────────────────────────────────────

# Phone/laptop vocal recordings arrive at wildly different levels (often very
# quiet), whereas MusicGen output is always peak-normalised.  Summing them as-is
# buries the singer.  So the vocal is normalised to a healthy RMS over the parts
# where it is actually singing, and the backing is set *relative* to that.
VOCAL_TARGET_RMS_DB = -18.0
BACKING_OFFSET_DB = -3.0          # backing RMS relative to the vocal, before ducking


def _voiced_mask(vocal_mono: np.ndarray, sr: int, rel_threshold: float = 0.1) -> np.ndarray:
    """Boolean per-sample mask of where the vocal is audibly present."""
    frame = max(int(sr * 0.05), 1)
    env = np.sqrt(np.convolve(vocal_mono.astype(np.float64) ** 2,
                              np.ones(frame) / frame, mode="same"))
    peak = float(env.max())
    if peak < 1e-6:
        return np.zeros_like(vocal_mono, dtype=bool)
    return env > peak * rel_threshold


def rms_db(x: np.ndarray, mask: np.ndarray | None = None) -> float:
    """RMS level in dBFS of a mono/stereo array, optionally over a sample mask."""
    x = np.asarray(x, dtype=np.float64)
    mono = x.mean(axis=0) if x.ndim == 2 else x
    if mask is not None and mask.any():
        mono = mono[mask[: mono.shape[-1]]]
    if mono.size == 0:
        return -120.0
    r = float(np.sqrt(np.mean(mono ** 2)))
    return 20.0 * np.log10(max(r, 1e-9))


def match_levels(
    vocal: np.ndarray,
    accompaniment: np.ndarray,
    sr: int,
    vocal_target_db: float = VOCAL_TARGET_RMS_DB,
    backing_offset_db: float = BACKING_OFFSET_DB,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Return ``(vocal, accompaniment, info)`` with the vocal normalised to
    ``vocal_target_db`` RMS over its voiced parts and the accompaniment set to
    ``vocal_target_db + backing_offset_db`` RMS over those same parts.
    """
    voc = _to_stereo(vocal)
    acc = _to_stereo(accompaniment)
    mask = _voiced_mask(voc.mean(axis=0), sr)

    v_db = rms_db(voc, mask)
    v_gain_db = float(np.clip(vocal_target_db - v_db, -24.0, 40.0)) if v_db > -100 else 0.0
    voc = voc * (10 ** (v_gain_db / 20.0))

    a_db = rms_db(acc, mask if mask.any() else None)
    a_target = vocal_target_db + backing_offset_db
    a_gain_db = float(np.clip(a_target - a_db, -40.0, 24.0)) if a_db > -100 else 0.0
    acc = acc * (10 ** (a_gain_db / 20.0))

    info = {
        "vocal_rms_db_in": round(v_db, 1), "vocal_gain_db": round(v_gain_db, 1),
        "backing_rms_db_in": round(a_db, 1), "backing_gain_db": round(a_gain_db, 1),
        "voiced_ratio": round(float(mask.mean()) if mask.size else 0.0, 3),
    }
    return voc.astype(np.float32), acc.astype(np.float32), info


# ──────────────────────────────────────────────────────────────────────────────
# Synchronise & mix
# ──────────────────────────────────────────────────────────────────────────────

def mix_vocal_over_accompaniment(
    vocal: np.ndarray,
    accompaniment: np.ndarray,
    sr: int,
    accompaniment_gain_db: float = 0.0,
    vocal_gain_db: float = 0.0,
    duck_depth_db: float = 6.0,
    level_match: bool = True,
) -> np.ndarray:
    """
    Synchronise and mix the original vocal on top of the accompaniment.

    The accompaniment is trimmed/padded to the exact length of the vocal so the
    two stay aligned from start to finish.  With ``level_match`` (default) the
    vocal is normalised and the backing is placed a few dB under it before the
    sidechain duck; the two ``*_gain_db`` arguments are trims on top of that.
    The vocal's *performance* is untouched — only its overall level changes.
    Returns a (2, samples) mix peaking below 0 dBFS.
    """
    voc = _to_stereo(vocal)
    acc = _to_stereo(accompaniment)

    n = voc.shape[-1]
    acc = _fit_length(acc, n)

    if level_match:
        voc, acc, _ = match_levels(voc, acc, sr)

    vocal_mono = voc.mean(axis=0)
    acc = sidechain_duck(acc, vocal_mono, sr, duck_depth_db=duck_depth_db)

    acc_gain = 10 ** (accompaniment_gain_db / 20.0)
    voc_gain = 10 ** (vocal_gain_db / 20.0)

    mix = acc * acc_gain + voc * voc_gain

    # Prevent clipping while preserving relative balance.
    peak = float(np.max(np.abs(mix)))
    if peak > 0.97:
        mix = mix * (0.97 / peak)
    return mix.astype(np.float32)


__all__ = [
    "carve_vocal_band",
    "sidechain_duck",
    "process_accompaniment",
    "match_levels",
    "rms_db",
    "mix_vocal_over_accompaniment",
]
