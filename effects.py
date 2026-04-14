"""
Audio post-processing pipeline for the music generator.

Applies a chain of perceptual effects optimised for sing-along backing
tracks:  compression → EQ → beat-transient boost → crescendo envelope
→ tremolo modulation → subtle reverb → stereo widening → normalisation.

All functions accept/return float32 numpy arrays.
Stereo arrays have shape (2, samples); mono arrays have shape (samples,).
"""

from __future__ import annotations

import numpy as np
import scipy.signal as sig
import warnings

try:
    import librosa
    _HAS_LIBROSA = True
except ImportError:
    _HAS_LIBROSA = False

# ──────────────────────────────────────────────────────────────────────────────
# Genre / mood → EQ preset
# ──────────────────────────────────────────────────────────────────────────────

_GENRE_EQ: dict[str, tuple[float, float, float]] = {
    # (bass_db, mid_db, treble_db)
    "pop":        (2.0,  0.0,  2.5),
    "rock":       (3.0,  1.0,  2.0),
    "jazz":       (1.0,  1.5,  1.0),
    "classical":  (0.0,  0.5,  2.0),
    "folk":       (1.0,  1.0,  1.5),
    "electronic": (5.0, -1.0,  3.0),
    "hip-hop":    (6.0, -1.0,  1.0),
    "r-and-b":    (3.0,  1.0,  1.5),
    "ambient":    (2.0,  0.0,  1.0),
    "reggae":     (4.0, -0.5,  1.0),
    "bossa nova": (1.0,  1.5,  2.0),
}

_MOOD_EQ: dict[str, tuple[float, float, float]] = {
    "happy":     (1.0,  0.0,  2.0),
    "sad":       (1.5, -0.5, -1.0),
    "energetic": (2.0,  0.0,  2.5),
    "calm":      (0.5, -0.5, -1.5),
    "romantic":  (1.0,  0.5,  0.5),
    "uplifting": (1.5,  0.0,  2.0),
    "mysterious":(-1.0, 1.0, -1.0),
    "aggressive":(4.0,  1.0,  2.0),
    "nostalgic": (1.0,  1.5, -0.5),
    "playful":   (1.0, -0.5,  2.5),
}

# ──────────────────────────────────────────────────────────────────────────────
# Individual effect implementations
# ──────────────────────────────────────────────────────────────────────────────

def _to_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 2:
        return audio.mean(axis=0)
    return audio


def apply_compression(
    audio: np.ndarray,
    threshold_db: float = -14.0,
    ratio: float = 4.0,
    makeup_db: float = 2.0,
) -> np.ndarray:
    """Soft-knee downward compressor with automatic make-up gain."""
    threshold_lin = 10 ** (threshold_db / 20.0)
    makeup = 10 ** (makeup_db / 20.0)

    abs_audio = np.abs(audio)
    gain = np.ones_like(abs_audio)

    over = abs_audio > threshold_lin
    excess = abs_audio[over] - threshold_lin
    compressed_excess = excess / ratio
    gain[over] = (threshold_lin + compressed_excess) / (abs_audio[over] + 1e-9)

    return np.clip(audio * gain * makeup, -1.0, 1.0)


def apply_eq(
    audio: np.ndarray,
    sr: int,
    bass_db: float = 0.0,
    mid_db: float = 0.0,
    treble_db: float = 0.0,
) -> np.ndarray:
    """Three-band shelving / peak EQ."""
    out = audio.astype(np.float64)
    nyq = sr / 2.0

    def _shelf_low(y: np.ndarray, cutoff: float, gain_db: float) -> np.ndarray:
        if abs(gain_db) < 0.1:
            return y
        b, a = sig.butter(2, cutoff / nyq, btype="low")
        band = sig.filtfilt(b, a, y)
        g = 10 ** (gain_db / 20.0) - 1.0
        return y + band * g

    def _peak(y: np.ndarray, centre: float, width: float, gain_db: float) -> np.ndarray:
        if abs(gain_db) < 0.1:
            return y
        low_f  = max(centre - width / 2, 20) / nyq
        high_f = min(centre + width / 2, nyq - 1) / nyq
        if low_f >= high_f or high_f >= 1.0:
            return y
        b, a = sig.butter(2, [low_f, high_f], btype="band")
        band = sig.filtfilt(b, a, y)
        g = 10 ** (gain_db / 20.0) - 1.0
        return y + band * g

    def _shelf_high(y: np.ndarray, cutoff: float, gain_db: float) -> np.ndarray:
        if abs(gain_db) < 0.1:
            return y
        b, a = sig.butter(2, cutoff / nyq, btype="high")
        band = sig.filtfilt(b, a, y)
        g = 10 ** (gain_db / 20.0) - 1.0
        return y + band * g

    if out.ndim == 2:
        for ch in range(out.shape[0]):
            out[ch] = _shelf_low(out[ch],  250.0, bass_db)
            out[ch] = _peak     (out[ch], 1500.0, 1800.0, mid_db)
            out[ch] = _shelf_high(out[ch], 5000.0, treble_db)
    else:
        out = _shelf_low (out,  250.0, bass_db)
        out = _peak      (out, 1500.0, 1800.0, mid_db)
        out = _shelf_high(out, 5000.0, treble_db)

    return np.clip(out, -1.5, 1.5).astype(np.float32)


def enhance_beat(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    Detect rhythmic onsets and apply a brief transient boost at each beat
    to make the track feel punchier and more drive-able for singing along.
    """
    if not _HAS_LIBROSA:
        return audio  # skip gracefully

    mono = _to_mono(audio).astype(np.float32)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            onset_env = librosa.onset.onset_strength(y=mono, sr=sr)
            _, beat_frames = librosa.beat.beat_track(
                onset_envelope=onset_env, sr=sr, units="frames"
            )
            beat_samples = librosa.frames_to_samples(beat_frames)
        except Exception:
            return audio

    win_len = max(int(sr * 0.018), 1)   # ~18 ms boost window
    boost   = 0.20                        # +20 % amplitude at each beat

    enhanced = audio.copy()

    def _boost_range(arr: np.ndarray) -> np.ndarray:
        for bs in beat_samples:
            end = min(int(bs) + win_len, arr.shape[-1])
            if end <= int(bs):
                continue
            window = np.hanning(end - int(bs)).astype(np.float32)
            if arr.ndim == 2:
                arr[:, int(bs):end] *= (1.0 + boost * window)
            else:
                arr[int(bs):end] *= (1.0 + boost * window)
        return arr

    enhanced = _boost_range(enhanced)
    return np.clip(enhanced, -1.0, 1.0)


def apply_crescendo(
    audio: np.ndarray,
    sr: int,
    pattern: str = "rise-fall",
) -> np.ndarray:
    """
    Impose a volume envelope over the track.

    Patterns
    --------
    rise       – starts at 50 %, ends at 100 %  (builds excitement)
    fall       – starts at 100 %, ends at 60 %  (fading outro)
    rise-fall  – rises to peak at 65 % then gently falls  (natural arc)
    natural    – S-curve ramp (smooth intro → sustain → gentle fade)
    verse-chorus – two-wave pattern mimicking verse/chorus dynamics
    """
    n = audio.shape[-1]
    t = np.linspace(0.0, 1.0, n, dtype=np.float32)

    if pattern == "rise":
        env = 0.50 + 0.50 * t

    elif pattern == "fall":
        env = 1.00 - 0.35 * t

    elif pattern == "rise-fall":
        peak_pos = 0.65
        rise = np.linspace(0.45, 1.0, int(n * peak_pos))
        fall = np.linspace(1.0,  0.72, n - int(n * peak_pos))
        env  = np.concatenate([rise, fall])

    elif pattern == "verse-chorus":
        # Three gentle waves simulating verse → chorus dynamic
        env = 0.70 + 0.28 * np.sin(2 * np.pi * 1.5 * t) ** 2
        # Extra boost in the second half
        env[n // 2:] *= 1.08

    else:  # "natural" – S-curve
        x   = np.linspace(-4.0, 4.0, n)
        sig_ = 1.0 / (1.0 + np.exp(-x))
        env  = 0.55 + 0.45 * (sig_ - sig_.min()) / (sig_.max() - sig_.min() + 1e-9)

    env = env[:n].astype(np.float32)

    if audio.ndim == 2:
        return audio * env[np.newaxis, :]
    return audio * env


def apply_tremolo(
    audio: np.ndarray,
    sr: int,
    rate_hz: float = 4.5,
    depth: float = 0.18,
) -> np.ndarray:
    """
    Amplitude modulation (tremolo) using a smooth LFO.
    Adds rhythmic expressiveness without changing pitch.
    """
    n = audio.shape[-1]
    t   = np.arange(n, dtype=np.float32) / sr
    lfo = (1.0 - depth) + depth * np.sin(2.0 * np.pi * rate_hz * t)

    if audio.ndim == 2:
        return audio * lfo[np.newaxis, :]
    return audio * lfo


def apply_reverb(
    audio: np.ndarray,
    sr: int,
    room_size: float = 0.25,
    wet: float = 0.12,
) -> np.ndarray:
    """
    Lightweight convolution reverb using an exponentially-decaying
    white-noise impulse response (Schroeder style).
    Keeps the mix transparent while adding a sense of acoustic space.
    """
    ir_len = max(int(sr * room_size), 64)
    rng    = np.random.default_rng(seed=42)          # reproducible IR

    t  = np.arange(ir_len, dtype=np.float64) / sr
    ir = np.exp(-6.0 * t) * rng.standard_normal(ir_len) * 0.08
    ir[0] = 1.0                                       # dry tap
    ir = (ir / (np.max(np.abs(ir)) + 1e-9)).astype(np.float32)

    def _convolve(channel: np.ndarray) -> np.ndarray:
        reverbed = sig.fftconvolve(channel.astype(np.float64), ir.astype(np.float64))
        reverbed = reverbed[:len(channel)].astype(np.float32)
        return channel * (1.0 - wet) + reverbed * wet

    if audio.ndim == 2:
        return np.stack([_convolve(audio[0]), _convolve(audio[1])])
    return _convolve(audio)


def apply_stereo_width(audio: np.ndarray, width: float = 1.15) -> np.ndarray:
    """
    Mid/side stereo widening.  width > 1 expands the stereo image,
    width < 1 narrows it.  Mono audio is returned unchanged.
    """
    if audio.ndim != 2 or audio.shape[0] != 2:
        return audio

    mid  = (audio[0] + audio[1]) * 0.5
    side = (audio[0] - audio[1]) * 0.5 * width

    left  = np.clip(mid + side, -1.0, 1.0)
    right = np.clip(mid - side, -1.0, 1.0)
    return np.stack([left, right])


def normalise(audio: np.ndarray, target_db: float = -1.0) -> np.ndarray:
    """Peak-normalise to target_db (default -1 dBFS)."""
    peak = np.max(np.abs(audio))
    if peak < 1e-6:
        return audio
    target_lin = 10 ** (target_db / 20.0)
    return audio * (target_lin / peak)


# ──────────────────────────────────────────────────────────────────────────────
# Main processing chain
# ──────────────────────────────────────────────────────────────────────────────

def process_audio(
    audio: np.ndarray,
    sr: int,
    genre: str | None = None,
    mood: str | None = None,
    crescendo_pattern: str = "rise-fall",
    tremolo_rate: float = 4.5,
    reverb_room: float = 0.25,
    stereo_width: float = 1.15,
) -> np.ndarray:
    """
    Full post-processing chain tailored for sing-along backing tracks.

    Order of operations
    -------------------
    1. Compression   – tame peaks, add punch
    2. EQ            – genre + mood frequency shaping
    3. Beat enhance  – transient boost at detected beats
    4. Crescendo     – dynamic volume arc across the track
    5. Tremolo       – rhythmic amplitude modulation
    6. Reverb        – acoustic space
    7. Stereo width  – immersive image
    8. Normalise     – final loudness to -1 dBFS
    """
    audio = audio.astype(np.float32)

    # ── 1. Compression ────────────────────────────────────────────────────────
    audio = apply_compression(audio, threshold_db=-14.0, ratio=4.0, makeup_db=2.0)

    # ── 2. EQ ─────────────────────────────────────────────────────────────────
    bass_db, mid_db, treble_db = _resolve_eq(genre, mood)
    audio = apply_eq(audio, sr, bass_db=bass_db, mid_db=mid_db, treble_db=treble_db)

    # ── 3. Beat enhancement ───────────────────────────────────────────────────
    audio = enhance_beat(audio, sr)

    # ── 4. Crescendo envelope ─────────────────────────────────────────────────
    audio = apply_crescendo(audio, sr, pattern=crescendo_pattern)

    # ── 5. Tremolo modulation ─────────────────────────────────────────────────
    audio = apply_tremolo(audio, sr, rate_hz=tremolo_rate, depth=0.12)

    # ── 6. Reverb (subtle) ────────────────────────────────────────────────────
    audio = apply_reverb(audio, sr, room_size=reverb_room, wet=0.12)

    # ── 7. Stereo widening ────────────────────────────────────────────────────
    audio = apply_stereo_width(audio, width=stereo_width)

    # ── 8. Normalise ──────────────────────────────────────────────────────────
    audio = normalise(audio, target_db=-1.0)

    return audio.astype(np.float32)


def _resolve_eq(genre: str | None, mood: str | None) -> tuple[float, float, float]:
    bass, mid, treble = 0.0, 0.0, 0.0

    if genre:
        gb, gm, gt = _GENRE_EQ.get(genre.lower(), (0.0, 0.0, 0.0))
        bass += gb; mid += gm; treble += gt

    if mood:
        mb, mm, mt = _MOOD_EQ.get(mood.lower(), (0.0, 0.0, 0.0))
        bass += mb; mid += mm; treble += mt

    # Clamp to reasonable range
    return (
        float(np.clip(bass,   -6, 9)),
        float(np.clip(mid,    -4, 6)),
        float(np.clip(treble, -6, 8)),
    )
