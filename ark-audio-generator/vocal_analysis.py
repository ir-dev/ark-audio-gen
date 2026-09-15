"""
Vocal performance analysis for the vocal-to-music pipeline.

Given a vocal-only recording (humming, singing, or improvised vocals) this
module extracts the *musical* characteristics of the performance so that a
complementary accompaniment can be arranged around it — rather than generating
generic background music.

Everything here is pure analysis (librosa + numpy + scipy).  No AI model is
loaded, so it is fast enough to run interactively (the ``/api/vocal/analyze``
endpoint) as well as inside the full generation job.

Detected characteristics
------------------------
* tempo (BPM) and a naive time-signature guess
* musical key + mode (major / minor) via Krumhansl–Schmuckler key finding
* pitch range, median pitch and register (from a pYIN fundamental contour)
* pitch movement / melodic contour summary
* phrasing — voiced segments and the pauses between them
* dynamics — loudness envelope and a suggested crescendo arc
* derived musical suggestions — genre, mood, supportive instruments and a
  diatonic chord progression that fits the detected key/mode

The result is a :class:`VocalAnalysis` dataclass whose :meth:`to_dict` output is
JSON-serialisable for the API/UI and consumable by :mod:`arrangement`.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np

try:
    import librosa
    _HAS_LIBROSA = True
except ImportError:  # pragma: no cover - librosa is a hard dependency in prod
    _HAS_LIBROSA = False


# ──────────────────────────────────────────────────────────────────────────────
# Music-theory constants
# ──────────────────────────────────────────────────────────────────────────────

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Krumhansl–Schmuckler key profiles (major / minor), indexed from the tonic.
_KS_MAJOR = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
)
_KS_MINOR = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
)

# Scale interval patterns (semitones from tonic).
_MAJOR_STEPS = [0, 2, 4, 5, 7, 9, 11]
_MINOR_STEPS = [0, 2, 3, 5, 7, 8, 10]  # natural minor

# Diatonic triad qualities per scale degree.
#   "" = major, "m" = minor, "dim" = diminished
_MAJOR_QUALITIES = ["", "m", "m", "", "", "m", "dim"]
_MINOR_QUALITIES = ["m", "dim", "", "m", "m", "", ""]

# Default diatonic progressions (0-based scale degrees) keyed by mood family.
# Extensible — future versions can add mood/genre-specific templates.
_MAJOR_PROGRESSIONS = {
    "default":   [0, 4, 5, 3],   # I  – V  – vi – IV  (pop workhorse)
    "uplifting": [0, 3, 4, 4],   # I  – IV – V  – V
    "calm":      [0, 5, 3, 4],   # I  – vi – IV – V
    "playful":   [0, 3, 0, 4],   # I  – IV – I  – V
}
_MINOR_PROGRESSIONS = {
    "default":   [0, 5, 2, 6],   # i  – VI – III – VII
    "sad":       [0, 3, 5, 4],   # i  – iv – VI  – v
    "mysterious":[0, 6, 5, 4],   # i  – VII– VI  – v
    "aggressive":[0, 4, 5, 0],   # i  – v  – VI  – i
}


# ──────────────────────────────────────────────────────────────────────────────
# Result container
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class VocalAnalysis:
    """Musical description of a vocal performance."""

    # Timing
    duration_sec: float = 0.0
    analysed_sec: float = 0.0          # portion actually analysed (may be capped)
    active_start_sec: float = 0.0      # where the singing/humming starts (within analysed part)
    active_end_sec: float = 0.0        # where it ends
    tempo_bpm: float = 100.0
    tempo_raw_bpm: float = 0.0         # tracker output before octave folding
    time_signature: str = "4/4"

    # Tonality
    key: str = "C"
    mode: str = "major"                # "major" | "minor"
    key_name: str = "C major"
    key_confidence: float = 0.0        # 0–1 correlation strength

    # Pitch
    pitch_min_note: str = ""
    pitch_max_note: str = ""
    pitch_median_note: str = ""
    pitch_median_hz: float = 0.0
    pitch_range_semitones: float = 0.0
    register: str = "mid"              # low | low-mid | mid | mid-high | high
    voiced_ratio: float = 0.0          # fraction of frames that are voiced

    # Melodic movement / phrasing
    contour: str = "steady"           # rising | falling | arching | wavy | steady
    note_density: float = 0.0          # onsets per second
    phrase_count: int = 0
    avg_phrase_sec: float = 0.0
    phrases: list = field(default_factory=list)   # [{"start":s,"end":e}, ...]

    # Dynamics
    dynamics: str = "steady"          # soft | steady | building | dynamic
    suggested_crescendo: str = "natural"

    # Derived musical suggestions
    suggested_genre: str = "pop"
    suggested_mood: str = "uplifting"
    suggested_instruments: list = field(default_factory=list)
    chord_progression: list = field(default_factory=list)      # ["C","G","Am","F"]
    chord_progression_roman: list = field(default_factory=list)
    melody_summary: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ──────────────────────────────────────────────────────────────────────────────
# Low-level helpers
# ──────────────────────────────────────────────────────────────────────────────

def _load_mono(source, sr: int, max_seconds: Optional[float]) -> tuple[np.ndarray, int, float]:
    """
    Load audio as mono at ``sr``.

    ``source`` may be a filesystem path or a pre-loaded (audio, sr) tuple / array.
    Returns ``(y, sr, full_duration_sec)`` where ``y`` is trimmed to
    ``max_seconds`` but ``full_duration_sec`` reflects the untrimmed length.
    """
    if isinstance(source, tuple):
        y, in_sr = source
        y = np.asarray(y, dtype=np.float32)
        if y.ndim == 2:
            y = y.mean(axis=0)
        if in_sr != sr:
            y = librosa.resample(y, orig_sr=in_sr, target_sr=sr)
    else:
        y, _ = librosa.load(str(source), sr=sr, mono=True)

    full_duration = float(len(y) / sr) if len(y) else 0.0

    if max_seconds is not None and full_duration > max_seconds:
        y = y[: int(max_seconds * sr)]

    return y.astype(np.float32), sr, full_duration


# Sung material almost always sits in this perceived-tempo window.  Beat
# trackers fed a *solo voice* (no drums, sparse onsets) very often land on a
# double- or half-time multiple of the real pulse — e.g. 152 BPM for a 76 BPM
# ballad — so anything outside the window is folded back by octaves.
TEMPO_MIN_BPM = 60.0
TEMPO_MAX_BPM = 140.0


def fold_tempo(bpm: float, lo: float = TEMPO_MIN_BPM, hi: float = TEMPO_MAX_BPM) -> float:
    """Fold double-/half-time tracker errors into a plausible sung-tempo range."""
    try:
        bpm = float(bpm)
    except (TypeError, ValueError):
        return 100.0
    if not np.isfinite(bpm) or bpm <= 0:
        return 100.0
    while bpm > hi:
        bpm /= 2.0
    while bpm < lo:
        bpm *= 2.0
    return bpm


def detect_active_region(
    y: np.ndarray,
    sr: int,
    top_db: float = 30.0,
    min_sec: float = 0.25,
) -> tuple[float, float]:
    """
    Return ``(start_sec, end_sec)`` of the part of a mono signal that actually
    contains singing/humming — i.e. from the first to the last non-silent
    stretch at least ``min_sec`` long.  Falls back to the whole signal when
    nothing is detected.
    """
    y = np.asarray(y, dtype=np.float32)
    if y.size == 0:
        return 0.0, 0.0
    total = len(y) / sr
    if float(np.max(np.abs(y))) < 1e-4:
        return 0.0, total
    try:
        intervals = librosa.effects.split(y, top_db=top_db)
    except Exception:
        return 0.0, total
    keep = [(s, e) for s, e in intervals if (e - s) / sr >= min_sec]
    if not keep:
        return 0.0, total
    return float(keep[0][0] / sr), float(keep[-1][1] / sr)


def _detect_key(chroma_mean: np.ndarray) -> tuple[str, str, float]:
    """Krumhansl–Schmuckler key finding from a mean chroma vector."""
    chroma_mean = np.asarray(chroma_mean, dtype=np.float64)
    if chroma_mean.sum() <= 0:
        return "C", "major", 0.0

    def _corr(a: np.ndarray, b: np.ndarray) -> float:
        a = a - a.mean()
        b = b - b.mean()
        denom = np.sqrt((a * a).sum() * (b * b).sum())
        return float((a * b).sum() / denom) if denom > 1e-9 else 0.0

    best = ("C", "major", -2.0)
    for tonic in range(12):
        maj = _corr(chroma_mean, np.roll(_KS_MAJOR, tonic))
        if maj > best[2]:
            best = (NOTE_NAMES[tonic], "major", maj)
        minr = _corr(chroma_mean, np.roll(_KS_MINOR, tonic))
        if minr > best[2]:
            best = (NOTE_NAMES[tonic], "minor", minr)

    key, mode, corr = best
    # Map correlation (~ -1..1) to a friendlier 0..1 confidence.
    confidence = float(np.clip((corr + 1.0) / 2.0, 0.0, 1.0))
    return key, mode, confidence


def _chord_progression(key: str, mode: str, mood: str) -> tuple[list, list]:
    """Return (chord_names, roman_numerals) for a diatonic progression."""
    root_pc = NOTE_NAMES.index(key) if key in NOTE_NAMES else 0
    steps = _MAJOR_STEPS if mode == "major" else _MINOR_STEPS
    qualities = _MAJOR_QUALITIES if mode == "major" else _MINOR_QUALITIES

    table = _MAJOR_PROGRESSIONS if mode == "major" else _MINOR_PROGRESSIONS
    degrees = table.get(mood, table["default"])

    roman_major = ["I", "ii", "iii", "IV", "V", "vi", "vii°"]
    roman_minor = ["i", "ii°", "III", "iv", "v", "VI", "VII"]
    roman_set = roman_major if mode == "major" else roman_minor

    names, romans = [], []
    for deg in degrees:
        pc = (root_pc + steps[deg]) % 12
        suffix = qualities[deg]
        names.append(NOTE_NAMES[pc] + suffix)
        romans.append(roman_set[deg])
    return names, romans


def _register_for_hz(median_hz: float) -> str:
    if median_hz <= 0:
        return "mid"
    if median_hz < 160:
        return "low"
    if median_hz < 220:
        return "low-mid"
    if median_hz < 330:
        return "mid"
    if median_hz < 500:
        return "mid-high"
    return "high"


def _describe_contour(f0: np.ndarray) -> str:
    """Summarise overall melodic movement from a voiced-only pitch track."""
    f0 = f0[np.isfinite(f0)]
    if f0.size < 4:
        return "steady"
    midi = librosa.hz_to_midi(f0)
    n = midi.size
    first, last = midi[: n // 3].mean(), midi[-n // 3:].mean()
    peak_pos = int(np.argmax(midi)) / max(n - 1, 1)
    spread = float(midi.max() - midi.min())
    drift = float(last - first)

    if spread < 2.0:
        return "steady"
    if drift > 2.5:
        return "rising"
    if drift < -2.5:
        return "falling"
    if 0.3 < peak_pos < 0.7 and spread > 4.0:
        return "arching"
    return "wavy"


# ──────────────────────────────────────────────────────────────────────────────
# Musical-suggestion heuristics
# ──────────────────────────────────────────────────────────────────────────────

def _suggest_mood(mode: str, tempo: float, dynamics: str) -> str:
    fast = tempo >= 120
    slow = tempo < 76
    if mode == "minor":
        if fast:
            return "mysterious" if dynamics != "dynamic" else "aggressive"
        if slow:
            return "sad"
        return "nostalgic"
    # major
    if fast:
        return "energetic" if dynamics == "dynamic" else "happy"
    if slow:
        return "calm" if dynamics == "soft" else "romantic"
    return "uplifting"


def _suggest_genre(mode: str, tempo: float, note_density: float) -> str:
    if tempo < 72 and note_density < 2.0:
        return "ambient"
    if tempo >= 128 and note_density >= 3.0:
        return "electronic" if mode == "minor" else "pop"
    if 96 <= tempo < 128:
        return "pop"
    if note_density < 2.5:
        return "folk"
    return "r-and-b" if mode == "minor" else "pop"


# Supportive instrument palettes chosen to sit *around* a lead vocal rather than
# competing with it (no lead melody instruments in the vocal register).
_GENRE_BACKING_INSTRUMENTS: dict[str, list[str]] = {
    "pop":        ["piano", "bass", "drums", "synth"],
    "rock":       ["electric guitar", "bass", "drums"],
    "jazz":       ["piano", "double bass", "brushed drums"],
    "classical":  ["strings", "piano", "cello"],
    "folk":       ["acoustic guitar", "bass", "light percussion"],
    "electronic": ["synth", "sub bass", "drum machine"],
    "hip-hop":    ["sub bass", "drums", "keys"],
    "r-and-b":    ["electric piano", "bass", "drums"],
    "ambient":    ["warm pads", "soft synth", "sub bass"],
    "reggae":     ["offbeat guitar", "bass", "drums"],
    "bossa-nova": ["nylon guitar", "double bass", "brushed drums"],
}


def _suggest_instruments(genre: str) -> list[str]:
    return list(_GENRE_BACKING_INSTRUMENTS.get(genre, ["piano", "bass", "drums"]))


# ──────────────────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────────────────

def analyze_vocal(
    source,
    sr: int = 22_050,
    max_seconds: float = 60.0,
) -> VocalAnalysis:
    """
    Analyse a vocal recording and return a :class:`VocalAnalysis`.

    Parameters
    ----------
    source      : path to an audio file, or an ``(audio_array, sr)`` tuple.
    sr          : analysis sample rate (22.05 kHz is plenty for pitch/tempo and
                  keeps pYIN fast).
    max_seconds : cap the analysed portion so interactive analysis stays quick.
                  ``duration_sec`` still reflects the full clip length.
    """
    if not _HAS_LIBROSA:
        raise RuntimeError("librosa is required for vocal analysis")

    y, sr, full_duration = _load_mono(source, sr, max_seconds)

    res = VocalAnalysis()
    res.duration_sec = round(full_duration, 2)

    if y.size == 0 or float(np.max(np.abs(y))) < 1e-4:
        # Silent / empty upload — return sensible defaults.
        res.analysed_sec = 0.0
        res.melody_summary = "no clear vocal detected"
        res.suggested_instruments = _suggest_instruments(res.suggested_genre)
        res.chord_progression, res.chord_progression_roman = _chord_progression(
            res.key, res.mode, res.suggested_mood
        )
        return res

    res.analysed_sec = round(float(len(y) / sr), 2)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # ── Trim leading/trailing silence for tonal analysis ──────────────────
        y_trim, _ = librosa.effects.trim(y, top_db=30)
        if y_trim.size < sr // 4:      # keep original if trim was too aggressive
            y_trim = y

        # ── Tempo & beats ─────────────────────────────────────────────────────
        try:
            onset_env = librosa.onset.onset_strength(y=y_trim, sr=sr)
            # start_bpm=90 centres the tracker's prior on song tempi instead of
            # librosa's 120 default, which pulls sparse vocal onsets upward.
            tempo, _ = librosa.beat.beat_track(
                onset_envelope=onset_env, sr=sr, start_bpm=90.0,
            )
            raw = float(np.atleast_1d(tempo)[0])
        except Exception:
            raw = 100.0
        if not np.isfinite(raw) or raw <= 0:
            raw = 100.0
        res.tempo_raw_bpm = round(raw, 1)
        res.tempo_bpm = round(fold_tempo(raw), 1)

        # ── Where the performance actually starts / ends ──────────────────────
        a_start, a_end = detect_active_region(y, sr)
        res.active_start_sec = round(a_start, 2)
        res.active_end_sec = round(a_end, 2)

        # ── Key / mode ────────────────────────────────────────────────────────
        try:
            chroma = librosa.feature.chroma_cqt(y=y_trim, sr=sr)
            res.key, res.mode, res.key_confidence = _detect_key(chroma.mean(axis=1))
            res.key_confidence = round(res.key_confidence, 3)
        except Exception:
            res.key, res.mode, res.key_confidence = "C", "major", 0.0
        res.key_name = f"{res.key} {res.mode}"

        # ── Pitch contour (pYIN) ──────────────────────────────────────────────
        try:
            fmin = librosa.note_to_hz("C2")
            fmax = librosa.note_to_hz("C7")
            f0, voiced_flag, _ = librosa.pyin(
                y_trim, fmin=fmin, fmax=fmax, sr=sr,
            )
            voiced = f0[np.isfinite(f0)]
            res.voiced_ratio = round(
                float(np.mean(voiced_flag)) if voiced_flag.size else 0.0, 3
            )
        except Exception:
            f0 = np.array([])
            voiced = np.array([])

        if voiced.size:
            lo = float(np.percentile(voiced, 5))
            hi = float(np.percentile(voiced, 95))
            med = float(np.median(voiced))
            res.pitch_min_note = str(librosa.hz_to_note(lo))
            res.pitch_max_note = str(librosa.hz_to_note(hi))
            res.pitch_median_note = str(librosa.hz_to_note(med))
            res.pitch_median_hz = round(med, 1)
            res.pitch_range_semitones = round(
                float(librosa.hz_to_midi(hi) - librosa.hz_to_midi(lo)), 1
            )
            res.register = _register_for_hz(med)
            res.contour = _describe_contour(f0)

        # ── Phrasing (voiced segments & pauses) ───────────────────────────────
        try:
            intervals = librosa.effects.split(y, top_db=30)
            phrases = [
                {"start": round(float(s / sr), 2), "end": round(float(e / sr), 2)}
                for s, e in intervals
                if (e - s) / sr >= 0.25          # ignore sub-250 ms blips
            ]
            res.phrases = phrases
            res.phrase_count = len(phrases)
            if phrases:
                res.avg_phrase_sec = round(
                    float(np.mean([p["end"] - p["start"] for p in phrases])), 2
                )
        except Exception:
            pass

        # ── Note density (onsets per second) ──────────────────────────────────
        try:
            onsets = librosa.onset.onset_detect(y=y_trim, sr=sr, units="time")
            span = max(res.analysed_sec, 1e-6)
            res.note_density = round(float(len(onsets) / span), 2)
        except Exception:
            res.note_density = 0.0

        # ── Dynamics / loudness arc ───────────────────────────────────────────
        try:
            rms = librosa.feature.rms(y=y_trim)[0]
            if rms.size >= 4:
                first, last = rms[: rms.size // 3].mean(), rms[-rms.size // 3:].mean()
                rel_var = float(np.std(rms) / (np.mean(rms) + 1e-9))
                if last > first * 1.25:
                    res.dynamics = "building"
                    res.suggested_crescendo = "rise"
                elif rel_var > 0.6:
                    res.dynamics = "dynamic"
                    res.suggested_crescendo = "verse-chorus"
                elif float(np.mean(rms)) < 0.05:
                    res.dynamics = "soft"
                    res.suggested_crescendo = "natural"
                else:
                    res.dynamics = "steady"
                    res.suggested_crescendo = "natural"
        except Exception:
            pass

    # ── Derived musical suggestions ──────────────────────────────────────────
    res.suggested_mood = _suggest_mood(res.mode, res.tempo_bpm, res.dynamics)
    res.suggested_genre = _suggest_genre(res.mode, res.tempo_bpm, res.note_density)
    res.suggested_instruments = _suggest_instruments(res.suggested_genre)
    res.chord_progression, res.chord_progression_roman = _chord_progression(
        res.key, res.mode, res.suggested_mood
    )
    res.melody_summary = _build_summary(res)

    return res


def _article(word: str) -> str:
    return "an" if word[:1].lower() in "aeiou" else "a"


def _build_summary(a: VocalAnalysis) -> str:
    """Human/prompt-friendly one-line description of the performance."""
    parts = [f"{_article(a.suggested_mood)} {a.suggested_mood} vocal melody in {a.key_name}"]
    parts.append(f"around {int(round(a.tempo_bpm))} BPM")
    if a.pitch_min_note and a.pitch_max_note:
        parts.append(f"spanning {a.pitch_min_note}–{a.pitch_max_note} ({a.register} register)")
    if a.contour != "steady":
        parts.append(f"with {_article(a.contour)} {a.contour} melodic contour")
    if a.phrase_count:
        parts.append(f"in {a.phrase_count} phrase(s)")
    return ", ".join(parts)


__all__ = ["VocalAnalysis", "analyze_vocal", "detect_active_region", "fold_tempo"]
