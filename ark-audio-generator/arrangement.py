"""
Arrangement planning for the vocal-to-music pipeline.

Turns a :class:`vocal_analysis.VocalAnalysis` (plus any optional user overrides)
into an :class:`ArrangementPlan`: the concrete recipe used to drive MusicGen
generation and the vocal-aware mix.

The plan is deliberately a plain data object so future versions can let users
override any field (genre, mood, instruments, tempo, key, arrangement) while the
auto-detected vocal characteristics remain the default — exactly the extension
point called for in the feature spec.

The MusicGen prompt is built here rather than through
:func:`prompt_builder.build_prompt`: that builder writes long "sing-along"
captions that ask for a lead melody, which is exactly what an *accompaniment*
must not have.  See :func:`_build_arrangement_prompt`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional

from vocal_analysis import VocalAnalysis


# ──────────────────────────────────────────────────────────────────────────────
# Plan container
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ArrangementPlan:
    """Concrete recipe for generating + mixing an accompaniment for a vocal."""

    # Musical identity (defaults come from the vocal, may be overridden)
    genre: str = "pop"
    mood: str = "uplifting"
    key: str = "C"
    mode: str = "major"
    tempo_bpm: float = 100.0
    instruments: list = field(default_factory=list)
    chord_progression: list = field(default_factory=list)

    # MusicGen generation controls
    prompt: str = ""
    guidance_scale: float = 3.5
    temperature: float = 1.0

    # Mixing / synchronisation controls.  Levels are *matched* automatically in
    # the mixer (vocal normalised, backing set relative to it); the two gain
    # fields are trims on top of that, so 0 dB means "the matched balance".
    crescendo: str = "natural"
    vocal_center_hz: float = 220.0     # vocal fundamental → where to carve space
    accompaniment_gain_db: float = 0.0
    vocal_gain_db: float = 0.0
    duck_depth_db: float = 6.0         # how hard the backing ducks under the vocal

    def to_dict(self) -> dict:
        return asdict(self)


# ──────────────────────────────────────────────────────────────────────────────
# Planner
# ──────────────────────────────────────────────────────────────────────────────

# Fields a caller (or future UI) may override while keeping every other
# auto-detected characteristic from the vocal.
_OVERRIDABLE = {
    "genre", "mood", "instruments", "tempo_bpm", "key", "mode",
    "guidance_scale", "temperature", "crescendo", "chord_progression",
}


def plan_arrangement(
    analysis: VocalAnalysis,
    overrides: Optional[dict] = None,
) -> ArrangementPlan:
    """
    Build an :class:`ArrangementPlan` from a vocal analysis.

    ``overrides`` (optional) lets a caller pin any of the fields in
    ``_OVERRIDABLE``; everything else is taken from the detected vocal.
    Empty / ``None`` override values are ignored so the UI can send blank
    "auto" fields safely.
    """
    overrides = {
        k: v for k, v in (overrides or {}).items()
        if k in _OVERRIDABLE and v not in (None, "", [])
    }

    plan = ArrangementPlan(
        genre=overrides.get("genre", analysis.suggested_genre),
        mood=overrides.get("mood", analysis.suggested_mood),
        key=overrides.get("key", analysis.key),
        mode=overrides.get("mode", analysis.mode),
        tempo_bpm=float(overrides.get("tempo_bpm", analysis.tempo_bpm)),
        instruments=list(overrides.get("instruments", analysis.suggested_instruments)),
        chord_progression=list(
            overrides.get("chord_progression", analysis.chord_progression)
        ),
        guidance_scale=float(overrides.get("guidance_scale", 3.0)),
        temperature=float(overrides.get("temperature", 1.0)),
        crescendo=overrides.get("crescendo", analysis.suggested_crescendo),
    )

    # Carve space around the detected vocal fundamental (fall back to register).
    plan.vocal_center_hz = float(analysis.pitch_median_hz or 220.0)

    plan.prompt = _build_arrangement_prompt(analysis, plan)
    return plan


# Backing-oriented style vocabulary.  MusicGen was trained on short, concrete
# captions (a dozen or two words), so every phrase here describes *sound*, not
# intent, and none of them invite a lead melody or a singer — that is what the
# uploaded vocal is for.
_GENRE_BACKING_STYLE: dict[str, str] = {
    "pop":        "clean pop production, steady drums, warm chords",
    "rock":       "driving rhythm guitars, solid drums, electric bass",
    "jazz":       "swinging rhythm section, comping piano, walking bass",
    "classical":  "orchestral strings and piano, gentle dynamics",
    "folk":       "acoustic guitar strumming, warm and organic",
    "electronic": "synth pads, programmed beat, pulsing bass",
    "hip-hop":    "boom bap drums, deep bass, sparse keys",
    "r-and-b":    "smooth soul groove, electric piano chords, laid-back drums",
    "ambient":    "soft evolving pads, spacious and slow",
    "reggae":     "offbeat skank guitar, deep bass, relaxed drums",
    "bossa-nova": "gentle bossa nova rhythm, nylon guitar, brushed drums",
}

_GENRE_LABEL: dict[str, str] = {
    "r-and-b": "R&B", "hip-hop": "hip hop", "bossa-nova": "bossa nova",
}

_MOOD_FEEL: dict[str, str] = {
    "happy":      "bright and joyful",
    "sad":        "melancholic and tender",
    "energetic":  "driving and energetic",
    "calm":       "gentle and relaxed",
    "romantic":   "warm and tender",
    "uplifting":  "warm and uplifting",
    "mysterious": "moody and atmospheric",
    "aggressive": "intense and powerful",
    "nostalgic":  "warm and bittersweet",
    "playful":    "light and bouncy",
}


def _build_arrangement_prompt(analysis: VocalAnalysis, plan: ArrangementPlan) -> str:
    """
    Compose a short, MusicGen-friendly caption for an *accompaniment*.

    Deliberately concise (≈ 30–40 T5 tokens): key, tempo, genre, instruments
    and feel.  The melodic detail does not go into the text at all — the model
    receives the melody directly through the chroma of the uploaded vocal.
    Words like "vocal", "melody" or "sing-along" are avoided on purpose: they
    make MusicGen synthesise a lead line (or vocal-like noises) that then
    fights the real singer.
    """
    genre_key = (plan.genre or "pop").lower().replace(" ", "-")
    genre_label = _GENRE_LABEL.get(genre_key, genre_key.replace("-", " "))
    style = _GENRE_BACKING_STYLE.get(genre_key, "steady rhythm section, warm chords")
    feel = _MOOD_FEEL.get((plan.mood or "").lower(), (plan.mood or "warm").lower())

    instruments = [i.strip() for i in (plan.instruments or []) if i and i.strip()]
    inst = ", ".join(instruments[:4]) if instruments else "piano, bass, drums"

    tempo = int(round(plan.tempo_bpm))
    parts = [
        f"{feel} {genre_label} instrumental backing track",
        inst,
        style,
        f"in {plan.key} {plan.mode}",
        f"{tempo} bpm",
        "steady groove, instrumental, no vocals",
    ]
    return ", ".join(parts)


__all__ = ["ArrangementPlan", "plan_arrangement"]
