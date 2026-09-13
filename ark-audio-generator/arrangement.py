"""
Arrangement planning for the vocal-to-music pipeline.

Turns a :class:`vocal_analysis.VocalAnalysis` (plus any optional user overrides)
into an :class:`ArrangementPlan`: the concrete recipe used to drive MusicGen
generation and the vocal-aware mix.

The plan is deliberately a plain data object so future versions can let users
override any field (genre, mood, instruments, tempo, key, arrangement) while the
auto-detected vocal characteristics remain the default — exactly the extension
point called for in the feature spec.

Prompt construction reuses :func:`prompt_builder.build_prompt` so the vocal flow
and the existing "describe melody" flow share one prompt vocabulary.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional

from prompt_builder import build_prompt
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

    # Mixing / synchronisation controls
    crescendo: str = "natural"
    vocal_center_hz: float = 220.0     # vocal fundamental → where to carve space
    accompaniment_gain_db: float = -3.0
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


def _build_arrangement_prompt(analysis: VocalAnalysis, plan: ArrangementPlan) -> str:
    """
    Compose the MusicGen text prompt for an *accompaniment* (not a full song).

    We reuse the shared prompt vocabulary, then append the vocal-specific
    conditioning: key, tempo, chord progression, and an explicit instruction to
    stay out of the lead vocal's way.
    """
    melody_hint = (
        analysis.melody_summary
        or f"a {plan.mood} vocal melody in {plan.key} {plan.mode}"
    )

    base = build_prompt(
        melody_description=melody_hint,
        genre=plan.genre,
        mood=plan.mood,
        instruments=plan.instruments,
        frequency_range=None,
        inferred={"genre": plan.genre, "mood": plan.mood},
    )

    tempo = int(round(plan.tempo_bpm))
    chords = " – ".join(plan.chord_progression) if plan.chord_progression else ""

    extras = [
        "instrumental accompaniment only",
        "no lead vocals, leave harmonic and midrange space for a solo singer",
        f"in the key of {plan.key} {plan.mode}",
        f"at {tempo} BPM",
    ]
    if chords:
        extras.append(f"chord progression {chords}")
    extras.append("supportive backing that follows and complements the vocal melody")

    return base + ", " + ", ".join(extras)


__all__ = ["ArrangementPlan", "plan_arrangement"]
