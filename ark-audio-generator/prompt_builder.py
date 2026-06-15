"""
Smart prompt construction for MusicGen.

Analyzes the user's melody description + optional parameters and builds
an optimised MusicGen text prompt.  When optional parameters are absent
the builder infers them from the melody text.
"""

from __future__ import annotations

# ──────────────────────────────────────────────────────────────────────────────
# Knowledge tables
# ──────────────────────────────────────────────────────────────────────────────

GENRE_DESCRIPTORS: dict[str, str] = {
    "pop":        "catchy hook, strong rhythmic pulse, accessible radio-friendly arrangement",
    "rock":       "driving electric guitars, powerful rhythm section, energetic groove",
    "jazz":       "swinging rhythm, rich jazz harmonies, improvised feel, walking bass",
    "classical":  "orchestral arrangement, structured phrasing, dynamic contrast",
    "folk":       "acoustic guitar, fingerpicking, warm natural tone, storytelling feel",
    "electronic": "synthesizer pads, programmed beat, electronic textures, pulse-driven",
    "hip-hop":    "strong sub-bass, punchy kick, rhythmic emphasis, urban groove",
    "country":    "twangy acoustic guitar, pedal steel, heartfelt melody",
    "r-and-b":    "smooth groove, soulful chord progressions, expressive phrasing",
    "ambient":    "evolving textures, spacious reverb, slow harmonic movement",
    "reggae":     "off-beat rhythm guitar, steady bass, relaxed groove",
    "bossa nova": "gentle samba rhythm, nylon guitar, warm jazz harmonies",
}

MOOD_DESCRIPTORS: dict[str, str] = {
    "happy":     "bright major key, upbeat feel, joyful",
    "sad":       "minor key, melancholic phrasing, emotional depth",
    "energetic": "fast tempo, driving beat, high energy, exciting",
    "calm":      "slow tempo, gentle dynamics, relaxing, peaceful",
    "romantic":  "tender melody, warm harmonies, expressive phrasing",
    "uplifting": "building energy, triumphant, inspiring, positive",
    "mysterious":"dark undertones, unexpected chord shifts, intriguing",
    "aggressive":"intense, powerful, fast-paced, heavy",
    "nostalgic": "warm retro tone, bittersweet, memorable melody",
    "playful":   "light, bouncy, fun, whimsical",
}

INSTRUMENT_MAP: dict[str, str] = {
    "piano":       "piano melody",
    "guitar":      "acoustic guitar",
    "electric guitar": "electric guitar riffs",
    "violin":      "violin strings",
    "viola":       "viola strings",
    "cello":       "cello",
    "drums":       "drum kit",
    "bass":        "bass guitar",
    "flute":       "flute melody",
    "trumpet":     "brass trumpet",
    "saxophone":   "saxophone",
    "synth":       "synthesizer",
    "synthesizer": "synthesizer pads",
    "harp":        "harp arpeggios",
    "ukulele":     "ukulele",
    "banjo":       "banjo",
    "organ":       "Hammond organ",
    "choir":       "choral vocals",
    "strings":     "string ensemble",
    "orchestra":   "full orchestral arrangement",
    "percussion":  "hand percussion",
    "marimba":     "marimba",
}

FREQ_MAP: dict[str, str] = {
    "bass":   "deep bass emphasis, sub-bass presence",
    "mid":    "warm midrange focus, balanced mix",
    "treble": "bright high-frequency shimmer, crisp detail",
    "full":   "full-spectrum mix, balanced bass, mids, and highs",
    "low":    "low-frequency warmth, bass-forward",
    "high":   "high-frequency clarity, airy top end",
}

# ──────────────────────────────────────────────────────────────────────────────
# Keyword inference
# ──────────────────────────────────────────────────────────────────────────────

_GENRE_KEYWORDS: dict[str, list[str]] = {
    "pop":        ["pop", "popular", "chart", "radio"],
    "rock":       ["rock", "guitar", "band", "riff"],
    "jazz":       ["jazz", "swing", "blues", "bebop", "improv"],
    "classical":  ["classical", "orchestral", "symphony", "concerto", "baroque"],
    "folk":       ["folk", "acoustic", "ballad", "fingerpick"],
    "electronic": ["electronic", "edm", "synth", "techno", "house", "trance"],
    "hip-hop":    ["hip-hop", "hip hop", "rap", "trap", "beat"],
    "r-and-b":    ["r&b", "rnb", "soul", "motown"],
    "ambient":    ["ambient", "atmospheric", "meditative", "drone"],
    "reggae":     ["reggae", "ska", "dancehall"],
    "bossa nova": ["bossa", "samba", "brazilian"],
}

_MOOD_KEYWORDS: dict[str, list[str]] = {
    "happy":     ["happy", "joyful", "cheerful", "bright", "fun", "jolly"],
    "sad":       ["sad", "melancholy", "blue", "mournful", "gloomy", "wistful"],
    "energetic": ["fast", "energetic", "upbeat", "lively", "exciting", "dynamic"],
    "calm":      ["slow", "calm", "peaceful", "gentle", "soft", "relaxing", "serene"],
    "romantic":  ["love", "romantic", "tender", "sweet", "affectionate"],
    "uplifting": ["uplifting", "inspiring", "triumphant", "powerful", "motivating"],
    "nostalgic": ["nostalgic", "retro", "old", "vintage", "memory"],
    "playful":   ["playful", "fun", "bouncy", "whimsical", "light"],
}


def infer_parameters(text: str) -> dict[str, str | None]:
    """Return best-guess genre and mood from a free-text description."""
    lower = text.lower()

    genre = None
    for g, keywords in _GENRE_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            genre = g
            break

    mood = None
    for m, keywords in _MOOD_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            mood = m
            break

    return {
        "genre": genre or "pop",      # pop is most sing-along friendly
        "mood":  mood  or "uplifting",
    }


# ──────────────────────────────────────────────────────────────────────────────
# Prompt assembly
# ──────────────────────────────────────────────────────────────────────────────

def build_prompt(
    melody_description: str,
    genre: str | None = None,
    mood: str | None = None,
    instruments: list[str] | None = None,
    frequency_range: str | None = None,
    inferred: dict | None = None,
) -> str:
    """
    Assemble an optimised MusicGen conditioning prompt.

    Explicitly provided parameters always take precedence over inferred ones.
    """
    parts: list[str] = []

    # ── Core sing-along identity ──────────────────────────────────────────────
    parts.append(
        "high-quality sing-along backing track, "
        "clear memorable melody, strong rhythmic pulse, "
        "professional studio recording"
    )

    # ── Genre ─────────────────────────────────────────────────────────────────
    effective_genre = (genre or (inferred or {}).get("genre") or "pop").lower()
    if effective_genre in GENRE_DESCRIPTORS:
        parts.append(GENRE_DESCRIPTORS[effective_genre])
    else:
        parts.append(effective_genre)

    # ── Mood ──────────────────────────────────────────────────────────────────
    effective_mood = (mood or (inferred or {}).get("mood") or "uplifting").lower()
    if effective_mood in MOOD_DESCRIPTORS:
        parts.append(MOOD_DESCRIPTORS[effective_mood])
    else:
        parts.append(effective_mood)

    # ── Instruments ───────────────────────────────────────────────────────────
    if instruments:
        inst_descs = [
            INSTRUMENT_MAP.get(i.lower().strip(), i.strip())
            for i in instruments
        ]
        parts.append("featuring " + ", ".join(inst_descs))

    # ── Frequency range ───────────────────────────────────────────────────────
    if frequency_range:
        freq_desc = FREQ_MAP.get(frequency_range.lower(), frequency_range)
        parts.append(freq_desc)

    # ── Rhythmic / dynamic characteristics (always present for sing-along) ────
    parts.append(
        "steady beat, clear harmonic progression, "
        "dynamic variation with crescendo, "
        "well-paced tempo suitable for singing along"
    )

    # ── Melody hint ───────────────────────────────────────────────────────────
    if melody_description and not melody_description.startswith("melody from audio"):
        parts.append(f"melody theme: {melody_description}")

    return ", ".join(parts)
