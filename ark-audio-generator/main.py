#!/usr/bin/env python3
"""
AI Sing-Along Music Generator
==============================
Generate a rhythmic, lively backing track from a melody description (or audio
file) in MP3 format using Meta's MusicGen.

Usage examples
--------------
# Text melody → auto-infer genre & mood
python main.py "a cheerful whistling melody in C major"

# Provide all parameters explicitly
python main.py "flowing jazz piano riff" \\
    --genre jazz --mood uplifting --instruments piano,bass,drums \\
    --duration 15 --output my_track.mp3

# Use an existing audio clip as the melody source (melody model)
python main.py /path/to/melody.wav \\
    --genre pop --mood happy --output sing_along.mp3
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

app     = typer.Typer(add_completion=False, help=__doc__)
console = Console()

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

_AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".aiff", ".m4a"}


def _is_audio_file(path_or_text: str) -> bool:
    p = Path(path_or_text)
    return p.suffix.lower() in _AUDIO_EXTS and p.exists()


def _save_mp3(audio: np.ndarray, sr: int, output_path: Path, bitrate: str = "192k") -> None:
    """Write numpy audio array to MP3 via soundfile + pydub."""
    import soundfile as sf
    from pydub import AudioSegment

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_wav = tmp.name

    try:
        # soundfile expects (samples, channels) for stereo
        if audio.ndim == 2:
            data = audio.T
        else:
            data = audio

        # Final peak normalisation before write
        peak = np.max(np.abs(data))
        if peak > 0:
            data = (data / peak * 0.97).astype(np.float32)

        sf.write(tmp_wav, data, sr, format="WAV", subtype="FLOAT")
        sound = AudioSegment.from_wav(tmp_wav)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sound.export(str(output_path), format="mp3", bitrate=bitrate)
    finally:
        try:
            os.unlink(tmp_wav)
        except OSError:
            pass


# ──────────────────────────────────────────────────────────────────────────────
# CLI command
# ──────────────────────────────────────────────────────────────────────────────

@app.command()
def generate(
    melody: str = typer.Argument(
        ...,
        help="Melody description (text) OR path to an audio file (wav/mp3/flac).",
    ),
    output: str = typer.Option(
        "output.mp3",
        "--output", "-o",
        help="Output file path (MP3).  Example: my_song.mp3",
    ),
    genre: Optional[str] = typer.Option(
        None, "--genre", "-g",
        help="Music genre: pop | rock | jazz | classical | folk | electronic | "
             "hip-hop | r-and-b | ambient | reggae | bossa-nova",
    ),
    mood: Optional[str] = typer.Option(
        None, "--mood", "-m",
        help="Emotional mood: happy | sad | energetic | calm | romantic | "
             "uplifting | mysterious | aggressive | nostalgic | playful",
    ),
    instruments: Optional[str] = typer.Option(
        None, "--instruments", "-i",
        help="Comma-separated instruments, e.g. 'piano,guitar,drums'",
    ),
    frequency_range: Optional[str] = typer.Option(
        None, "--frequency", "-f",
        help="Frequency emphasis: bass | mid | treble | full",
    ),
    duration: float = typer.Option(
        20.0, "--duration", "-d",
        help="Output length in seconds (1 – 20).  Longer = slower on CPU.",
    ),
    crescendo: str = typer.Option(
        "rise-fall", "--crescendo",
        help="Volume arc pattern: rise | fall | rise-fall | natural | verse-chorus",
    ),
    model_id: Optional[str] = typer.Option(
        None, "--model-id",
        help="Override HuggingFace model ID (advanced).  "
             "Defaults to facebook/musicgen-small or facebook/musicgen-melody.",
    ),
    bitrate: str = typer.Option(
        "192k", "--bitrate",
        help="MP3 export bitrate (e.g. 128k, 192k, 320k).",
    ),
    no_effects: bool = typer.Option(
        False, "--no-effects",
        help="Skip post-processing effects chain.",
    ),
    guidance_scale: float = typer.Option(
        3.5, "--guidance",
        help="Classifier-free guidance scale (higher = closer to prompt, 1–10).",
    ),
    temperature: float = typer.Option(
        1.05, "--temperature",
        help="Sampling temperature (0.8 crisp → 1.2 more random).",
    ),
) -> None:
    """Generate a rhythmic, lively sing-along backing track using AI."""

    # ── Banner ────────────────────────────────────────────────────────────────
    console.print(
        Panel.fit(
            "[bold cyan]AI Sing-Along Music Generator[/bold cyan]\n"
            "[dim]Powered by Meta MusicGen · HuggingFace Transformers[/dim]",
            border_style="cyan",
        )
    )

    # ── Validate / clamp parameters ───────────────────────────────────────────
    duration = float(np.clip(duration, 1.0, 20.0))

    inst_list: list[str] | None = (
        [i.strip() for i in instruments.split(",") if i.strip()]
        if instruments
        else None
    )

    # ── Detect melody input type ──────────────────────────────────────────────
    is_audio_input   = _is_audio_file(melody)
    use_melody_model = is_audio_input  # use melody-conditioned model when audio supplied

    if is_audio_input:
        melody_path       = melody
        melody_description = "melody from audio file"
        console.print(f"[green]Melody source  :[/green] audio file → {melody}")
    else:
        melody_path       = None
        melody_description = melody
        console.print(f"[green]Melody source  :[/green] text description")

    # ── Infer missing parameters ──────────────────────────────────────────────
    from prompt_builder import build_prompt, infer_parameters
    inferred = infer_parameters(melody_description)

    effective_genre = genre or inferred["genre"]
    effective_mood  = mood  or inferred["mood"]

    # ── Parameter summary table ───────────────────────────────────────────────
    tbl = Table(box=None, show_header=False, padding=(0, 2))
    tbl.add_column(style="dim",  no_wrap=True)
    tbl.add_column(style="cyan")
    tbl.add_row("Genre",          f"{effective_genre}  {'[dim](inferred)[/dim]' if not genre else ''}")
    tbl.add_row("Mood",           f"{effective_mood}   {'[dim](inferred)[/dim]' if not mood  else ''}")
    tbl.add_row("Instruments",    ", ".join(inst_list) if inst_list else "[dim]auto[/dim]")
    tbl.add_row("Frequency",      frequency_range or "[dim]auto[/dim]")
    tbl.add_row("Duration",       f"{duration} s")
    tbl.add_row("Crescendo",      crescendo)
    tbl.add_row("Guidance scale", str(guidance_scale))
    tbl.add_row("Output",         output)
    console.print(tbl)

    # ── Build text prompt ─────────────────────────────────────────────────────
    prompt = build_prompt(
        melody_description=melody_description,
        genre=genre,
        mood=mood,
        instruments=inst_list,
        frequency_range=frequency_range,
        inferred=inferred,
    )

    # ── Run generation pipeline ───────────────────────────────────────────────
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    ) as progress:

        # 1. Load model
        task = progress.add_task("[yellow]Loading AI model…[/yellow]", total=None)
        from generator import MusicGenerator
        gen = MusicGenerator(
            use_melody_model=use_melody_model,
            model_id=model_id,
        )
        progress.update(task, description="[green]Model loaded.[/green]")

        # 2. Generate
        progress.update(task, description="[yellow]Generating music… (this takes a while on CPU)[/yellow]")
        audio, sr = gen.generate(
            prompt=prompt,
            melody_path=melody_path,
            duration=duration,
            guidance_scale=guidance_scale,
            temperature=temperature,
        )
        progress.update(task, description="[green]Music generated.[/green]")

        # 3. Post-processing
        if not no_effects:
            progress.update(task, description="[yellow]Applying audio effects…[/yellow]")
            from effects import process_audio
            audio = process_audio(
                audio,
                sr,
                genre=effective_genre,
                mood=effective_mood,
                crescendo_pattern=crescendo,
            )
            progress.update(task, description="[green]Effects applied.[/green]")

        # 4. Export
        progress.update(task, description="[yellow]Exporting MP3…[/yellow]")
        out_path = Path(output)
        _save_mp3(audio, sr, out_path, bitrate=bitrate)
        progress.update(task, description="[bold green]Done![/bold green]")

    # ── Success summary ───────────────────────────────────────────────────────
    size_kb = out_path.stat().st_size // 1024
    console.print(
        Panel(
            f"[bold green]Track saved successfully![/bold green]\n\n"
            f"  File   : [cyan]{out_path.absolute()}[/cyan]\n"
            f"  Size   : [cyan]{size_kb} KB[/cyan]\n"
            f"  Length : [cyan]{duration} s[/cyan]\n"
            f"  Format : [cyan]MP3 {bitrate}[/cyan]",
            border_style="green",
            title="[bold]Output[/bold]",
        )
    )


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app()
