#!/usr/bin/env python3
"""
Vocal → Music CLI
=================
Turn a vocal-only recording (humming / singing / improvisation) into a fully
arranged track: the tool analyses the vocal, builds a complementary
accompaniment with MusicGen, and mixes your original vocal back on top.

Kept separate from ``main.py`` so the existing single-command CLI
(``python main.py "<melody description>"``) keeps working unchanged.

Usage
-----
# Fully automatic — everything inferred from the vocal
python vocalize.py my_humming.wav

# Analyse only (fast, no generation)
python vocalize.py my_humming.wav --analyze-only

# Override some of the auto-detected settings
python vocalize.py my_singing.wav --genre jazz --mood calm \\
    --instruments "piano,double bass,brushed drums" -o song.mp3
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

app = typer.Typer(add_completion=False, help=__doc__)
console = Console()


def _print_analysis(a) -> None:
    tbl = Table(box=None, show_header=False, padding=(0, 2))
    tbl.add_column(style="dim", no_wrap=True)
    tbl.add_column(style="cyan")
    tbl.add_row("Duration",   f"{a.duration_sec} s")
    if a.active_end_sec:
        tbl.add_row("Singing",    f"{a.active_start_sec}s → {a.active_end_sec}s")
    tbl.add_row("Key / mode", f"{a.key_name}  (confidence {a.key_confidence})")
    tbl.add_row("Tempo",      f"{a.tempo_bpm} BPM")
    tbl.add_row("Pitch range", f"{a.pitch_min_note}–{a.pitch_max_note}  ({a.register})")
    tbl.add_row("Contour",    a.contour)
    tbl.add_row("Phrases",    f"{a.phrase_count} (avg {a.avg_phrase_sec}s)")
    tbl.add_row("Dynamics",   a.dynamics)
    tbl.add_row("Genre",      f"{a.suggested_genre} (suggested)")
    tbl.add_row("Mood",       f"{a.suggested_mood} (suggested)")
    tbl.add_row("Instruments", ", ".join(a.suggested_instruments))
    tbl.add_row("Chords",     " – ".join(a.chord_progression))
    console.print(tbl)


def _write_mp3(audio: np.ndarray, sr: int, out_path: Path, bitrate: str = "192k") -> None:
    """Export a (2, samples) float array to MP3 (shared ffmpeg helper)."""
    from audio_io import write_mp3
    write_mp3(audio, sr, out_path, bitrate=bitrate)


@app.command()
def vocalize(
    vocal: str = typer.Argument(..., help="Path to a vocal-only audio file (wav/mp3/flac/…)."),
    output: str = typer.Option("vocal_track.mp3", "--output", "-o", help="Output MP3 for the full mix."),
    genre: Optional[str] = typer.Option(None, "--genre", "-g", help="Override auto-detected genre."),
    mood: Optional[str] = typer.Option(None, "--mood", "-m", help="Override auto-detected mood."),
    instruments: Optional[str] = typer.Option(None, "--instruments", "-i", help="Comma-separated instruments."),
    tempo: Optional[float] = typer.Option(None, "--tempo", help="Override detected tempo (BPM)."),
    crescendo: Optional[str] = typer.Option(None, "--crescendo", help="rise | fall | rise-fall | natural | verse-chorus"),
    guidance_scale: float = typer.Option(3.0, "--guidance", help="Classifier-free guidance (1–10)."),
    bitrate: str = typer.Option("192k", "--bitrate", help="MP3 export bitrate."),
    analyze_only: bool = typer.Option(False, "--analyze-only", help="Only analyse the vocal; skip generation."),
) -> None:
    """Analyse a vocal recording and generate a complementary arrangement."""

    console.print(
        Panel.fit(
            "[bold cyan]Vocal → Music[/bold cyan]\n"
            "[dim]Analyse a vocal · arrange · synchronise · mix[/dim]",
            border_style="cyan",
        )
    )

    if not Path(vocal).exists():
        console.print(f"[red]File not found:[/red] {vocal}")
        raise typer.Exit(code=1)

    # ── Analyse ───────────────────────────────────────────────────────────────
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                  TimeElapsedColumn(), console=console, transient=True) as prog:
        t = prog.add_task("[yellow]Analysing vocal…[/yellow]", total=None)
        from vocal_analysis import analyze_vocal
        analysis = analyze_vocal(vocal)
        prog.update(t, description="[green]Analysis complete.[/green]")

    console.print("\n[bold]Detected musical characteristics[/bold]")
    _print_analysis(analysis)
    console.print(f"\n[dim]{analysis.melody_summary}[/dim]\n")

    if analyze_only:
        console.print("[green]Analyse-only mode — done.[/green]")
        return

    # ── Generate + mix ────────────────────────────────────────────────────────
    overrides = {
        "genre": genre, "mood": mood, "tempo_bpm": tempo, "crescendo": crescendo,
        "guidance_scale": guidance_scale,
        "instruments": [i.strip() for i in instruments.split(",")] if instruments else None,
    }
    overrides = {k: v for k, v in overrides.items() if v not in (None, "", [])}

    from vocal_pipeline import run_vocal_to_music

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                  TimeElapsedColumn(), console=console, transient=False) as prog:
        task = prog.add_task("[yellow]Starting…[/yellow]", total=None)

        def _cb(pct: int, msg: str) -> None:
            prog.update(task, description=f"[yellow]{msg}[/yellow] ({pct}%)")

        result = run_vocal_to_music(vocal, overrides=overrides, progress=_cb)
        prog.update(task, description="[yellow]Exporting…[/yellow]")

        out_mix = Path(output)
        out_acc = out_mix.with_name(out_mix.stem + "_accompaniment" + out_mix.suffix)
        _write_mp3(result.mix, result.sample_rate, out_mix, bitrate=bitrate)
        _write_mp3(result.accompaniment, result.sample_rate, out_acc, bitrate=bitrate)
        prog.update(task, description="[bold green]Done![/bold green]")

    if result.warnings:
        for w in result.warnings:
            console.print(f"[yellow]⚠️  {w}[/yellow]")

    console.print(
        Panel(
            f"[bold green]Track ready![/bold green]\n\n"
            f"  Full mix     : [cyan]{out_mix.absolute()}[/cyan]\n"
            f"  Music only   : [cyan]{out_acc.absolute()}[/cyan]\n"
            f"  Key / tempo  : [cyan]{result.plan.key} {result.plan.mode} · "
            f"{int(round(result.plan.tempo_bpm))} BPM[/cyan]\n"
            f"  Genre / mood : [cyan]{result.plan.genre} · {result.plan.mood}[/cyan]\n"
            f"  Source window: [cyan]{result.window_start_sec}s → {result.window_end_sec}s "
            f"of {result.source_duration_sec}s[/cyan]\n"
            f"  Prompt       : [dim]{result.plan.prompt}[/dim]\n"
            f"  Segments     : [cyan]{result.segments}"
            f"{' (chained by audio continuation)' if result.continuation_used else ''}[/cyan]",
            border_style="green", title="[bold]Output[/bold]",
        )
    )


if __name__ == "__main__":
    app()
