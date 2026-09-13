"""
Shared audio-export helpers.

Extracted from ``api.py`` so both the HTTP layer and the background job worker
can write MP3s without importing each other (avoids a circular import).
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np


def find_ffmpeg() -> str:
    """
    Locate the ffmpeg binary.

    Search order:
      1. FFMPEG_PATH env-var  (set by startup.sh on Azure)
      2. PATH via shutil.which
      3. Common Linux paths
      4. Homebrew path on macOS
    """
    import shutil

    candidate = (
        os.environ.get("FFMPEG_PATH")
        or shutil.which("ffmpeg")
        or "/home/bin/ffmpeg"        # Azure cached location
        or "/usr/bin/ffmpeg"         # standard Linux
        or "/usr/local/bin/ffmpeg"   # Homebrew macOS
    )
    if candidate and Path(candidate).exists():
        return candidate
    raise FileNotFoundError(
        "ffmpeg not found. On Azure ensure startup.sh ran; locally run: brew install ffmpeg"
    )


def write_mp3(audio: np.ndarray, sr: int, out_path: Path, bitrate: str = "192k") -> None:
    """
    Write a numpy audio array to an MP3 file.

    Uses ffmpeg via subprocess directly — this avoids pydub's internal
    ffprobe auto-detection which fails when ffmpeg is not on the system PATH
    (e.g. Azure App Service containers).
    """
    import soundfile as sf
    import subprocess

    ffmpeg = find_ffmpeg()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_wav = tmp.name
    try:
        data = audio.T if audio.ndim == 2 else audio
        peak = np.max(np.abs(data))
        if peak > 0:
            data = (data / peak * 0.97).astype(np.float32)
        sf.write(tmp_wav, data, sr, format="WAV", subtype="FLOAT")

        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Call ffmpeg directly — no pydub, no ffprobe probe step
        result = subprocess.run(
            [
                ffmpeg, "-y",           # overwrite without asking
                "-i", tmp_wav,          # input WAV
                "-codec:a", "libmp3lame",
                "-b:a", bitrate,
                "-q:a", "2",            # VBR quality hint (1=best, 9=worst)
                str(out_path),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed:\n{result.stderr[-1000:]}")
    finally:
        try:
            os.unlink(tmp_wav)
        except OSError:
            pass


__all__ = ["find_ffmpeg", "write_mp3"]
