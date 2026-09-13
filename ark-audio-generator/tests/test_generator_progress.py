"""
Tests for the per-token progress hook and the vocal-pipeline progress wiring.

These validate the actual "track generation progress" mechanism without loading
any MusicGen model:

  * ``MusicGenerator._build_progress_criteria`` is exercised with plain torch
    tensors that mimic the growing decoder input.
  * ``run_vocal_to_music`` is driven with a stub generator so the real
    orchestration + per-segment progress mapping runs end to end.
"""

import numpy as np
import pytest


# ──────────────────────────────────────────────────────────────────────────────
# StoppingCriteria progress hook
# ──────────────────────────────────────────────────────────────────────────────

def test_no_callback_returns_none():
    from generator import MusicGenerator
    assert MusicGenerator._build_progress_criteria(100, None) is None


def test_progress_criteria_reports_increasing_fraction():
    import torch
    from generator import MusicGenerator

    seen: list[float] = []
    crit_list = MusicGenerator._build_progress_criteria(10, seen.append)
    crit = crit_list[0]

    # First call establishes the baseline length (5) → fraction 0.
    out0 = crit(torch.zeros((1, 5), dtype=torch.long), None)
    assert out0.dtype == torch.bool and tuple(out0.shape) == (1,)
    assert not bool(out0.any())          # never requests a stop

    # +5 tokens → 5/10 = 0.5 ; way past the end → clamped to 1.0.
    crit(torch.zeros((1, 10), dtype=torch.long), None)
    crit(torch.zeros((1, 999), dtype=torch.long), None)

    assert seen[0] == pytest.approx(0.0)
    assert seen[1] == pytest.approx(0.5)
    assert seen[2] == pytest.approx(1.0)


def test_progress_criteria_batched_returns_one_flag_per_sequence():
    import torch
    from generator import MusicGenerator

    crit = MusicGenerator._build_progress_criteria(50, lambda _f: None)[0]
    out = crit(torch.zeros((3, 8), dtype=torch.long), None)
    assert tuple(out.shape) == (3,)
    assert not bool(out.any())


# ──────────────────────────────────────────────────────────────────────────────
# Vocal pipeline progress mapping (stub generator, no real model / ffmpeg)
# ──────────────────────────────────────────────────────────────────────────────

class _StubGenerator:
    """Cheap stand-in for MusicGenerator that emits progress and silence."""
    sample_rate = 32000

    def generate(self, prompt, melody_path=None, duration=1.0,
                 guidance_scale=3.0, temperature=1.0, progress_cb=None):
        if progress_cb:
            for f in (0.0, 0.5, 1.0):
                progress_cb(f)
        n = max(int(duration * self.sample_rate), 16)
        return np.zeros((2, n), dtype=np.float32), self.sample_rate


def _write_sine(path, seconds=2.0, sr=32000, freq=220.0):
    import soundfile as sf
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    y = (0.2 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    sf.write(str(path), y, sr, format="WAV")


def test_vocal_pipeline_reports_bounded_monotonic_progress(tmp_path):
    from vocal_pipeline import run_vocal_to_music

    vocal = tmp_path / "vocal.wav"
    _write_sine(vocal)

    events: list[tuple[int, str]] = []

    result = run_vocal_to_music(
        str(vocal),
        overrides={"genre": "folk"},
        generator_factory=_StubGenerator,
        progress=lambda pct, msg: events.append((int(pct), msg)),
    )

    pcts = [p for p, _ in events]
    # Progress is non-decreasing and stays within the pipeline's 0–94 band
    # (export happens after the last callback).
    assert pcts == sorted(pcts)
    assert 0 <= min(pcts) and max(pcts) <= 94

    # The accompaniment stage emitted a per-token percentage inside its 30–75 band.
    gen_pcts = [p for p, m in events if m.startswith("Generating accompaniment")]
    assert gen_pcts, "expected accompaniment progress events"
    assert all(30 <= p <= 75 for p in gen_pcts)

    # Sane result shapes.
    assert result.accompaniment.shape[0] == 2
    assert result.mix.shape[0] == 2
    assert result.segments >= 1
