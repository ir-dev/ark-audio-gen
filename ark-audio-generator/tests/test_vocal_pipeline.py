"""
Unit tests for the vocal-to-music pipeline that run without MusicGen.

They cover the things that turned a real upload into "irrelevant noise":
a silent count-in eating the generation window, an octave-doubled tempo, a
long contradictory prompt, unrelated clips glued together, and a vocal mixed
far below the backing.
"""

import inspect

import numpy as np
import pytest
import soundfile as sf

SR = 32_000


def _tone(seconds, freq=220.0, amp=0.2, sr=SR):
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _write(path, y, sr=SR):
    sf.write(str(path), y, sr, format="WAV", subtype="FLOAT")


class _RecordingStub:
    """Minimal generator: records calls, returns low-level noise of the right length."""

    sample_rate = SR

    def __init__(self):
        self.calls = []

    def generate(self, prompt, melody_path, duration, guidance_scale, temperature,
                 progress_cb=None):
        mel, sr = sf.read(melody_path)
        self.calls.append({"prompt": prompt, "duration": duration,
                           "melody_seconds": len(mel) / sr})
        if progress_cb:
            progress_cb(0.5)
            progress_cb(1.0)
        n = int(duration * SR)
        rng = np.random.default_rng(0)
        return rng.standard_normal((2, n)).astype(np.float32) * 0.2, SR


class _ContinuationStub(_RecordingStub):
    """Generator that also accepts an audio prompt, like the real one."""

    def generate(self, prompt, melody_path, duration, guidance_scale, temperature,
                 progress_cb=None, continuation=None):
        audio, sr = super().generate(prompt, melody_path, duration, guidance_scale,
                                     temperature, progress_cb)
        self.calls[-1]["continuation_seconds"] = (
            continuation[0].shape[-1] / continuation[1] if continuation else 0.0
        )
        return audio, sr


# ── analysis helpers ──────────────────────────────────────────────────────────

def test_fold_tempo_corrects_octave_errors():
    from vocal_analysis import fold_tempo
    assert fold_tempo(152.0) == 76.0          # the real-world failure
    assert fold_tempo(40.0) == 80.0
    assert fold_tempo(100.0) == 100.0
    assert fold_tempo(float("nan")) == 100.0
    assert fold_tempo(0) == 100.0


def test_detect_active_region_skips_silence():
    from vocal_analysis import detect_active_region
    y = np.concatenate([np.zeros(3 * SR, np.float32), _tone(2.0), np.zeros(SR, np.float32)])
    start, end = detect_active_region(y, SR)
    assert start == pytest.approx(3.0, abs=0.1)
    assert end == pytest.approx(5.0, abs=0.1)


def test_detect_active_region_on_silence_returns_whole():
    from vocal_analysis import detect_active_region
    assert detect_active_region(np.zeros(SR, np.float32), SR) == (0.0, 1.0)


def test_analysis_reports_active_region_and_folded_tempo(tmp_path):
    from vocal_analysis import analyze_vocal
    y = np.concatenate([np.zeros(2 * SR, np.float32), _tone(3.0)])
    p = tmp_path / "v.wav"; _write(p, y)
    a = analyze_vocal(str(p))
    assert a.active_start_sec == pytest.approx(2.0, abs=0.15)
    assert 60.0 <= a.tempo_bpm <= 140.0
    assert "active_start_sec" in a.to_dict()


# ── window selection ──────────────────────────────────────────────────────────

def test_choose_window_caps_singing_not_silence():
    from vocal_pipeline import choose_window
    # 13 s of silence, singing from 13 s to 100 s, 30 s cap.
    s, e, trunc = choose_window(13.0, 100.0, 120.0, 30.0)
    assert s == pytest.approx(12.5)
    assert e == pytest.approx(42.5)
    assert trunc is True
    # Short clip: nothing truncated, post-roll clamped to the file.
    s, e, trunc = choose_window(0.2, 5.0, 5.5, 30.0)
    assert (s, e, trunc) == (0.0, 5.5, False)


# ── arrangement prompt ────────────────────────────────────────────────────────

def test_arrangement_prompt_is_short_and_instrumental():
    from arrangement import plan_arrangement
    from vocal_analysis import VocalAnalysis
    a = VocalAnalysis(key="A", mode="major", key_name="A major", tempo_bpm=76.0,
                      suggested_genre="pop", suggested_mood="happy",
                      suggested_instruments=["piano", "bass", "drums", "synth"],
                      melody_summary="a happy vocal melody in A major")
    prompt = plan_arrangement(a).prompt
    words = prompt.replace(",", " ").split()
    assert len(words) <= 40, prompt
    assert "A major" in prompt and "76 bpm" in prompt
    assert "instrumental" in prompt
    for banned in ("vocal melody", "sing-along", "catchy hook", "melody theme"):
        assert banned not in prompt


# ── mixing ────────────────────────────────────────────────────────────────────

def test_match_levels_puts_quiet_vocal_on_top():
    from vocal_mixer import match_levels, rms_db
    vocal = np.stack([_tone(4.0, amp=0.01)] * 2)                # very quiet phone take
    rng = np.random.default_rng(1)
    backing = rng.standard_normal((2, 4 * SR)).astype(np.float32) * 0.5   # loud
    v, b, info = match_levels(vocal, backing, SR)
    assert rms_db(v) == pytest.approx(-18.0, abs=0.5)
    assert rms_db(b) < rms_db(v)
    assert info["vocal_gain_db"] > 20


def test_mix_keeps_alignment_and_headroom():
    from vocal_mixer import mix_vocal_over_accompaniment
    vocal = np.stack([_tone(3.0)] * 2)
    backing = np.ones((2, int(2.5 * SR)), np.float32) * 0.3     # shorter → padded
    mix = mix_vocal_over_accompaniment(vocal, backing, SR)
    assert mix.shape == vocal.shape
    assert np.max(np.abs(mix)) <= 0.97 + 1e-6
    assert np.isfinite(mix).all()


# ── pipeline ──────────────────────────────────────────────────────────────────

def test_pipeline_skips_silent_count_in(tmp_path):
    from vocal_pipeline import run_vocal_to_music
    y = np.concatenate([np.zeros(4 * SR, np.float32), _tone(6.0, freq=261.6)])
    p = tmp_path / "v.wav"; _write(p, y)
    stub = _RecordingStub()

    res = run_vocal_to_music(str(p), generator_factory=lambda: stub)

    assert res.window_start_sec == pytest.approx(3.5, abs=0.15)
    assert res.window_end_sec == pytest.approx(10.0, abs=0.05)
    assert res.source_duration_sec == pytest.approx(10.0, abs=0.05)
    assert res.segments == 1
    win = res.window_end_sec - res.window_start_sec
    # One generation call, sized to the sung window, not to the whole file.
    assert len(stub.calls) == 1
    assert stub.calls[0]["duration"] == pytest.approx(win, abs=0.05)
    assert stub.calls[0]["melody_seconds"] == pytest.approx(win, abs=0.05)
    assert any("silence" in w for w in res.warnings)
    # Output length == window length (allow the analysis frame hop of 512 samples).
    assert res.mix.shape[-1] == res.accompaniment.shape[-1]
    assert abs(res.mix.shape[-1] - win * SR) <= 512
    assert np.isfinite(res.mix).all()


def test_pipeline_caps_by_sung_seconds(tmp_path):
    from vocal_pipeline import run_vocal_to_music
    y = np.concatenate([np.zeros(5 * SR, np.float32), _tone(20.0)])
    p = tmp_path / "v.wav"; _write(p, y)
    stub = _RecordingStub()
    res = run_vocal_to_music(str(p), generator_factory=lambda: stub, max_total_seconds=8.0)
    assert res.truncated is True
    assert res.window_start_sec == pytest.approx(4.5, abs=0.15)
    assert res.window_end_sec == pytest.approx(12.5, abs=0.15)
    assert abs(res.mix.shape[-1] - 8.0 * SR) <= 512


def test_pipeline_chains_windows_with_continuation(tmp_path):
    from vocal_pipeline import run_vocal_to_music
    p = tmp_path / "v.wav"; _write(p, _tone(12.0))
    stub = _ContinuationStub()
    res = run_vocal_to_music(str(p), generator_factory=lambda: stub,
                             max_total_seconds=30.0, segment_seconds=5.0)
    assert res.segments == 3
    assert res.continuation_used is True
    assert stub.calls[0]["continuation_seconds"] == 0.0
    assert all(c["continuation_seconds"] > 0 for c in stub.calls[1:])
    # Output stays exactly as long as the vocal — no drift from crossfades.
    assert res.accompaniment.shape[-1] == int(round(12.0 * SR))
    assert res.mix.shape[-1] == int(round(12.0 * SR))


def test_pipeline_crossfades_when_generator_cannot_continue(tmp_path):
    from vocal_pipeline import run_vocal_to_music
    p = tmp_path / "v.wav"; _write(p, _tone(12.0))
    stub = _RecordingStub()
    res = run_vocal_to_music(str(p), generator_factory=lambda: stub, segment_seconds=5.0)
    assert res.segments == 3
    assert res.continuation_used is False
    # Non-final windows were generated with the 0.5 s overlap.
    assert stub.calls[0]["duration"] > stub.calls[-1]["duration"]
    assert res.accompaniment.shape[-1] == int(round(12.0 * SR))


def test_pipeline_tolerates_minimal_stub_signature(tmp_path):
    """The README's bare stub (no progress_cb) must still work."""
    from vocal_pipeline import run_vocal_to_music
    p = tmp_path / "v.wav"; _write(p, _tone(2.0))

    class Bare:
        sample_rate = SR
        def generate(self, prompt, melody_path, duration, guidance_scale, temperature):
            return np.zeros((2, int(duration * SR)), np.float32), SR

    res = run_vocal_to_music(str(p), generator_factory=Bare)
    assert res.mix.shape[0] == 2


def test_real_generator_accepts_continuation_kwarg():
    """Guards the pipeline's feature detection against a signature change."""
    from generator import MusicGenerator
    assert "continuation" in inspect.signature(MusicGenerator.generate).parameters
