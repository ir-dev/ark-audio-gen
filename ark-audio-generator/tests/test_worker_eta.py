"""
Tests that the job worker publishes live ETA and token-count telemetry.

The generator is stubbed (no MusicGen load); we only care that the worker
translates the per-token progress fraction into ``tokens_done`` / ``tokens_total``
and an extrapolated ``eta_seconds`` on the job, which the tracking page renders.
"""

import importlib
import time
from pathlib import Path

import numpy as np
import pytest


class _StubGen:
    """Emits a few progress ticks (with a small delay so an ETA can form)."""
    sample_rate = 32000

    def generate(self, prompt, melody_path=None, duration=15.0,
                 guidance_scale=3.5, temperature=1.05, progress_cb=None):
        if progress_cb:
            progress_cb(0.0)
            time.sleep(0.5)      # let wall-clock elapse so the ETA is meaningful
            progress_cb(0.25)
            progress_cb(1.0)
        return np.zeros((2, 1600), dtype=np.float32), self.sample_rate


@pytest.fixture()
def worker_ctx(tmp_path, monkeypatch):
    monkeypatch.setenv("ARK_DB_PATH", str(tmp_path / "jobs.db"))
    monkeypatch.setenv("ARK_DISABLE_WORKER", "1")

    import job_store
    import job_worker
    import effects
    importlib.reload(job_store)
    importlib.reload(job_worker)
    job_store.init_db()

    out = tmp_path / "generated"
    out.mkdir()
    monkeypatch.setattr(job_worker, "OUTPUT_DIR", out)
    monkeypatch.setattr(job_worker, "write_mp3",
                        lambda audio, sr, path: Path(path).write_bytes(b"ID3"))
    monkeypatch.setattr(job_worker, "_make_generator", lambda use_melody: _StubGen())
    monkeypatch.setattr(effects, "process_audio", lambda audio, sr, **kw: audio)

    # Record every ``result`` dict written while the job runs.
    captured: list[dict] = []
    orig_update = job_store.update_job

    def spy(job_id, **fields):
        if isinstance(fields.get("result"), dict):
            captured.append(fields["result"])
        orig_update(job_id, **fields)

    monkeypatch.setattr(job_store, "update_job", spy)

    return job_store, job_worker, captured


def test_text_job_reports_token_count_and_eta(worker_ctx):
    store, worker, captured = worker_ctx

    jid = store.create_job("text", {"melody": "a bright cheerful tune", "duration": 15.0})
    job = store.claim_next_job()
    worker.process_text(job)

    assert store.get_job(jid)["status"] == store.DONE

    # 15 s → 15*50 + 4 = 754 tokens, reported on each live progress tick.
    live = [r for r in captured if "tokens_total" in r]
    assert live, "expected live token-count progress updates"
    assert all(r["tokens_total"] == 754 for r in live)
    assert any(r["tokens_done"] > 0 for r in live)
    # tokens_done never exceeds the total.
    assert all(0 <= r["tokens_done"] <= 754 for r in live)

    # ETA is extrapolated once generation is under way.
    assert any(r.get("eta_seconds", 0) > 0 for r in captured)


def test_status_view_flattens_token_telemetry(worker_ctx):
    """Live token/ETA telemetry surfaces on the public status view for the UI."""
    store, worker, _ = worker_ctx

    jid = store.create_job("text", {"melody": "another happy melody", "duration": 15.0})
    store.claim_next_job()
    store.update_job(jid, status=store.PROCESSING, progress=42,
                     message="Generating music… 25%",
                     result={"tokens_done": 189, "tokens_total": 754, "eta_seconds": 12})

    view = store.public_view(store.get_job(jid))
    assert view["tokens_done"] == 189
    assert view["tokens_total"] == 754
    assert view["eta_seconds"] == 12
    assert view["progress"] == 42
