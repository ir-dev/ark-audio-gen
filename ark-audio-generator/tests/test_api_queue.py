"""
End-to-end queue tests through the FastAPI app.

The heavy MusicGen processors are replaced with fast fakes so we exercise the
real submit → enqueue → claim → status → download → delete flow without loading
any model.
"""

import importlib
import io
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient


def _wav_bytes(seconds: float = 0.5, sr: int = 16000) -> bytes:
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    y = (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, y, sr, format="WAV")
    return buf.getvalue()


@pytest.fixture()
def ctx(tmp_path, monkeypatch):
    monkeypatch.setenv("ARK_DB_PATH", str(tmp_path / "jobs.db"))
    monkeypatch.setenv("ARK_DISABLE_WORKER", "1")
    monkeypatch.setenv("ARK_JOB_TTL_SECONDS", str(24 * 3600))

    import job_store
    import job_worker
    import api
    importlib.reload(job_store)
    importlib.reload(job_worker)
    importlib.reload(api)

    out = tmp_path / "generated"
    (out / "uploads").mkdir(parents=True)
    monkeypatch.setattr(api, "OUTPUT_DIR", out)
    monkeypatch.setattr(api, "UPLOAD_DIR", out / "uploads")
    monkeypatch.setattr(job_worker, "OUTPUT_DIR", out)

    # ── Fast fake processors ──────────────────────────────────────────────────
    def fake_text(job):
        jid = job["id"]
        job_store.update_job(jid, status=job_store.PROCESSING, progress=30, message="gen")
        f = out / f"{jid}.mp3"
        f.write_bytes(b"ID3" + b"\x00" * 2048)
        job_store.update_job(
            jid, status=job_store.DONE, progress=100, message="ready",
            file=str(f),
            result={"mode": "text", "genre": "pop", "mood": "happy", "duration": 15.0},
        )

    def fake_vocal(job):
        jid = job["id"]
        vp = job["params"]["vocal_path"]
        assert Path(vp).exists(), "upload should be saved before the worker runs"
        f = out / f"{jid}.mp3"
        fa = out / f"{jid}_accompaniment.mp3"
        f.write_bytes(b"ID3" + b"\x00" * 2048)
        fa.write_bytes(b"ID3" + b"\x00" * 2048)
        Path(vp).unlink(missing_ok=True)
        job_store.update_job(
            jid, status=job_store.DONE, progress=100, message="ready",
            file=str(f), file_accompaniment=str(fa),
            result={"mode": "vocal", "genre": "folk", "key": "C major",
                    "tempo": 100, "duration": 10.0},
        )

    monkeypatch.setattr(job_worker, "PROCESSORS",
                        {"text": fake_text, "vocal": fake_vocal})

    with TestClient(api.app) as client:
        yield client, job_store, job_worker


def test_health(ctx):
    client, *_ = ctx
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_text_job_end_to_end(ctx):
    client, store, worker = ctx

    r = client.post("/api/generate", json={"melody": "a cheerful whistling tune"})
    assert r.status_code == 200
    jid = r.json()["job_id"]

    # Immediately queued, not yet processed.
    s = client.get(f"/api/status/{jid}").json()
    assert s["status"] == "queued"
    assert s["progress"] == 0
    assert s["queue_position"] == 1

    assert worker.run_one() is True          # worker drains it
    assert worker.run_one() is False         # queue now empty

    s = client.get(f"/api/status/{jid}").json()
    assert s["status"] == "done"
    assert s["progress"] == 100
    assert s["genre"] == "pop"

    # Download works.
    d = client.get(f"/api/download/{jid}")
    assert d.status_code == 200
    assert d.headers["content-type"] == "audio/mpeg"

    # Appears in the tracking list.
    jobs = client.get("/api/jobs").json()["jobs"]
    assert any(j["id"] == jid and j["status"] == "done" for j in jobs)


def test_vocal_job_end_to_end(ctx):
    client, store, worker = ctx

    files = {"file": ("take.wav", _wav_bytes(), "audio/wav")}
    r = client.post("/api/vocal/generate", files=files, data={"genre": "folk"})
    assert r.status_code == 200
    jid = r.json()["job_id"]

    s = client.get(f"/api/status/{jid}").json()
    assert s["status"] == "queued"
    assert s["mode"] == "vocal"

    assert worker.run_one() is True

    s = client.get(f"/api/status/{jid}").json()
    assert s["status"] == "done"
    assert s["has_accompaniment"] is True
    assert s["key"] == "C major"

    # Both variants downloadable.
    assert client.get(f"/api/download/{jid}?variant=mix").status_code == 200
    assert client.get(f"/api/download/{jid}?variant=accompaniment").status_code == 200


def test_queue_positions_and_serialisation(ctx):
    client, store, worker = ctx

    a = client.post("/api/generate", json={"melody": "first tune here"}).json()["job_id"]
    b = client.post("/api/generate", json={"melody": "second tune here"}).json()["job_id"]

    assert client.get(f"/api/status/{b}").json()["queue_position"] == 2

    worker.run_one()                          # processes A (FIFO)
    assert client.get(f"/api/status/{a}").json()["status"] == "done"
    # B still queued, now at the front.
    sb = client.get(f"/api/status/{b}").json()
    assert sb["status"] == "queued"
    assert sb["queue_position"] == 1

    worker.run_one()
    assert client.get(f"/api/status/{b}").json()["status"] == "done"


def test_processor_failure_marks_error(ctx):
    client, store, worker = ctx

    def boom(job):
        raise RuntimeError("model blew up")

    worker.PROCESSORS["text"] = boom
    jid = client.post("/api/generate", json={"melody": "will fail here"}).json()["job_id"]

    assert worker.run_one() is True
    s = client.get(f"/api/status/{jid}").json()
    assert s["status"] == "error"
    assert "model blew up" in s["message"]
    assert s["progress"] == 0


def test_status_404_for_unknown(ctx):
    client, *_ = ctx
    assert client.get("/api/status/nope").status_code == 404


def test_delete_removes_job_and_files(ctx):
    client, store, worker = ctx
    jid = client.post("/api/generate", json={"melody": "delete me please"}).json()["job_id"]
    worker.run_one()

    file_path = Path(store.get_job(jid)["file"])
    assert file_path.exists()

    r = client.delete(f"/api/job/{jid}")
    assert r.status_code == 200
    assert not file_path.exists()
    assert client.get(f"/api/status/{jid}").status_code == 404


def test_upload_rejects_bad_extension(ctx):
    client, *_ = ctx
    files = {"file": ("notes.txt", b"x" * 4096, "text/plain")}
    r = client.post("/api/vocal/generate", files=files)
    assert r.status_code == 400
