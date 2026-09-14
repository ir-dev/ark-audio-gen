"""Unit tests for the durable job store / FIFO queue."""

import time


def test_create_and_get(store):
    jid = store.create_job("text", {"melody": "a happy tune"})
    job = store.get_job(jid)
    assert job is not None
    assert job["status"] == store.QUEUED
    assert job["progress"] == 0
    assert job["params"]["melody"] == "a happy tune"
    assert job["mode"] == "text"


def test_get_missing_returns_none(store):
    assert store.get_job("does-not-exist") is None


def test_update_job_persists_and_encodes_result(store):
    jid = store.create_job("text", {"melody": "x"})
    store.update_job(jid, status=store.PROCESSING, progress=42, message="halfway")
    store.update_job(jid, status=store.DONE, progress=100,
                     result={"genre": "pop", "duration": 15.0}, file="generated/x.mp3")
    job = store.get_job(jid)
    assert job["status"] == store.DONE
    assert job["progress"] == 100
    assert job["result"] == {"genre": "pop", "duration": 15.0}
    assert job["file"] == "generated/x.mp3"


def test_update_ignores_unknown_fields(store):
    jid = store.create_job("text", {"melody": "x"})
    store.update_job(jid, status=store.PROCESSING, bogus="nope", id="hacked")
    job = store.get_job(jid)
    assert job["id"] == jid          # PK not mutated
    assert "bogus" not in job


def test_claim_is_fifo(store):
    ids = [store.create_job("text", {"melody": str(i)}) for i in range(3)]
    claimed = [store.claim_next_job()["id"] for _ in range(3)]
    assert claimed == ids            # oldest-first
    assert store.claim_next_job() is None   # queue drained


def test_claim_flips_to_processing(store):
    jid = store.create_job("text", {"melody": "x"})
    job = store.claim_next_job()
    assert job["id"] == jid
    assert job["status"] == store.PROCESSING
    # Persisted, not just returned.
    assert store.get_job(jid)["status"] == store.PROCESSING


def test_queue_position(store):
    ids = [store.create_job("text", {"melody": str(i)}) for i in range(3)]
    # Nothing claimed yet: positions 1,2,3.
    positions = [store.queue_position(store.get_job(i)) for i in ids]
    assert positions == [1, 2, 3]

    # Claim the first → it's processing; the next queued job is now position 2
    # (1 running ahead + 0 queued ahead + 1).
    store.claim_next_job()
    assert store.queue_position(store.get_job(ids[1])) == 2
    assert store.queue_position(store.get_job(ids[2])) == 3
    # A processing/done job has position 0.
    assert store.queue_position(store.get_job(ids[0])) == 0


def test_recover_orphans_requeues(store):
    # An interrupted (processing) job is put back on the queue, not failed, so
    # the worker re-runs it from the top after a restart.
    jid = store.create_job("text", {"melody": "y"})
    store.claim_next_job()                      # now processing (attempt 1)
    assert store.get_job(jid)["status"] == store.PROCESSING

    n = store.recover_orphans()
    assert n == 1
    job = store.get_job(jid)
    assert job["status"] == store.QUEUED
    assert job["progress"] == 0
    assert "resum" in job["message"].lower()
    assert job.get("error") is None

    # It's genuinely re-claimable, and each claim counts as an attempt.
    reclaimed = store.claim_next_job()
    assert reclaimed["id"] == jid
    assert reclaimed["attempts"] == 2

    # Queued jobs are untouched by recovery.
    qid = store.create_job("text", {"melody": "z"})
    assert store.recover_orphans() == 1         # only the reclaimed one above
    assert store.get_job(qid)["status"] == store.QUEUED


def test_recover_orphans_gives_up_after_max_attempts(store, monkeypatch):
    # A job that keeps getting interrupted is eventually failed rather than
    # looping forever (guards against a job that crashes the container).
    monkeypatch.setattr(store, "_MAX_ATTEMPTS", 2)
    jid = store.create_job("text", {"melody": "y"})

    store.claim_next_job()                      # attempt 1
    assert store.recover_orphans() == 1
    assert store.get_job(jid)["status"] == store.QUEUED

    store.claim_next_job()                      # attempt 2 (== cap)
    assert store.recover_orphans() == 1
    job = store.get_job(jid)
    assert job["status"] == store.ERROR
    assert "repeatedly" in job["message"].lower()


def test_delete_job(store):
    jid = store.create_job("text", {"melody": "x"})
    removed = store.delete_job(jid)
    assert removed["id"] == jid
    assert store.get_job(jid) is None
    assert store.delete_job(jid) is None        # idempotent


def test_list_jobs_newest_first(store):
    ids = [store.create_job("text", {"melody": str(i)}) for i in range(3)]
    listed = [j["id"] for j in store.list_jobs()]
    assert listed == list(reversed(ids))


def test_cleanup_expired(store):
    jid = store.create_job("text", {"melody": "old"})
    # Force the row to look old.
    with store._connect() as conn:
        conn.execute("UPDATE jobs SET created_at=? WHERE id=?",
                     (time.time() - 10_000, jid))
    fresh = store.create_job("text", {"melody": "new"})

    removed = store.cleanup_expired(ttl_seconds=3600)
    assert [r["id"] for r in removed] == [jid]
    assert store.get_job(jid) is None
    assert store.get_job(fresh) is not None


def test_public_view_hides_internal_fields(store):
    jid = store.create_job("vocal", {"vocal_path": "/tmp/secret.wav"})
    store.update_job(jid, status=store.DONE, progress=100,
                     file="generated/a.mp3", file_accompaniment="generated/a_acc.mp3",
                     result={"genre": "jazz", "key": "C major"})
    view = store.public_view(store.get_job(jid))

    assert "params" not in view          # hides the server-side upload path
    assert "file" not in view
    assert "file_accompaniment" not in view
    assert view["has_accompaniment"] is True
    assert view["genre"] == "jazz"       # result flattened in
    assert view["key"] == "C major"
    assert view["id"] == jid


def test_public_view_includes_queue_position_when_queued(store):
    store.create_job("text", {"melody": "a"})
    jid = store.create_job("text", {"melody": "b"})
    view = store.public_view(store.get_job(jid))
    assert view["queue_position"] == 2
