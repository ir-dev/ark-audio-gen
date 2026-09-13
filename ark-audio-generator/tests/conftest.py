"""
Shared pytest fixtures.

Every test runs against an isolated temp SQLite DB (``ARK_DB_PATH``) and with the
background worker disabled (``ARK_DISABLE_WORKER=1``) so the queue can be driven
deterministically via ``job_worker.run_one()``.
"""

import importlib
import os
import sys
from pathlib import Path

import pytest

# Make the project root importable (tests/ is a subdir).
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture()
def store(tmp_path, monkeypatch):
    """Fresh, isolated job_store bound to a temp DB."""
    db = tmp_path / "jobs.db"
    monkeypatch.setenv("ARK_DB_PATH", str(db))
    monkeypatch.setenv("ARK_DISABLE_WORKER", "1")

    import job_store
    importlib.reload(job_store)          # pick up env; reset monotonic clock
    job_store.init_db()
    return job_store
