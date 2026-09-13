"""
Durable, SQLite-backed job store **and** FIFO queue for generation jobs.

Why this exists
---------------
The previous design kept jobs in a plain in-memory ``dict`` and kicked off
generation from a FastAPI ``BackgroundTask`` serialised by an in-process
``Semaphore(1)``.  That has two fatal flaws on Azure App Service:

  * **No durability** — any worker recycle / OOM wipes every job, so status
    cannot be tracked from a separate screen later on.
  * **No real queue** — a single slow or hung job holds the semaphore forever
    and every later submission sits at ``"pending"`` with no visibility (the
    exact symptom we debugged).

This module replaces both with a persistent queue on disk.  A single
background worker (see :mod:`job_worker`) atomically claims the oldest queued
job, so submission is fully decoupled from processing.

Implementation notes
--------------------
* Pure stdlib (``sqlite3``) — no new dependency.
* WAL journal + ``busy_timeout`` make it safe for the FastAPI request threads
  to read while the worker thread writes.
* The DB lives on Azure's persistent ``/home`` (path via ``ARK_DB_PATH``),
  so it survives container restarts.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Optional

# Default DB location (persistent on Azure — generated/ sits under
# /home/site/wwwroot).  Resolved per-connection via :func:`_db_path` so tests
# can point ``ARK_DB_PATH`` at a temp file without re-importing this module.
_DEFAULT_DB_PATH = "generated/jobs.db"


def _db_path() -> Path:
    return Path(os.environ.get("ARK_DB_PATH", _DEFAULT_DB_PATH))

# Job lifecycle states.
QUEUED = "queued"
PROCESSING = "processing"
DONE = "done"
ERROR = "error"

# Fields a caller is allowed to mutate via :func:`update_job`.
_MUTABLE = {
    "status", "progress", "message", "result", "error",
    "file", "file_accompaniment",
}

# Keys never exposed to the browser by :func:`public_view`.
_INTERNAL = {"params", "file", "file_accompaniment", "result"}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id                  TEXT PRIMARY KEY,
    mode                TEXT NOT NULL,
    status              TEXT NOT NULL,
    progress            INTEGER NOT NULL DEFAULT 0,
    message             TEXT NOT NULL DEFAULT '',
    params              TEXT NOT NULL DEFAULT '{}',
    result              TEXT NOT NULL DEFAULT '{}',
    error               TEXT,
    file                TEXT,
    file_accompaniment  TEXT,
    created_at          REAL NOT NULL,
    updated_at          REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_jobs_status_created ON jobs(status, created_at);
"""


# ──────────────────────────────────────────────────────────────────────────────
# Connections
# ──────────────────────────────────────────────────────────────────────────────

# Strictly-monotonic timestamps so ``created_at`` is a total order even when two
# jobs are enqueued within the same clock tick — makes FIFO claiming and queue
# positions deterministic.
_ts_lock = threading.Lock()
_last_ts = 0.0


def _now() -> float:
    global _last_ts
    with _ts_lock:
        t = time.time()
        if t <= _last_ts:
            t = _last_ts + 1e-6
        _last_ts = t
        return t


def _connect() -> sqlite3.Connection:
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def init_db() -> None:
    """Create the schema (idempotent)."""
    with _connect() as conn:
        conn.executescript(_SCHEMA)


def recover_orphans() -> int:
    """
    Repair jobs left mid-flight by a previous (crashed/recycled) worker.

    A job stuck in ``processing`` cannot be resumed — the in-memory model
    state is gone — so it is marked ``error`` with a clear message.  Jobs still
    ``queued`` are left untouched; the worker will pick them up normally.

    Returns the number of orphaned jobs repaired.
    """
    now = time.time()
    with _connect() as conn:
        cur = conn.execute(
            "UPDATE jobs SET status=?, error=?, message=?, updated_at=? "
            "WHERE status=?",
            (
                ERROR,
                "Interrupted by a server restart.",
                "Interrupted by a server restart — please resubmit.",
                now,
                PROCESSING,
            ),
        )
        return cur.rowcount


# ──────────────────────────────────────────────────────────────────────────────
# Row helpers
# ──────────────────────────────────────────────────────────────────────────────

def _row_to_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    d["params"] = json.loads(d.get("params") or "{}")
    d["result"] = json.loads(d.get("result") or "{}")
    return d


# ──────────────────────────────────────────────────────────────────────────────
# CRUD
# ──────────────────────────────────────────────────────────────────────────────

def create_job(mode: str, params: dict) -> str:
    """Enqueue a new job and return its id."""
    job_id = str(uuid.uuid4())
    now = _now()
    with _connect() as conn:
        conn.execute(
            "INSERT INTO jobs "
            "(id, mode, status, progress, message, params, result, created_at, updated_at) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (job_id, mode, QUEUED, 0, "Queued…", json.dumps(params), "{}", now, now),
        )
    return job_id


def update_job(job_id: str, **fields: Any) -> None:
    """Update mutable columns of a job.  ``result`` may be a dict (JSON-encoded)."""
    fields = {k: v for k, v in fields.items() if k in _MUTABLE}
    if not fields:
        return
    if "result" in fields and not isinstance(fields["result"], str):
        fields["result"] = json.dumps(fields["result"])
    fields["updated_at"] = time.time()
    cols = ", ".join(f"{k}=?" for k in fields)
    vals = list(fields.values()) + [job_id]
    with _connect() as conn:
        conn.execute(f"UPDATE jobs SET {cols} WHERE id=?", vals)


def get_job(job_id: str) -> Optional[dict]:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
    return _row_to_dict(row) if row else None


def list_jobs(limit: int = 50) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [_row_to_dict(r) for r in rows]


def delete_job(job_id: str) -> Optional[dict]:
    """Delete a job row; return the job (pre-delete) so the caller can bin files."""
    job = get_job(job_id)
    if job is None:
        return None
    with _connect() as conn:
        conn.execute("DELETE FROM jobs WHERE id=?", (job_id,))
    return job


def claim_next_job() -> Optional[dict]:
    """
    Atomically claim the oldest queued job, flipping it to ``processing``.

    Uses ``BEGIN IMMEDIATE`` so that even with multiple workers exactly one can
    claim a given job.  Returns the claimed job (already marked processing) or
    ``None`` when the queue is empty.
    """
    conn = _connect()
    try:
        conn.isolation_level = None  # manual transaction control
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT * FROM jobs WHERE status=? ORDER BY created_at ASC LIMIT 1",
            (QUEUED,),
        ).fetchone()
        if row is None:
            conn.execute("ROLLBACK")
            return None
        now = time.time()
        conn.execute(
            "UPDATE jobs SET status=?, message=?, progress=?, updated_at=? WHERE id=?",
            (PROCESSING, "Starting…", 1, now, row["id"]),
        )
        conn.execute("COMMIT")
    finally:
        conn.close()

    job = _row_to_dict(row)
    job["status"] = PROCESSING
    job["progress"] = 1
    return job


# ──────────────────────────────────────────────────────────────────────────────
# Queue introspection
# ──────────────────────────────────────────────────────────────────────────────

def queue_position(job: dict) -> int:
    """
    1-based position of a queued job in line (counts a running job as ahead).

    Returns 0 for jobs that are not queued.
    """
    if job.get("status") != QUEUED:
        return 0
    with _connect() as conn:
        ahead = conn.execute(
            "SELECT COUNT(*) FROM jobs WHERE status=? AND created_at < ?",
            (QUEUED, job["created_at"]),
        ).fetchone()[0]
        running = conn.execute(
            "SELECT COUNT(*) FROM jobs WHERE status=?", (PROCESSING,)
        ).fetchone()[0]
    return ahead + running + 1


def cleanup_expired(ttl_seconds: float) -> list[dict]:
    """Delete jobs older than ``ttl_seconds``; return them so files can be removed."""
    cutoff = time.time() - ttl_seconds
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM jobs WHERE created_at < ?", (cutoff,)
        ).fetchall()
        conn.execute("DELETE FROM jobs WHERE created_at < ?", (cutoff,))
    return [_row_to_dict(r) for r in rows]


# ──────────────────────────────────────────────────────────────────────────────
# Presentation
# ──────────────────────────────────────────────────────────────────────────────

def public_view(job: dict) -> dict:
    """Strip internal fields and flatten ``result`` for the browser/API."""
    out = {k: v for k, v in job.items() if k not in _INTERNAL}
    out.update(job.get("result") or {})
    if job.get("status") == QUEUED:
        out["queue_position"] = queue_position(job)
    out["has_accompaniment"] = bool(job.get("file_accompaniment"))
    if not out.get("error"):
        out.pop("error", None)
    return out


__all__ = [
    "QUEUED", "PROCESSING", "DONE", "ERROR",
    "init_db", "recover_orphans", "create_job", "update_job", "get_job",
    "list_jobs", "delete_job", "claim_next_job", "queue_position",
    "cleanup_expired", "public_view",
]
