"""
Single source of truth for where the service keeps state on disk.

Everything persistent — the SQLite job DB, generated tracks, and uploaded
vocals — lives under one **data root** so it survives not just a container
restart but a *redeploy*.

Why a dedicated root (and not ``generated/`` under the app)
----------------------------------------------------------
On Azure App Service only ``/home`` is persistent, and a ``--clean`` ZIP deploy
wipes ``/home/site/wwwroot``.  If state lived under the app directory it would
be erased on every deploy.  So the data root must sit *outside* wwwroot —
``startup.sh`` points ``ARK_DATA_DIR`` at ``/home/data``.  Locally (where
``startup.sh`` isn't sourced) it falls back to a ``generated/`` folder beside
the code, so the dev experience is unchanged.

Overrides
---------
* ``ARK_DATA_DIR`` — the data root (default ``generated``).
* ``ARK_DB_PATH``  — the DB file specifically; still honoured so the test suite
  can point it at an isolated temp file without moving everything else.
"""

from __future__ import annotations

import os
from pathlib import Path

_DEFAULT_DATA_DIR = "generated"


def data_dir() -> Path:
    """Root under which all persistent state lives (override: ``ARK_DATA_DIR``)."""
    return Path(os.environ.get("ARK_DATA_DIR", _DEFAULT_DATA_DIR))


def output_dir() -> Path:
    """Directory that finished tracks (``{job_id}.mp3``) are written to."""
    return data_dir()


def upload_dir() -> Path:
    """Directory for transiently-stored uploaded vocals."""
    return data_dir() / "uploads"


def db_path() -> Path:
    """SQLite job-DB path.  ``ARK_DB_PATH`` wins if set (used by the tests)."""
    return Path(os.environ.get("ARK_DB_PATH") or (data_dir() / "jobs.db"))


def ensure_dirs() -> None:
    """Create the data root and its subdirectories (idempotent)."""
    output_dir().mkdir(parents=True, exist_ok=True)
    upload_dir().mkdir(parents=True, exist_ok=True)


__all__ = ["data_dir", "output_dir", "upload_dir", "db_path", "ensure_dirs"]
