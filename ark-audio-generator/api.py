"""
FastAPI web application for the AI Sing-Along Music Generator.

Routes
------
GET    /                          Serve index.html (UI)
GET    /jobs.html                 Job tracking screen (served statically)
POST   /api/generate             Enqueue a text→music job → {job_id}
POST   /api/vocal/analyze        Fast, model-free vocal analysis
POST   /api/vocal/generate       Enqueue a vocal→music job → {job_id}
GET    /api/status/{job_id}      Poll one job's status
GET    /api/jobs                 List recent jobs (powers the tracking screen)
GET    /api/download/{job_id}    Stream the finished MP3 (?variant=mix|accompaniment)
GET    /api/health               Azure load-balancer probe
DELETE /api/job/{job_id}         Client-side cleanup (optional)

Submission is **decoupled from processing**: a job is written to a durable
SQLite queue (:mod:`job_store`) and a single background worker
(:mod:`job_worker`) drains it FIFO.  Because the queue and status live on disk,
a job can be tracked from a separate screen at any later time, and a slow job
can never wedge the service invisibly.
"""

from __future__ import annotations

import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

import job_store
import job_worker

# ──────────────────────────────────────────────────────────────────────────────
# App setup
# ──────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(_app: FastAPI):
    # ── Startup: prepare the queue and launch the worker ──────────────────────
    job_store.init_db()
    job_store.recover_orphans()      # re-queue any jobs a prior crash left mid-flight
    _purge_expired()
    job_worker.start()               # begins draining the queue (unless disabled)
    yield
    # ── Shutdown: stop the worker thread cleanly ──────────────────────────────
    job_worker.stop()


app = FastAPI(
    title="AI Sing-Along Music Generator",
    description="Generate rhythmic, lively sing-along backing tracks using Meta MusicGen.",
    version="2.0.0",
    lifespan=lifespan,
)

OUTPUT_DIR = Path("generated")
OUTPUT_DIR.mkdir(exist_ok=True)

# Uploaded vocals live here transiently (removed once a job finishes).
UPLOAD_DIR = OUTPUT_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# Accepted vocal upload formats / size cap.
_VOCAL_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".aiff", ".m4a", ".aac"}
MAX_UPLOAD_BYTES = 25 * 1024 * 1024   # 25 MB

# How long to keep finished jobs (and their files) before cleanup.
JOB_TTL = float(os.environ.get("ARK_JOB_TTL_SECONDS", str(24 * 3600)))   # 24 h


# ──────────────────────────────────────────────────────────────────────────────
# Request / response schemas
# ──────────────────────────────────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    melody: str = Field(..., min_length=3, max_length=600,
                        description="Melody description or absolute path to an audio file.")
    genre: Optional[str] = Field(None, description="pop | rock | jazz | classical | folk | electronic | hip-hop | r-and-b | ambient | reggae | bossa-nova")
    mood: Optional[str] = Field(None, description="happy | sad | energetic | calm | romantic | uplifting | mysterious | aggressive | nostalgic | playful")
    instruments: Optional[str] = Field(None, description="Comma-separated list, e.g. 'piano,guitar,drums'")
    frequency_range: Optional[str] = Field(None, description="bass | mid | treble | full")
    duration: float = Field(default=15.0, ge=5.0, le=20.0, description="Length in seconds (5–20)")
    crescendo: str = Field(default="rise-fall", description="rise | fall | rise-fall | natural | verse-chorus")
    guidance_scale: float = Field(default=3.5, ge=1.0, le=10.0)
    temperature: float = Field(default=1.05, ge=0.5, le=1.5)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _remove_job_files(job: dict) -> None:
    """Delete every on-disk artefact referenced by a job."""
    for key in ("file", "file_accompaniment"):
        val = job.get(key)
        if val:
            try:
                Path(val).unlink(missing_ok=True)
            except OSError:
                pass


def _purge_expired() -> None:
    """Remove jobs (and their output files) older than :data:`JOB_TTL`."""
    for job in job_store.cleanup_expired(JOB_TTL):
        _remove_job_files(job)


def _build_overrides(
    genre: Optional[str],
    mood: Optional[str],
    instruments: Optional[str],
    tempo_bpm: Optional[float],
    guidance_scale: Optional[float],
    temperature: Optional[float],
    crescendo: Optional[str],
) -> dict:
    """Assemble arrangement overrides from optional form fields (blanks ignored)."""
    inst_list = (
        [i.strip() for i in instruments.split(",") if i.strip()]
        if instruments else None
    )
    overrides = {
        "genre": genre,
        "mood": mood,
        "instruments": inst_list,
        "tempo_bpm": tempo_bpm,
        "guidance_scale": guidance_scale,
        "temperature": temperature,
        "crescendo": crescendo,
    }
    return {k: v for k, v in overrides.items() if v not in (None, "", [])}


async def _save_vocal_upload(file: UploadFile) -> str:
    """Validate and persist an uploaded vocal file; return its temp path."""
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in _VOCAL_EXTS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format '{suffix or '?'}'. "
                   f"Use one of: {', '.join(sorted(_VOCAL_EXTS))}",
        )
    data = await file.read()
    if len(data) < 1024:
        raise HTTPException(status_code=400, detail="Upload is empty or too small.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large (max {MAX_UPLOAD_BYTES // (1024 * 1024)} MB).",
        )
    fd, path = tempfile.mkstemp(suffix=suffix, dir=str(UPLOAD_DIR))
    with os.fdopen(fd, "wb") as fh:
        fh.write(data)
    return path


def _analyze_vocal_sync(vocal_path: str) -> dict:
    """Run analysis only (no generation); used by /api/vocal/analyze."""
    try:
        from vocal_analysis import analyze_vocal
        return analyze_vocal(vocal_path).to_dict()
    finally:
        try:
            Path(vocal_path).unlink(missing_ok=True)
        except OSError:
            pass


# ──────────────────────────────────────────────────────────────────────────────
# API routes  (register BEFORE static-file mount)
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/api/health", tags=["ops"])
async def health():
    jobs = job_store.list_jobs(limit=1000)
    active = sum(1 for j in jobs if j["status"] in (job_store.QUEUED, job_store.PROCESSING))
    return {"status": "ok", "jobs": len(jobs), "active": active}


@app.post("/api/generate", tags=["generation"])
async def start_generation(req: GenerateRequest):
    _purge_expired()
    job_id = job_store.create_job("text", req.model_dump())
    return {"job_id": job_id}


@app.post("/api/vocal/analyze", tags=["vocal"])
async def analyze_vocal_endpoint(file: UploadFile = File(...)):
    """
    Analyse an uploaded vocal recording and return its musical characteristics.

    Fast, model-free — intended for the UI to preview detected key/tempo/mood
    before committing to a (slow) generation job.
    """
    vocal_path = await _save_vocal_upload(file)
    analysis = await run_in_threadpool(_analyze_vocal_sync, vocal_path)
    return {"analysis": analysis}


@app.post("/api/vocal/generate", tags=["vocal"])
async def start_vocal_generation(
    file: UploadFile = File(...),
    genre: Optional[str] = Form(None),
    mood: Optional[str] = Form(None),
    instruments: Optional[str] = Form(None),
    tempo_bpm: Optional[float] = Form(None),
    guidance_scale: Optional[float] = Form(None),
    temperature: Optional[float] = Form(None),
    crescendo: Optional[str] = Form(None),
):
    """
    Upload a vocal recording and enqueue a complementary-accompaniment job.

    All arrangement fields are optional — anything omitted is auto-detected from
    the vocal.  Returns ``{job_id}``; poll ``/api/status/{job_id}`` for progress
    (or watch it on ``/jobs.html``), then download via ``/api/download/{job_id}``
    (``?variant=mix`` default, or ``?variant=accompaniment``).
    """
    _purge_expired()
    vocal_path = await _save_vocal_upload(file)
    overrides = _build_overrides(
        genre, mood, instruments, tempo_bpm, guidance_scale, temperature, crescendo,
    )
    job_id = job_store.create_job(
        "vocal", {"vocal_path": vocal_path, "overrides": overrides}
    )
    return {"job_id": job_id}


@app.get("/api/status/{job_id}", tags=["generation"])
async def get_status(job_id: str):
    job = job_store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job_store.public_view(job)


@app.get("/api/jobs", tags=["generation"])
async def list_jobs(limit: int = Query(50, ge=1, le=200)):
    """Recent jobs, newest first — powers the /jobs.html tracking screen."""
    _purge_expired()
    return {"jobs": [job_store.public_view(j) for j in job_store.list_jobs(limit)]}


@app.get("/api/download/{job_id}", tags=["generation"])
async def download_track(job_id: str, variant: str = Query("mix")):
    job = job_store.get_job(job_id)
    if job is None or job.get("status") != job_store.DONE:
        raise HTTPException(status_code=404, detail="Track not ready or not found")

    if variant == "accompaniment":
        file_path = job.get("file_accompaniment")
        filename = "ark_accompaniment.mp3"
    else:
        file_path = job.get("file")
        filename = "ark_vocal_track.mp3" if job.get("mode") == "vocal" \
            else "ai_singalong_track.mp3"

    if not file_path or not Path(file_path).exists():
        raise HTTPException(status_code=404, detail="File missing on server")
    return FileResponse(
        path=file_path,
        media_type="audio/mpeg",
        filename=filename,
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@app.delete("/api/job/{job_id}", tags=["generation"])
async def delete_job(job_id: str):
    job = job_store.delete_job(job_id)
    if job:
        _remove_job_files(job)
    return {"deleted": job_id}


# ──────────────────────────────────────────────────────────────────────────────
# Static files (SPA fallback) — MUST be last
# ──────────────────────────────────────────────────────────────────────────────

app.mount("/", StaticFiles(directory="static", html=True), name="static")


# ──────────────────────────────────────────────────────────────────────────────
# Run directly:  python api.py
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    print(f"\n  Open your browser at  http://localhost:{port}\n")
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=True)
