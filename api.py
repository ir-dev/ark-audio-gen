"""
FastAPI web application for the AI Sing-Along Music Generator.

Routes
------
GET  /                          Serve index.html (UI)
POST /api/generate              Submit a generation job → {job_id}
GET  /api/status/{job_id}       Poll job status
GET  /api/download/{job_id}     Stream the finished MP3
GET  /api/health                Azure load-balancer probe
DELETE /api/job/{job_id}        Client-side cleanup (optional)

Generation runs in a thread-pool so no HTTP request ever blocks the event
loop. The client polls /api/status every 3 s until status == "done"|"error".
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# ──────────────────────────────────────────────────────────────────────────────
# App setup
# ──────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="AI Sing-Along Music Generator",
    description="Generate rhythmic, lively sing-along backing tracks using Meta MusicGen.",
    version="1.0.0",
)

OUTPUT_DIR = Path("generated")
OUTPUT_DIR.mkdir(exist_ok=True)

# One generation at a time on CPU (prevents OOM on small Azure SKUs)
_gen_semaphore = threading.Semaphore(1)

# Thread pool – keep separate from the asyncio event loop
_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="gen")

# In-memory job store  {job_id: {...}}
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()

# How long (seconds) to keep finished jobs before cleanup
JOB_TTL = 3600  # 1 hour


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
# Job utilities
# ──────────────────────────────────────────────────────────────────────────────

def _set_job(job_id: str, **fields) -> None:
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id].update(fields)
        else:
            _jobs[job_id] = {"created_at": time.time(), **fields}


def _get_job(job_id: str) -> dict | None:
    with _jobs_lock:
        return dict(_jobs[job_id]) if job_id in _jobs else None


def _cleanup_expired() -> None:
    """Remove jobs (and their output files) that are older than JOB_TTL."""
    cutoff = time.time() - JOB_TTL
    with _jobs_lock:
        expired = [jid for jid, j in _jobs.items() if j.get("created_at", 0) < cutoff]
        for jid in expired:
            fp = _jobs[jid].get("_file")
            if fp:
                try:
                    Path(fp).unlink(missing_ok=True)
                except OSError:
                    pass
            del _jobs[jid]


# ──────────────────────────────────────────────────────────────────────────────
# Core generation (runs in thread pool)
# ──────────────────────────────────────────────────────────────────────────────

def _generate_sync(job_id: str, req: GenerateRequest) -> None:
    """Blocking generation pipeline.  Called from ThreadPoolExecutor."""

    _gen_semaphore.acquire()
    try:
        # ── Imports (deferred so startup is instant) ──────────────────────────
        from effects import process_audio
        from generator import MusicGenerator
        from prompt_builder import build_prompt, infer_parameters

        # ── Detect melody type ────────────────────────────────────────────────
        _AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".aiff"}
        p = Path(req.melody)
        is_audio_file = p.suffix.lower() in _AUDIO_EXTS and p.exists()
        melody_path = str(p) if is_audio_file else None
        melody_desc = "melody from audio file" if is_audio_file else req.melody

        # ── Build prompt ──────────────────────────────────────────────────────
        inferred = infer_parameters(melody_desc)
        inst_list = (
            [i.strip() for i in req.instruments.split(",") if i.strip()]
            if req.instruments
            else None
        )
        prompt = build_prompt(
            melody_description=melody_desc,
            genre=req.genre,
            mood=req.mood,
            instruments=inst_list,
            frequency_range=req.frequency_range,
            inferred=inferred,
        )
        effective_genre = req.genre or inferred["genre"]
        effective_mood = req.mood or inferred["mood"]

        # ── Generate ──────────────────────────────────────────────────────────
        _set_job(job_id, status="processing", message="Loading AI model…", progress=10)
        gen = MusicGenerator(use_melody_model=is_audio_file)
        _set_job(job_id, message="Generating music… (2–10 min on CPU)", progress=30)
        audio, sr = gen.generate(
            prompt=prompt,
            melody_path=melody_path,
            duration=req.duration,
            guidance_scale=req.guidance_scale,
            temperature=req.temperature,
        )

        # ── Effects ───────────────────────────────────────────────────────────
        _set_job(job_id, message="Applying audio effects…", progress=80)
        audio = process_audio(
            audio, sr,
            genre=effective_genre,
            mood=effective_mood,
            crescendo_pattern=req.crescendo,
        )

        # ── Export MP3 ────────────────────────────────────────────────────────
        _set_job(job_id, message="Exporting to MP3…", progress=92)
        out_file = OUTPUT_DIR / f"{job_id}.mp3"
        _write_mp3(audio, sr, out_file)

        _set_job(
            job_id,
            status="done",
            message="Your track is ready!",
            progress=100,
            _file=str(out_file),
            genre=effective_genre,
            mood=effective_mood,
            duration=req.duration,
        )

    except Exception as exc:
        _set_job(
            job_id,
            status="error",
            message=f"Generation failed: {exc}",
            progress=0,
            _detail=traceback.format_exc(),
        )
    finally:
        _gen_semaphore.release()


def _write_mp3(audio: np.ndarray, sr: int, out_path: Path, bitrate: str = "192k") -> None:
    import soundfile as sf
    from pydub import AudioSegment

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_wav = tmp.name
    try:
        data = audio.T if audio.ndim == 2 else audio
        peak = np.max(np.abs(data))
        if peak > 0:
            data = (data / peak * 0.97).astype(np.float32)
        sf.write(tmp_wav, data, sr, format="WAV", subtype="FLOAT")
        sound = AudioSegment.from_wav(tmp_wav)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sound.export(str(out_path), format="mp3", bitrate=bitrate)
    finally:
        try:
            os.unlink(tmp_wav)
        except OSError:
            pass


# ──────────────────────────────────────────────────────────────────────────────
# API routes  (register BEFORE static-file mount)
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/api/health", tags=["ops"])
async def health():
    return {"status": "ok", "jobs": len(_jobs)}


@app.post("/api/generate", tags=["generation"])
async def start_generation(req: GenerateRequest, background_tasks: BackgroundTasks):
    _cleanup_expired()

    job_id = str(uuid.uuid4())
    _set_job(job_id, status="pending", message="Queued…", progress=0)

    # Submit to thread pool via async background task
    async def _submit():
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(_executor, _generate_sync, job_id, req)

    background_tasks.add_task(_submit)
    return {"job_id": job_id}


@app.get("/api/status/{job_id}", tags=["generation"])
async def get_status(job_id: str):
    job = _get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    # Strip internal fields before returning to client
    return {k: v for k, v in job.items() if not k.startswith("_")}


@app.get("/api/download/{job_id}", tags=["generation"])
async def download_track(job_id: str):
    job = _get_job(job_id)
    if job is None or job.get("status") != "done":
        raise HTTPException(status_code=404, detail="Track not ready or not found")
    file_path = job.get("_file")
    if not file_path or not Path(file_path).exists():
        raise HTTPException(status_code=404, detail="File missing on server")
    return FileResponse(
        path=file_path,
        media_type="audio/mpeg",
        filename="ai_singalong_track.mp3",
        headers={"Content-Disposition": "attachment; filename=ai_singalong_track.mp3"},
    )


@app.delete("/api/job/{job_id}", tags=["generation"])
async def delete_job(job_id: str):
    with _jobs_lock:
        job = _jobs.pop(job_id, None)
    if job:
        fp = job.get("_file")
        if fp:
            try:
                Path(fp).unlink(missing_ok=True)
            except OSError:
                pass
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
