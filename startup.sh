#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# Azure App Service startup script
#
# This script is set as the "Startup Command" in Azure Portal:
#   Configuration → General settings → Startup Command → bash startup.sh
#
# Key decisions:
#   --workers 1     : CPU-only; one worker avoids duplicate model loads & OOM
#   --threads 4     : Handle concurrent HTTP requests within the worker
#   --timeout 600   : 10 min – generation can take ~5–10 min on CPU
#   --preload       : Load app once at startup so workers share model memory
#   PORT            : Azure injects PORT (default 8000)
# ──────────────────────────────────────────────────────────────────────────────

set -e

# ── Cache HuggingFace models in Azure persistent storage (/home is persistent)
export HF_HOME="${HF_HOME:-/home/.cache/huggingface}"
export TRANSFORMERS_CACHE="${HF_HOME}"
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
export HF_HUB_DISABLE_PROGRESS_BARS=1

# ── Create output directory for generated MP3s
mkdir -p generated
mkdir -p "${HF_HOME}"

# ── Resolve port (Azure sets PORT or WEBSITES_PORT)
PORT="${PORT:-${WEBSITES_PORT:-8000}}"

echo "======================================"
echo " AI Sing-Along Music Generator"
echo " Starting on port ${PORT}"
echo " HF cache: ${HF_HOME}"
echo "======================================"

# ── Install ffmpeg (not pre-installed in Azure App Service Python containers) ──
# We cache the binaries in /home/bin (persistent) so apt-get only runs once.
FFMPEG_CACHE="/home/bin"
mkdir -p "${FFMPEG_CACHE}"

if [ ! -f "${FFMPEG_CACHE}/ffmpeg" ]; then
    echo "[startup] ffmpeg not found in cache — installing via apt-get..."
    apt-get update -qq
    apt-get install -y -qq --no-install-recommends ffmpeg
    # Copy to persistent /home/bin so future restarts skip apt-get entirely
    cp "$(which ffmpeg)"  "${FFMPEG_CACHE}/ffmpeg"
    cp "$(which ffprobe)" "${FFMPEG_CACHE}/ffprobe"
    echo "[startup] ffmpeg installed and cached at ${FFMPEG_CACHE}"
else
    echo "[startup] ffmpeg found in cache — skipping apt-get"
fi

# Put the cached binaries on PATH so pydub and subprocesses can find them
export PATH="${FFMPEG_CACHE}:${PATH}"
# Also export explicit paths so api.py can use them directly
export FFMPEG_PATH="${FFMPEG_CACHE}/ffmpeg"
export FFPROBE_PATH="${FFMPEG_CACHE}/ffprobe"

# ── Pre-warm: download MusicGen model before accepting traffic ─────────────────
# Runs once; subsequent starts skip download because the cache already exists.
# This avoids a slow first-user experience and prevents mid-request timeouts.
echo "[startup] Pre-warming facebook/musicgen-small model..."
python3 - <<'PYEOF'
import os, sys
os.environ.setdefault("HF_HOME", "/home/.cache/huggingface")
os.environ.setdefault("TRANSFORMERS_CACHE", "/home/.cache/huggingface")
try:
    from transformers import AutoProcessor, MusicgenForConditionalGeneration
    print("  Downloading / verifying processor...")
    AutoProcessor.from_pretrained("facebook/musicgen-small")
    print("  Downloading / verifying model weights...")
    MusicgenForConditionalGeneration.from_pretrained(
        "facebook/musicgen-small", torch_dtype="auto"
    )
    print("[startup] Model ready.")
except Exception as e:
    print(f"[startup] WARNING: model pre-warm failed ({e}). Will retry on first request.")
    sys.exit(0)   # non-fatal – gunicorn still starts
PYEOF

exec gunicorn api:app \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind "0.0.0.0:${PORT}" \
  --workers 1 \
  --threads 4 \
  --timeout 600 \
  --keep-alive 30 \
  --access-logfile - \
  --error-logfile - \
  --log-level info
