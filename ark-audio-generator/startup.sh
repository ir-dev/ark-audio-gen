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

# ── Install static ffmpeg (self-contained, no shared-library dependencies) ─────
#
# WHY static and not apt-get:
#   Azure App Service containers are ephemeral. On every container recycle or
#   base-image update, apt-installed .so files (/usr/lib/libavdevice.so etc.)
#   are wiped. A cached dynamic binary then fails with "cannot open shared
#   object file". A statically-compiled binary bundles everything inside the
#   single executable and survives in /home (persistent storage) indefinitely.
#
# WHY we test execution, not just file existence:
#   After a container update the old binary may still exist in /home/bin but
#   silently fail. We run `ffmpeg -version` to confirm it actually works.
# ──────────────────────────────────────────────────────────────────────────────
FFMPEG_CACHE="/home/bin"
mkdir -p "${FFMPEG_CACHE}"

# Test if the cached binary actually executes successfully
if "${FFMPEG_CACHE}/ffmpeg" -version >/dev/null 2>&1; then
    echo "[startup] ffmpeg OK (static binary verified at ${FFMPEG_CACHE}/ffmpeg)"
else
    echo "[startup] ffmpeg missing or broken — downloading static build..."

    # Static build from johnvansickle.com — no .so dependencies, Linux x86_64
    FFMPEG_URL="https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz"
    TMP_ARCHIVE="/tmp/ffmpeg-static.tar.xz"
    TMP_DIR="/tmp/ffmpeg-static"

    echo "[startup]   Downloading ${FFMPEG_URL}..."
    curl -fsSL --retry 3 --retry-delay 5 "${FFMPEG_URL}" -o "${TMP_ARCHIVE}"

    echo "[startup]   Extracting..."
    rm -rf "${TMP_DIR}" && mkdir -p "${TMP_DIR}"
    tar -xf "${TMP_ARCHIVE}" -C "${TMP_DIR}" --strip-components=1

    cp "${TMP_DIR}/ffmpeg"  "${FFMPEG_CACHE}/ffmpeg"
    cp "${TMP_DIR}/ffprobe" "${FFMPEG_CACHE}/ffprobe"
    chmod +x "${FFMPEG_CACHE}/ffmpeg" "${FFMPEG_CACHE}/ffprobe"

    rm -rf "${TMP_ARCHIVE}" "${TMP_DIR}"

    # Final sanity check
    if "${FFMPEG_CACHE}/ffmpeg" -version >/dev/null 2>&1; then
        echo "[startup] Static ffmpeg installed and verified at ${FFMPEG_CACHE}"
    else
        echo "[startup] ERROR: ffmpeg still not working after download. Check logs."
        exit 1
    fi
fi

# Expose to all child processes (gunicorn workers, subprocesses in api.py)
export PATH="${FFMPEG_CACHE}:${PATH}"
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
