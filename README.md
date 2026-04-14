# AI Sing-Along Music Generator

Generate rhythmic, lively sing-along backing tracks from any melody description using
**Meta MusicGen** — running entirely on CPU, delivered as a web application.

---

## Features

| Feature | Detail |
|---|---|
| AI model | Meta `facebook/musicgen-small` (text) or `facebook/musicgen-melody` (audio-conditioned) |
| Output | MP3, 192 kbps, stereo, up to 20 seconds |
| Parameters | Genre, mood, instruments, frequency range, crescendo pattern, guidance scale, temperature |
| Smart defaults | All optional fields are inferred from the melody description |
| Post-processing | Compressor → 3-band EQ → Beat enhancement → Crescendo envelope → Tremolo → Reverb → Stereo widening |
| Web UI | Single-page app, 12 regional sample melodies, real-time progress |
| API | FastAPI, async job queue, polling-based progress |
| Deployment | Azure App Service (Linux, Python 3.11) |

---

## Project structure

```
audio-gen/
├── api.py                ← FastAPI app (web + REST API)
├── generator.py          ← MusicGen inference wrapper
├── effects.py            ← Audio post-processing chain
├── prompt_builder.py     ← Smart prompt construction & parameter inference
├── main.py               ← CLI entry point (optional, standalone)
├── startup.sh            ← Azure App Service startup command
├── requirements.txt      ← Python dependencies
├── .gitignore
├── README.md             ← This file
└── static/               ← Frontend (served by FastAPI)
    ├── index.html
    ├── style.css
    └── app.js
```

---

## Prerequisites

| Tool | Version | Install |
|---|---|---|
| Python | 3.9 – 3.11 | https://python.org |
| ffmpeg | any recent | `brew install ffmpeg` (Mac) or `apt install ffmpeg` (Linux) |
| Git | any | https://git-scm.com |

> **Azure only:** No local ffmpeg install needed — it is pre-installed on the App Service Linux image.

---

## Local setup (step-by-step)

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd audio-gen
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate          # Mac / Linux
# .venv\Scripts\activate           # Windows PowerShell
```

### 3. Install PyTorch (CPU-only build — ~250 MB vs 2.5 GB GPU version)

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 4. Install all other dependencies

```bash
pip install -r requirements.txt
```

### 5. Verify the install

```bash
# Syntax-check all modules
python3 -m py_compile effects.py generator.py prompt_builder.py api.py main.py
echo "All OK"

# Confirm CLI help renders
python3 main.py --help
```

### 6. Start the web server locally

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

Open your browser at **http://localhost:8000**

### 7. (Optional) Run the CLI directly

```bash
# Text melody → auto infer everything
python3 main.py "a cheerful whistling tune in C major" --duration 10

# Full control
python3 main.py "jazz piano riff with walking bass" \
    --genre jazz --mood uplifting \
    --instruments "piano,bass,drums" \
    --duration 15 --output my_track.mp3
```

---

## API reference

All endpoints are also documented at **http://localhost:8000/docs** (Swagger UI).

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/health` | Health check (used by Azure load balancer) |
| `POST` | `/api/generate` | Submit generation job → `{ "job_id": "..." }` |
| `GET` | `/api/status/{job_id}` | Poll status → `{ "status", "message", "progress" }` |
| `GET` | `/api/download/{job_id}` | Stream the finished MP3 |
| `DELETE` | `/api/job/{job_id}` | Clean up a job and its file |

### POST /api/generate — request body

```json
{
  "melody": "A cheerful whistling tune in C major",
  "genre": "pop",
  "mood": "happy",
  "instruments": "piano,guitar,drums",
  "frequency_range": "full",
  "duration": 15,
  "crescendo": "rise-fall",
  "guidance_scale": 3.5,
  "temperature": 1.05
}
```

All fields except `melody` are optional.

---

## Azure App Service deployment

### Step 1 — Create a resource group and App Service plan

```bash
az group create --name rg-singalong --location eastus

az appservice plan create \
  --name plan-singalong \
  --resource-group rg-singalong \
  --sku B2 \
  --is-linux
```

> **Recommended SKU:** B2 or higher (2 vCPU, 3.5 GB RAM). B1 will work but generation is slower.

### Step 2 — Create the Web App

```bash
az webapp create \
  --resource-group rg-singalong \
  --plan plan-singalong \
  --name singalong-ai \
  --runtime "PYTHON:3.11"
```

### Step 3 — Configure the startup command

```bash
az webapp config set \
  --resource-group rg-singalong \
  --name singalong-ai \
  --startup-file "bash startup.sh"
```

### Step 4 — Enable build-during-deployment (so Azure runs pip install)

```bash
az webapp config appsettings set \
  --resource-group rg-singalong \
  --name singalong-ai \
  --settings \
    SCM_DO_BUILD_DURING_DEPLOYMENT=true \
    HF_HOME=/home/.cache/huggingface \
    TRANSFORMERS_CACHE=/home/.cache/huggingface \
    PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu
```

> Setting `PIP_EXTRA_INDEX_URL` tells the Azure build agent to pull the CPU-only
> PyTorch wheel, avoiding the large CUDA download.

### Step 5 — Deploy via Git or ZIP

**Option A — Local Git push:**
```bash
az webapp deployment source config-local-git \
  --resource-group rg-singalong \
  --name singalong-ai

# Add the Azure remote and push
git remote add azure <deployment-url-from-above>
git push azure main
```

**Option B — ZIP deploy (faster):**
```bash
zip -r deploy.zip . \
  --exclude ".venv/*" \
  --exclude "generated/*" \
  --exclude ".git/*" \
  --exclude "__pycache__/*"

az webapp deploy \
  --resource-group rg-singalong \
  --name singalong-ai \
  --src-path deploy.zip
```

### Step 6 — Open the app

```bash
az webapp browse --resource-group rg-singalong --name singalong-ai
```

URL format: `https://singalong-ai.azurewebsites.net`

---

## Time expectations on CPU

| Track length | Model | Approximate time |
|---|---|---|
| 8 s | musicgen-small | 2 – 4 min |
| 15 s | musicgen-small | 4 – 8 min |
| 20 s | musicgen-small | 6 – 12 min |
| 20 s | musicgen-melody (audio input) | 10 – 20 min |

The first request downloads and caches the model (~300 MB). Subsequent requests use the cache.

---

## Sample melody prompts (global diversity)

| Region | Sample prompt |
|---|---|
| 🇮🇳 India | `A cheerful Bollywood melody with sitar-style ornaments, rising chorus, and a joyful festive feel rooted in North Indian classical music` |
| 🌍 West Africa | `A lively West African highlife melody with kora-style arpeggios, talking drum rhythm, and a bright call-and-response vocal hook` |
| 🇯🇵 Japan | `A peaceful Japanese melody in the pentatonic scale with shakuhachi-style breathy flute, koto arpeggios, and soft taiko drum accents` |
| 🇧🇷 Brazil | `A warm bossa nova melody with nylon guitar chord comping, syncopated bass, light shaker groove and a tender romantic feel` |
| 🇮🇪 Celtic | `A lively Irish jig melody with fiddle runs, tin whistle descant, steady bodhrán beat and an infectious dance-floor energy` |
| 🇸🇦 Middle East | `A flowing Arabic melody with oud slides and ornaments, darbuka rhythm, rich string pads and a mysterious minor-scale character` |
| 🇰🇷 Korea | `A punchy K-Pop chorus melody with bright synth pads, powerful snare-heavy beat, catchy hook and an uplifting triumphant feel` |
| 🇺🇸 Gospel | `A soulful gospel anthem with Hammond organ chords, rich choir harmonies, steady gospel beat and a powerful uplifting melody` |
| 🇪🇸 Flamenco | `An intense flamenco melody with fast guitar rasgueado, cajón accents, clapping palmas and passionate Phrygian scale character` |
| 🇲🇽 Mexico | `A bright mariachi melody with lead trumpet, violin harmonies, guitarrón bass and a joyful celebratory feel in major key` |
| 🇷🇺 Slavic | `A lively Slavic folk dance melody with balalaika tremolo, accordion bass-chord pattern and energetic stomping rhythm` |
| 🌊 Pacific | `A breezy Hawaiian melody with ukulele strumming, gentle steel guitar slides, soft bass and a warm relaxed island atmosphere` |

---

## Validation commands (run by Claude during development)

The following commands were executed to verify the project. Run them at any time to confirm your setup is healthy.

```bash
# 1. Check Python version (requires 3.9+)
python3 --version

# 2. Verify ffmpeg is installed (required for MP3 export)
ffmpeg -version | head -1

# 3. Syntax-check all Python modules
python3 -m py_compile effects.py && echo "effects.py OK"
python3 -m py_compile generator.py && echo "generator.py OK"
python3 -m py_compile prompt_builder.py && echo "prompt_builder.py OK"
python3 -m py_compile api.py && echo "api.py OK"
python3 -m py_compile main.py && echo "main.py OK"

# 4. Smoke-test prompt builder + effects chain (no model download required)
python3 - <<'EOF'
from prompt_builder import build_prompt, infer_parameters
from effects import process_audio
import numpy as np

inf = infer_parameters("a happy whistling pop melody")
print("inferred:", inf)

prompt = build_prompt(
    melody_description="happy whistling pop melody",
    instruments=["piano", "guitar"],
    inferred=inf,
)
print("prompt:", prompt[:120], "...")

# 2-second dummy stereo signal at 32kHz
sr = 32000
dummy = np.random.randn(2, sr * 2).astype(np.float32) * 0.3
out = process_audio(dummy, sr, genre="pop", mood="happy", crescendo_pattern="rise-fall")
print("effects output shape:", out.shape, "peak:", round(float(np.max(np.abs(out))), 3))
print("ALL OK")
EOF

# 5. Confirm CLI help renders
python3 main.py --help

# 6. Confirm FastAPI app loads (no model download, just import check)
python3 -c "import api; print('FastAPI app loaded OK')"

# 7. List installed packages relevant to this project
pip show torch transformers fastapi uvicorn gunicorn librosa pydub soundfile
```

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `ModuleNotFoundError: No module named 'numpy'` | Run `pip install -r requirements.txt` inside the venv |
| `pydub.exceptions.CouldntDecodeError` | `ffmpeg` is not on PATH. Install with `brew install ffmpeg` or `apt install ffmpeg` |
| Generation takes > 15 min | Use `--duration 8` to reduce length. musicgen-small on CPU averages 1 min per second of audio |
| Azure: 503 timeout during generation | Normal — generation is async. The UI polls for status; the Azure 230s request timeout does not apply to background jobs |
| Azure: Out of memory | Upgrade to B3 or P1v3 SKU. musicgen-small requires ~1.5 GB RAM during inference |
| First request very slow | Model downloads ~300 MB on first request. Subsequent requests use the `/home/.cache` persistent cache |
| `CUDA not available` warning | Expected on CPU-only machines. The app is designed to run on CPU |

---

## Developer and AI attribution

- Author: **Immanuel R** (Along with Claude Code)
- Contact: raj@immanuel.co
---

## License

This project uses **Meta MusicGen** via HuggingFace Transformers, released under the
[CC-BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) licence.
Generated audio is for personal / educational use.
