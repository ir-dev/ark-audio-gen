# AI Sing-Along Music Generator

Generate rhythmic, lively sing-along backing tracks from any melody description using
**Meta MusicGen** — running entirely on CPU, delivered as a web application.

Two ways in:

* **Describe a melody** — type a description, get a backing track.
* **Turn my vocal into music** — upload a vocal-only recording (humming, singing,
  improvisation) and the app analyses your performance, arranges a complementary
  accompaniment around it, and mixes your original vocal back on top. See
  [Vocal → Music](#vocal--music-pipeline).

---

## Features

| Feature | Detail |
|---|---|
| AI model | Meta `facebook/musicgen-small` (text) or `facebook/musicgen-melody` (audio-conditioned) |
| Output | MP3, 192 kbps, stereo — up to 20 s (describe mode) / 30 s of singing (vocal mode, raisable) |
| Parameters | Genre, mood, instruments, frequency range, crescendo pattern, guidance scale, temperature |
| Smart defaults | All optional fields are inferred from the melody description |
| **Vocal → Music** | Upload a vocal; auto-detect key, tempo, pitch range, phrasing, mood; arrange + synchronise + mix |
| Post-processing | Compressor → 3-band EQ → Beat enhancement → Crescendo envelope → Tremolo → Reverb → Stereo widening |
| Web UI | Single-page app, mode switcher, 12 regional sample melodies, drag-and-drop vocal upload, real-time progress |
| API | FastAPI, async job queue, polling-based progress |
| Deployment | Azure App Service (Linux, Python 3.11) |

---

## Project structure

```
audio-gen/
├── api.py                ← FastAPI app (web + REST API, both modes)
├── generator.py          ← MusicGen inference wrapper
├── effects.py            ← Audio post-processing chain (shared primitives)
├── prompt_builder.py     ← Smart prompt construction & parameter inference
├── main.py               ← CLI: describe-a-melody (optional, standalone)
│
│   ── Vocal → Music pipeline ──
├── vocal_analysis.py     ← Analyse a vocal: key, tempo, pitch, phrasing, mood
├── arrangement.py        ← Turn the analysis into a generation + mix plan
├── vocal_mixer.py        ← Vocal-aware EQ carve, ducking, sync & mix
├── vocal_pipeline.py     ← Orchestrator (analyse → arrange → generate → mix)
├── vocalize.py           ← CLI: vocal-to-music (optional, standalone)
│
├── startup.sh            ← Azure App Service startup command
├── requirements.txt      ← Python dependencies
├── .gitignore
├── README.md             ← This file
└── static/               ← Frontend (served by FastAPI)
    ├── index.html        ← SPA with a mode switcher (describe / upload vocal)
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
| `POST` | `/api/generate` | Submit a describe-a-melody job → `{ "job_id": "..." }` |
| `POST` | `/api/vocal/analyze` | Upload a vocal (multipart `file`) → `{ "analysis": {...} }` (fast, no generation) |
| `POST` | `/api/vocal/generate` | Upload a vocal (multipart `file` + optional overrides) → `{ "job_id": "..." }` |
| `GET` | `/api/status/{job_id}` | Poll status → `{ "status", "message", "progress", … }` (vocal jobs also return `analysis`, `plan`, `warnings`) |
| `GET` | `/api/download/{job_id}` | Stream the finished MP3. `?variant=mix` (default) or `?variant=accompaniment` |
| `DELETE` | `/api/job/{job_id}` | Clean up a job and its files |

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

## Vocal → Music pipeline

Upload a **vocal-only recording** and the app builds music around it, in this flow:

```
Upload Vocal → Find the sung region → Analyse Vocal → Extract Musical Structure
             → Generate Arrangement → Synchronise with Vocal → Preview → Export
```

The output starts where your singing starts: a silent count-in at the head of
the recording is skipped (and reported back as a warning / badge), so the whole
generation budget goes to the part that has a melody in it.

### What it detects (`vocal_analysis.py`)

Approximate melody & pitch movement, pitch range / register, tempo (BPM),
phrasing (voiced segments and pauses), musical key + mode (Krumhansl–Schmuckler
key finding), melodic contour, dynamics, and — derived from those — a suggested
genre, mood, supportive instrument palette and diatonic chord progression.

### How it arranges & mixes

* **`arrangement.py`** turns the analysis into a plan and a *short* MusicGen
  caption (genre, instruments, key, tempo, feel, "instrumental"). The melody
  itself is not described in words — the model hears it directly through the
  chroma of your vocal. Tempo from a solo voice is octave-folded into a
  60–140 BPM window because beat trackers routinely report double time for a
  ballad (152 BPM for a 76 BPM song). Override anything you disagree with.
* **`generator.py`** (the melody-conditioned `musicgen-melody` model) generates
  the accompaniment in **one 30 s pass** — MusicGen's native clip length and
  exactly the span its melody conditioning covers. If you raise the cap above
  30 s, each further window is generated as an **audio continuation** of the
  previous one (the last 3 s are fed back as a decoder prompt), so the music
  stays one piece instead of two unrelated clips crossfaded together.
* **`vocal_mixer.py`** processes the accompaniment with a vocal-friendly effects
  chain (reusing `effects.py`), **carves EQ space** around the vocal's
  fundamental and presence band so instruments don't mask the voice,
  **level-matches** the two (the vocal is normalised to −18 dBFS RMS over its
  sung parts and the backing set 3 dB under it — phone recordings are often
  20 dB quieter than MusicGen output), **sidechain-ducks** the backing under
  the vocal, then **synchronises** (trims/pads the accompaniment to the exact
  vocal length) and mixes the *original* vocal performance back on top.

Two files come out of every job: the **full mix** (`?variant=mix`) and the
**accompaniment only** (`?variant=accompaniment`).

### Overrides & limits

Everything is auto-detected, but any of `genre`, `mood`, `instruments`,
`tempo_bpm`, `crescendo`, `guidance_scale` may be sent to override the detected
value (blank = keep auto). Generation length is bounded by two env vars so CPU
runtime stays sane:

| Env var | Default | Meaning |
|---|---|---|
| `ARK_VOCAL_MAX_SECONDS` | `30` | Max seconds **of singing** scored (silence before it doesn't count; longer is truncated with a warning) |
| `ARK_VOCAL_SEGMENT_SECONDS` | `30` | Generation window (≤ 30; only matters when the cap is raised above it) |
| `ARK_VOCAL_CONTINUATION_SECONDS` | `3` | Audio handed from one window to the next as a continuation prompt |
| `ARK_VOCAL_SCAN_SECONDS` | `300` | How much of a long upload is scanned for the sung region |
| `ARK_DEVICE` / `ARK_FORCE_CPU` | cpu | Torch device (`mps` on Apple Silicon, `cuda`); see *Memory matters* below |
| `ARK_DTYPE` | fp32 (CPU) / fp16 (MPS) | Weight precision; half precision halves the melody model to ~3 GB |

### CLI

```bash
# Fully automatic
python vocalize.py my_humming.wav

# Analyse only (fast, no generation)
python vocalize.py my_humming.wav --analyze-only

# With overrides
python vocalize.py my_singing.wav --genre jazz --mood calm \
    --instruments "piano,double bass,brushed drums" -o song.mp3
```

### REST (curl)

```bash
# Analyse (fast)
curl -X POST http://localhost:8000/api/vocal/analyze -F "file=@my_humming.wav"

# Generate; then poll status and download both variants
JOB=$(curl -s -X POST http://localhost:8000/api/vocal/generate \
        -F "file=@my_humming.wav" -F "genre=folk" | python3 -c "import sys,json;print(json.load(sys.stdin)['job_id'])")
curl -s http://localhost:8000/api/status/$JOB
curl -s "http://localhost:8000/api/download/$JOB?variant=mix"          -o mix.mp3
curl -s "http://localhost:8000/api/download/$JOB?variant=accompaniment" -o music_only.mp3
```

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

## Time expectations — and why memory matters more than CPU

| Track length | Model | Approximate time |
|---|---|---|
| 8 s | musicgen-small | 2 – 4 min |
| 15 s | musicgen-small | 4 – 8 min |
| 20 s | musicgen-small | 6 – 12 min |
| 30 s (vocal mode) | musicgen-melody, weights resident in RAM | see the table in `key_points.txt` |
| 30 s (vocal mode) | musicgen-melody, weights **paged from disk** | hours to a day — do not run it like this |

The first request downloads and caches the model (musicgen-small ≈ 300 MB,
musicgen-melody ≈ 3 GB). Subsequent requests use the cache.

**`musicgen-melody` is a 1.55 B-parameter model: 6.2 GB of weights in float32,
3.1 GB in half precision.** MusicGen decodes autoregressively — every one of
the 1 500 tokens in a 30 s clip touches every weight — so if those weights do
not fit in free RAM the OS pages them from disk on every step and a generation
that should take minutes takes a day. This is *not* fixed by more CPU cores.
Concretely:

* An Azure **B2 plan has 3.5 GB of RAM**: the melody model cannot be resident
  there at all. Use a plan with ≥ 8 GB (P1v3 or above) and `ARK_DTYPE=bfloat16`,
  or a GPU host. Text mode with `musicgen-small` (1.2 GB) is fine on B2.
* On a 16 GB laptop, close the VM / browser tabs before generating, or run on
  the GPU in half precision (`ARK_DEVICE=mps`, the default precision there is
  fp16) so only ~3 GB has to be resident.
* Watch the per-token timing printed by the worker (`Generating accompaniment…
  12%`): if the ETA reads in hours, the model is swapping — stop and free
  memory rather than wait.

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
# Vocal → Music modules
python3 -m py_compile vocal_analysis.py arrangement.py vocal_mixer.py \
    vocal_pipeline.py vocalize.py && echo "vocal pipeline OK"

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
python3 vocalize.py --help

# 5b. Smoke-test the vocal pipeline end-to-end WITHOUT MusicGen
#     (synthesises a hum, analyses it, and runs the mix with a stub generator)
python3 - <<'EOF'
import numpy as np, soundfile as sf, tempfile
from vocal_analysis import analyze_vocal
from arrangement import plan_arrangement
from vocal_pipeline import run_vocal_to_music, VocalPipelineResult

sr = 32000
t = np.linspace(0, 6, 6*sr, endpoint=False)
notes = [261.63, 293.66, 329.63, 349.23, 329.63, 293.66]
y = np.zeros_like(t); seg = len(t)//len(notes)
for i, f in enumerate(notes):
    s, e = i*seg, (i+1)*seg; tt = t[s:e]
    y[s:e] = (np.sin(2*np.pi*f*tt) + 0.3*np.sin(2*np.pi*2*f*tt)) * np.hanning(len(tt))
path = tempfile.mktemp(suffix=".wav"); sf.write(path, (0.6*y).astype(np.float32), sr)

a = analyze_vocal(path); print("key:", a.key_name, "| tempo:", a.tempo_bpm, "| mood:", a.suggested_mood)
print("chords:", a.chord_progression, "| summary:", a.melody_summary)

class StubGen:
    sample_rate = sr
    def generate(self, prompt, melody_path, duration, guidance_scale, temperature):
        n = int(duration*sr); return (np.random.randn(2, n).astype("float32")*0.2, sr)

res = run_vocal_to_music(path, generator_factory=lambda: StubGen())
assert res.mix.shape[0] == 2 and np.isfinite(res.mix).all()
print("mix shape:", res.mix.shape, "| segments:", res.segments, "| peak:", round(float(np.max(np.abs(res.mix))),3))
print("window:", res.window_start_sec, "→", res.window_end_sec, "s of", res.source_duration_sec, "s | prompt:", res.plan.prompt)
print("VOCAL PIPELINE OK")
EOF

# 5c. Unit tests for the vocal pipeline, queue and worker (no model download)
pytest -q tests/

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
