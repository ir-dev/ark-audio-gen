# ArkSpeechToText

An ASP.NET Core MVC web app that wraps a **CPU-hosted, lightweight speech-to-text**
engine. Transcription runs entirely on the CPU via
[Whisper.net](https://github.com/sandrohanea/whisper.net) (whisper.cpp bindings)
using a small `ggml` model — no GPU and no cloud API required.

## How it works

```
.wav upload ─▶ NAudio (downmix to mono + resample to 16 kHz)
            ─▶ Whisper.net (whisper.cpp, CPU inference)
            ─▶ transcript + time-stamped segments
```

- **Engine:** `Whisper.net` + `Whisper.net.Runtime` (bundled native whisper.cpp libs for Windows/Linux/macOS).
- **Audio:** `NAudio`'s fully-managed `WdlResamplingSampleProvider` converts any WAV
  (any sample rate / channel count) to the 16 kHz mono float PCM Whisper needs —
  cross-platform, no native codecs.
- **Model:** the `ggml-<size>.bin` model is downloaded once from Hugging Face on
  first use and cached in `App_Data/`. Default is `base` (~150 MB).

## Run

```bash
cd ark-audio-translator
dotnet run
```

Then open the printed URL (e.g. `http://localhost:5234`). The first transcription
downloads the model, so it takes longer; subsequent runs use the cached file.

## Endpoints

| Method | Route              | Purpose                                  |
|--------|--------------------|------------------------------------------|
| GET    | `/`                | Upload UI (Transcription/Index)          |
| POST   | `/Transcription`   | Form upload, renders transcript          |
| POST   | `/api/transcribe`  | JSON API — multipart `file` (+ `language`) |

Example API call:

```bash
curl -X POST http://localhost:5234/api/transcribe \
  -F "file=@sample.wav" -F "language=auto"
```

```json
{
  "text": "(upbeat music)",
  "language": "en",
  "durationSeconds": 10.02,
  "processingSeconds": 0.71,
  "segments": [{ "start": 0, "end": 2.58, "text": "(upbeat music)" }]
}
```

## Configuration (`appsettings.json`)

```json
"Whisper": {
  "ModelSize": "base",      // tiny | base | small | medium | large-v3 (+ .en variants)
  "ModelDirectory": "App_Data",
  "DefaultLanguage": "auto",
  "Threads": 0               // 0 = auto
}
```

Pick a smaller model (`tiny`) for faster/lighter inference, or a larger one for
better accuracy.

## Input format

**WAV** (`.wav`), **MP3** (`.mp3`), **OGG Vorbis** (`.ogg`) and **Opus**
(`.opus`, or Opus-in-Ogg `.ogg`) are accepted. All decoders are fully managed,
so they work cross-platform without native codecs:

| Format | Decoder |
|--------|---------|
| WAV    | NAudio `WaveFileReader` |
| MP3    | [NLayer](https://github.com/naudio/NLayer) |
| OGG Vorbis | [NVorbis](https://github.com/NAudio/Vorbis) (`NAudio.Vorbis`) |
| Opus   | [Concentus](https://github.com/lostromb/concentus) (`Concentus.OggFile`) |

For `.ogg` uploads the codec is sniffed (`OpusHead` magic) and routed to the
right decoder. WAV/MP3/Vorbis are downmixed to mono and resampled to 16 kHz;
Opus is decoded natively to 16 kHz mono.

To transcribe other formats (m4a, flac, …), convert first, e.g.:

```bash
ffmpeg -i input.m4a output.wav
```
