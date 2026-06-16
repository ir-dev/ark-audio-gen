# Ark Transcribe (ArkSpeechToText)

A developer-first **SaaS** for **CPU-hosted, lightweight speech-to-text**.
Transcription runs entirely on the CPU via
[Whisper.net](https://github.com/sandrohanea/whisper.net) (whisper.cpp bindings)
using a small `ggml` model — no GPU and no cloud API required. Access to the
transcriber and the REST API is gated behind a passwordless email account.

## How it works

```
.wav/.mp3/.ogg/.opus upload
            ─▶ managed decoders (NAudio / NLayer / NVorbis / Concentus)
            ─▶ downmix to mono + resample to 16 kHz
            ─▶ Whisper.net (whisper.cpp, CPU inference)
            ─▶ transcript + time-stamped segments
```

- **Engine:** `Whisper.net` + `Whisper.net.Runtime` (bundled native whisper.cpp libs for Windows/Linux/macOS).
- **Audio:** `NAudio`'s fully-managed `WdlResamplingSampleProvider` converts any
  audio to the 16 kHz mono float PCM Whisper needs — cross-platform, no native codecs.
- **Model:** the `ggml-<size>.bin` model is downloaded once from Hugging Face on
  first use and cached in `App_Data/`. Default is `base` (~150 MB).

## SaaS features

- **Registered-users only.** The transcriber UI and the API both require an
  account. Sign-in is **passwordless** — an email one-time code (OTP) protected
  by an inline SVG **CAPTCHA**.
- **No-email fallback.** If SMTP isn't configured (empty `Smtp:Host`), the OTP is
  **shown on screen** after the CAPTCHA instead of being emailed — handy for local
  dev and air-gapped deployments.
- **Personal API keys.** Every account gets a unique, high-entropy `ark_…` key.
  Each user picks the **HTTP header name** the key is sent in (default
  `X-Ark-Api-Key`) and can regenerate the key from their profile.
- **Built-in API console.** The transcriber generates a ready-to-run `curl` for the
  file/language you selected, with your key pre-filled.
- **Dark / light theme.** Dark by default, toggle in the navbar (persisted in
  `localStorage`).
- **SQLite storage.** Users and OTPs live in `App_Data/ark-stt.db` (EF Core,
  created automatically on first run).

## Run

```bash
cd ark-audio-stt
dotnet run
```

Open the printed URL (e.g. `http://localhost:5234`). You'll land on the marketing
page; click **Sign in / Sign up**, enter any email, solve the CAPTCHA, then enter
the on-screen code (no SMTP configured by default). The first transcription
downloads the Whisper model, so it takes longer; subsequent runs use the cache.

## Endpoints

| Method | Route                | Auth            | Purpose                                    |
|--------|----------------------|-----------------|--------------------------------------------|
| GET    | `/`                  | anonymous       | Landing / marketing page                   |
| GET    | `/Account/Login`     | anonymous       | Email + CAPTCHA sign-in                     |
| GET    | `/Account/Verify`    | anonymous       | Enter OTP (shown here in no-email mode)     |
| GET    | `/Account/Profile`   | cookie          | API key, custom header name, regenerate     |
| GET/POST | `/Transcription`   | cookie          | Upload UI + live API console                |
| POST   | `/api/transcribe`    | **API key**     | JSON API — multipart `file` (+ `language`)  |

Example API call (header name + key come from your profile):

```bash
curl -X POST http://localhost:5234/api/transcribe \
  -H "X-Ark-Api-Key: ark_your_personal_key" \
  -F "file=@sample.wav" \
  -F "language=auto"
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

The key must arrive under **exactly** the header name configured on your profile;
sending it under any other name is rejected with `401`.

## Configuration (`appsettings.json`)

```json
"ConnectionStrings": {
  "Default": "Data Source=App_Data/ark-stt.db"
},
"Smtp": {
  "Host": "",                 // empty => OTP shown on screen instead of emailed
  "Port": 587,
  "EnableSsl": true,
  "User": "",
  "Password": "",
  "FromAddress": "no-reply@ark-transcribe.immanuel.co",
  "FromName": "Ark Transcribe"
},
"Whisper": {
  "ModelSize": "base",        // tiny | base | small | medium | large-v3 (+ .en variants)
  "ModelDirectory": "App_Data",
  "DefaultLanguage": "auto",
  "Threads": 0                // 0 = auto
}
```

Fill in the `Smtp` section to email codes in production. Pick a smaller model
(`tiny`) for faster/lighter inference, or a larger one for better accuracy.

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
