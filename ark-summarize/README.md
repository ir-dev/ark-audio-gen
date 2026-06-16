# Ark Summarize

A CPU-hosted, privacy-first **text intelligence** SaaS built with ASP.NET Core MVC.
Paste any text and get back clean JSON: the inferred **intent**, a concise extractive
**summary**, and the **entities** mentioned in the text.

Everything runs on the CPU with a fully self-contained engine — **no GPU, no model
download, and no third-party AI provider**. Your text never leaves the server.

```json
{
  "intent": "complaint",
  "intentConfidence": 0.82,
  "summary": "The payment gateway has been down since Monday and customers cannot check out.",
  "entities": [
    { "text": "Acme Corp", "type": "organization" },
    { "text": "support@acme.com", "type": "email" },
    { "text": "Monday", "type": "date" }
  ],
  "keywords": ["payment", "gateway", "checkout"]
}
```

## Features

- **Intent detection** — cue-based classifier (question, request, complaint, praise,
  transactional, instruction, opinion, announcement, statement).
- **Extractive summary** — term-frequency sentence ranking with a positional prior;
  faithful to the source (no hallucinations). Length controlled by `maxSentences` (1–10).
- **Entity extraction** — names, organizations, locations, emails, URLs, money, dates,
  times, phone numbers and percentages.
- **SaaS auth** — passwordless email one-time-passcode (OTP) sign-in with CAPTCHA.
  When SMTP isn't configured, the OTP is shown **on screen** so the app is usable
  out of the box.
- **Developer API** — `POST /api/summarize`, authenticated by a personal API key sent
  in a **custom HTTP header name you choose** on your profile page.
- **Live API console** — the summarizer page generates the exact `curl` command for the
  text and settings you entered.
- **Dark / light theme** — dark by default, toggle persisted in `localStorage`.
- **SQLite** persistence (created automatically on first run).

## Engines

Two interchangeable CPU engines implement `ISummarizationService`; pick one or run several
and **compare them side by side** in the UI. The API selects one per call via the optional
`model` field (default `lexical`).

| key          | engine                  | summary                        | intent                       | notes |
|--------------|-------------------------|--------------------------------|------------------------------|-------|
| `lexical`    | Ark Lexical Engine *(default)* | frequency-based **extractive** | rule/cue based               | no model, instant, fully offline |
| `minilm`     | MiniLM Semantic (ONNX)  | embedding-centroid **extractive** | zero-shot embedding similarity | `all-MiniLM-L6-v2` (int8 ONNX, ~23 MB), ~5–10 ms warm |
| `abstractive`| Abstractive (DistilBART, ONNX) | **abstractive** — *generates* new condensed text | rule/cue based | `distilbart-cnn-6-6` (int8 ONNX, ~270 MB) downloaded once; ~0.5–1 s warm |

The two extractive engines select existing sentences, so they can't shorten a single-sentence
input (they echo it). The **abstractive** engine generates a brand-new, condensed summary, so it
genuinely compresses short text too. Swap `Abstractive:ModelRepo` to `Xenova/distilbart-xsum-6-6`
for even more aggressive single-sentence abstraction.

Entities and keywords are deterministic and shared across engines (`Services/SharedNlp.cs`).
ONNX models + vocab are pulled from Hugging Face and cached under `App_Data/models/`; everything
then runs locally on the CPU — no GPU, no cloud. Add more engines by implementing
`ISummarizationService` and registering it — the UI and API pick them up automatically.

## Running locally

```bash
cd ark-summarize
dotnet run
```

Then open the printed URL (e.g. `http://localhost:5244`). On first run the SQLite
database is created under `App_Data/`.

### Sign-in without email

If the `Smtp` section in `appsettings.json` has an empty `Host` (the default), the app
runs in **no-email mode**: after you submit your email + CAPTCHA, the one-time code is
displayed directly on the verification screen. To send real emails, fill in the SMTP
settings.

## API

```bash
curl -X POST http://localhost:5244/api/summarize \
  -H "X-Ark-Api-Key: ark_your_personal_key" \
  -H "Content-Type: application/json" \
  -d '{"text":"Acme Corp reported that revenue grew 24% to $5M in 2025.","maxSentences":2,"model":"minilm"}'

# List available engines:
curl http://localhost:5244/api/models -H "X-Ark-Api-Key: ark_your_personal_key"
```

- The header **name** (`X-Ark-Api-Key` by default) is configurable per user on the
  profile page; the **value** is your generated key (regenerable any time).
- Request body: `{ "text": "...", "maxSentences": 3, "model": "lexical" }`
  (`maxSentences` optional 1–10; `model` optional, defaults to `lexical`; unknown
  models return 400 with the list of valid keys).
- `GET /api/models` returns the available engine keys.
- Max input length: 100,000 characters.

## Project layout

```
Controllers/   Home, Account (OTP auth + profile), Summarize (UI + /api/summarize)
Services/      ISummarizationService + SummarizationService (the CPU engine)
Services/Auth/ OTP, CAPTCHA, SMTP email, API-key generation & auth handler
Data/          EF Core DbContext (SQLite)
Models/        Domain + view models
Views/         Razor views (landing, account, summarizer) + dark/light theme
wwwroot/       Static assets (site.css, site.js, bootstrap/jquery)
```

Built by [Immanuel R](https://immanuel.co).
