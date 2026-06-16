# ArkTextTranslator

An ASP.NET Core MVC web app that takes text in **any language**, **detects the
language**, and **translates it** to a target language — running entirely on the
**CPU**, with no cloud API required.

## How it works

```
text ─▶ NTextCat (offline n-gram language detection, CPU)
     ─▶ NLLB-200 seq2seq via ONNX Runtime (CPU inference)
     ─▶ translated text
```

- **Language detection:** [NTextCat](https://github.com/ivanakcheurov/ntextcat) —
  fully-managed character n-gram identification. The bundled `Core14` profile covers
  ~280 languages. No download, effectively instant.
- **Translation (default):** Meta's **NLLB-200** (No Language Left Behind, distilled
  600M) run on the CPU through [ONNX Runtime](https://onnxruntime.ai). The ONNX model
  + SentencePiece tokenizer are downloaded once from Hugging Face on first use and
  cached in `App_Data/`. The default `_quantized` (int8) weights keep it responsive on
  CPU (~890 MB total download).
- **Translation (alternative):** a [LibreTranslate](https://libretranslate.com) HTTP
  server, selectable via configuration.

### Tokenization

NLLB ids are the SentencePiece ids plus a fairseq offset of **+1** (verified against
the model's `tokenizer.json`). Each request encodes as
`[source-lang] + pieces(+1) + </s>`, then greedily decodes while forcing the
target-language token first (NLLB's `decoder_start_token_id = 2`,
`forced_bos_token_id = target`). Long inputs are split into sentences so responses
stay snappy and within the model's context window.

## Run

```bash
cd ark-audio-translator
dotnet run
```

Open the printed URL (e.g. `http://localhost:5234`). **The first translation downloads
the model (~890 MB), so it takes a while; subsequent runs use the cached files.**

## Endpoints

| Method | Route             | Purpose                                     |
|--------|-------------------|---------------------------------------------|
| GET    | `/`               | Translate UI (Translation/Index)            |
| POST   | `/Translation`    | Form submit, renders the translation        |
| POST   | `/api/translate`  | JSON API                                     |

Example API call:

```bash
curl -X POST http://localhost:5234/api/translate \
  -H "Content-Type: application/json" \
  -d '{"text":"Bonjour le monde","source":"auto","target":"eng_Latn"}'
```

```json
{
  "translatedText": "Hello world",
  "detectedLanguage": "French",
  "detectionConfidence": 0.42,
  "sourceLanguage": "fra_Latn",
  "targetLanguage": "eng_Latn",
  "engine": "NLLB-200 (ONNX, int8)",
  "processingSeconds": 1.31
}
```

`source` / `target` are [FLORES-200](https://github.com/facebookresearch/flores) codes
(e.g. `eng_Latn`, `fra_Latn`, `hin_Deva`, `zho_Hans`). Use `"auto"` (or omit) for
`source` to auto-detect.

## Configuration (`appsettings.json`)

```json
"Translation": {
  "Provider": "Onnx",                 // "Onnx" (offline CPU) or "LibreTranslate"
  "Onnx": {
    "ModelRepo": "Xenova/nllb-200-distilled-600M",
    "QuantizationSuffix": "_quantized", // "" = fp32 (bigger, slightly better), "_quantized" = int8
    "ModelDirectory": "App_Data",
    "Threads": 0,                       // 0 = let ONNX Runtime decide
    "MaxOutputTokens": 256
  },
  "LibreTranslate": {
    "Endpoint": "http://localhost:5000",
    "ApiKey": ""
  }
}
```

### Using LibreTranslate instead

Set `"Provider": "LibreTranslate"` and run a server:

```bash
docker run -ti --rm -p 5000:5000 libretranslate/libretranslate
```

Language detection still runs locally; only the translation step is delegated.

## Notes

- The first run loads the ONNX encoder/decoder into memory after download — expect a
  one-time delay; subsequent translations are fast.
- Pick `"QuantizationSuffix": ""` for full-precision (fp32) weights if you prefer
  quality over footprint/speed (~3.5 GB download).
- The encoder/decoder tensor names are resolved at runtime, so the app is robust to
  minor differences between ONNX exports.
