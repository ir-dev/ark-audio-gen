# Architecture

Ark Speech is an offline-first Android app that turns a push-to-talk recording into structured JSON:

```
Push-to-talk ─▶ Audio capture ─▶ Validation ─▶ VAD ─▶ Language ID ─▶ Model routing ─▶ ASR
            ─▶ Transcript review (user) ─▶ Intent ─▶ Field extraction ─▶ Clarification (user) ─▶ JSON
```

Every stage is a small, replaceable component behind an interface. Nothing leaves the device.

## Modules

| module | what | depends on |
|---|---|---|
| `:core` | Pure Kotlin/JVM. Audio types and validation, WAV codec, language registry, VAD/LID/ASR contracts, model routing, configuration schema + versioned repository, intent classifier, extraction engine, pipeline orchestrator, output contract, model catalogue schema, safe logging. | kotlinx-serialization, kotlinx-coroutines |
| `:app` | Android. Microphone capture, engine adapters (sherpa-onnx, ONNX Runtime), model store/downloader, Compose UI, settings, retention, instrumented tests and the benchmark harness. | `:core`, sherpa-onnx AAR, onnxruntime-android, Compose |

`:core` has no Android dependency, so the pipeline logic, parsers, validation and configuration code
run under plain JUnit on the build machine (81 tests, `./gradlew :core:test`). Engine adapters are the
only code that touches native runtimes.

## Packages (`:core`, `co.immanuel.arkspeech.core`)

| package | contents |
|---|---|
| `audio` | `PcmAudio` (16 kHz mono float), `Resampler`, `WavCodec` (PCM/float WAV in, PCM16 out; everything else is `UnsupportedAudioFormatException`), `AudioValidator` → `AudioQualityReport` (duration bounds, silence, low level, clipping ratio, frame-energy SNR, speech ratio, DC offset) with `AudioValidationPolicy` thresholds. |
| `vad` | `VoiceActivityDetector` → `SpeechSegment`s; `EnergyVad` fallback (JVM-testable). |
| `lang` | `Languages` (22 scheduled languages + English with script and per-engine codes), `LanguageDetector` contract, `SegmentedLanguageIdentifier` (pieces ≤ 15 s, duration-weighted mean of log-probabilities, time share, code-mix and uncertainty decision per `LanguagePolicy`), `LanguageDetection`. |
| `asr` | `SpeechRecognizer` / `RecognizerFactory` contracts, `Transcript`, `RoutingTable` + `ModelRouter` (language → model pack, with fallbacks to what is installed), `TranscriptQualityEstimator` (heuristic confidence). |
| `config` | `ExtractionDefinition` schema (intent, languages, triggers, fields, validation, extraction hints, examples, threshold, fallback), `DefinitionValidator`, `DefinitionRepository` + `FileDefinitionRepository` (immutable versions on disk, history, rollback, import/export bundles). |
| `nlu` | `TextNormalizer`/`NormalizedText` (NFC, native digits → ASCII, script-aware tokens), `Lexicons` (numbers, dates, times, currency, durations in en/hi/ta/te/kn/bn/mr/gu), typed parsers (`NumberParser`, `DateParser`, `TimeParser`, `PhoneParser`, `AmountParser`, `DurationParser`), `LexicalIntentClassifier`, `ExtractionEngine` → `FieldResult` with status (`EXTRACTED`, `MISSING`, `LOW_CONFIDENCE`, `AMBIGUOUS`, `INVALID`, `DEFAULTED`, `USER_PROVIDED`). |
| `pipeline` | `SpeechPipeline` (stage methods + `runAll`), `PipelineRun` (mutable state the UI drives), `PipelineResult` (the output contract). |
| `model` | `ModelCatalog`/`ModelPack` schema (engine id, files with SHA-256, licence, tier, RAM estimate). |
| `log` | `SafeLog` + `Redactor`: structured metadata only, sensitive keys dropped, values truncated. |

## Packages (`:app`, `co.immanuel.arkspeech`)

| package | contents |
|---|---|
| `capture` | `PushToTalkRecorder`: `AudioRecord` (VOICE_RECOGNITION source, 16 kHz mono PCM16, optional NoiseSuppressor/AGC), records only between start and stop, hard stop at 3 minutes, level meter, cancel wipes the buffer. No service, no wake lock; the Activity keeps the screen on only while recording and cancels recording in `onStop`. |
| `engine` | `SherpaVad` (Silero VAD via sherpa-onnx), `AnuvaniLanguageDetector` (ONNX Runtime; CIIL's published `audio`/`audio_length` → `logits` contract, 512-sample tail pad, softmax), `WhisperLanguageDetector` (sherpa-onnx SLID fallback, vote-based), `SherpaOfflineRecognizer` + `SherpaRecognizerFactory` (Whisper, Omnilingual CTC, Dolphin CTC, NeMo CTC/IndicConformer packs through one class). |
| `model` | `ModelStore` (packs under `Android/data/<pkg>/files/models/<id>/`, install state, SHA-256 verify), `ModelDownloader` (system `DownloadManager`, Wi-Fi only by default). |
| `app` | `AppGraph` (hand-wired DI, lazy engine construction from installed packs, `CachingRecognizerFactory` keeps one ASR model resident, releases on memory pressure), `Settings`, `Retention` (writes only with consent), `DeviceInfo`, `AndroidSafeLog`, `ArkApp`. |
| `ui` | Compose: `RecordScreen` (hold-to-talk or tap, timer, level, 3-minute cap, cancel/retry, WAV import), `ReviewScreen` (language confidence + override, code-mix notice, editable transcript, re-transcribe), `ResultScreen` (intent, fields with status chips, clarifications, consent, timings, JSON copy/share), `ModelsScreen`, `AdminScreen` (definitions, enable/disable, JSON editor with validation and live test, history/rollback, import/export, routing), `SettingsScreen`. |

## Data flow in one interaction

1. **Capture**: the recorder fills an in-memory PCM16 buffer; at release (or 3:00) it yields `PcmAudio`.
2. **Validate** (`AudioValidator`): REJECT stops here (`AUDIO_REJECTED`, with the issues) — no model runs on
   silence, noise or a 0.3 s blip. WARN continues and the warnings are carried into the result.
3. **VAD** (`SherpaVad`): speech segments, ≤ 25 s each, silence dropped. None → `NO_SPEECH`.
4. **Language ID** (`SegmentedLanguageIdentifier` over `AnuvaniLanguageDetector`): each segment is cut into
   ≤ 15 s pieces, classified, aggregated. The decision carries confidence, alternatives, per-segment labels
   and time share; `uncertain` (confidence < 0.60 or margin < 0.15) or `isCodeMixed` (two languages ≥ 25 %
   of speech time) produce a `LANGUAGE` clarification. The user can override on the review screen; a client
   that knows the language passes `languageOverride` and skips the stage.
5. **Route** (`ModelRouter`): `routing.byLanguage[lang]` → `defaultModelId` → `fallbackModelIds`, first one
   installed. None → `MODEL_UNAVAILABLE`.
6. **ASR** (`SherpaOfflineRecognizer`): each segment decoded, joined in order, script preserved. Whisper gets
   the language as a decoding hint (so code-mixed speech does not flip language mid-utterance); CTC models
   need none. Confidence is heuristic (coverage, speaking rate, repetition, script agreement) and labelled so.
7. **Review** (user): confirm/override language, correct the transcript. Edits re-run understanding only.
8. **Intent** (`LexicalIntentClassifier`): administrator-defined trigger phrases/patterns per language, with
   English triggers always included for code-mixed speech; score = 1 − (1+hits)^−1.5; per-definition
   threshold and fallback (`ASK_USER`, `REJECT`, `ACCEPT_LOW_CONFIDENCE`); near ties are `AMBIGUOUS`.
9. **Extract** (`ExtractionEngine`): per field, candidates from patterns → anchors (before/after) → typed
   scanners / enum synonyms; spans already explained by another field are not reused; one confident value →
   `EXTRACTED`, competing values → `AMBIGUOUS`, weak → `LOW_CONFIDENCE`, invalid → `INVALID`, nothing →
   `MISSING`. Free-text fields are never guessed: no anchor/pattern, no value.
10. **Clarify** (user): missing/ambiguous/low-confidence/invalid fields become `Clarification`s; answers are
    parsed and validated by the same typed parsers (`USER_PROVIDED`).
11. **Result**: `PipelineResult` JSON (docs/OUTPUT_CONTRACT.md). Audio and transcript are kept only if the
    user ticked the consent boxes; otherwise they live in memory until "New recording".

## Replacing a model or runtime

* **Another ASR model that sherpa-onnx can load**: add a pack to `app/src/main/assets/models/catalog.json`
  with `engine` one of `sherpa-whisper`, `sherpa-omnilingual-ctc`, `sherpa-dolphin-ctc`, `sherpa-nemo-ctc`
  and the file names in `params`; no code change. Other sherpa families (SenseVoice, Zipformer transducer,
  Paraformer…) need one more `when` branch in `SherpaOfflineRecognizer.modelConfig`.
* **Another LID model**: implement `LanguageDetector.classify(piece) → List<LanguageScore>`; everything
  else (piecing, aggregation, thresholds, UI) is shared.
* **Another runtime** (whisper.cpp, TFLite, ExecuTorch): implement `SpeechRecognizer`/`RecognizerFactory` in
  a new `engine` package and register it in `AppGraph`.
* **An ML intent classifier or an on-device LLM extractor**: implement `IntentClassifier` or wrap
  `ExtractionEngine`; the configuration schema already carries descriptions, examples and instructions for
  such a component.

## Privacy, battery, reliability

* Audio is processed from memory; nothing is written unless the user consents on the result screen.
  Retained files are app-private, excluded from backups (`data_extraction_rules.xml`) and deletable in Settings.
* Logs carry stage names, durations, counts and ids; `Redactor` drops transcript/value/audio keys.
* No service, no wake lock, no background work: recording happens only while the screen is on and the
  button is held/toggled; the Activity cancels recording when it stops. Models load lazily, the ASR model
  stays resident only when the setting allows and is dropped under memory pressure (`onTrimMemory`).
* The only network use is the system `DownloadManager` fetching model packs on request (Wi-Fi only by default).
* Every engine failure is caught and reported as a status with a warning, never a crash; routing falls back
  to any installed model; the UI offers retry at every failure point.
