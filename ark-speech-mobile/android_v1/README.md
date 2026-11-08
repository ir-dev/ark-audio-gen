# Ark Speech — offline push-to-talk understanding for Indian languages

An Android app that records speech while you hold a button, checks the audio, finds the speech,
identifies the language (22 scheduled languages of India + English), transcribes it on the device in
the language's own script, works out the intent and pulls out administrator-defined fields, and
returns a structured JSON result with confidences and processing status. Everything runs offline;
nothing is uploaded; nothing is kept without consent.

```
Push-to-talk → capture → validation → VAD → language ID → model routing → ASR
            → transcript review → intent → field extraction → clarification → JSON
```

| | |
|---|---|
| Runtimes | sherpa-onnx 2.0.1 (VAD, ASR), ONNX Runtime 1.31.1 (language ID) — both on-device, CPU |
| Default models | Silero VAD (0.6 MB) · CIIL Anuvani-LID int8 (161 MB) · Omnilingual ASR 300M CTC int8 (349 MB) |
| Optional models | Whisper tiny/base/small · Dolphin base · IndicConformer 600M per language |
| Languages | as bn brx doi en gu hi kn ks kok mai ml mni mr ne or pa sa sat sd ta te ur |
| Android | minSdk 26, targetSdk 36, arm64-v8a / armeabi-v7a / x86_64, Jetpack Compose |
| Build | Gradle 8.14.3, AGP 8.13.2, Kotlin 2.2.21 (`gradle/libs.versions.toml`) |

Documentation: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) · [docs/MODEL_EVALUATION.md](docs/MODEL_EVALUATION.md) ·
[docs/ADMIN_CONFIG.md](docs/ADMIN_CONFIG.md) · [docs/OUTPUT_CONTRACT.md](docs/OUTPUT_CONTRACT.md) ([schema](docs/output-schema.json)) ·
[docs/BENCHMARK.md](docs/BENCHMARK.md) · [samples/](samples/README.md) · [testdata/audio](testdata/audio/README.md)

## Build and run

Prerequisites: JDK 17+ (Android Studio's bundled JDK works), Android SDK with platform 36 and
build-tools 36 (`ANDROID_HOME` or `local.properties`), a device or emulator on Android 8.0+.

```bash
cd ark-speech-mobile/android_v1
scripts/fetch-runtime.sh                 # sherpa-onnx AAR → app/libs (37 MB, SHA-256 checked)
./gradlew :core:test                     # 81 JVM unit tests (parsers, validation, config, pipeline)
./gradlew :app:assembleDebug             # app/build/outputs/apk/debug/app-debug.apk (3 ABIs, ~180 MB)
./gradlew -ParkAbi=arm64-v8a :app:assembleDebug   # one ABI for a dev device/emulator (~68 MB)
adb install -r app/build/outputs/apk/debug/app-debug.apk
./gradlew :app:assembleRelease           # minified (R8) release build, unsigned
```

Open the app once (so its storage directory exists), then install the default models:

```bash
scripts/fetch-models.sh --list           # every pack with size, tier and licence
scripts/fetch-models.sh --push           # default tier: silero-vad anuvani-lid-int8 omnilingual-300m-ctc-int8 (≈510 MB)
scripts/fetch-models.sh --push whisper-tiny-int8 dolphin-base-ctc-int8   # optional extras
```

Models can also be downloaded inside the app (Models tab; system download manager, Wi-Fi only by
default). The Models tab shows the exact directory the app reads
(`Android/data/co.immanuel.arkspeech/files/models/<pack-id>/`); on emulator images that block
`adb push` there, copy into the app's internal `files/models/<pack-id>/` with `run-as` instead — both
locations are honoured.

### Instrumented tests and benchmark

```bash
./gradlew :app:assembleDebug :app:assembleDebugAndroidTest
scripts/instrumented-tests.sh             # VAD, Anuvani-LID, end-to-end multilingual clips, benchmark (skip without models)
scripts/instrumented-tests.sh co.immanuel.arkspeech.PipelineE2ETest   # one class
scripts/benchmark.sh                      # BenchmarkTest + dumpsys meminfo/batterystats → benchmarks/results/<stamp>/
```

On Play-Store emulator images the 6 GB data partition fills quickly: `adb install` fails below roughly
500 MB free. Remove one large pack, install, and push it back (`scripts/fetch-models.sh --push <id>`).

`./gradlew :app:connectedDebugAndroidTest` works too, but Gradle uninstalls the app afterwards and
Android deletes the app's external files with it — including the model packs. The script installs with
`adb` and runs `am instrument` directly, so models survive between runs. For A/B runs it accepts
`-e asrModel <pack-id>` through `am instrument` (see `TestSupport.kt`).

The test clips in `testdata/audio/` were synthesised with macOS voices (`scripts/gen-test-audio.sh`);
they exercise the whole chain in English, Hindi, Tamil, Telugu, Kannada and Bengali plus silence,
noise, clipping, a 0.3 s blip, a 44.1 kHz stereo file and an unsupported M4A.

## Using the app

1. **Record**: hold TALK (or tap to toggle in Settings). The timer counts towards the 3:00 cap, the
   meter shows level, and recording stops by itself at the cap; Cancel discards. A WAV file can be
   processed instead of the microphone for testing.
2. **Review**: the detected language with its probability (and alternatives when uncertain, a notice
   when the speech mixes languages); pick another language and re-transcribe if needed; correct the
   transcript text.
3. **Result**: intent and score, every field with its status (extracted / confirm? / ambiguous /
   missing / invalid), questions for anything missing or doubtful (type an answer or pick an option),
   consent boxes for keeping the recording or transcript, per-stage timings, and the JSON (copy/share).
4. **Models**: install, verify (SHA-256) and remove packs; see which engine each stage uses.
5. **Admin** (optional PIN): define, edit, enable/disable, version, roll back, import and export
   extraction definitions; test a sentence against a definition; edit speech-model routing.
6. **Settings**: hold-vs-tap, default consent, retained files, threads, keep-models-loaded,
   Wi-Fi-only downloads, admin PIN, device info.

## Project layout

```
core/      pure Kotlin: audio validation, language registry, VAD/LID/ASR contracts, routing,
           configuration schema + versioned repository, parsers, intent + extraction, pipeline, contract
app/       Android: capture, sherpa-onnx + ONNX Runtime adapters, model store/downloader, Compose UI,
           instrumented tests and benchmark harness
samples/   four administrator definitions (appointment, civic issue, payment, stock) + default routing
testdata/  multilingual WAV clips and edge cases used by both test suites
scripts/   fetch-runtime.sh, fetch-models.sh, gen-test-audio.sh, benchmark.sh
docs/      architecture, model evaluation, admin configuration, output contract, benchmark
```

## Privacy and resource notes

* Audio is held in memory and processed on the device; it is written to app-private storage only when
  the user ticks "keep" on the result screen, and those files are excluded from backups and deletable.
* Logs contain stage names, timings, counts and ids; transcript text and values are redacted.
* No service, wake lock or background processing: the microphone is open only while the button is
  held and the screen is on; leaving the app cancels the recording.
* Models load lazily; the speech model stays resident between recordings only if allowed in Settings
  and is released under memory pressure. The only network traffic is model download on request.

## What has been verified

| check | result |
|---|---|
| `./gradlew :core:test` | 81 JVM tests pass: audio validation on the real clips, WAV decoding, parsers in 8 languages, classifier, extraction on every sample example, versioned repository, redaction, pipeline with fakes |
| `./gradlew :app:assembleDebug` / `:app:assembleRelease` | debug (68 MB single-ABI, 180 MB three-ABI) and R8-minified release APKs build |
| Instrumented, Android 16 arm64 emulator with Silero + Anuvani + Omnilingual installed | 9 tests: VAD (3), Anuvani-LID (3), end-to-end edge cases and multilingual clips (2), benchmark (1) pass; per-clip outputs in `benchmarks/results/` after `scripts/instrumented-tests.sh` |
| Benchmark on that emulator (Apple M5 host, not a phone) | warm 0.08–0.12× real time end to end, ≈1.0 GB peak PSS with both default models resident; see docs/BENCHMARK.md |
| Physical mid-range phone | **not yet run** — `scripts/benchmark.sh` is the procedure; emulator numbers only show proportions |

## Known limits

* Transcript confidence is heuristic (sherpa-onnx offline recognisers expose no token probabilities).
* Intent and entity extraction are rule-driven and transparent; coverage comes from the definitions'
  phrases, synonyms, anchors and patterns (deepest built-in lexicons: English, Hindi, Tamil).
  An ML classifier or on-device LLM extractor can be slotted in behind the same interfaces.
* IndicConformer packs are 628 MB per language (community export duplicates the encoder); they are
  for 6 GB+ devices and one or two languages.
* Accuracy and latency figures depend on the phone; see docs/BENCHMARK.md for what was measured and how.
