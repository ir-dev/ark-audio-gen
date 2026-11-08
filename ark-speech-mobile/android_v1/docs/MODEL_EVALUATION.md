# Model and runtime evaluation

Evaluated for an offline, Indian-language-first push-to-talk app on ordinary mid-range Android phones
(4–6 GB RAM, Cortex-A7x cores, no reliable NPU path). Facts below were checked against the published
model cards, repositories and release assets on 8 November 2026; sizes are the actual file sizes.

## Decision in one table

| stage | chosen | why | runner-up |
|---|---|---|---|
| Runtime (VAD + ASR) | **sherpa-onnx 2.0.1** (Apache-2.0), prebuilt Android AAR | one Kotlin API for VAD, Whisper, Omnilingual, Dolphin, NeMo/IndicConformer and 20+ other families; model swap = catalogue entry; no NDK build | whisper.cpp (MIT): token probabilities and LID probabilities, but Whisper-only and needs an NDK build |
| Runtime (LID) | **ONNX Runtime Android 1.31.1** (MIT) | Anuvani-LID ships as a plain ONNX graph with a documented I/O contract; upstream sherpa-onnx has no loader for it (CIIL's kit patches sherpa-onnx 1.13.8) | CIIL's patched sherpa-onnx libraries (would pin the whole app to a fork) |
| VAD | **Silero VAD** (MIT), 0.6 MB | standard, tiny, robust to noise; used by both CIIL's and sherpa's pipelines | TEN-VAD (Apache-2.0) |
| Language ID | **CIIL Anuvani-LID int8** (Apache-2.0), 161 MB | the only model whose label set is exactly the 22 scheduled languages + Indian English, calibrated (ECE 0.008), rejects music/non-Indian speech, built for phones | Whisper's built-in LID (15 of 23 languages, no probabilities through sherpa-onnx) |
| ASR default | **Omnilingual ASR 300M CTC int8** (Apache-2.0), 349 MB | one model covering all 23 languages in their own scripts, no language hint, strongest recent open multilingual CTC at this size | Whisper small int8 (358 MB, 15 of 23 languages, hallucinates on silence) |
| ASR per-language upgrade | **IndicConformer 600M** per-language int8 packs (MIT), 628 MB each | best published accuracy on Indic benchmarks; routed per language where a deployment can afford it | Dolphin base CTC int8 (99 MB, 10 of our languages, no English) |

The architecture does not depend on these picks: every row is a catalogue entry or an adapter class.

## Candidates examined

### Spoken language identification

| model | licence | languages | size | gives probabilities? | mobile path | assessment |
|---|---|---|---|---|---|---|
| **CIIL Anuvani-LID** (115.6M Conformer, arXiv 2610.20263) | Apache-2.0 (code + weights) | 22 scheduled + Indian English (23 labels) | int8 161 MB, fp32 443 MB | yes, calibrated (ECE 0.008 after temperature); music/non-Indian rejected at AUROC 0.89–0.91 | ONNX graph: `audio` [N,S] f32 + `audio_length` [N] i64 → `logits` [N,23]; `mobile_inference/` kit with Android/iOS samples on a patched sherpa-onnx; CPU int8 default, QNN/NNAPI variants | **Chosen.** Reported 0.914 mean macro-F1 over 37 conditions and 0.897 over all 23 languages; 24 points ahead of Gemini at 1 s. CIIL's own measurements: 141 ms per 2 s piece and 988 ms per 15 s piece on an x86 emulator at 4 threads; peak RSS 307 MB (2 s pieces) to 687 MB (30 s); we cap pieces at 15 s (≈474 MB). Accuracy on 0.25–1 s clips is poor (24–74 %), so the app waits for ≥ 2 s of speech before trusting a decision. Not yet measured by CIIL on a physical phone. |
| Whisper LID via sherpa-onnx `SpokenLanguageIdentification` | MIT | 99 Whisper languages: 15 of our 23 (no Bodo, Dogri, Kashmiri, Konkani, Maithili, Manipuri, Odia, Santali) | shares the ASR pack (tiny 99 MB / base 153 MB) | no: the API returns one code; we derive a vote share across pieces | sherpa-onnx | Kept as the fallback when Anuvani is not installed. Known to confuse Hindi/Urdu/Marathi and to label Hinglish as English. |
| Meta MMS-LID (126 / 256 / 4017 languages) | **CC-BY-NC 4.0** | all 23 | 1 GB+ | yes | ONNX export possible | Excluded: non-commercial licence, large. |
| SpeechBrain VoxLingua107 ECAPA | Apache-2.0 | 107 languages, ~12 of our 23 | ~85 MB | yes, uncalibrated | ONNX export | Weak on Indian languages (little Indian data), missing 11 of our labels. |
| Dolphin's built-in LID | Apache-2.0 | 40 Eastern languages (hi, ur, bn, ta, te, gu, or, ne, ks, mr of ours; no English) | part of the ASR model | via decoder tokens | sherpa-onnx (CTC export has no LID output) | Not usable as a separate, calibrated decision. |

### Speech recognition

| model | licence | our languages | int8 size | RAM (est./measured) | sherpa-onnx | assessment |
|---|---|---|---|---|---|---|
| **Omnilingual ASR 300M CTC** (Meta, wav2vec2 + CTC, 1,600+ languages, shared 10k SentencePiece vocab) | Apache-2.0 | all 23 (scripts verified in the token file: Devanagari, Bengali, Gurmukhi, Gujarati, Odia, Tamil, Telugu, Kannada, Malayalam, Arabic) | 349 MB (model 348.4 + tokens) | ~700 MB (to measure) | native loader (`omnilingual`) | **Default.** No language hint, so code-mixed speech is transcribed as spoken. Accepts pieces ≤ 40 s (VAD caps at 25 s). It is the smallest Omnilingual size; Meta publishes per-language CER only for the 7B LLM model (hi 3.6, ta 5.2, te 6.5, bn 2.8), so the 300M CTC accuracy must come from our benchmark. The 1B int8 (750 MB) is the next step up for high-end phones. |
| **Whisper** tiny / base / small (OpenAI) | MIT | 15 of 23 + English | 99 / 153 / 358 MB | 300 / 450 / 900 MB | native (`whisper`), language hint, 30 s window | Strong English, weak and hallucination-prone on Indian languages below `medium` (1.5 GB, too heavy). Kept as optional packs and as the LID fallback carrier. |
| **AI4Bharat IndicConformer 600M multilingual** (hybrid CTC + RNNT) | MIT (gated, auto-approved on Hugging Face) | all 22 scheduled languages (no English) | official ONNX: shared encoder with external weights + per-language `joint_post_net_*` (0.6 MB each) + 5,633-token CTC head; community sherpa-onnx int8 packs: **628 MB per language** (encoder duplicated with a language mask) | ~1.4 GB per loaded language (to measure) | `nemo_ctc` loader (community packs) | Best Indic accuracy available; too large to be the default on a mid-range phone and needs the language first (which our LID-first design provides). Offered as `heavy` per-language packs (hi, ta in the catalogue; others are the same recipe) routed through `routing.byLanguage`. A proper port that shares the encoder across languages would cut storage to ~650 MB total; that is export work, not app work. |
| **Dolphin base CTC** (DataoceanAI + Tsinghua, E-Branchformer, 40 Eastern languages + 22 Chinese dialects) | Apache-2.0 | hi ur bn ta te gu or ne ks mr (no English, no kn/ml/pa/as…) | 99 MB (model 98.9 + tokens) | ~350 MB | native (`dolphin`) | Lightest Indic option; 33 % average WER on its own test sets for `base`. English must be routed elsewhere. Optional pack. |
| Vosk (Kaldi) | Apache-2.0 | hi, gu, en-IN only (small models 40–50 MB; large 1–1.5 GB) | — | low | separate runtime | No Tamil/Telugu/Bengali/Kannada…, no LID; rejected for coverage. |
| Google on-device `SpeechRecognizer` / ML Kit | proprietary | depends on device/language packs | — | — | — | Not controllable, needs Google services, no offline guarantee, no model choice; rejected on the "replaceable models" requirement. |
| NVIDIA Parakeet (CC-BY-4.0), Moonshine (MIT), SenseVoice (Apache-2.0) | various | English / European / zh-ja-ko | — | — | native | No Indian coverage. |

### Runtimes

| runtime | licence | pros | cons |
|---|---|---|---|
| **sherpa-onnx 2.0.1** | Apache-2.0 | prebuilt AAR (37 MB static-link variant: arm64-v8a, armeabi-v7a, x86_64), Kotlin API, VAD + LID + ASR + many model families, active (release 26 Oct 2026) | AAR is not on Maven Central (fetched by script, SHA-256 pinned); offline recognisers expose no token probabilities, so transcript confidence is heuristic |
| **ONNX Runtime Android 1.31.1** | MIT | Maven Central, runs any ONNX graph (Anuvani-LID), NNAPI/QNN providers available | duplicate `libonnxruntime.so` with the regular sherpa AAR — solved by sherpa's static-link variant |
| whisper.cpp 1.9.5 | MIT | token and language probabilities, GGML quantisation, very mature | Whisper only; app must build native code with the NDK; no Indic-specific models |
| CIIL's patched sherpa-onnx 1.13.8 | Apache-2.0 | Anuvani `LidPipeline` ready-made | a fork pinned to an older sherpa-onnx; conflicts with the current AAR |

## Licence summary

All shipped components are permissive and commercial-use friendly: sherpa-onnx (Apache-2.0), ONNX
Runtime (MIT), Silero VAD (MIT), Anuvani-LID (Apache-2.0 code and weights), Omnilingual ASR (Apache-2.0;
its corpus is CC-BY-4.0 but is not shipped), Whisper (MIT), Dolphin (Apache-2.0), IndicConformer (MIT).
MMS-LID (CC-BY-NC) was excluded for licence reasons. Each pack's licence and source are recorded in
`app/src/main/assets/models/catalog.json` and shown in the Models tab. Qualcomm QNN runtime libraries
(needed only for Anuvani's NPU variant) carry Qualcomm's own licence and are not included.

## Footprint of the default profile

| pack | files | on disk |
|---|---|---|
| silero-vad | 1 | 0.6 MB |
| anuvani-lid-int8 | 1 | 161.2 MB |
| omnilingual-300m-ctc-int8 | 2 | 348.5 MB |
| app APK (debug, arm64 + armv7 + x86_64 native libs) | | see BENCHMARK.md |
| **total models** | | **≈ 510 MB** |

Peak RAM with LID and ASR loaded sequentially is the larger of the two (≈ 700 MB class); with "keep
models loaded" on, both stay resident. The app releases models on `onTrimMemory` and the setting is
off by default on `isLowRamDevice` devices through the Models screen's fit check.

## A finding from the emulator: SME kernels and the hypervisor

The Android emulator on an Apple-silicon Mac passes the host CPU's feature flags into the guest
(`/proc/cpuinfo` lists `sme`, `sme2`, `sme2p1`…). ONNX Runtime 1.31.1 for Android ships KleidiAI
micro-kernels for SME/SME2 and dispatches to them on those flags, but the hypervisor does not execute
SME instructions, so the very first `OrtSession.run` dies with SIGILL. ORT's session option
`mlas.disable_kleidiai=1` avoids the dispatch; the app sets it automatically on emulator images and
offers it as a Settings kill switch ("Disable accelerated CPU kernels") for phones that misreport
their CPU. The sherpa-onnx static build contains no SME kernels, so VAD and ASR were unaffected.
This matters for anyone benchmarking ORT-based models on an Apple-hosted emulator; on a physical
phone with SME-capable cores (2025+ Arm C1/X925 class) the kernels are legitimate and faster.

## What the emulator runs showed (8 November 2026)

Full numbers and per-clip tables are in `docs/BENCHMARK.md`; the conclusions that affect the picks:

* **Anuvani-LID behaves as advertised.** 10 of 12 synthetic clips identified at 0.66–1.00; the two
  misses (Indian-English TTS voices → Gujarati 0.56–0.59) were flagged `uncertain` by the policy, and
  white noise scored 0.59. Identical output across three runs; 80–330 ms per 4–9 s clip warm.
* **Omnilingual 300M CTC int8 is the right default for Tamil and Telugu and a weak one for Hindi.**
  Tamil and Telugu clips reached full extraction; Hindi came out in Devanagari in 2 of 5 clips and in
  Urdu script in 3 (the same spoken language, no hint to choose the script); Kannada and Bengali were
  unusable on these voices. Routing Hindi to Whisper small instead was measured and rejected (next
  point); for Hindi-heavy deployments the answer is an IndicConformer pack via `routing.byLanguage`,
  and a Urdu→Devanagari script normaliser for Omnilingual output is the obvious next piece of work.
* **Whisper tiny is the English model, not an Indic one.** English appointment: full extraction,
  status OK, 175–205 ms ASR for 8.7 s; every Indic clip unusable. Whisper small is the smallest
  Whisper worth routing Hindi to; IndicConformer-hi remains the accuracy ceiling at 628 MB.
* **CTC models need leading silence**: every clip lost its first syllable until 300 ms of zeros were
  padded around each piece. They also fuse and misspell words, so lexical matching gained a
  substring pass and a bounded, script-aware fuzzy pass that can only ask, never decide.
* **Whisper small with the language hint writes Hindi in Devanagari** ("रमेश को 2500 रूपे योपी आई से…",
  "कल सुबहे दस भजे ड़वक्तर शर्मा के साथ…") and transcribed the English appointment perfectly (status OK,
  intent 0.85, four fields). But every Indic transcript stopped after 4–5 s of speech: Whisper's decoder
  has a fixed token budget per window and Indic scripts cost several byte-level BPE tokens per character.
  Feeding 5 s pieces did not change it (run 5), so this is the decoder stopping early on these voices,
  not the window budget; Whisper small also costs ~4× Omnilingual's CPU time and 1.27 GB PSS beside
  Anuvani. It is therefore not routed for Hindi by default. Wrong hints are destructive (an English clip decoded as Gujarati became "వార్ల్ల్ల…"), which is
  another reason the language decision must be confirmed when it is uncertain.
* **Memory**: ≈1.0 GB peak PSS with Anuvani int8 and Omnilingual int8 resident, ≈0.7 GB with Whisper
  tiny instead. On 4 GB phones keep "keep models loaded" off or route to the lighter pack.
* **ONNX Runtime 1.31.1 on an Apple-hosted emulator needs `mlas.disable_kleidiai`** (SME kernels vs
  the hypervisor); irrelevant on real phones but it cost an afternoon and is now a Settings switch.

## What still needs measuring

Accuracy and latency numbers above are the publishers' own, on their hardware. `docs/BENCHMARK.md`
describes the on-device benchmark harness (`scripts/benchmark.sh`) and records what has been measured
so far; a real mid-range phone is required for CPU, RAM, latency and battery figures that mean anything.
