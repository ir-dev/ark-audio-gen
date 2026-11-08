# Benchmark

What to measure, how the harness measures it, and the numbers obtained so far. Figures from the
development machine's emulator are **not** phone figures; the section "Results" says exactly where
each number came from.

## Metrics and how they are captured

| metric | how |
|---|---|
| Model download / storage size | file sizes in `catalog.json` (verified SHA-256); on device `du -sh Android/data/co.immanuel.arkspeech/files/models/*` (`scripts/benchmark.sh` saves `model-storage.txt`) |
| APK size | `app/build/outputs/apk/debug/app-debug.apk`; the native libraries dominate (sherpa-onnx 24 MB + ONNX Runtime 34 MB per ABI). A release App Bundle delivers one ABI per device |
| Peak RAM | `Debug.getPss()` sampled every 50 ms during each run by `BenchmarkTest` (`peakPssMb`), plus `dumpsys meminfo <pkg>` after the run; baseline PSS before any model loads is recorded |
| CPU usage | `Process.getElapsedCpuTime()` delta per run (`cpuMs`); compare with wall time to see parallelism (threads setting) |
| Inference time | per-stage wall time from the result's `processing.stages` (validate, vad, lid, route, asr, intent, extract) and `totalMs`; real-time factor = total / audio duration; first run is the cold start (model load included) |
| Battery | `BatteryManager.BATTERY_PROPERTY_CHARGE_COUNTER` delta over the benchmark (µAh; `null` where the device does not report it) and `dumpsys batterystats <pkg>` after a `--reset`. Meaningful only unplugged, screen state constant, over a longer session (e.g. 50 recordings) |

Harness: `app/src/androidTest/.../BenchmarkTest.kt` (3 clips × 3 runs: English, Hindi, Tamil
appointments, 4–9 s each), run by `scripts/benchmark.sh`, which also pulls `benchmark.json`,
`benchmark.md`, `meminfo.txt`, `batterystats.txt` and `model-storage.txt` into
`benchmarks/results/<timestamp>/`.

Procedure on a phone:

1. `./gradlew :app:installDebug`, open the app once, `scripts/fetch-models.sh --push`.
2. Unplug, screen on, no other foreground apps. `scripts/benchmark.sh`.
3. For battery: run 50 recordings via the UI or loop the test, read `batterystats.txt`
   ("Estimated power use" and the app's CPU time) and the charge counter delta.
4. Repeat with Settings → threads = 2 and 4, and with "keep models loaded" off, to see the
   latency/RAM trade-off.

Target device class: 4–6 GB RAM, Snapdragon 6xx/7xx or Dimensity 7xx/8xx class, Android 12–15.

## Publisher-reported reference points (not our measurements)

* Anuvani-LID int8 (CIIL, x86_64 emulator, 4 threads): 141 ms per 2 s piece, 533 ms per 8 s,
  988 ms per 15 s; peak RSS 307 / 384 / 474 MB for 2 / 8 / 15 s pieces; 687 MB at 30 s; load 0.6 s.
* Omnilingual ASR 300M CTC: Meta reports RTF 0.001 on a server GPU; CPU int8 phone figures are not
  published — ours are below.
* Whisper small int8 on sherpa-onnx is typically 1–3× real time on a mid-range phone CPU.

## Results

### Run 1 — Android emulator on an Apple M5 MacBook (not a phone)

Hardware: Android Studio emulator 36.3.10, `Medium_Phone_API_36.1` image (Android 16, arm64, Google
Play), started with `-memory 4096 -cores 4`, hypervisor-accelerated on an Apple M5 (10 cores, 16 GB).
Engines: Silero VAD · Anuvani-LID int8 (ONNX Runtime, KleidiAI disabled, 4 threads) · Omnilingual
300M CTC int8 (sherpa-onnx, 4 threads), both models kept resident. Measured 2026-11-08 with
`BenchmarkTest`; each clip 3 runs, the first run of the first clip includes model loading.

| clip | audio | run | total ms | RTF | VAD ms | LID ms | ASR ms | peak PSS MB | CPU ms |
|---|---|---|---|---|---|---|---|---|---|
| en_appointment | 8.7 s | cold | 2040 | 0.23 | 71 | 703 | 1251 | 897 | 3982 |
| en_appointment | 8.7 s | warm | 925 / 808 | 0.11 / 0.09 | 40 / 47 | 224 / 209 | 649 / 532 | 1015 | 3077 / 2761 |
| hi_appointment | 8.8 s | warm | 1174 / 819 / 780 | 0.13 / 0.09 / 0.09 | 248 / 70 / 47 | 308 / 207 / 206 | 605 / 532 / 515 | 1017 | 3545 / 2720 / 2680 |
| ta_appointment | 4.4 s | warm | 379 / 367 / 365 | 0.09 / 0.08 / 0.08 | 15 / 14 / 13 | 97 / 94 / 82 | 254 / 248 / 261 | 1015 | 1307 / 1295 / 1276 |

Reading the numbers:

* Warm end-to-end latency is about **0.09× real time** (8.7 s of speech understood in 0.8 s);
  intent + extraction cost 0–2 ms, validation 1–4 ms. Everything is the two neural stages.
* Cold start adds ≈1.2 s: Anuvani-LID load ≈0.5 s, Omnilingual load ≈0.7 s (the first ASR call also
  pays ONNX Runtime graph optimisation).
* **Peak PSS ≈ 1.0 GB** with both int8 models resident (baseline process ≈ 130 MB). CIIL's own
  figure for Anuvani int8 alone is 307–474 MB at 2–15 s pieces; the Omnilingual 300M int8 session
  accounts for the rest. With "keep models loaded" off, peak falls to the larger of the two.
* CPU time ≈ 3.3× wall time: the 4 inference threads are well used; on a phone with 2 big + 6 small
  cores expect a lower ratio and 2–4× these wall times.
* Battery: the emulator reports no charge counter (`batteryChargeCounterDeltaUah = null`); see the
  procedure above for a phone.
* These are emulator numbers on a laptop-class CPU. They show stage proportions and memory, not phone
  latency. A Snapdragon 6-series phone should be expected around 2–5 s warm for a 9 s utterance;
  this is the measurement still to be made.

Storage on the device: silero-vad 0.6 MB · anuvani-lid-int8 161 MB · omnilingual-300m-ctc-int8
349 MB · (optional) whisper-tiny-int8 99 MB. Debug APK 180 MB (three ABIs × two native runtimes);
a release App Bundle ships one ABI.

### Run 2 — same emulator, ASR model comparison (end-to-end clips)

Same device and settings; `scripts/instrumented-tests.sh` with the default routing (Omnilingual 300M
CTC int8) and then `ASR_MODEL=whisper-tiny-int8` (Whisper decodes with the detected language as the
hint). Thirteen synthetic clips, intent correct = the expected definition was chosen (possibly as a
"please confirm" candidate); fields = values extracted automatically out of the definition's fields.

| clip | expected | Omnilingual 300M: status / intent / fields | Whisper tiny: status / intent / fields |
|---|---|---|---|
| en_appointment | book_appointment | needs clarification / ✓ / 3 of 5 | **OK / ✓ / 4 of 5** |
| en_payment | record_payment | needs clarification / ✓ (via language fallback) / 3 of 5 | needs clarification / ✓ / 4 of 5 |
| en_issue | report_issue | no intent (Urdu-script output) | needs clarification / ✓ / 2 of 4 |
| hi_appointment | book_appointment | no intent (trigger misspelled "अपौइंतमेंत") | no intent (digits and noise) |
| hi_payment | record_payment | needs clarification / ✓ / 2 of 5 | no intent |
| hi_issue | report_issue | no intent (Urdu script) | no intent |
| hi_codemix | book_appointment | no intent (Urdu script) | needs clarification / ✓ / 2 of 5 |
| hi_appointment_noisy | book_appointment | no intent (mixed scripts) | no intent |
| ta_appointment | book_appointment | needs clarification / ✓ / 1 of 5 | no intent |
| ta_issue | report_issue | no intent ("பிறச்சனை" misspelled) | no intent |
| te_appointment | book_appointment | **OK / ✓ / 2 of 5** (date and time) | no intent |
| kn_stock | stock_update | no intent (mixed scripts) | no intent |
| bn_payment | record_payment | no intent | no intent |
| **intent found** | | **5 / 13** | **4 / 13** |
| warm ASR time, 8.7 s clip | | 460–590 ms | 175–205 ms |
| peak PSS with LID resident | | ≈ 1040 MB | ≈ 710 MB |

What the comparison says: Whisper tiny is the better English model and useless for Indian languages
at this size; Omnilingual 300M is usable for Tamil and Telugu, inconsistent for Hindi (Devanagari in
2 of 5 clips, Urdu script in 3) and poor for Kannada and Bengali on these synthetic voices. Neither
small model is good enough for Hindi; that is what the per-language routing is for (Whisper small or
an IndicConformer pack for `hi`). Two robustness fixes came out of this run and are in the code:
300 ms of silence padded around each piece before CTC decoding (the first syllable was being lost on
every clip), and bounded fuzzy matching of trigger words and enum synonyms with script-aware folding
of retroflex/dental, sibilant and vowel-length confusions (a fuzzy hit can only produce a
"please confirm", never an automatic acceptance).

### Run 3 — same emulator, Omnilingual routing, after the robustness fixes

Same clips with silence padding, fused-word and fuzzy matching, the cross-language intent fallback
and the padded VAD segments in place. `PipelineE2ETest` passes its thresholds (13/13 transcripts,
language top-2 ≥ 60 %, intents ≥ 40 %).

| clip | status | intent (how) | fields extracted |
|---|---|---|---|
| en_appointment | needs clarification | book_appointment (matched) | doctor, time, phone (3/5) |
| en_payment | needs clarification | record_payment (language fallback → confirm) | payee, date, mode; amount low-confidence (no currency word heard) |
| en_issue | needs clarification | stock_update suggested at 0.46 — a **fuzzy false positive** ("स्टॉप" ≈ "स्टॉक" in a Devanagari transliteration of English); surfaced as a question, not accepted | 0/4 |
| hi_appointment | needs clarification | book_appointment (fuzzy "अपौइंतमेंत" → confirm) | date 2026-11-09, time 10:00 (2/5) |
| hi_payment | needs clarification | record_payment (matched) | payee रमेश, date 2026-11-05; mode upi low-confidence (fuzzy "यूपआई") |
| hi_issue, hi_codemix | no intent | Urdu-script transcript | — |
| hi_appointment_noisy | no intent | mixed-script transcript | — |
| ta_appointment | needs clarification | book_appointment (matched) | doctor (1/5); date/time lost to fused words ("காலைப்பத்து") |
| ta_issue | **OK** | report_issue (fuzzy "பிறச்சனை") | issue_type water, location "காந்தி சாலைப பெருந்து நீடுத்தம்", severity high (3/4) |
| te_appointment | **OK** | book_appointment (matched, 0.85) | date 2026-11-09, time 10:00 (2/5) |
| kn_stock, bn_payment | no intent | unusable transcripts from the 300M model on these voices | — |
| **intent found** | | **8 / 13** (5 before the fixes) | |

Confirmed again with the final build at 10:28 (all 9 instrumented tests pass; 8/13 intents; warm
0.07–0.09 RTF; peak PSS 1.04 GB). Raw files: `docs/benchmarks/emulator-2026-11-08/`.

### Run 4 — same emulator, Whisper small int8 for every language (hinted)

`ASR_MODEL=whisper-small-int8`, before the 5 s piece splitting. ASR 1.0–3.2 s per clip (3–6× slower
than Whisper tiny, 2–4× slower than Omnilingual), peak PSS not re-measured.

| clip | result |
|---|---|
| en_appointment | **OK**, book_appointment 0.85, doctor Sharma, 2026-11-09, 10:00, phone (4/5): "Book an appointment with Dr. Sharma for tomorrow at 10 in the morning…" |
| en_payment, en_issue | LID said Gujarati (uncertain) → Gujarati hint → repetitive garbage ("వార్ల్ల్ల…"); a wrong hint is worse than no hint |
| hi_appointment, hi_payment, hi_issue, hi_codemix | **Devanagari**, readable, but cut after ~4–5 s ("रमेश को 2500 रूपे योपी आई से भे", "कल सुबहे दस भजे ड़वक्तर शर्मा के साथ अपाट्वे") → no intent |
| ta_appointment, ta_issue | readable Tamil, also truncated ("நாளை,காலை 10 மணிக்கு டாக்திர் கும") |
| te, kn, bn | garbage |

The truncation is Whisper's per-window token budget against byte-level Indic tokenisation; the
recogniser now splits segments into ≤ 5 s pieces at quiet points before decoding with Whisper.

### Run 5 — same emulator, Hindi routed to Whisper small (5 s pieces), everything else Omnilingual

Full suite with the sample routing temporarily set to `hi → whisper-small-int8`; three models installed
(Anuvani, Omnilingual, Whisper small). VAD and LID tests pass, the end-to-end test **fails** its intent
threshold (5/13), because every Hindi clip still stops after ~3 s of speech even in 5 s pieces
("उजी कल सुबहेदस भजे ड़वक् मेरा फों नमबर…", "रमेश को 2500 रूपे योपी आई से भे") — so the early stop is the
decoder giving up on these voices, not the window budget. Omnilingual's Hindi (2 of 5 intents, run 3)
is better than Whisper small's (0 of 5) on this material, and the sample routing now has **no Hindi
override**; the per-language mechanism stays for IndicConformer packs.

| clip | run | total ms | RTF | ASR ms | LID ms | peak PSS MB | CPU ms | ASR model |
|---|---|---|---|---|---|---|---|---|
| en_appointment (8.7 s) | cold | 2213 | 0.25 | 1278 | 875 | 920 | 4380 | Omnilingual |
| en_appointment | warm | 716 / 734 | 0.08 | 507 / 509 | 154 / 157 | 1046 | 2611 / 2700 | Omnilingual |
| hi_appointment (8.8 s) | first (loads Whisper) | 3471 | 0.39 | 3205 | 200 | 1183 | 10074 | Whisper small |
| hi_appointment | warm | 2405 / 2154 | 0.27 / 0.24 | 1965 / 1967 | 391 / 139 | 1266 / 1240 | 8758 / 8180 | Whisper small |
| ta_appointment (4.4 s) | first (reloads Omnilingual) | 989 | 0.23 | 895 | 73 | 1225 | 1941 | Omnilingual |
| ta_appointment | warm | 333 / 331 | 0.08 | 234 / 241 | 77 / 69 | 931 / 927 | 1263 / 1262 | Omnilingual |

Two operational numbers from this run: **switching ASR models costs 0.7–1.2 s** (the cache keeps one
model resident, so alternating languages reloads), and **Whisper small int8 is ~4× the CPU time of
Omnilingual 300M int8** per second of audio (≈ 1 s per 4 s of speech on this emulator; expect 2–4×
that on a mid-range phone, i.e. slower than real time). Peak PSS reached 1.27 GB with Whisper small
loaded next to Anuvani — too much for 4 GB phones unless "keep models loaded" is off.

### Accuracy observations from the end-to-end clips (run 1)

Anuvani-LID: 10 of 12 synthetic clips identified correctly with 0.66–1.00 probability; the two
misses (en_payment, en_issue — macOS Indian-English voices) came back as Gujarati at 0.56–0.59 and
were flagged `uncertain`, exactly the case the manual override exists for; white noise scored 0.59,
below the confidence threshold. Omnilingual 300M CTC int8 transcribed Tamil and Telugu well, English
usably, and Hindi/Kannada/Bengali poorly on these TTS voices, writing Hindi speech in Urdu script
in 3 of 5 Hindi clips (no language hint possible with a CTC model). That drove two changes: the
sample routing sends `hi`/`ur` to a hinted Whisper pack when one is installed, and the lexical
matcher tolerates fused words. The comparison with Whisper tiny is recorded below.

