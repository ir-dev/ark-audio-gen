# Structured output contract (`schemaVersion` 1.0)

`PipelineResult` is the JSON a client receives for one push-to-talk interaction. It is produced by
`co.immanuel.arkspeech.core.pipeline.PipelineResult.toJson()`; the machine-readable schema is
`docs/output-schema.json`. All fields are present (defaults encoded); `null` means "not determined".

```json
{
  "schemaVersion": "1.0",
  "requestId": "6c0f1a7e-…",
  "status": "OK",
  "audio": { "durationMs": 8830, "sampleRate": 16000, "rmsDbfs": -22.4, "peakDbfs": -3.1, "clippingRatio": 0.0,
             "estimatedSnrDb": 38.2, "speechRatio": 0.71, "dcOffset": 0.0, "issues": [], "verdict": "OK" },
  "speechSegments": [ { "startMs": 420, "endMs": 8610 } ],
  "language": { "code": "hi", "confidence": 0.93, "alternatives": [ { "code": "ur", "probability": 0.04 } ],
                "perSegment": [ { "startMs": 420, "endMs": 8610, "code": "hi", "probability": 0.93 } ],
                "timeShare": { "hi": 1.0 }, "isCodeMixed": false, "uncertain": false, "supported": true,
                "source": "AUTO", "engine": "ort-anuvani-lid", "durationMs": 8190 },
  "transcript": { "text": "मुझे कल सुबह दस बजे डॉक्टर शर्मा के साथ अपॉइंटमेंट बुक करनी है…", "language": "hi",
                  "segments": [ { "startMs": 420, "endMs": 8610, "text": "…" } ], "confidence": 0.82,
                  "confidenceSource": "heuristic", "modelId": "omnilingual-300m-ctc-int8",
                  "engine": "sherpa-omnilingual-ctc", "script": "Devanagari", "edited": false },
  "intent": { "status": "MATCHED", "best": { "definitionId": "book_appointment", "intentName": "book_appointment",
              "definitionVersion": 1, "score": 0.65, "matchedPhrases": ["अपॉइंटमेंट"], "matchedPatterns": 0, "negativeHits": 0 },
              "alternatives": [], "confidence": 0.65, "engine": "lexical-v1" },
  "fields": {
    "doctor_name":      { "name": "doctor_name", "type": "STRING", "required": false, "status": "EXTRACTED", "value": "शर्मा", "raw": "शर्मा", "confidence": 0.8, "candidates": [ … ], "extra": {} },
    "appointment_date": { "name": "appointment_date", "type": "DATE", "required": true, "status": "EXTRACTED", "value": "2026-11-09", "raw": "कल", "confidence": 0.9, "candidates": [ … ], "extra": {} },
    "appointment_time": { "…": "10:00" },
    "phone":            { "…": "9876543210" },
    "department":       { "name": "department", "type": "ENUM", "required": false, "status": "MISSING", "confidence": 0.0, "candidates": [] }
  },
  "missingRequired": [],
  "clarifications": [],
  "warnings": [],
  "processing": { "startedAt": "2026-11-08T04:31:02.114Z", "totalMs": 4120,
                  "stages": [ { "stage": "validate", "ms": 3, "status": "OK" }, { "stage": "vad", "ms": 61, "status": "OK", "detail": "1 segments, 8190 ms speech" },
                              { "stage": "lid", "ms": 640, "status": "OK", "detail": "hi 0.93" }, { "stage": "route", "ms": 0, "status": "OK", "detail": "omnilingual-300m-ctc-int8" },
                              { "stage": "asr", "ms": 3300, "status": "OK", "detail": "71 chars, conf 0.82" }, { "stage": "intent", "ms": 4, "status": "MATCHED", "detail": "book_appointment 0.65" },
                              { "stage": "extract", "ms": 9, "status": "OK", "detail": "4/5 extracted" } ],
                  "vadEngine": "sherpa-silero-vad", "lidEngine": "ort-anuvani-lid", "asrEngine": "sherpa-omnilingual-ctc",
                  "asrModelId": "omnilingual-300m-ctc-int8", "nluEngine": "lexical-v1+rules-v1",
                  "device": { "manufacturer": "…", "model": "…", "sdk": "36", "abi": "arm64-v8a", "cores": "8", "totalRamMb": "5800", "lowRamDevice": "false" } },
  "retention": { "consentToRetainAudio": false, "consentToRetainTranscript": false, "audioRetained": false, "transcriptRetained": false, "processedOnDevice": true }
}
```

## `status`

| value | meaning | client action |
|---|---|---|
| `OK` | intent matched; every required field extracted or provided and valid | act on `fields` |
| `NEEDS_CLARIFICATION` | usable, but `clarifications` is non-empty (language, intent or fields) | ask the user, call the step API, re-read |
| `AUDIO_REJECTED` | `audio.issues` explains (SILENT, NOISY, TOO_SHORT, TOO_LONG, CLIPPING, EMPTY, UNSUPPORTED_FORMAT) | re-record |
| `NO_SPEECH` | VAD found nothing (or ASR produced nothing) | re-record |
| `UNSUPPORTED_LANGUAGE` | detected language outside the supported set | pick a language manually |
| `MODEL_UNAVAILABLE` | no installed ASR pack in the routing chain | install a model |
| `NO_INTENT` | transcript present, no definition matched (or its fallback is REJECT) | offer intents manually |
| `FAILED` | an engine threw; see `warnings` | retry |
| `IN_PROGRESS` | only in intermediate UI snapshots | — |

## Field status (`fields[*].status`)

`EXTRACTED` · `MISSING` · `LOW_CONFIDENCE` (value present, confirm) · `AMBIGUOUS` (`candidates` lists the
options, `value` null) · `INVALID` (`reason` says why) · `DEFAULTED` (administrator default applied, explicit)
· `USER_PROVIDED` (entered or confirmed during clarification).

Canonical value formats: DATE `YYYY-MM-DD`; TIME `HH:mm` (24 h); PHONE 10 digits; AMOUNT plain number with
`extra.currency` (`INR`); NUMBER/INTEGER plain number; DURATION ISO-8601 (`PT2H30M`, `P3D`); BOOLEAN
`true`/`false`; ENUM the configured `value`; STRING as spoken (original script and case).

## Confidence

* `language.confidence` is a calibrated probability from Anuvani-LID (`engine` = `ort-anuvani-lid`); with
  the Whisper fallback it is a duration-weighted vote share (`sherpa-whisper-lid`).
* `transcript.confidence` is heuristic (`confidenceSource` = `heuristic`) for the sherpa-onnx offline
  recognisers, which expose no token probabilities; `user` after an edit.
* `intent.confidence` is the lexical score; `fields[*].confidence` reflects the evidence source
  (pattern 0.92, anchor 0.8–0.88, scanner 0.5–0.95, synonym 0.9–0.95, user 1.0).

## Clarifications

`kind` ∈ `LANGUAGE`, `INTENT`, `FIELD_MISSING`, `FIELD_AMBIGUOUS`, `FIELD_LOW_CONFIDENCE`, `FIELD_INVALID`;
`prompt` is a ready-to-show question; `options` lists codes/ids/values to offer; `candidates` carries the
raw spans for ambiguous values. The step API (`SpeechPipeline.overrideLanguage`, `editTranscript`,
`extract(forcedDefinitionId)`, `provideFieldValue`, `confirmFieldValue`) resolves them.
