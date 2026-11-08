# Administrator configuration

Administrators define what the app can understand without an app update. Every definition is a JSON
document in the schema below, stored on the device as immutable versions
(`files/config/definitions/<id>/v<N>.json` + a `current` pointer), managed from the **Admin** tab
(optionally PIN-protected in Settings) or by importing a bundle. Four worked definitions ship in
`samples/definitions/` and are seeded on first launch.

## Lifecycle

| action | what happens |
|---|---|
| **New / Edit → Save** | `DefinitionValidator` checks the document; a new version `N+1` is written and becomes current. The change note and author go into the history. |
| **Enable / Disable** | recorded as a version too; disabled definitions are ignored by the pipeline. |
| **History → Restore** | copies version `K` forward as version `N+1` ("Rolled back to version K"); nothing is ever overwritten. |
| **Import** | a `ConfigBundle` (or a single definition file) is validated as a whole first; if anything is invalid nothing is imported. Each definition becomes a new version of its id; optional routing and policies are applied. |
| **Export** | writes the current versions, routing and policies as one bundle (`ark-speech-config.json`) for backup, review or distribution to other devices (MDM, email, USB). |
| **Delete** | removes every version of an id (confirmation required). |
| **Test** | in the editor: type a transcript, pick a language and see the intent score, matched phrases and every field's status, value and reason. |

## Definition schema

```jsonc
{
  "id": "book_appointment",                 // snake_case, stable; versions are per id
  "intent": { "name": "book_appointment", "description": "Book a clinic or doctor appointment" },
  "version": 1,                             // assigned by the repository on save
  "enabled": true,
  "languages": ["en", "hi", "ta"],          // empty = all 23 languages
  "triggers": {
    "phrases": { "en": ["book an appointment", "appointment"], "hi": ["अपॉइंटमेंट", "appointment"], "*": [] },
    "patterns": ["(?i)appoint\\w*"],        // Java regex on the lower-cased transcript
    "negativePhrases": { "en": ["cancel"] } // subtract from the score
  },
  "fields": [ /* see below */ ],
  "confidenceThreshold": 0.5,               // minimum intent score
  "fallback": "ASK_USER",                   // ASK_USER | REJECT | ACCEPT_LOW_CONFIDENCE
  "examples": [ { "language": "hi", "text": "…", "expected": { "doctor_name": "शर्मा" }, "note": "" } ],
  "instructions": "Free text for people (and for a future LLM extractor).",
  "updatedAt": "", "updatedBy": "", "changeNote": ""   // filled by the repository
}
```

Phrases are matched after normalisation: Unicode NFC, case folding, punctuation removed, native digits
(Devanagari, Bengali, Tamil, Telugu, Kannada, Malayalam, Gujarati, Gurmukhi, Odia, Arabic-Indic) converted
to ASCII, at word boundaries that respect Indic combining marks. For code-mixed speech the English
phrases are always checked in addition to the detected language's.

### Field

```jsonc
{
  "name": "appointment_date",              // snake_case, unique in the definition
  "description": "Appointment date",       // shown to the user in clarification prompts
  "type": "DATE",                          // STRING INTEGER NUMBER BOOLEAN DATE TIME PHONE AMOUNT ENUM DURATION
  "required": true,
  "validation": {
    "regex": "^[6-9][0-9]{9}$",            // on the canonical value
    "allowedValues": [ { "value": "upi", "synonyms": { "en": ["upi", "gpay"], "hi": ["यूपीआई"] }, "description": "" } ],  // ENUM only
    "min": 1, "max": 100000,               // INTEGER NUMBER AMOUNT
    "minLength": 3, "maxLength": 80,       // STRING
    "maxDaysFromToday": 365                // DATE
  },
  "extraction": {
    "patterns": ["(?i)ref\\s*(?<value>[A-Z0-9]{6,})"],   // named group `value` (or group 1)
    "anchors": { "en": ["with doctor", "doctor"], "hi": ["डॉक्टर"] },       // value follows the phrase
    "anchorsBefore": { "hi": ["के पास"], "ta": ["அருகில்"] },             // value precedes the phrase (postpositions)
    "stopWords": { "hi": ["के"], "*": ["for", "at"] },  // end an anchored STRING value
    "maxTokens": 2,                                     // words taken for an anchored STRING
    "temporalDirection": "FUTURE",                      // FUTURE | PAST | UNSPECIFIED: resolves "kal", weekdays, year
    "instructions": "", "examples": []
  },
  "confidenceThreshold": 0.6,              // per-field override of the acceptance threshold
  "defaultValue": null                     // if set, applied explicitly as DEFAULTED when nothing is found
}
```

### How values are found (in order, per field)

1. **patterns** — regex captures, confidence 0.92; the capture is then parsed by the field's type.
2. **anchors / anchorsBefore** — the value next to a configured phrase: for typed fields the nearest
   parsable value in a short window (0.88); for STRING up to `maxTokens` words until punctuation, a stop
   word or an intent trigger word (0.8).
3. **type scanner** — the whole transcript: dates (numeric, "5 November", relative words and weekdays in
   en/hi/ta/te/kn/bn/mr/gu), times (am/pm, "सुबह दस बजे", "பத்து மணிக்கு", "साढ़े दस"), phone numbers
   (digits or spoken digits, +91/0 prefixes), amounts (number + ₹/rupees/रुपये/ரூபாய்/টাকা…), numbers
   (digits or words incl. lakh/crore, Hindi 1–99 words, Hinglish), durations, ENUM synonyms.
   STRING and BOOLEAN fields have no scanner: without a pattern or anchor they stay MISSING.

Spans already explained by a more specific field (a date, an amount, a phone) are not reused for
generic numbers or names; an ambiguous ENUM whose option lies inside another field's value ("Road" in
"Gandhi Road bus stop") is re-decided without it.

### Decision per field

| situation | status |
|---|---|
| one value, confidence ≥ threshold, valid | `EXTRACTED` |
| one value below threshold (e.g. "10 baje" with no AM/PM) | `LOW_CONFIDENCE` → user confirms |
| distinct values within 0.25 confidence of each other (Hindi "kal" with `UNSPECIFIED` direction) | `AMBIGUOUS` → user chooses |
| value fails validation | `INVALID` with `reason` |
| nothing found, `defaultValue` set | `DEFAULTED` |
| nothing found | `MISSING` (→ clarification if required) |

The engine never fills a value it did not see. Ambiguity and low confidence are surfaced, not resolved
silently; `temporalDirection` is the administrator's explicit instruction for cases like "kal".

## Validation rules the editor enforces

ids and field names snake_case; known language codes; at least one trigger phrase or pattern; valid
regular expressions; unique field names; ENUM fields have allowed values (and only ENUM fields do);
min ≤ max; thresholds in [0, 1]; examples reference existing fields; bundle schema version 1.x and unique ids.

## Routing and policies (also in the bundle)

* `routing` — `{"defaultModelId", "byLanguage": {"hi": "indicconformer-hi-int8"}, "fallbackModelIds": [...]}`
  chooses the ASR pack per language (Admin → Speech model routing). Pack ids are in `catalog.json`.
* `languagePolicy` — `minConfidence` (0.60), `minMargin` (0.15), `codeMixMinShare` (0.25), `allowedLanguages`.
* `audioPolicy` — duration bounds, silence/quiet levels, clipping and SNR thresholds.

## Language coverage of the built-in lexicons

| language | numbers | dates (relative, months, weekdays) | times | currency |
|---|---|---|---|---|
| English, Hindi (Devanagari + romanised), Tamil | full | full | full | yes |
| Telugu, Kannada, Bengali, Marathi, Gujarati | core (units, tens, hundred, thousand, lakh, crore) | relative days, months, weekdays (te/kn/bn) | periods + hour markers | yes |
| other scheduled languages | ASCII/native digits only | numeric dates | numeric times | ₹/Rs |

Coverage grows through configuration — synonyms, anchors, patterns and stop words are per language in
every definition — and through `Lexicons.kt` for shared vocabulary.
