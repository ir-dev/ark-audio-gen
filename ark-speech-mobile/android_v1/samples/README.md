# Sample administrator configuration

`definitions/*.json` are four extraction definitions in the schema described in
`docs/ADMIN_CONFIG.md`. They are seeded into the app on first launch (Admin → Import also accepts
them as a bundle) and exercised by the unit tests in `core/src/test` and the instrumented tests in
`app/src/androidTest`.

| id | intent | languages | required fields |
|---|---|---|---|
| book_appointment | clinic booking | en hi ta te kn bn mr gu | appointment_date |
| report_issue | civic complaint | all | issue_type, location |
| record_payment | payment log | en hi ta te kn bn | amount, payee |
| stock_update | inventory movement | all | action, quantity, item |

`routing.json` is the default language → ASR model routing: everything to the Omnilingual 300M
pack, with Whisper packs as fallbacks when it is not installed. A deployment that installs an
IndicConformer language pack adds e.g. `"byLanguage": {"hi": "indicconformer-hi-int8"}`.
