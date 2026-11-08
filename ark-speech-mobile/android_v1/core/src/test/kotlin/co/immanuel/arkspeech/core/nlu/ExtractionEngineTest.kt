package co.immanuel.arkspeech.core.nlu

import co.immanuel.arkspeech.core.config.AllowedValue
import co.immanuel.arkspeech.core.config.ExtractionHints
import co.immanuel.arkspeech.core.config.FieldDefinition
import co.immanuel.arkspeech.core.config.FieldType
import co.immanuel.arkspeech.core.config.FieldValidation
import co.immanuel.arkspeech.core.config.IntentSpec
import co.immanuel.arkspeech.core.config.IntentTriggers
import co.immanuel.arkspeech.core.config.TemporalDirection
import co.immanuel.arkspeech.core.config.ExtractionDefinition
import co.immanuel.arkspeech.core.support.TestSupport
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull
import kotlin.test.assertTrue

class ExtractionEngineTest {
    private val engine = ExtractionEngine(TestSupport.clock)

    @Test fun `every sample example extracts its expected values`() {
        val failures = ArrayList<String>()
        for (def in TestSupport.definitions()) for (ex in def.examples) {
            val r = engine.extract(NormalizedText(ex.text), def, ex.language)
            for ((field, expected) in ex.expected) {
                val f = r.fields.getValue(field)
                if (f.value != expected || f.status !in setOf(FieldStatus.EXTRACTED, FieldStatus.LOW_CONFIDENCE)) {
                    failures += "${def.id}/${ex.language}.$field: expected '$expected' got '${f.value}' (${f.status}, ${f.reason}, cands=${f.candidates.map { it.raw }})"
                }
            }
        }
        assertTrue(failures.isEmpty(), failures.joinToString("\n"))
    }

    @Test fun `appointment date resolves forward from the fixed clock`() {
        val def = TestSupport.definition("book_appointment")
        val en = engine.extract(NormalizedText(def.examples[0].text), def, "en")
        assertEquals("2026-11-09", en.fields.getValue("appointment_date").value)
        assertEquals(FieldStatus.EXTRACTED, en.fields.getValue("appointment_date").status)
        val hi = engine.extract(NormalizedText(def.examples[1].text), def, "hi")
        assertEquals("2026-11-09", hi.fields.getValue("appointment_date").value, "कल resolved with FUTURE direction")
        assertTrue(hi.missingRequired.isEmpty(), "${hi.missingRequired}")
    }

    @Test fun `hindi payment date is not re-used as the payee`() {
        val def = TestSupport.definition("record_payment")
        val r = engine.extract(NormalizedText("रमेश को दो हज़ार पाँच सौ रुपये यूपीआई से भेजे हैं, पाँच नवंबर को।"), def, "hi")
        assertEquals("रमेश", r.fields.getValue("payee").value)
        assertEquals(FieldStatus.EXTRACTED, r.fields.getValue("payee").status, "${r.fields.getValue("payee")}")
        assertEquals("2026-11-05", r.fields.getValue("payment_date").value)
        assertEquals("2500", r.fields.getValue("amount").value)
        assertEquals("INR", r.fields.getValue("amount").extra["currency"])
    }

    @Test fun `missing required field is reported not guessed`() {
        val def = TestSupport.definition("record_payment")
        val r = engine.extract(NormalizedText("রমেশকে পাঁচই নভেম্বর ইউপিআই-এর মাধ্যমে দুই হাজার পাঁচশো টাকা পাঠানো হয়েছে।"), def, "bn")
        assertEquals(FieldStatus.MISSING, r.fields.getValue("payee").status)
        assertEquals(listOf("payee"), r.missingRequired)
        assertTrue(r.needsClarification)
        assertEquals("upi", r.fields.getValue("payment_mode").value)
    }

    private fun single(field: FieldDefinition) = ExtractionDefinition("t", IntentSpec("t"), triggers = IntentTriggers(mapOf("*" to listOf("x"))), fields = listOf(field))

    @Test fun `ambiguous relative date asks instead of guessing`() {
        val def = single(FieldDefinition("when", type = FieldType.DATE, required = true, extraction = ExtractionHints(temporalDirection = TemporalDirection.UNSPECIFIED)))
        val f = engine.extract(NormalizedText("कल मिलते हैं"), def, "hi").fields.getValue("when")
        assertEquals(FieldStatus.AMBIGUOUS, f.status)
        assertEquals(setOf("2026-11-07", "2026-11-09"), f.candidates.map { it.value }.toSet())
        assertNull(f.value)
    }

    @Test fun `validation failure is INVALID with a reason`() {
        val def = single(FieldDefinition("phone", type = FieldType.PHONE, required = true, validation = FieldValidation(regex = "^[6-9][0-9]{9}$")))
        val f = engine.extract(NormalizedText("my number is 1234567890"), def, "en").fields.getValue("phone")
        assertEquals(FieldStatus.INVALID, f.status)
        assertEquals("1234567890", f.value)
        assertTrue(f.reason!!.contains("pattern"))
    }

    @Test fun `numeric bounds are enforced`() {
        val def = single(FieldDefinition("qty", type = FieldType.INTEGER, required = true, validation = FieldValidation(min = 1.0, max = 100.0)))
        assertEquals(FieldStatus.INVALID, engine.extract(NormalizedText("add 500 bags"), def, "en").fields.getValue("qty").status)
        assertEquals(FieldStatus.EXTRACTED, engine.extract(NormalizedText("add 50 bags"), def, "en").fields.getValue("qty").status)
    }

    @Test fun `plain string fields are never guessed without anchors`() {
        val def = single(FieldDefinition("name", type = FieldType.STRING, required = true))
        val f = engine.extract(NormalizedText("Ramesh Kumar from Hosur"), def, "en").fields.getValue("name")
        assertEquals(FieldStatus.MISSING, f.status)
    }

    @Test fun `explicit administrator default is marked DEFAULTED`() {
        val def = single(FieldDefinition("mode", type = FieldType.ENUM, validation = FieldValidation(allowedValues = listOf(AllowedValue("cash"), AllowedValue("upi"))), defaultValue = "cash"))
        val f = engine.extract(NormalizedText("paid Ramesh 500"), def, "en").fields.getValue("mode")
        assertEquals(FieldStatus.DEFAULTED, f.status)
        assertEquals("cash", f.value)
    }

    @Test fun `low confidence time is flagged for confirmation`() {
        val def = single(FieldDefinition("t", type = FieldType.TIME))
        val f = engine.extract(NormalizedText("come at 10 baje"), def, "hi").fields.getValue("t")
        assertEquals(FieldStatus.LOW_CONFIDENCE, f.status)
        assertEquals("10:00", f.value)
    }

    @Test fun `user answers are parsed and validated per type`() {
        val date = FieldDefinition("d", type = FieldType.DATE, required = true, extraction = ExtractionHints(temporalDirection = TemporalDirection.FUTURE))
        assertEquals("2026-11-10", engine.userValue(date, "day after tomorrow").value)
        assertEquals(FieldStatus.USER_PROVIDED, engine.userValue(date, "12/11/2026").status)
        assertEquals(FieldStatus.INVALID, engine.userValue(date, "yesterday").status)
        val en = FieldDefinition("m", type = FieldType.ENUM, validation = FieldValidation(allowedValues = listOf(AllowedValue("upi", mapOf("hi" to listOf("यूपीआई"))), AllowedValue("cash"))))
        assertEquals("upi", engine.userValue(en, "यूपीआई").value)
        assertEquals("cash", engine.userValue(en, "Cash").value)
        assertEquals(FieldStatus.INVALID, engine.userValue(en, "cheque").status)
        val amt = FieldDefinition("a", type = FieldType.AMOUNT, validation = FieldValidation(min = 1.0))
        assertEquals("2500", engine.userValue(amt, "दो हज़ार पाँच सौ").value)
        assertEquals(FieldStatus.MISSING, engine.userValue(amt, "   ").status)
    }
}

class MisspelledStopWordTest {
    @Test fun `anchored name stops at a misspelled intent word`() {
        val def = co.immanuel.arkspeech.core.support.TestSupport.definition("book_appointment")
        val r = ExtractionEngine(co.immanuel.arkspeech.core.support.TestSupport.clock).extract(NormalizedText("நாளைக் காலைப்பத்து மணிக்கு டாக்டர் குமாருடன் அப்பாயன்டமென்ட் புக் செய்யவேண்டும்"), def, "ta")
        assertEquals("குமாருடன்", r.fields.getValue("doctor_name").value)
    }
}
