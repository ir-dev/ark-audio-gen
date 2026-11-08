package co.immanuel.arkspeech.core.nlu

import co.immanuel.arkspeech.core.support.TestSupport
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

class IntentClassifierTest {
    private val defs = TestSupport.definitions()
    private val clf = LexicalIntentClassifier()

    private fun classify(text: String, lang: String?) = clf.classify(NormalizedText(text), lang, defs)

    @Test fun `sample examples classify to their own definition`() {
        for (def in defs) for (ex in def.examples) {
            val r = classify(ex.text, ex.language)
            assertEquals(def.id, r.best?.definitionId, "${def.id}/${ex.language}: $r")
            assertTrue(r.status == IntentStatus.MATCHED || r.status == IntentStatus.USER_SELECTED, "${def.id}/${ex.language}: ${r.status} ${r.best}")
        }
    }

    @Test fun `unrelated text has no intent`() {
        assertEquals(IntentStatus.NO_MATCH, classify("the weather is lovely in Mysuru today", "en").status)
    }

    @Test fun `negative phrase lowers the score`() {
        val positive = classify("book an appointment with the dentist", "en")
        val negative = classify("cancel my appointment with the dentist", "en")
        assertTrue(negative.confidence < positive.confidence, "pos=${positive.confidence} neg=${negative.confidence}")
    }

    @Test fun `two intents with equal evidence are ambiguous`() {
        val r = classify("appointment payment", "en")
        assertEquals(IntentStatus.AMBIGUOUS, r.status, "$r")
        assertTrue(r.alternatives.isNotEmpty())
    }

    @Test fun `language scoping excludes definitions`() {
        // record_payment is not configured for Marathi; report_issue applies to every language.
        val r = classify("भुगतान समस्या", "mr")
        assertEquals("report_issue", r.best?.definitionId, "$r")
    }

    @Test fun `english triggers still fire for code-mixed speech`() {
        val r = classify("मुझे Doctor Sharma के साथ kal appointment book करनी है", "hi")
        assertEquals("book_appointment", r.best?.definitionId)
        assertNotNull(r.best?.matchedPhrases?.firstOrNull { it.contains("appointment") })
    }

    @Test fun `disabled definitions are ignored`() {
        val disabled = defs.map { if (it.id == "book_appointment") it.copy(enabled = false) else it }
        val r = clf.classify(NormalizedText("book an appointment"), "en", disabled)
        assertTrue(r.best?.definitionId != "book_appointment", "$r")
    }
}

class FusedWordTest {
    @Test fun `fused CTC output still triggers long phrases`() {
        val defs = co.immanuel.arkspeech.core.support.TestSupport.definitions()
        val r = LexicalIntentClassifier().classify(NormalizedText("காந்தி சாலைப் பெருந்து நிருத்்தம் அருகில் தண்ணீர் வசதி பிரச்சனையஉள்ளது இது அவசரம"), "ta", defs)
        assertEquals("report_issue", r.best?.definitionId, "$r")
        assertTrue(r.best!!.score > 0.4f)
    }

    @Test fun `routing sample parses with its note`() {
        val json = java.io.File(co.immanuel.arkspeech.core.support.TestSupport.samplesDir.parentFile, "routing.json").readText()
        val t = co.immanuel.arkspeech.core.config.ConfigJson.pretty.decodeFromString(co.immanuel.arkspeech.core.asr.RoutingTable.serializer(), json)
        assertEquals("omnilingual-300m-ctc-int8", t.defaultModelId)
        assertEquals(listOf("omnilingual-300m-ctc-int8", "whisper-small-int8", "whisper-tiny-int8"), t.candidates("hi"))
    }
}

class FuzzyMatchTest {
    private val defs = co.immanuel.arkspeech.core.support.TestSupport.definitions()

    @Test fun `misspelled trigger words surface the intent for confirmation not acceptance`() {
        val r = LexicalIntentClassifier().classify(NormalizedText("कुझी कल सुबह दस बजे दौक्तर शर्मा के साथ अपौइंतमेंत बुत करनी है"), "hi", defs)
        assertEquals("book_appointment", r.best?.definitionId, "$r")
        assertEquals(IntentStatus.LOW_CONFIDENCE, r.status)
        assertTrue(r.best!!.matchedPhrases.any { it.endsWith("~") })
        val ta = LexicalIntentClassifier().classify(NormalizedText("காந்தி சாலைப பெருந்து அருகில் தண்ணீர் வசதி பிறச்சனை உள்ளது இது அவசரம்"), "ta", defs)
        assertEquals("report_issue", ta.best?.definitionId, "$ta")
    }

    @Test fun `fuzzy matching does not invent intents`() {
        assertEquals(IntentStatus.NO_MATCH, LexicalIntentClassifier().classify(NormalizedText("the weather is lovely in Mysuru today and the garden is green"), "en", defs).status)
        assertTrue(Fuzzy.similar("अपौइंतमेंत", "अपॉइंटमेंट"))
        assertTrue(Fuzzy.similar("பிறச்சனை", "பிரச்சனை"))
        assertTrue(!Fuzzy.similar("payment", "patient"))
        assertTrue(!Fuzzy.similar("book", "boot"))
    }

    @Test fun `misspelled enum synonym is a low confidence candidate`() {
        val def = co.immanuel.arkspeech.core.support.TestSupport.definition("record_payment")
        val r = ExtractionEngine(co.immanuel.arkspeech.core.support.TestSupport.clock).extract(NormalizedText("रमेश को दू हजार पांच सौ रूप यूपआई से भेजे हैं पाच नवंबर को"), def, "hi")
        val mode = r.fields.getValue("payment_mode")
        assertEquals("upi", mode.value, "$mode")
        assertEquals(FieldStatus.LOW_CONFIDENCE, mode.status)
    }
}
