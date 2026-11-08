package co.immanuel.arkspeech.core.pipeline

import co.immanuel.arkspeech.core.asr.RecognizerFactory
import co.immanuel.arkspeech.core.asr.RoutingTable
import co.immanuel.arkspeech.core.asr.SpeechRecognizer
import co.immanuel.arkspeech.core.asr.Transcript
import co.immanuel.arkspeech.core.asr.TranscriptSegment
import co.immanuel.arkspeech.core.audio.PcmAudio
import co.immanuel.arkspeech.core.config.FileDefinitionRepository
import co.immanuel.arkspeech.core.lang.LanguageDetector
import co.immanuel.arkspeech.core.lang.LanguageScore
import co.immanuel.arkspeech.core.nlu.FieldStatus
import co.immanuel.arkspeech.core.nlu.IntentStatus
import co.immanuel.arkspeech.core.support.TestSupport
import co.immanuel.arkspeech.core.vad.SpeechSegment
import co.immanuel.arkspeech.core.vad.VoiceActivityDetector
import kotlinx.coroutines.test.runTest
import java.nio.file.Files
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

class SpeechPipelineTest {

    private class FakeVad : VoiceActivityDetector {
        override val engineId = "fake-vad"
        override fun segments(audio: PcmAudio) = if (audio.durationMs < 300) emptyList() else listOf(SpeechSegment(0, audio.durationMs))
    }

    private class FakeDetector(val scores: List<LanguageScore>) : LanguageDetector {
        override val engineId = "fake-lid"
        override suspend fun classify(piece: PcmAudio) = scores
    }

    private class FakeRecognizer(override val modelId: String, val text: String, val lang: String?) : SpeechRecognizer {
        override val engineId = "fake-asr"
        override val languages = setOf("*")
        var closed = false
        override suspend fun transcribe(audio: PcmAudio, segments: List<SpeechSegment>, language: String?) =
            Transcript(text, lang ?: language, segments.map { TranscriptSegment(it.startMs, it.endMs, text) }, 0f, "heuristic", modelId, engineId)
        override fun close() { closed = true }
    }

    private class FakeFactory(val installed: Set<String>, val text: String, val lang: String? = null) : RecognizerFactory {
        val created = ArrayList<FakeRecognizer>()
        override fun availableModelIds() = installed
        override fun create(modelId: String) = FakeRecognizer(modelId, text, lang).also { created += it }
    }

    private fun repo() = FileDefinitionRepository(Files.createTempDirectory("ark").toFile(), TestSupport.clock).also { r -> TestSupport.definitions().forEach { r.save(it) } }

    private fun deps(factory: RecognizerFactory, detector: LanguageDetector? = FakeDetector(listOf(LanguageScore("hi", 0.93f), LanguageScore("ur", 0.04f), LanguageScore("mr", 0.03f)))) =
        PipelineDependencies(FakeVad(), detector, factory, RoutingTable("omni", fallbackModelIds = listOf("whisper")), repo(), clock = TestSupport.clock)

    private val hindiAppointment = "मुझे कल सुबह दस बजे डॉक्टर शर्मा के साथ अपॉइंटमेंट बुक करनी है। मेरा फ़ोन नंबर नौ आठ सात छह पाँच चार तीन दो एक शून्य है।"

    @Test fun `happy path produces OK with every required field`() = runTest {
        val factory = FakeFactory(setOf("omni"), hindiAppointment)
        val result = SpeechPipeline(deps(factory)).runAll(TestSupport.syntheticSpeech(5000))
        assertEquals(PipelineStatus.OK, result.status, result.toJson())
        assertEquals("hi", result.language!!.code)
        assertFalse(result.language!!.uncertain)
        assertEquals("book_appointment", result.intent!!.best!!.definitionId)
        assertEquals("2026-11-09", result.fields.getValue("appointment_date").value)
        assertEquals("10:00", result.fields.getValue("appointment_time").value)
        assertEquals("9876543210", result.fields.getValue("phone").value)
        assertEquals("शर्मा", result.fields.getValue("doctor_name").value)
        assertEquals(FieldStatus.MISSING, result.fields.getValue("department").status)
        assertTrue(result.missingRequired.isEmpty())
        assertTrue(result.clarifications.isEmpty())
        assertEquals("omni", result.processing.asrModelId)
        assertTrue(result.processing.stages.map { it.stage }.containsAll(listOf("validate", "vad", "lid", "route", "asr", "intent", "extract")))
        assertTrue(factory.created.single().closed, "recogniser must be released")
        assertEquals(OUTPUT_SCHEMA_VERSION, result.schemaVersion)
    }

    @Test fun `result JSON round trips`() = runTest {
        val result = SpeechPipeline(deps(FakeFactory(setOf("omni"), hindiAppointment))).runAll(TestSupport.syntheticSpeech(3000))
        val back = PipelineResult.fromJson(result.toJson())
        assertEquals(result.fields.keys, back.fields.keys)
        assertEquals(result.status, back.status)
        assertEquals(result.requestId, back.requestId)
    }

    @Test fun `rejected audio stops before any model runs`() = runTest {
        val factory = FakeFactory(setOf("omni"), hindiAppointment)
        val result = SpeechPipeline(deps(factory)).runAll(PcmAudio(FloatArray(16000 * 4), 16000))
        assertEquals(PipelineStatus.AUDIO_REJECTED, result.status)
        assertTrue(factory.created.isEmpty())
        assertEquals(listOf("validate"), result.processing.stages.map { it.stage })
    }

    @Test fun `uncertain language yields a clarification and still transcribes`() = runTest {
        val detector = FakeDetector(listOf(LanguageScore("hi", 0.45f), LanguageScore("ur", 0.40f), LanguageScore("pa", 0.15f)))
        val result = SpeechPipeline(deps(FakeFactory(setOf("omni"), hindiAppointment), detector)).runAll(TestSupport.syntheticSpeech(3000))
        assertEquals(PipelineStatus.NEEDS_CLARIFICATION, result.status)
        assertTrue(result.language!!.uncertain)
        val c = result.clarifications.single { it.kind == ClarificationKind.LANGUAGE }
        assertEquals(listOf("hi", "ur", "pa"), c.options)
        assertNotNull(result.transcript)
    }

    @Test fun `language override skips detection and is reported`() = runTest {
        val result = SpeechPipeline(deps(FakeFactory(setOf("omni"), hindiAppointment))).runAll(TestSupport.syntheticSpeech(3000), languageOverride = "hi")
        assertEquals(co.immanuel.arkspeech.core.lang.LanguageSource.USER_OVERRIDE, result.language!!.source)
        assertTrue(result.processing.stages.none { it.stage == "lid" })
        assertEquals(PipelineStatus.OK, result.status)
    }

    @Test fun `routing falls back to an installed model`() = runTest {
        val factory = FakeFactory(setOf("whisper"), hindiAppointment)
        val result = SpeechPipeline(deps(factory)).runAll(TestSupport.syntheticSpeech(3000))
        assertEquals("whisper", result.processing.asrModelId)
        assertEquals(PipelineStatus.OK, result.status)
    }

    @Test fun `no installed model is a clean MODEL_UNAVAILABLE`() = runTest {
        val result = SpeechPipeline(deps(FakeFactory(emptySet(), hindiAppointment))).runAll(TestSupport.syntheticSpeech(3000))
        assertEquals(PipelineStatus.MODEL_UNAVAILABLE, result.status)
        assertTrue(result.warnings.any { it.contains("No installed ASR model") })
    }

    @Test fun `step by step flow with transcript edit and clarification`() = runTest {
        val pipeline = SpeechPipeline(deps(FakeFactory(setOf("omni"), "রমেশকে পাঁচই নভেম্বর ইউপিআই-এর মাধ্যমে দুই হাজার পাঁচশো টাকা পাঠানো হয়েছে।"), FakeDetector(listOf(LanguageScore("bn", 0.97f)))))
        val run = pipeline.begin(TestSupport.syntheticSpeech(3000))
        pipeline.detectLanguage(run)
        pipeline.transcribe(run)
        pipeline.extract(run)
        assertEquals(PipelineStatus.NEEDS_CLARIFICATION, run.status)
        var result = pipeline.result(run)
        assertEquals(listOf("payee"), result.missingRequired)
        assertEquals(ClarificationKind.FIELD_MISSING, result.clarifications.single().kind)

        pipeline.provideFieldValue(run, "payee", "Ramesh")
        result = pipeline.result(run)
        assertEquals(PipelineStatus.OK, result.status, result.toJson())
        assertEquals(FieldStatus.USER_PROVIDED, result.fields.getValue("payee").status)

        // The user fixes the transcript; extraction re-runs on the corrected text.
        pipeline.editTranscript(run, "Record a payment of three thousand rupees to Suresh by cash")
        pipeline.extract(run)
        result = pipeline.result(run)
        assertTrue(result.transcript!!.edited)
        assertEquals("3000", result.fields.getValue("amount").value)
        assertEquals("Suresh", result.fields.getValue("payee").value)
        assertEquals("cash", result.fields.getValue("payment_mode").value)
    }

    @Test fun `no matching intent is NO_INTENT`() = runTest {
        val result = SpeechPipeline(deps(FakeFactory(setOf("omni"), "आज मौसम बहुत अच्छा है"))).runAll(TestSupport.syntheticSpeech(3000))
        assertEquals(PipelineStatus.NO_INTENT, result.status)
        assertEquals(IntentStatus.NO_MATCH, result.intent!!.status)
    }

    @Test fun `user can force an intent`() = runTest {
        val pipeline = SpeechPipeline(deps(FakeFactory(setOf("omni"), "Ramesh 2500 rupees")))
        val run = pipeline.begin(TestSupport.syntheticSpeech(3000))
        pipeline.overrideLanguage(run, "en")
        pipeline.transcribe(run)
        pipeline.extract(run, forcedDefinitionId = "record_payment")
        assertEquals(IntentStatus.USER_SELECTED, run.intent!!.status)
        assertEquals("2500", run.extraction!!.fields.getValue("amount").value)
    }

    @Test fun `retention flags reflect consent and nothing is kept by default`() = runTest {
        val result = SpeechPipeline(deps(FakeFactory(setOf("omni"), hindiAppointment))).runAll(TestSupport.syntheticSpeech(3000))
        assertFalse(result.retention.audioRetained)
        assertFalse(result.retention.transcriptRetained)
        assertTrue(result.retention.processedOnDevice)
    }
}
