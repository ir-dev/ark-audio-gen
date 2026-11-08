package co.immanuel.arkspeech.core.nlu

import co.immanuel.arkspeech.core.pipeline.PipelineResult
import co.immanuel.arkspeech.core.support.TestSupport
import kotlin.test.Test
import kotlin.test.assertEquals

/** Replays a transcript captured on the emulator through the JVM classifier to compare behaviour. */
class DeviceTranscriptTest {
    @Test fun `device tamil transcript matches report_issue`() {
        val json = javaClass.getResource("/device_ta_issue.json")!!.readText()
        val r = PipelineResult.fromJson(json)
        val text = NormalizedText(r.transcript!!.text)
        println("tokens=" + text.tokens.map { it.text })
        val c = LexicalIntentClassifier().classify(text, r.language!!.code, TestSupport.definitions())
        println("classification=$c")
        assertEquals("report_issue", c.best?.definitionId)
    }
}
