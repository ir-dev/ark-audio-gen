package co.immanuel.arkspeech

import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import co.immanuel.arkspeech.core.audio.PcmAudio
import co.immanuel.arkspeech.core.pipeline.PipelineStatus
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

/**
 * The whole chain on the bundled multilingual clips with the real installed models. Writes one JSON
 * result per clip to the app's external files dir (e2e/) so the outputs can be inspected and attached
 * to the benchmark report.
 */
@RunWith(AndroidJUnit4::class)
class PipelineE2ETest {
    @Before fun models() { TestSupport.requireVad(); TestSupport.requireAsr() }

    private val outDir: File by lazy { File(TestSupport.targetContext.getExternalFilesDir(null), "e2e").apply { mkdirs() } }

    @Test fun edgeCasesAreRejectedBeforeAnyModelRuns(): Unit = runBlocking {
        val p = TestSupport.graph.pipeline()
        for ((name, issue) in listOf("edge_silence_4s.wav" to "SILENT", "edge_noise_4s.wav" to "NOISY", "edge_too_short.wav" to "TOO_SHORT")) {
            val r = p.runAll(TestSupport.wav(name))
            assertEquals(name, PipelineStatus.AUDIO_REJECTED, r.status)
            assertTrue("$name issues=${r.audio?.issues}", r.audio!!.issues.any { it.name == issue })
            assertEquals(name, listOf("validate"), r.processing.stages.map { it.stage })
        }
        val empty = p.runAll(PcmAudio.empty())
        assertEquals(PipelineStatus.AUDIO_REJECTED, empty.status)
    }

    @Test fun multilingualClipsProduceTranscriptsAndIntents(): Unit = runBlocking {
        val p = TestSupport.graph.pipeline()
        val hasLid = TestSupport.graph.store.installed(co.immanuel.arkspeech.core.model.ModelKind.LID).isNotEmpty()
        val report = StringBuilder()
        var transcripts = 0; var languages = 0; var intents = 0
        for ((name, expected) in TestSupport.speechClips) {
            val (lang, intent) = expected
            // Without a language identifier the expected language is supplied, as a client that knows it would.
            val r = if (hasLid) p.runAll(TestSupport.wav(name)) else p.runAll(TestSupport.wav(name), languageOverride = lang)
            File(outDir, name.removeSuffix(".wav") + ".json").writeText(r.toJson())
            val text = r.transcript?.text ?: ""
            if (text.isNotBlank()) transcripts++
            val langOk = r.language?.code == lang || r.language?.alternatives?.take(2)?.any { it.code == lang } == true
            if (langOk) languages++
            val intentOk = r.intent?.best?.definitionId == intent
            if (intentOk) intents++
            report.append("$name: status=${r.status} lang=${r.language?.code}(${"%.2f".format(r.language?.confidence ?: 0f)}) intent=${r.intent?.best?.definitionId}(${"%.2f".format(r.intent?.confidence ?: 0f)}) fields=${r.fields.values.joinToString { "${it.name}=${it.value ?: "-"}/${it.status.name.take(4)}" }} asr=${r.processing.stages.firstOrNull { it.stage == "asr" }?.ms}ms text='${text.take(80)}'\n")
        }
        Log.i("PipelineE2ETest", "\n$report")
        File(outDir, "summary.txt").writeText(report.toString())
        val n = TestSupport.speechClips.size
        assertTrue("transcripts $transcripts/$n\n$report", transcripts == n)
        assertTrue("language top-2 hits $languages/$n\n$report", languages >= n * 0.6)
        assertTrue("intent hits $intents/$n\n$report", intents >= n * 0.4)
    }
}
