package co.immanuel.arkspeech

import android.content.Context
import androidx.test.platform.app.InstrumentationRegistry
import co.immanuel.arkspeech.app.AppGraph
import co.immanuel.arkspeech.core.audio.PcmAudio
import co.immanuel.arkspeech.core.audio.WavCodec
import co.immanuel.arkspeech.core.model.ModelKind
import org.junit.Assume.assumeTrue

/** Shared helpers for the instrumented tests. Models are never bundled; tests skip cleanly without them. */
object TestSupport {
    val targetContext: Context get() = InstrumentationRegistry.getInstrumentation().targetContext
    private val testContext: Context get() = InstrumentationRegistry.getInstrumentation().context

    val graph: AppGraph by lazy {
        AppGraph(targetContext).also { g ->
            // `-e asrModel whisper-tiny-int8` routes every language to one pack for A/B runs; without the
            // argument the bundled default routing is restored, because a previous run's override persists
            // in the app's repository on the device.
            val override = InstrumentationRegistry.getArguments().getString("asrModel")
            val routing = if (override != null) co.immanuel.arkspeech.core.asr.RoutingTable(override, emptyMap(), listOf("omnilingual-300m-ctc-int8"))
            else co.immanuel.arkspeech.core.config.ConfigJson.pretty.decodeFromString(
                co.immanuel.arkspeech.core.asr.RoutingTable.serializer(), targetContext.assets.open("routing.json").bufferedReader().readText())
            g.repository.saveRouting(routing)
            g.invalidateEngines()
        }
    }

    fun wav(name: String): PcmAudio = testContext.assets.open("audio/$name").use { WavCodec.decode(it).audio }

    fun wavBytes(name: String): ByteArray = testContext.assets.open("audio/$name").use { it.readBytes() }

    fun requireVad() = assumeTrue("Silero VAD pack not installed (scripts/fetch-models.sh --push silero-vad)", graph.store.installed(ModelKind.VAD).isNotEmpty())
    fun requireLid() = assumeTrue("Anuvani-LID pack not installed", graph.store.installed(ModelKind.LID).isNotEmpty())
    fun requireAsr() = assumeTrue("No ASR pack installed", graph.store.installed(ModelKind.ASR).isNotEmpty())

    /** Clip name → (expected language, expected intent definition id). */
    val speechClips: Map<String, Pair<String, String>> = linkedMapOf(
        "en_appointment.wav" to ("en" to "book_appointment"),
        "en_payment.wav" to ("en" to "record_payment"),
        "en_issue.wav" to ("en" to "report_issue"),
        "hi_appointment.wav" to ("hi" to "book_appointment"),
        "hi_payment.wav" to ("hi" to "record_payment"),
        "hi_issue.wav" to ("hi" to "report_issue"),
        "hi_codemix.wav" to ("hi" to "book_appointment"),
        "hi_appointment_noisy.wav" to ("hi" to "book_appointment"),
        "ta_appointment.wav" to ("ta" to "book_appointment"),
        "ta_issue.wav" to ("ta" to "report_issue"),
        "te_appointment.wav" to ("te" to "book_appointment"),
        "kn_stock.wav" to ("kn" to "stock_update"),
        "bn_payment.wav" to ("bn" to "record_payment"),
    )
}
