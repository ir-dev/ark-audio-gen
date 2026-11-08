package co.immanuel.arkspeech

import android.content.Context
import android.os.BatteryManager
import android.os.Debug
import android.os.Process
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import co.immanuel.arkspeech.core.config.ConfigJson
import co.immanuel.arkspeech.core.model.ModelKind
import kotlinx.coroutines.runBlocking
import kotlinx.serialization.Serializable
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong
import kotlin.concurrent.thread

/**
 * On-device benchmark: runs the pipeline on three clips (English, Hindi, Tamil) several times and
 * records per-stage latency, cold vs warm model load, peak PSS, CPU time, battery counter delta and the
 * installed model footprint. Output: external files dir `benchmark/benchmark.json` (+ logcat table).
 * Run via scripts/benchmark.sh, which also captures dumpsys meminfo/batterystats.
 */
@RunWith(AndroidJUnit4::class)
class BenchmarkTest {
    @Before fun models() { TestSupport.requireVad(); TestSupport.requireAsr() }

    @Serializable data class StageMs(val stage: String, val ms: Long)
    @Serializable data class RunMeasure(val clip: String, val iteration: Int, val cold: Boolean, val totalMs: Long, val stages: List<StageMs>, val audioMs: Long, val realTimeFactor: Double, val peakPssMb: Double, val cpuMs: Long, val status: String, val language: String?, val intent: String?)
    @Serializable data class Report(
        val device: Map<String, String>, val engines: Map<String, String?>, val installedPacks: Map<String, Long>, val threads: Int,
        val runs: List<RunMeasure>, val batteryChargeCounterDeltaUah: Long?, val wallClockMs: Long, val baselinePssMb: Double,
    )

    @Test fun benchmark(): Unit = runBlocking {
        val ctx = TestSupport.targetContext
        val graph = TestSupport.graph
        val clips = listOf("en_appointment.wav", "hi_appointment.wav", "ta_appointment.wav")
        val bm = ctx.getSystemService(Context.BATTERY_SERVICE) as BatteryManager
        fun charge(): Long? = bm.getLongProperty(BatteryManager.BATTERY_PROPERTY_CHARGE_COUNTER).takeIf { it != Long.MIN_VALUE && it != 0L }
        val chargeBefore = charge()
        val wallStart = System.currentTimeMillis()
        graph.releaseEngines()
        System.gc(); Thread.sleep(500)
        val baselinePss = Debug.getPss() / 1024.0
        val runs = ArrayList<RunMeasure>()
        for ((ci, clip) in clips.withIndex()) {
            val audio = TestSupport.wav(clip)
            repeat(3) { i ->
                val cold = ci == 0 && i == 0
                val peak = AtomicLong(0); val stop = AtomicBoolean(false)
                val sampler = thread(start = true) { while (!stop.get()) { peak.set(maxOf(peak.get(), Debug.getPss())); Thread.sleep(50) } }
                val cpu0 = Process.getElapsedCpuTime()
                val r = graph.pipeline().runAll(audio)
                val cpu = Process.getElapsedCpuTime() - cpu0
                stop.set(true); sampler.join()
                runs += RunMeasure(clip, i, cold, r.processing.totalMs, r.processing.stages.map { StageMs(it.stage, it.ms) }, audio.durationMs,
                    r.processing.totalMs.toDouble() / audio.durationMs, peak.get() / 1024.0, cpu, r.status.name, r.language?.code, r.intent?.best?.definitionId)
                Log.i("Benchmark", "$clip #$i ${if (cold) "cold" else "warm"} total=${r.processing.totalMs}ms rtf=${"%.2f".format(r.processing.totalMs.toDouble() / audio.durationMs)} peakPss=${peak.get() / 1024}MB cpu=${cpu}ms stages=${r.processing.stages.joinToString { "${it.stage}:${it.ms}" }}")
            }
        }
        val report = Report(
            device = graph.deviceInfo,
            engines = mapOf("vad" to graph.engineSummary().vad, "lid" to graph.engineSummary().lid, "asr" to graph.engineSummary().asrDefault),
            installedPacks = graph.catalog.packs.filter { graph.store.isInstalled(it) }.associate { it.id to graph.store.installedBytes(it) },
            threads = graph.settings.inferenceThreads, runs = runs, batteryChargeCounterDeltaUah = chargeBefore?.let { b -> charge()?.let { it - b } },
            wallClockMs = System.currentTimeMillis() - wallStart, baselinePssMb = baselinePss,
        )
        val out = File(ctx.getExternalFilesDir(null), "benchmark").apply { mkdirs() }
        File(out, "benchmark.json").writeText(ConfigJson.pretty.encodeToString(Report.serializer(), report))
        Log.i("Benchmark", "report written to ${File(out, "benchmark.json")}")
        // Also a markdown table for the docs.
        val md = StringBuilder("| clip | run | total ms | RTF | asr ms | lid ms | peak PSS MB | CPU ms |\n|---|---|---|---|---|---|---|---|\n")
        runs.forEach { r -> md.append("| ${r.clip} | ${r.iteration}${if (r.cold) " (cold)" else ""} | ${r.totalMs} | ${"%.2f".format(r.realTimeFactor)} | ${r.stages.firstOrNull { it.stage == "asr" }?.ms ?: "-"} | ${r.stages.firstOrNull { it.stage == "lid" }?.ms ?: "-"} | ${"%.0f".format(r.peakPssMb)} | ${r.cpuMs} |\n") }
        File(out, "benchmark.md").writeText(md.toString())
        Log.i("Benchmark", "\n$md")
    }
}
