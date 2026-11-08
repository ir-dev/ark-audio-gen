package co.immanuel.arkspeech.core.support

import co.immanuel.arkspeech.core.audio.PcmAudio
import co.immanuel.arkspeech.core.audio.WavCodec
import co.immanuel.arkspeech.core.config.ConfigJson
import co.immanuel.arkspeech.core.config.ExtractionDefinition
import co.immanuel.arkspeech.core.util.Clock
import java.io.File
import java.time.Instant
import java.time.ZoneId
import kotlin.math.PI
import kotlin.math.sin
import kotlin.random.Random

/** Paths and helpers shared by the core tests. Gradle runs tests from the module directory. */
object TestSupport {
    private val repoRoot: File = sequenceOf(File("."), File(".."), File("../..")).map { it.absoluteFile.normalize() }
        .first { File(it, "samples/definitions").isDirectory }

    val samplesDir = File(repoRoot, "samples/definitions")
    val audioDir = File(repoRoot, "testdata/audio")

    /** Sunday 8 November 2026, 10:00 IST: every relative-date expectation in the tests is pinned to this. */
    val clock = Clock.Fixed(Instant.parse("2026-11-08T04:30:00Z"), ZoneId.of("Asia/Kolkata"))

    fun definitions(): List<ExtractionDefinition> = samplesDir.listFiles { f -> f.extension == "json" }!!.sortedBy { it.name }
        .map { ConfigJson.pretty.decodeFromString(ExtractionDefinition.serializer(), it.readText()) }

    fun definition(id: String): ExtractionDefinition = definitions().first { it.id == id }

    fun wav(name: String): PcmAudio = WavCodec.decode(File(audioDir, name).readBytes()).audio

    fun bytes(name: String): ByteArray = File(audioDir, name).readBytes()

    /** Bursty tone so the validator sees speech-like dynamics (loud frames and quiet frames). */
    fun syntheticSpeech(durationMs: Int, sampleRate: Int = 16_000, seed: Int = 1): PcmAudio {
        val rnd = Random(seed)
        val n = (sampleRate.toLong() * durationMs / 1000).toInt()
        val out = FloatArray(n)
        var i = 0
        while (i < n) {
            val burst = sampleRate * (250 + rnd.nextInt(400)) / 1000
            val gap = sampleRate * (120 + rnd.nextInt(200)) / 1000
            val f = 120.0 + rnd.nextInt(200)
            for (k in 0 until burst) {
                if (i + k >= n) break
                val env = sin(PI * k / burst)
                out[i + k] = (0.35 * env * (sin(2 * PI * f * k / sampleRate) + 0.4 * sin(2 * PI * 3.1 * f * k / sampleRate)) + 0.004 * rnd.nextDouble(-1.0, 1.0)).toFloat()
            }
            i += burst + gap
        }
        return PcmAudio(out, sampleRate)
    }
}
