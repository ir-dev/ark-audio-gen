package co.immanuel.arkspeech.core.vad

import co.immanuel.arkspeech.core.audio.PcmAudio
import kotlinx.serialization.Serializable
import kotlin.math.log10
import kotlin.math.sqrt

/** One stretch of speech found by a VAD, in milliseconds from the start of the recording. */
@Serializable
data class SpeechSegment(val startMs: Long, val endMs: Long) {
    val durationMs: Long get() = endMs - startMs
}

interface VoiceActivityDetector {
    val engineId: String
    fun segments(audio: PcmAudio): List<SpeechSegment>
}

/**
 * Energy-based VAD: frames whose energy exceeds the noise floor by a margin are speech. It is the
 * JVM-testable fallback used when the Silero model is not installed; the app normally uses
 * sherpa-onnx's Silero VAD, which is far better on noisy audio.
 */
class EnergyVad(
    private val frameMs: Int = 30,
    private val marginDb: Double = 9.0,
    private val minSpeechMs: Long = 250,
    private val minSilenceMs: Long = 300,
    private val padMs: Long = 100,
    private val maxSegmentMs: Long = 30_000,
) : VoiceActivityDetector {
    override val engineId = "energy-vad"

    override fun segments(audio: PcmAudio): List<SpeechSegment> {
        val frameLen = audio.sampleRate * frameMs / 1000
        val frames = audio.sampleCount / frameLen
        if (frames < 3) return emptyList()
        val db = DoubleArray(frames)
        for (f in 0 until frames) {
            var e = 0.0
            val base = f * frameLen
            for (i in 0 until frameLen) {
                val v = audio.samples[base + i].toDouble()
                e += v * v
            }
            db[f] = 10 * log10((sqrt(e / frameLen) + 1e-9).let { it * it })
        }
        val sorted = db.sortedArray()
        val floor = sorted[(sorted.size * 0.15).toInt().coerceIn(0, sorted.size - 1)]
        val gate = maxOf(floor + marginDb, -55.0)
        val raw = ArrayList<SpeechSegment>()
        var inSpeech = false
        var start = 0L
        var lastSpeechEnd = 0L
        for (f in 0 until frames) {
            val t = f.toLong() * frameMs
            if (db[f] > gate) {
                if (!inSpeech) { inSpeech = true; start = t }
                lastSpeechEnd = t + frameMs
            } else if (inSpeech && t - lastSpeechEnd >= minSilenceMs) {
                raw += SpeechSegment(start, lastSpeechEnd)
                inSpeech = false
            }
        }
        if (inSpeech) raw += SpeechSegment(start, lastSpeechEnd)
        val total = audio.durationMs
        return raw.filter { it.durationMs >= minSpeechMs }
            .map { SpeechSegment((it.startMs - padMs).coerceAtLeast(0), (it.endMs + padMs).coerceAtMost(total)) }
            .flatMap { splitLong(it) }
    }

    private fun splitLong(seg: SpeechSegment): List<SpeechSegment> {
        if (seg.durationMs <= maxSegmentMs) return listOf(seg)
        val out = ArrayList<SpeechSegment>()
        var s = seg.startMs
        while (s < seg.endMs) {
            val e = minOf(s + maxSegmentMs, seg.endMs)
            out += SpeechSegment(s, e)
            s = e
        }
        return out
    }
}
