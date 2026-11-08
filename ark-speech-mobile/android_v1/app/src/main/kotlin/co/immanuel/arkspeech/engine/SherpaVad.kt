package co.immanuel.arkspeech.engine

import co.immanuel.arkspeech.core.audio.PcmAudio
import co.immanuel.arkspeech.core.vad.SpeechSegment
import co.immanuel.arkspeech.core.vad.VoiceActivityDetector
import com.k2fsa.sherpa.onnx.SileroVadModelConfig
import com.k2fsa.sherpa.onnx.Vad
import com.k2fsa.sherpa.onnx.VadModelConfig
import java.io.File

/**
 * Silero VAD through sherpa-onnx. Segments are capped at [maxSpeechSeconds] so every downstream
 * model (Whisper's 30 s window, Omnilingual's 40 s limit) gets a piece it can take whole.
 */
class SherpaVad(
    private val modelFile: File,
    private val threads: Int = 1,
    private val threshold: Float = 0.5f,
    private val minSilenceSeconds: Float = 0.3f,
    private val minSpeechSeconds: Float = 0.25f,
    private val maxSpeechSeconds: Float = 25f,
    /** Silero reports speech a little after it starts; pad both ends so the first syllable survives. */
    private val padMs: Long = 250,
) : VoiceActivityDetector, AutoCloseable {

    override val engineId = "sherpa-silero-vad"

    private val vad: Vad by lazy {
        Vad(
            config = VadModelConfig(
                sileroVadModelConfig = SileroVadModelConfig(
                    model = modelFile.absolutePath,
                    threshold = threshold,
                    minSilenceDuration = minSilenceSeconds,
                    minSpeechDuration = minSpeechSeconds,
                    windowSize = WINDOW,
                    maxSpeechDuration = maxSpeechSeconds,
                ),
                sampleRate = PcmAudio.TARGET_SAMPLE_RATE,
                numThreads = threads,
                provider = "cpu",
            ),
        )
    }

    @Synchronized
    override fun segments(audio: PcmAudio): List<SpeechSegment> {
        require(audio.sampleRate == PcmAudio.TARGET_SAMPLE_RATE) { "VAD expects 16 kHz audio" }
        val v = vad
        v.reset()
        val out = ArrayList<SpeechSegment>()
        val samples = audio.samples
        var offset = 0
        val window = FloatArray(WINDOW)
        while (offset + WINDOW <= samples.size) {
            System.arraycopy(samples, offset, window, 0, WINDOW)
            v.acceptWaveform(window)
            drain(v, out)
            offset += WINDOW
        }
        if (offset < samples.size) {
            // Zero-pad the tail window so the last few hundred ms are not lost.
            window.fill(0f)
            System.arraycopy(samples, offset, window, 0, samples.size - offset)
            v.acceptWaveform(window)
        }
        v.flush()
        drain(v, out)
        val total = audio.durationMs
        val padded = out.map { SpeechSegment((it.startMs - padMs).coerceIn(0, total), (it.endMs + padMs).coerceIn(0, total)) }.filter { it.durationMs > 0 }
        // Padding can make neighbours overlap; merge those so no audio is transcribed twice.
        val merged = ArrayList<SpeechSegment>()
        for (s in padded) {
            val last = merged.lastOrNull()
            if (last != null && s.startMs <= last.endMs) merged[merged.size - 1] = SpeechSegment(last.startMs, maxOf(last.endMs, s.endMs)) else merged += s
        }
        return merged
    }

    private fun drain(v: Vad, out: MutableList<SpeechSegment>) {
        while (!v.empty()) {
            val seg = v.front()
            val startMs = seg.start * 1000L / PcmAudio.TARGET_SAMPLE_RATE
            val endMs = startMs + seg.samples.size * 1000L / PcmAudio.TARGET_SAMPLE_RATE
            out += SpeechSegment(startMs, endMs)
            v.pop()
        }
    }

    override fun close() {
        runCatching { vad.release() }
    }

    companion object {
        const val WINDOW = 512
    }
}
