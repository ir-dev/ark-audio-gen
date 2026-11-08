package co.immanuel.arkspeech.engine

import co.immanuel.arkspeech.core.audio.PcmAudio
import co.immanuel.arkspeech.core.lang.LanguageDetector
import co.immanuel.arkspeech.core.lang.LanguageScore
import co.immanuel.arkspeech.core.lang.Languages
import com.k2fsa.sherpa.onnx.SpokenLanguageIdentification
import com.k2fsa.sherpa.onnx.SpokenLanguageIdentificationConfig
import com.k2fsa.sherpa.onnx.SpokenLanguageIdentificationWhisperConfig
import java.io.File

/**
 * Fallback language identification with a Whisper model through sherpa-onnx. The API returns only
 * the winning language, so each piece votes with probability 1.0 and the core aggregator turns the
 * votes into a share; this is weaker than Anuvani-LID (no calibration, Whisper knows 15 of the 23
 * languages) and is used only when the Anuvani pack is not installed.
 */
class WhisperLanguageDetector(encoder: File, decoder: File, threads: Int = 2) : LanguageDetector, AutoCloseable {
    override val engineId = "sherpa-whisper-lid"
    override val maxPieceMs: Long = 30_000

    private val slid = SpokenLanguageIdentification(
        config = SpokenLanguageIdentificationConfig(
            whisper = SpokenLanguageIdentificationWhisperConfig(encoder = encoder.absolutePath, decoder = decoder.absolutePath, tailPaddings = -1),
            numThreads = threads,
            provider = "cpu",
        ),
    )

    override suspend fun classify(piece: PcmAudio): List<LanguageScore> = synchronized(this) { classifyLocked(piece) }

    private fun classifyLocked(piece: PcmAudio): List<LanguageScore> {
        val stream = slid.createStream()
        try {
            stream.acceptWaveform(piece.samples, piece.sampleRate)
            val raw = slid.compute(stream).trim().lowercase()
            if (raw.isEmpty()) return emptyList()
            val code = Languages.fromWhisperCode(raw)?.code ?: raw
            return listOf(LanguageScore(code, 1f))
        } finally {
            stream.release()
        }
    }

    override fun close() {
        runCatching { slid.release() }
    }
}
