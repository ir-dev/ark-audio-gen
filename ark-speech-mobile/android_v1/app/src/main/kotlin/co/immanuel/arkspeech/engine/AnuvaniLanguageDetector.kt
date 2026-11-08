package co.immanuel.arkspeech.engine

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import co.immanuel.arkspeech.core.audio.PcmAudio
import co.immanuel.arkspeech.core.lang.LanguageDetector
import co.immanuel.arkspeech.core.lang.LanguageScore
import co.immanuel.arkspeech.core.lang.Languages
import java.io.File
import java.nio.FloatBuffer
import java.nio.LongBuffer
import kotlin.math.exp

/**
 * CIIL Anuvani-LID (Apache-2.0) through ONNX Runtime, following the published model contract:
 * inputs `audio` [N, S] float32 in [-1, 1] at 16 kHz and `audio_length` [N] int64; output `logits`
 * [N, 23]; softmax gives calibrated probabilities. Each piece is followed by 512 zeros, as the kit
 * does, so the mel front end's last frames see silence. Pieces are capped at 15 s to keep peak RAM
 * near 470 MB on budget phones (CIIL's own measurement); the aggregator in core handles longer speech.
 */
class AnuvaniLanguageDetector(
    private val modelFile: File,
    private val threads: Int = 2,
    override val maxPieceMs: Long = 15_000,
    /**
     * Turns off ONNX Runtime's KleidiAI micro-kernels (SME/SME2/i8mm). Needed on the Android emulator on
     * Apple-silicon hosts, where the guest advertises SME but the hypervisor cannot execute it (SIGILL),
     * and available as a Settings kill switch for any device that misreports its CPU features.
     */
    private val disableKleidiAi: Boolean = false,
) : LanguageDetector, AutoCloseable {

    override val engineId = "ort-anuvani-lid"
    override val minPieceMs: Long = 400

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession by lazy {
        val opts = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(threads)
            setInterOpNumThreads(1)
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            setMemoryPatternOptimization(true)
            if (disableKleidiAi) addConfigEntry("mlas.disable_kleidiai", "1")
        }
        env.createSession(modelFile.absolutePath, opts)
    }

    /** Labels in output order, read from the model's metadata when present, else the published list. */
    private val labels: List<String> by lazy {
        val meta = runCatching { session.metadata.customMetadata["labels"] }.getOrNull()
        meta?.split(',')?.map { it.trim() }?.takeIf { it.size == 23 } ?: DEFAULT_LABELS
    }

    override suspend fun classify(piece: PcmAudio): List<LanguageScore> = synchronized(this) { classifyLocked(piece) }

    private fun classifyLocked(piece: PcmAudio): List<LanguageScore> {
        require(piece.sampleRate == PcmAudio.TARGET_SAMPLE_RATE) { "Anuvani-LID expects 16 kHz audio" }
        val n = piece.sampleCount
        if (n < MIN_SAMPLES) return emptyList()
        val padded = FloatArray(n + TAIL_PAD)
        System.arraycopy(piece.samples, 0, padded, 0, n)
        val audio = OnnxTensor.createTensor(env, FloatBuffer.wrap(padded), longArrayOf(1, padded.size.toLong()))
        val length = OnnxTensor.createTensor(env, LongBuffer.wrap(longArrayOf(n.toLong())), longArrayOf(1))
        try {
            session.run(mapOf("audio" to audio, "audio_length" to length)).use { out ->
                @Suppress("UNCHECKED_CAST")
                val logits = (out[0].value as Array<FloatArray>)[0]
                val probs = softmax(logits)
                return probs.indices.sortedByDescending { probs[it] }.map { i ->
                    val code = Languages.fromAnuvaniLabel(labels[i])?.code ?: labels[i]
                    LanguageScore(code, probs[i])
                }
            }
        } finally {
            audio.close(); length.close()
        }
    }

    private fun softmax(x: FloatArray): FloatArray {
        val max = x.maxOrNull() ?: 0f
        val e = FloatArray(x.size) { exp((x[it] - max).toDouble()).toFloat() }
        val sum = e.sum()
        return FloatArray(x.size) { e[it] / sum }
    }

    override fun close() {
        runCatching { session.close() }
    }

    companion object {
        const val TAIL_PAD = 512
        /** Below 0.1 s the graph is unreliable (NaN under 160 samples), per the model contract. */
        const val MIN_SAMPLES = 1600
        val DEFAULT_LABELS = listOf(
            "assamese", "bengali", "bodo", "dogri", "english", "gujarati", "hindi", "kannada", "kashmiri", "konkani", "maithili",
            "malayalam", "manipuri", "marathi", "nepali", "odia", "punjabi", "sanskrit", "santali", "sindhi", "tamil", "telugu", "urdu",
        )
    }
}
