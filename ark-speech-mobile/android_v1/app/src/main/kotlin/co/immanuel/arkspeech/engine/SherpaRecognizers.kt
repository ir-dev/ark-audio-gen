package co.immanuel.arkspeech.engine

import co.immanuel.arkspeech.core.asr.RecognizerFactory
import co.immanuel.arkspeech.core.asr.SpeechRecognizer
import co.immanuel.arkspeech.core.asr.Transcript
import co.immanuel.arkspeech.core.asr.TranscriptSegment
import co.immanuel.arkspeech.core.audio.PcmAudio
import co.immanuel.arkspeech.core.lang.Languages
import co.immanuel.arkspeech.core.model.ModelKind
import co.immanuel.arkspeech.core.model.ModelPack
import co.immanuel.arkspeech.core.vad.SpeechSegment
import co.immanuel.arkspeech.model.ModelStore
import com.k2fsa.sherpa.onnx.FeatureConfig
import com.k2fsa.sherpa.onnx.OfflineDolphinModelConfig
import com.k2fsa.sherpa.onnx.OfflineModelConfig
import com.k2fsa.sherpa.onnx.OfflineNemoEncDecCtcModelConfig
import com.k2fsa.sherpa.onnx.OfflineOmnilingualAsrCtcModelConfig
import com.k2fsa.sherpa.onnx.OfflineRecognizer
import com.k2fsa.sherpa.onnx.OfflineRecognizerConfig
import com.k2fsa.sherpa.onnx.OfflineWhisperModelConfig
import java.io.File

/**
 * One sherpa-onnx offline recogniser wrapping any model family sherpa-onnx can load. The pack's
 * `engine` chooses the config shape; the rest of the app never sees the difference.
 */
class SherpaOfflineRecognizer(
    private val pack: ModelPack,
    private val dir: File,
    private val threads: Int,
) : SpeechRecognizer {

    override val modelId: String = pack.id
    override val engineId: String = pack.engine
    override val languages: Set<String> = pack.languages.toSet()

    private val isWhisper = pack.engine == ENGINE_WHISPER
    private var currentWhisperLanguage: String = ""

    private fun file(key: String): String = File(dir, pack.params[key] ?: error("pack ${pack.id} lacks param '$key'")).absolutePath

    private fun modelConfig(whisperLanguage: String): OfflineModelConfig = when (pack.engine) {
        ENGINE_WHISPER -> OfflineModelConfig(
            whisper = OfflineWhisperModelConfig(encoder = file("encoder"), decoder = file("decoder"), language = whisperLanguage, task = "transcribe", tailPaddings = 1000),
            tokens = file("tokens"), modelType = "whisper", numThreads = threads, provider = "cpu",
        )
        ENGINE_OMNILINGUAL -> OfflineModelConfig(
            omnilingual = OfflineOmnilingualAsrCtcModelConfig(model = file("model")),
            tokens = file("tokens"), numThreads = threads, provider = "cpu",
        )
        ENGINE_DOLPHIN -> OfflineModelConfig(
            dolphin = OfflineDolphinModelConfig(model = file("model")),
            tokens = file("tokens"), numThreads = threads, provider = "cpu",
        )
        ENGINE_NEMO_CTC -> OfflineModelConfig(
            nemo = OfflineNemoEncDecCtcModelConfig(model = file("model")),
            tokens = file("tokens"), modelType = "nemo_ctc", numThreads = threads, provider = "cpu",
        )
        else -> error("Unsupported engine '${pack.engine}' for pack ${pack.id}")
    }

    private fun config(whisperLanguage: String) = OfflineRecognizerConfig(
        featConfig = FeatureConfig(sampleRate = PcmAudio.TARGET_SAMPLE_RATE, featureDim = 80),
        modelConfig = modelConfig(whisperLanguage),
        decodingMethod = "greedy_search",
    )

    private val recognizer: OfflineRecognizer by lazy { OfflineRecognizer(config = config(currentWhisperLanguage)) }

    override suspend fun transcribe(audio: PcmAudio, segments: List<SpeechSegment>, language: String?): Transcript = synchronized(this) { transcribeLocked(audio, segments, language) }

    private fun transcribeLocked(audio: PcmAudio, segments: List<SpeechSegment>, language: String?): Transcript {
        require(audio.sampleRate == PcmAudio.TARGET_SAMPLE_RATE)
        if (isWhisper) {
            // Whisper decodes in a fixed language; "" lets it detect per piece (used when unknown).
            val wl = language?.let { Languages.get(it)?.whisperCode } ?: ""
            if (wl != currentWhisperLanguage) {
                currentWhisperLanguage = wl
                recognizer.setConfig(config(wl))
            }
        }
        val raw = if (segments.isEmpty()) listOf(SpeechSegment(0, audio.durationMs)) else segments
        // Whisper is given pieces of at most WHISPER_MAX_PIECE_MS, split at quiet points, to stay well inside its
        // 30 s window. (Measured: Whisper small int8 still stopped after ~3 s of Hindi speech even with 5 s pieces,
        // so the early stop is not the window budget; see docs/BENCHMARK.md run 5.)
        val pieces = if (isWhisper) raw.flatMap { splitAtQuietPoints(audio, it, WHISPER_MAX_PIECE_MS) } else raw
        val out = ArrayList<TranscriptSegment>()
        var detected: String? = null
        for (seg in pieces) {
            val piece = audio.slice(seg.startMs, seg.endMs)
            if (piece.sampleCount < PcmAudio.TARGET_SAMPLE_RATE / 10) continue
            val stream = recognizer.createStream()
            try {
                stream.acceptWaveform(withSilencePadding(piece.samples), piece.sampleRate)
                recognizer.decode(stream)
                val r = recognizer.getResult(stream)
                val text = r.text.trim()
                if (text.isNotEmpty()) out += TranscriptSegment(seg.startMs, seg.endMs, text)
                if (detected == null && r.lang.isNotBlank()) detected = r.lang.trim('<', '>', '|').let { Languages.fromWhisperCode(it)?.code ?: it }
            } finally {
                stream.release()
            }
        }
        val text = out.joinToString(" ") { it.text }
        return Transcript(text = text, language = language ?: detected, segments = out, confidence = 0f, confidenceSource = "heuristic", modelId = modelId, engine = engineId)
    }

    /**
     * Splits a long segment into pieces of at most [maxMs], cutting at the lowest-energy 20 ms frame within the
     * middle half of the piece so words are not severed at an arbitrary offset.
     */
    private fun splitAtQuietPoints(audio: PcmAudio, seg: SpeechSegment, maxMs: Long): List<SpeechSegment> {
        if (seg.durationMs <= maxMs) return listOf(seg)
        val sr = audio.sampleRate
        val frame = sr * 20 / 1000
        val startIdx = (seg.startMs * sr / 1000).toInt(); val endIdx = (seg.endMs * sr / 1000).toInt().coerceAtMost(audio.sampleCount)
        val lo = startIdx + (endIdx - startIdx) / 4; val hi = endIdx - (endIdx - startIdx) / 4
        var best = (startIdx + endIdx) / 2; var bestE = Double.MAX_VALUE
        var i = lo
        while (i + frame <= hi) {
            var e = 0.0
            for (k in i until i + frame) { val v = audio.samples[k].toDouble(); e += v * v }
            if (e < bestE) { bestE = e; best = i + frame / 2 }
            i += frame
        }
        val cutMs = best * 1000L / sr
        return splitAtQuietPoints(audio, SpeechSegment(seg.startMs, cutMs), maxMs) + splitAtQuietPoints(audio, SpeechSegment(cutMs, seg.endMs), maxMs)
    }

    /** CTC models trained on utterances with leading silence drop the first syllable of a piece that starts mid-word. */
    private fun withSilencePadding(samples: FloatArray): FloatArray {
        val pad = PcmAudio.TARGET_SAMPLE_RATE * SILENCE_PAD_MS / 1000
        val out = FloatArray(samples.size + 2 * pad)
        System.arraycopy(samples, 0, out, pad, samples.size)
        return out
    }

    override fun close() {
        runCatching { recognizer.release() }
    }

    companion object {
        const val SILENCE_PAD_MS = 300
        /** Longest piece handed to Whisper (its window is 30 s; shorter pieces bound latency per piece). */
        const val WHISPER_MAX_PIECE_MS = 12_000L
        const val ENGINE_WHISPER = "sherpa-whisper"
        const val ENGINE_OMNILINGUAL = "sherpa-omnilingual-ctc"
        const val ENGINE_DOLPHIN = "sherpa-dolphin-ctc"
        const val ENGINE_NEMO_CTC = "sherpa-nemo-ctc"
        val SUPPORTED = setOf(ENGINE_WHISPER, ENGINE_OMNILINGUAL, ENGINE_DOLPHIN, ENGINE_NEMO_CTC)
    }
}

/** Creates recognisers for installed ASR packs. Loading is lazy and per call; the caller closes them. */
class SherpaRecognizerFactory(private val store: ModelStore, private val threads: Int) : RecognizerFactory {
    override fun availableModelIds(): Set<String> =
        store.catalog.byKind(ModelKind.ASR).filter { it.engine in SherpaOfflineRecognizer.SUPPORTED && store.isInstalled(it) }.map { it.id }.toSet()

    override fun create(modelId: String): SpeechRecognizer {
        val pack = store.catalog.get(modelId) ?: throw IllegalArgumentException("Unknown model pack $modelId")
        check(store.isInstalled(pack)) { "Model pack $modelId is not installed" }
        return SherpaOfflineRecognizer(pack, store.dir(pack), threads)
    }
}
