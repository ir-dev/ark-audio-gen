package co.immanuel.arkspeech.core.asr

import co.immanuel.arkspeech.core.audio.PcmAudio
import co.immanuel.arkspeech.core.lang.Languages
import co.immanuel.arkspeech.core.vad.SpeechSegment
import kotlinx.serialization.Serializable

@Serializable
data class TranscriptSegment(val startMs: Long, val endMs: Long, val text: String)

/**
 * A transcript in the language's own script. [confidence] is heuristic unless the engine exposes
 * token probabilities (see [TranscriptQualityEstimator]); [confidenceSource] says which.
 */
@Serializable
data class Transcript(
    val text: String,
    val language: String?,
    val segments: List<TranscriptSegment> = emptyList(),
    val confidence: Float = 0f,
    val confidenceSource: String = "heuristic",
    val modelId: String = "",
    val engine: String = "",
    val script: String? = Languages.dominantScript(text),
    val edited: Boolean = false,
) {
    val isBlank: Boolean get() = text.isBlank()
}

/** One ASR engine instance, already loaded with one model pack. Created by a [RecognizerFactory]. */
interface SpeechRecognizer : AutoCloseable {
    val modelId: String
    val engineId: String

    /** Languages the loaded model can produce; "*" means any. */
    val languages: Set<String>

    /**
     * Transcribe each speech segment of [audio] and join them in order. [language] is the language to
     * decode in when the model takes a hint (Whisper); engines without a hint ignore it.
     */
    suspend fun transcribe(audio: PcmAudio, segments: List<SpeechSegment>, language: String?): Transcript

    fun supports(language: String): Boolean = "*" in languages || language in languages
}

/** Creates recognisers for model packs. The app module implements this over sherpa-onnx. */
interface RecognizerFactory {
    /** Model packs that are installed and loadable right now. */
    fun availableModelIds(): Set<String>
    fun create(modelId: String): SpeechRecognizer
}
