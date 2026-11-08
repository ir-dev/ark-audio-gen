package co.immanuel.arkspeech.core.pipeline

import co.immanuel.arkspeech.core.asr.Transcript
import co.immanuel.arkspeech.core.audio.AudioQualityReport
import co.immanuel.arkspeech.core.config.ConfigJson
import co.immanuel.arkspeech.core.lang.LanguageDetection
import co.immanuel.arkspeech.core.nlu.Candidate
import co.immanuel.arkspeech.core.nlu.FieldResult
import co.immanuel.arkspeech.core.nlu.IntentClassification
import co.immanuel.arkspeech.core.vad.SpeechSegment
import kotlinx.serialization.Serializable

/** Version of the structured output contract (docs/OUTPUT_CONTRACT.md). */
const val OUTPUT_SCHEMA_VERSION = "1.0"

@Serializable
enum class PipelineStatus {
    /** Intent matched, every required field extracted and valid. */
    OK,
    /** Usable, but the user must confirm or supply something (language, intent or fields). */
    NEEDS_CLARIFICATION,
    /** Audio failed validation (silence, noise, too short, unsupported format). */
    AUDIO_REJECTED,
    /** Audio was fine but the VAD found no speech. */
    NO_SPEECH,
    /** Language detected but not supported by any installed model. */
    UNSUPPORTED_LANGUAGE,
    /** No installed ASR model for the language. */
    MODEL_UNAVAILABLE,
    /** Transcript produced but no definition matched. */
    NO_INTENT,
    /** An engine threw; see warnings. */
    FAILED,
    /** Work in progress (the UI's intermediate state). */
    IN_PROGRESS,
}

@Serializable
enum class ClarificationKind { LANGUAGE, INTENT, FIELD_MISSING, FIELD_AMBIGUOUS, FIELD_LOW_CONFIDENCE, FIELD_INVALID }

/** A question the client should put to the user before the result can be acted on. */
@Serializable
data class Clarification(
    val kind: ClarificationKind,
    /** Field name for FIELD_* kinds. */
    val field: String? = null,
    val prompt: String,
    /** Choices to offer (language codes, intent definition ids, enum values). */
    val options: List<String> = emptyList(),
    val candidates: List<Candidate> = emptyList(),
)

@Serializable
data class StageTiming(val stage: String, val ms: Long, val status: String, val detail: String? = null)

@Serializable
data class ProcessingInfo(
    val startedAt: String,
    val totalMs: Long,
    val stages: List<StageTiming>,
    val vadEngine: String? = null,
    val lidEngine: String? = null,
    val asrEngine: String? = null,
    val asrModelId: String? = null,
    val nluEngine: String? = null,
    val device: Map<String, String> = emptyMap(),
)

/** What happened to the audio and transcript: nothing is kept unless the user consented. */
@Serializable
data class RetentionInfo(
    val consentToRetainAudio: Boolean = false,
    val consentToRetainTranscript: Boolean = false,
    val audioRetained: Boolean = false,
    val transcriptRetained: Boolean = false,
    val processedOnDevice: Boolean = true,
)

/**
 * The structured output of one push-to-talk interaction. Everything a client needs: what was heard,
 * in which language, what the user meant, the extracted values with their confidence and status, what
 * still needs clarifying and how long each stage took.
 */
@Serializable
data class PipelineResult(
    val schemaVersion: String = OUTPUT_SCHEMA_VERSION,
    val requestId: String,
    val status: PipelineStatus,
    val audio: AudioQualityReport? = null,
    val speechSegments: List<SpeechSegment> = emptyList(),
    val language: LanguageDetection? = null,
    val transcript: Transcript? = null,
    val intent: IntentClassification? = null,
    val fields: Map<String, FieldResult> = emptyMap(),
    val missingRequired: List<String> = emptyList(),
    val clarifications: List<Clarification> = emptyList(),
    val warnings: List<String> = emptyList(),
    val processing: ProcessingInfo,
    val retention: RetentionInfo = RetentionInfo(),
) {
    fun toJson(pretty: Boolean = true): String =
        (if (pretty) ConfigJson.pretty else ConfigJson.compact).encodeToString(serializer(), this)

    companion object {
        fun fromJson(json: String): PipelineResult = ConfigJson.pretty.decodeFromString(serializer(), json)
    }
}
