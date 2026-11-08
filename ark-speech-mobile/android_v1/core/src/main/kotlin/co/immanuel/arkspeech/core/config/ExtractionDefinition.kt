package co.immanuel.arkspeech.core.config

import co.immanuel.arkspeech.core.asr.RoutingTable
import co.immanuel.arkspeech.core.audio.AudioValidationPolicy
import co.immanuel.arkspeech.core.lang.LanguagePolicy
import kotlinx.serialization.Serializable

/** Current version of the configuration document format; bump when the schema changes incompatibly. */
const val CONFIG_SCHEMA_VERSION = "1.0"

@Serializable
enum class FieldType { STRING, INTEGER, NUMBER, BOOLEAN, DATE, TIME, PHONE, AMOUNT, ENUM, DURATION }

/** What to do when the intent score is below the definition's threshold. */
@Serializable
enum class FallbackBehaviour {
    /** Show the best guesses and let the user pick (default). */
    ASK_USER,
    /** Report no intent; the client decides. */
    REJECT,
    /** Proceed with the best-scoring intent, flagged low confidence. */
    ACCEPT_LOW_CONFIDENCE,
}

/** Resolves ambiguous relative dates such as Hindi "kal" (tomorrow or yesterday). */
@Serializable
enum class TemporalDirection { FUTURE, PAST, UNSPECIFIED }

@Serializable
data class IntentSpec(
    val name: String,
    val description: String = "",
)

/**
 * Evidence that an utterance expresses the intent. Phrases are matched after normalisation (case,
 * punctuation, Unicode NFC, native digits to ASCII); patterns are Java regular expressions applied to
 * the normalised text. Keys of the maps are language codes; "*" applies to every language.
 */
@Serializable
data class IntentTriggers(
    val phrases: Map<String, List<String>> = emptyMap(),
    val patterns: List<String> = emptyList(),
    /** Phrases that argue against this intent (e.g. "cancel" for book_appointment). */
    val negativePhrases: Map<String, List<String>> = emptyMap(),
)

/** One permitted value of an ENUM field and the ways people say it, per language ("*" = any). */
@Serializable
data class AllowedValue(
    val value: String,
    val synonyms: Map<String, List<String>> = emptyMap(),
    val description: String = "",
)

@Serializable
data class FieldValidation(
    val regex: String? = null,
    val allowedValues: List<AllowedValue> = emptyList(),
    val min: Double? = null,
    val max: Double? = null,
    val minLength: Int? = null,
    val maxLength: Int? = null,
    /** For DATE fields: reject dates outside this many days from today (null = no bound). */
    val maxDaysFromToday: Int? = null,
)

/**
 * How to find the field in the transcript. All three sources are optional and tried in order:
 * explicit patterns (regex with a `value` named group, or group 1), anchor phrases (the value follows
 * the phrase), then the type's own scanner (dates, numbers, phones, amounts, enum synonyms). STRING
 * fields without patterns or anchors are never guessed.
 */
@Serializable
data class ExtractionHints(
    val patterns: List<String> = emptyList(),
    val anchors: Map<String, List<String>> = emptyMap(),
    /** Phrases the value comes *before* (postpositions: Hindi "के पास", Tamil "அருகில்"), per language. */
    val anchorsBefore: Map<String, List<String>> = emptyMap(),
    /** Words that end an anchored STRING value (per language, "*" for all). */
    val stopWords: Map<String, List<String>> = emptyMap(),
    /** Maximum tokens taken after an anchor for STRING values. */
    val maxTokens: Int = 3,
    val temporalDirection: TemporalDirection = TemporalDirection.UNSPECIFIED,
    val instructions: String = "",
    val examples: List<String> = emptyList(),
)

@Serializable
data class FieldDefinition(
    val name: String,
    val description: String = "",
    val type: FieldType = FieldType.STRING,
    val required: Boolean = false,
    val validation: FieldValidation = FieldValidation(),
    val extraction: ExtractionHints = ExtractionHints(),
    /** Minimum candidate confidence to accept without asking; null uses the definition's threshold. */
    val confidenceThreshold: Float? = null,
    /** Only used when the administrator explicitly allows a default; otherwise missing stays missing. */
    val defaultValue: String? = null,
)

/** A worked example that documents the definition and doubles as a regression test for it. */
@Serializable
data class ExtractionExample(
    val language: String,
    val text: String,
    val expected: Map<String, String> = emptyMap(),
    val note: String = "",
)

/**
 * One administrator-managed extraction definition: an intent, the languages it applies to, how to
 * recognise it and which fields to pull out. Immutable once saved; a change creates the next version.
 */
@Serializable
data class ExtractionDefinition(
    val id: String,
    val intent: IntentSpec,
    val version: Int = 1,
    val enabled: Boolean = true,
    /** Language codes this definition applies to; empty means all languages. */
    val languages: List<String> = emptyList(),
    val triggers: IntentTriggers = IntentTriggers(),
    val fields: List<FieldDefinition> = emptyList(),
    /** Minimum intent score to accept the match. */
    val confidenceThreshold: Float = 0.5f,
    val fallback: FallbackBehaviour = FallbackBehaviour.ASK_USER,
    val examples: List<ExtractionExample> = emptyList(),
    val instructions: String = "",
    val updatedAt: String = "",
    val updatedBy: String = "",
    val changeNote: String = "",
) {
    fun appliesTo(language: String?): Boolean =
        languages.isEmpty() || language == null || language in languages

    val requiredFields: List<FieldDefinition> get() = fields.filter { it.required }
}

/** Everything an administrator can import or export in one file. */
@Serializable
data class ConfigBundle(
    val schemaVersion: String = CONFIG_SCHEMA_VERSION,
    val definitions: List<ExtractionDefinition> = emptyList(),
    val routing: RoutingTable? = null,
    val languagePolicy: LanguagePolicy? = null,
    val audioPolicy: AudioValidationPolicy? = null,
    val exportedAt: String = "",
    val note: String = "",
)
