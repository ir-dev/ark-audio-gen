package co.immanuel.arkspeech.core.nlu

import kotlinx.serialization.Serializable

/** One possible value for a field, with where it came from. Several candidates = ambiguity to resolve. */
@Serializable
data class Candidate(
    /** Canonical value: ISO date, HH:mm, digits-only phone, plain number, enum value, ISO-8601 duration. */
    val value: String,
    /** The words as spoken, for the review screen. */
    val raw: String,
    val confidence: Float,
    /** pattern | anchor | scanner | synonym | default | user */
    val source: String,
    val start: Int = -1,
    val end: Int = -1,
    val extra: Map<String, String> = emptyMap(),
    val note: String? = null,
) {
    fun overlaps(s: Int, e: Int): Boolean = start >= 0 && end > start && s < end && e > start
}
