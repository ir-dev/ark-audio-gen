package co.immanuel.arkspeech.core.nlu

import co.immanuel.arkspeech.core.config.ExtractionDefinition
import co.immanuel.arkspeech.core.config.FallbackBehaviour
import kotlinx.serialization.Serializable
import kotlin.math.pow

@Serializable
enum class IntentStatus { MATCHED, LOW_CONFIDENCE, AMBIGUOUS, NO_MATCH, USER_SELECTED }

@Serializable
data class IntentCandidate(
    val definitionId: String,
    val intentName: String,
    val definitionVersion: Int,
    val score: Float,
    val matchedPhrases: List<String> = emptyList(),
    val matchedPatterns: Int = 0,
    val negativeHits: Int = 0,
)

@Serializable
data class IntentClassification(
    val status: IntentStatus,
    val best: IntentCandidate?,
    val alternatives: List<IntentCandidate> = emptyList(),
    val confidence: Float = best?.score ?: 0f,
    val engine: String = "lexical-v1",
) {
    companion object {
        fun none() = IntentClassification(IntentStatus.NO_MATCH, null)
    }
}

/** Classifies a transcript into one of the active definitions' intents. Replaceable by an ML model behind the same contract. */
interface IntentClassifier {
    val engineId: String
    fun classify(text: NormalizedText, language: String?, definitions: List<ExtractionDefinition>): IntentClassification
}

/**
 * Transparent lexical classifier: counts administrator-configured trigger phrases and patterns.
 * Score = 1 - (1 + hits)^-1.5, so one phrase gives 0.65, two give 0.81, three 0.88; negative phrases
 * subtract. Every match is reported so an administrator can see why an intent won or lost.
 */
class LexicalIntentClassifier(private val ambiguityMargin: Float = 0.10f) : IntentClassifier {
    override val engineId = "lexical-v1"
    private companion object { const val FUSED_MIN_CHARS = 5 }

    override fun classify(text: NormalizedText, language: String?, definitions: List<ExtractionDefinition>): IntentClassification {
        val scored = definitions.filter { it.enabled && it.appliesTo(language) }.map { score(text, language, it) }
            .filter { it.score > 0f }.sortedByDescending { it.score }
        if (scored.isEmpty()) return IntentClassification.none()
        val best = scored.first()
        val def = definitions.first { it.id == best.definitionId }
        val second = scored.getOrNull(1)
        val status = when {
            second != null && second.score >= def.confidenceThreshold && best.score - second.score < ambiguityMargin -> IntentStatus.AMBIGUOUS
            best.score >= def.confidenceThreshold -> IntentStatus.MATCHED
            def.fallback == FallbackBehaviour.ACCEPT_LOW_CONFIDENCE -> IntentStatus.LOW_CONFIDENCE
            else -> IntentStatus.LOW_CONFIDENCE
        }
        return IntentClassification(status, best, scored.drop(1).take(4), best.score, engineId)
    }

    private fun score(text: NormalizedText, language: String?, def: ExtractionDefinition): IntentCandidate {
        val matched = ArrayList<String>()
        var hits = 0f
        for (phrase in phrasesFor(def.triggers.phrases, language)) {
            val p = TextNormalizer.phrase(phrase)
            if (p.isEmpty()) continue
            if (text.containsPhrase(p)) {
                matched += phrase
                hits += if (p.contains(' ')) 1.5f else 1f
            } else if (p.length >= FUSED_MIN_CHARS && text.lower.contains(p)) {
                // CTC recognisers often glue words together ("பிரச்சனையஉள்ளது"); a long phrase found inside
                // a token still counts, at reduced weight.
                matched += phrase
                hits += 0.6f
            } else if (!p.contains(' ') && Fuzzy.findWord(text, p) != null) {
                // A misspelled trigger word ("अपौइंतमेंत"). Weight 0.5 alone scores 0.46: below any sane
                // threshold, so the user is asked rather than the intent assumed.
                matched += "$phrase~"
                hits += 0.5f
            }
        }
        var patterns = 0
        for (pat in def.triggers.patterns) {
            val re = runCatching { Regex(pat, RegexOption.IGNORE_CASE) }.getOrNull() ?: continue
            if (re.containsMatchIn(text.lower)) { patterns++; hits += 1.5f }
        }
        var negatives = 0
        for (phrase in phrasesFor(def.triggers.negativePhrases, language)) {
            val p = TextNormalizer.phrase(phrase)
            if (p.isNotEmpty() && text.containsPhrase(p)) { negatives++; hits -= 1.5f }
        }
        val score = if (hits <= 0f) 0f else (1.0 - (1.0 + hits).pow(-1.5)).toFloat()
        return IntentCandidate(def.id, def.intent.name, def.version, score, matched, patterns, negatives)
    }

    private fun phrasesFor(map: Map<String, List<String>>, language: String?): List<String> {
        val out = ArrayList<String>()
        map["*"]?.let { out += it }
        if (language != null) map[language]?.let { out += it }
        else map.filterKeys { it != "*" }.values.forEach { out += it }
        // Code-mixed speech: English triggers are always worth checking alongside the detected language.
        if (language != null && language != "en") map["en"]?.let { out += it }
        return out.distinct()
    }
}
