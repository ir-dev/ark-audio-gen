package co.immanuel.arkspeech.core.lang

import co.immanuel.arkspeech.core.audio.PcmAudio
import co.immanuel.arkspeech.core.vad.SpeechSegment
import kotlinx.serialization.Serializable
import kotlin.math.exp
import kotlin.math.ln

@Serializable
data class LanguageScore(val code: String, val probability: Float)

/** Language of one stretch of speech, as classified by the detector. */
@Serializable
data class SegmentLanguage(val startMs: Long, val endMs: Long, val code: String, val probability: Float)

@Serializable
enum class LanguageSource { AUTO, USER_OVERRIDE, DEFINITION_DEFAULT }

/**
 * The language decision for a recording. [confidence] is a calibrated probability when the engine
 * provides one (Anuvani-LID) and a vote share otherwise (Whisper); [engine] says which.
 */
@Serializable
data class LanguageDetection(
    val code: String?,
    val confidence: Float,
    val alternatives: List<LanguageScore> = emptyList(),
    val perSegment: List<SegmentLanguage> = emptyList(),
    /** Share of speech time per language, for code-mixed recordings. */
    val timeShare: Map<String, Float> = emptyMap(),
    val isCodeMixed: Boolean = false,
    val uncertain: Boolean = false,
    val supported: Boolean = true,
    val source: LanguageSource = LanguageSource.AUTO,
    val engine: String = "",
    val durationMs: Long = 0,
) {
    companion object {
        fun override(code: String, engine: String = "user"): LanguageDetection =
            LanguageDetection(code, 1f, source = LanguageSource.USER_OVERRIDE, engine = engine, supported = Languages.isSupported(code))

        fun none(engine: String): LanguageDetection =
            LanguageDetection(null, 0f, uncertain = true, supported = false, engine = engine)
    }
}

/** A spoken-language identification engine. Implementations live in the app module (ONNX Runtime, sherpa-onnx). */
interface LanguageDetector {
    val engineId: String

    /** Probabilities over the languages the engine knows, for one piece of speech, largest first. */
    suspend fun classify(piece: PcmAudio): List<LanguageScore>

    /** Longest piece the engine should be given at once, in milliseconds. */
    val maxPieceMs: Long get() = 15_000

    /** Shortest piece worth classifying. */
    val minPieceMs: Long get() = 400
}

/** Thresholds that turn engine probabilities into a decision the UI can act on. */
@Serializable
data class LanguagePolicy(
    /** Below this the UI must ask the user to confirm or pick the language. */
    val minConfidence: Float = 0.60f,
    /** If the runner-up is this close, treat the decision as uncertain too. */
    val minMargin: Float = 0.15f,
    /** Two languages each holding at least this share of speech time means code-mixed speech. */
    val codeMixMinShare: Float = 0.25f,
    /** Languages the app accepts; null means all 23. The detector's other labels are still reported as alternatives. */
    val allowedLanguages: Set<String>? = null,
)

/**
 * Runs a [LanguageDetector] over VAD speech segments and aggregates the result the way CIIL's
 * long-audio procedure does: pieces of at most [LanguageDetector.maxPieceMs], then a duration-weighted
 * mean of per-piece log-probabilities. The engine itself is interchangeable.
 */
class SegmentedLanguageIdentifier(
    private val detector: LanguageDetector,
    private val policy: LanguagePolicy = LanguagePolicy(),
) {
    suspend fun identify(audio: PcmAudio, segments: List<SpeechSegment>): LanguageDetection {
        val pieces = pieces(audio, segments)
        if (pieces.isEmpty()) return LanguageDetection.none(detector.engineId)

        val weightedLog = HashMap<String, Double>()
        val share = HashMap<String, Double>()
        val perSegment = ArrayList<SegmentLanguage>()
        var totalMs = 0.0
        for (piece in pieces) {
            val scores = detector.classify(piece.audio)
            if (scores.isEmpty()) continue
            val w = piece.durationMs.toDouble()
            totalMs += w
            for (s in scores) {
                val lp = ln(s.probability.toDouble().coerceAtLeast(1e-9))
                weightedLog[s.code] = (weightedLog[s.code] ?: 0.0) + w * lp
            }
            val top = scores.first()
            share[top.code] = (share[top.code] ?: 0.0) + w
            perSegment += SegmentLanguage(piece.startMs, piece.endMs, top.code, top.probability)
        }
        if (totalMs <= 0.0) return LanguageDetection.none(detector.engineId)

        // exp(mean log p) per language, renormalised so the values read as one distribution.
        val unnorm = weightedLog.mapValues { exp(it.value / totalMs) }
        val z = unnorm.values.sum().coerceAtLeast(1e-12)
        val ranked = unnorm.map { LanguageScore(it.key, (it.value / z).toFloat()) }.sortedByDescending { it.probability }
        val allowed = policy.allowedLanguages
        val eligible = if (allowed == null) ranked else ranked.filter { it.code in allowed }
        val top = eligible.firstOrNull() ?: return LanguageDetection(
            code = null, confidence = 0f, alternatives = ranked.take(5), perSegment = perSegment,
            uncertain = true, supported = false, engine = detector.engineId, durationMs = totalMs.toLong(),
        )
        val second = eligible.getOrNull(1)?.probability ?: 0f
        val timeShare = share.mapValues { (it.value / totalMs).toFloat() }
        val bigShares = timeShare.values.count { it >= policy.codeMixMinShare }
        val uncertain = top.probability < policy.minConfidence || (top.probability - second) < policy.minMargin
        return LanguageDetection(
            code = top.code,
            confidence = top.probability,
            alternatives = ranked.filter { it.code != top.code }.take(5),
            perSegment = perSegment,
            timeShare = timeShare,
            isCodeMixed = bigShares >= 2,
            uncertain = uncertain,
            supported = Languages.isSupported(top.code),
            engine = detector.engineId,
            durationMs = totalMs.toLong(),
        )
    }

    private class Piece(val audio: PcmAudio, val startMs: Long, val endMs: Long) {
        val durationMs get() = endMs - startMs
    }

    /** Cut segments into engine-sized pieces; drop pieces too short to carry evidence. */
    private fun pieces(audio: PcmAudio, segments: List<SpeechSegment>): List<Piece> {
        val out = ArrayList<Piece>()
        val max = detector.maxPieceMs
        for (seg in segments) {
            var start = seg.startMs
            while (start < seg.endMs) {
                var end = minOf(start + max, seg.endMs)
                // A short tail is folded into one full piece ending at the segment end (CIIL's min_tail rule).
                if (seg.endMs - end in 1 until 1000) end = seg.endMs
                if (end - start >= detector.minPieceMs) out += Piece(audio.slice(start, end), start, end)
                start = end
            }
        }
        return out
    }
}
