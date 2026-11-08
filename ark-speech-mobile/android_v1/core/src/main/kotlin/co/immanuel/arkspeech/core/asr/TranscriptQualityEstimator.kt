package co.immanuel.arkspeech.core.asr

import co.immanuel.arkspeech.core.lang.Languages
import co.immanuel.arkspeech.core.vad.SpeechSegment
import kotlin.math.min

/**
 * Heuristic transcript confidence in [0, 1] for engines that give no token probabilities (sherpa-onnx's
 * offline recognisers). Combines: coverage (segments that produced text), speaking-rate plausibility
 * (characters per second of speech, script-aware), repetition (a hallmark of Whisper hallucination) and
 * script agreement with the expected language. Reported as `confidenceSource = "heuristic"`.
 */
object TranscriptQualityEstimator {

    fun estimate(transcript: Transcript, speech: List<SpeechSegment>, expectedLanguage: String?): Float {
        if (transcript.text.isBlank()) return 0f
        val speechMs = speech.sumOf { it.durationMs }.coerceAtLeast(1)
        val coverage = if (transcript.segments.isEmpty()) 1f else
            transcript.segments.count { it.text.isNotBlank() }.toFloat() / transcript.segments.size

        val letters = transcript.text.count { Character.isLetterOrDigit(it) }
        val cps = letters * 1000f / speechMs
        // Indic abugidas encode a syllable per 1-3 code points; Latin ~ 12-16 letters/s at speed.
        val rate = when {
            cps < 1.5f -> 0.3f
            cps < 4f -> 0.7f
            cps <= 22f -> 1f
            cps <= 30f -> 0.6f
            else -> 0.2f
        }

        val tokens = transcript.text.split(Regex("\\s+")).filter { it.isNotBlank() }
        val repetition = if (tokens.size < 6) 1f else {
            val trigrams = tokens.windowed(3).map { it.joinToString(" ") }
            val distinct = trigrams.toSet().size.toFloat() / trigrams.size
            when {
                distinct > 0.8f -> 1f
                distinct > 0.5f -> 0.7f
                else -> 0.3f
            }
        }

        val scriptAgreement = run {
            val lang = expectedLanguage?.let { Languages.get(it) } ?: return@run 1f
            val script = transcript.script ?: return@run 1f
            when {
                script == lang.script -> 1f
                // Code-mixed speech legitimately lands in Latin for an Indic language, or vice versa.
                script == "Latin" || lang.script == "Latin" -> 0.8f
                else -> 0.5f
            }
        }
        return min(1f, 0.25f * coverage + 0.35f * rate + 0.25f * repetition + 0.15f * scriptAgreement)
    }
}
