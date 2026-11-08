package co.immanuel.arkspeech.core.audio

import kotlinx.serialization.Serializable
import kotlin.math.abs
import kotlin.math.log10
import kotlin.math.sqrt

/** What can be wrong with a recording. Reported in the output contract so the client can explain it. */
@Serializable
enum class AudioIssue {
    EMPTY, TOO_SHORT, TOO_LONG, SILENT, TOO_QUIET, CLIPPING, NOISY, LOW_SPEECH_RATIO, DC_OFFSET, UNSUPPORTED_FORMAT,
}

@Serializable
enum class AudioVerdict { OK, WARN, REJECT }

/** Thresholds for [AudioValidator]. Defaults are tuned for 16 kHz phone microphone speech. */
@Serializable
data class AudioValidationPolicy(
    val minDurationMs: Long = 500,
    /** Push-to-talk stops at 3 minutes; anything longer can only arrive from an imported file. */
    val maxDurationMs: Long = 180_000,
    val silenceRmsDbfs: Double = -50.0,
    val quietRmsDbfs: Double = -36.0,
    val clippingRatioWarn: Double = 0.002,
    val clippingRatioReject: Double = 0.02,
    val snrDbWarn: Double = 12.0,
    val snrDbReject: Double = 4.0,
    val minSpeechRatio: Double = 0.04,
    val dcOffsetWarn: Double = 0.05,
    val frameMs: Int = 20,
)

@Serializable
data class AudioQualityReport(
    val durationMs: Long,
    val sampleRate: Int,
    val rmsDbfs: Double,
    val peakDbfs: Double,
    val clippingRatio: Double,
    val estimatedSnrDb: Double,
    val speechRatio: Double,
    val dcOffset: Double,
    val issues: List<AudioIssue>,
    val verdict: AudioVerdict,
) {
    val isUsable: Boolean get() = verdict != AudioVerdict.REJECT
}

/**
 * Deterministic signal-level checks that run before any model: duration bounds, silence, low level,
 * clipping, a frame-energy SNR estimate and a speech-activity ratio. Cheap (one pass over the samples)
 * and fully testable on the JVM. The VAD and models run only when the verdict is not REJECT.
 */
class AudioValidator(private val policy: AudioValidationPolicy = AudioValidationPolicy()) {

    fun validate(audio: PcmAudio): AudioQualityReport {
        val n = audio.sampleCount
        val issues = mutableListOf<AudioIssue>()
        if (n == 0) {
            return AudioQualityReport(0, audio.sampleRate, NEG_INF_DB, NEG_INF_DB, 0.0, 0.0, 0.0, 0.0, listOf(AudioIssue.EMPTY), AudioVerdict.REJECT)
        }
        var sumSq = 0.0
        var sum = 0.0
        var peak = 0f
        var clipped = 0
        for (s in audio.samples) {
            sumSq += (s * s).toDouble()
            sum += s
            val a = abs(s)
            if (a > peak) peak = a
            if (a >= CLIP_LEVEL) clipped++
        }
        val rms = sqrt(sumSq / n)
        val dc = sum / n
        val rmsDb = toDb(rms)
        val peakDb = toDb(peak.toDouble())
        val clipRatio = clipped.toDouble() / n

        // Frame energies for an SNR estimate: the loud tenth of frames stands for speech, the quiet tenth
        // for the noise floor. On pure noise both are similar and the estimate collapses toward 0 dB.
        val frameLen = (audio.sampleRate * policy.frameMs / 1000).coerceAtLeast(1)
        val frameCount = n / frameLen
        var snrDb = 0.0
        var speechRatio = 0.0
        if (frameCount >= 5) {
            val energies = DoubleArray(frameCount)
            for (f in 0 until frameCount) {
                var e = 0.0
                val base = f * frameLen
                for (i in 0 until frameLen) {
                    val v = audio.samples[base + i].toDouble()
                    e += v * v
                }
                energies[f] = e / frameLen
            }
            val sorted = energies.sortedArray()
            val noise = percentile(sorted, 0.10).coerceAtLeast(1e-10)
            val loud = percentile(sorted, 0.90).coerceAtLeast(1e-10)
            snrDb = 10 * log10(loud / noise)
            val speechGate = noise * 10.0 // 10 dB above the noise floor
            speechRatio = energies.count { it > speechGate }.toDouble() / frameCount
        }

        val durationMs = audio.durationMs
        if (durationMs < policy.minDurationMs) issues += AudioIssue.TOO_SHORT
        if (durationMs > policy.maxDurationMs) issues += AudioIssue.TOO_LONG
        when {
            rmsDb <= policy.silenceRmsDbfs -> issues += AudioIssue.SILENT
            rmsDb <= policy.quietRmsDbfs -> issues += AudioIssue.TOO_QUIET
        }
        if (clipRatio >= policy.clippingRatioWarn) issues += AudioIssue.CLIPPING
        if (AudioIssue.SILENT !in issues && frameCount >= 5) {
            if (snrDb < policy.snrDbWarn) issues += AudioIssue.NOISY
            if (speechRatio < policy.minSpeechRatio) issues += AudioIssue.LOW_SPEECH_RATIO
        }
        if (abs(dc) >= policy.dcOffsetWarn) issues += AudioIssue.DC_OFFSET

        val verdict = when {
            AudioIssue.TOO_SHORT in issues || AudioIssue.TOO_LONG in issues || AudioIssue.SILENT in issues -> AudioVerdict.REJECT
            clipRatio >= policy.clippingRatioReject -> AudioVerdict.REJECT
            AudioIssue.NOISY in issues && snrDb < policy.snrDbReject -> AudioVerdict.REJECT
            issues.isEmpty() -> AudioVerdict.OK
            else -> AudioVerdict.WARN
        }
        return AudioQualityReport(durationMs, audio.sampleRate, round2(rmsDb), round2(peakDb), clipRatio, round2(snrDb), round2(speechRatio), round2(dc), issues, verdict)
    }

    private fun percentile(sorted: DoubleArray, p: Double): Double {
        if (sorted.isEmpty()) return 0.0
        val idx = ((sorted.size - 1) * p).toInt().coerceIn(0, sorted.size - 1)
        return sorted[idx]
    }

    private fun toDb(v: Double): Double = if (v <= 1e-9) NEG_INF_DB else 20 * log10(v)
    private fun round2(v: Double) = Math.round(v * 100.0) / 100.0

    companion object {
        const val CLIP_LEVEL = 0.985f
        /** Stands in for -infinity dBFS so the report stays JSON-serialisable. */
        const val NEG_INF_DB = -120.0
    }
}
