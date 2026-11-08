package co.immanuel.arkspeech.core.audio

/**
 * Mono PCM audio as normalised floats in [-1, 1].
 *
 * Every engine in this app (VAD, language identification, ASR) consumes 16 kHz mono float PCM, so the
 * capture and decode layers convert into this type once and everything downstream shares it.
 */
class PcmAudio(val samples: FloatArray, val sampleRate: Int) {

    init {
        require(sampleRate > 0) { "sampleRate must be positive" }
    }

    val sampleCount: Int get() = samples.size

    val durationMs: Long get() = samples.size * 1000L / sampleRate

    val isEmpty: Boolean get() = samples.isEmpty()

    /** A copy of the samples between the two instants (clamped to the clip). */
    fun slice(startMs: Long, endMs: Long): PcmAudio {
        val from = (startMs * sampleRate / 1000).toInt().coerceIn(0, samples.size)
        val to = (endMs * sampleRate / 1000).toInt().coerceIn(from, samples.size)
        return PcmAudio(samples.copyOfRange(from, to), sampleRate)
    }

    /** Linear resample to [target] Hz; returns this instance when no change is needed. */
    fun resampledTo(target: Int): PcmAudio =
        if (target == sampleRate) this else PcmAudio(Resampler.resample(samples, sampleRate, target), target)

    /** Little-endian signed 16-bit PCM (what WAV writers and `AudioRecord` use). */
    fun toPcm16Bytes(): ByteArray {
        val out = ByteArray(samples.size * 2)
        var i = 0
        for (s in samples) {
            val v = Math.round(s * 32768f).coerceIn(-32768, 32767)
            out[i++] = (v and 0xFF).toByte()
            out[i++] = ((v shr 8) and 0xFF).toByte()
        }
        return out
    }

    companion object {
        /** The sample rate every model in this app expects. */
        const val TARGET_SAMPLE_RATE = 16_000

        fun empty(sampleRate: Int = TARGET_SAMPLE_RATE) = PcmAudio(FloatArray(0), sampleRate)

        /** Interleaved little-endian signed 16-bit PCM, any channel count, down-mixed to mono. */
        fun fromPcm16(bytes: ByteArray, sampleRate: Int, channels: Int, length: Int = bytes.size): PcmAudio {
            require(channels > 0) { "channels must be positive" }
            val frames = length / (2 * channels)
            val out = FloatArray(frames)
            var idx = 0
            for (f in 0 until frames) {
                var acc = 0f
                for (c in 0 until channels) {
                    val lo = bytes[idx].toInt() and 0xFF
                    val hi = bytes[idx + 1].toInt()
                    acc += ((hi shl 8) or lo).toShort() / 32768f
                    idx += 2
                }
                out[f] = acc / channels
            }
            return PcmAudio(out, sampleRate)
        }

        fun fromShorts(shorts: ShortArray, sampleRate: Int, length: Int = shorts.size): PcmAudio {
            val out = FloatArray(length)
            for (i in 0 until length) out[i] = shorts[i] / 32768f
            return PcmAudio(out, sampleRate)
        }
    }
}

/** Linear-interpolation resampler: adequate for speech models that only need 16 kHz input. */
object Resampler {
    fun resample(input: FloatArray, from: Int, to: Int): FloatArray {
        if (from == to || input.isEmpty()) return input.copyOf()
        val ratio = from.toDouble() / to
        val outLen = (input.size / ratio).toInt().coerceAtLeast(1)
        val out = FloatArray(outLen)
        for (i in 0 until outLen) {
            val pos = i * ratio
            val i0 = pos.toInt().coerceAtMost(input.size - 1)
            val i1 = (i0 + 1).coerceAtMost(input.size - 1)
            val frac = (pos - i0).toFloat()
            out[i] = input[i0] * (1f - frac) + input[i1] * frac
        }
        return out
    }
}
