package co.immanuel.arkspeech.core.audio

import java.io.ByteArrayOutputStream
import java.io.InputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

class UnsupportedAudioFormatException(message: String) : Exception(message)

/** Result of decoding a WAV file: the audio converted to 16 kHz mono plus what the file originally was. */
class DecodedWav(
    val audio: PcmAudio,
    val originalSampleRate: Int,
    val originalChannels: Int,
    val bitsPerSample: Int,
)

/**
 * Minimal RIFF/WAVE reader and writer. Reads PCM 8/16/24/32-bit integer and 32-bit float, mono or
 * multi-channel, any sample rate, and converts to 16 kHz mono. Anything else (MP3, M4A, OGG, Opus,
 * WAV with compressed codecs) is reported as [UnsupportedAudioFormatException] so the caller can show
 * a clear "unsupported format" message instead of feeding garbage to the models.
 */
object WavCodec {

    fun decode(bytes: ByteArray, targetSampleRate: Int = PcmAudio.TARGET_SAMPLE_RATE): DecodedWav {
        if (bytes.size < 12) throw UnsupportedAudioFormatException("File too small to be a WAV file")
        val buf = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        if (ascii(bytes, 0) != "RIFF" || ascii(bytes, 8) != "WAVE") {
            throw UnsupportedAudioFormatException(sniff(bytes))
        }
        var pos = 12
        var format = -1
        var channels = 0
        var sampleRate = 0
        var bits = 0
        var dataStart = -1
        var dataLen = 0
        while (pos + 8 <= bytes.size) {
            val id = ascii(bytes, pos)
            val size = buf.getInt(pos + 4)
            val body = pos + 8
            when (id) {
                "fmt " -> {
                    if (size < 16) throw UnsupportedAudioFormatException("Malformed WAV fmt chunk")
                    format = buf.getShort(body).toInt() and 0xFFFF
                    channels = buf.getShort(body + 2).toInt() and 0xFFFF
                    sampleRate = buf.getInt(body + 4)
                    bits = buf.getShort(body + 14).toInt() and 0xFFFF
                    if (format == 0xFFFE && size >= 40) {
                        // WAVE_FORMAT_EXTENSIBLE: the sub-format GUID's first two bytes hold the real tag.
                        format = buf.getShort(body + 24).toInt() and 0xFFFF
                    }
                }
                "data" -> {
                    dataStart = body
                    dataLen = minOf(size.coerceAtLeast(0), bytes.size - body)
                }
            }
            pos = body + size + (size and 1)
            if (dataStart >= 0 && format >= 0) break
        }
        if (format < 0) throw UnsupportedAudioFormatException("WAV file has no fmt chunk")
        if (dataStart < 0) throw UnsupportedAudioFormatException("WAV file has no data chunk")
        if (channels <= 0 || sampleRate <= 0) throw UnsupportedAudioFormatException("WAV header has invalid channel count or sample rate")
        val frameBytes = channels * bits / 8
        if (frameBytes <= 0) throw UnsupportedAudioFormatException("WAV header has invalid bit depth")
        val frames = dataLen / frameBytes
        val mono = FloatArray(frames)
        when {
            format == 1 && bits == 16 -> for (f in 0 until frames) {
                var acc = 0f
                for (c in 0 until channels) acc += buf.getShort(dataStart + (f * channels + c) * 2) / 32768f
                mono[f] = acc / channels
            }
            format == 1 && bits == 8 -> for (f in 0 until frames) {
                var acc = 0f
                for (c in 0 until channels) acc += ((bytes[dataStart + f * channels + c].toInt() and 0xFF) - 128) / 128f
                mono[f] = acc / channels
            }
            format == 1 && bits == 24 -> for (f in 0 until frames) {
                var acc = 0f
                for (c in 0 until channels) {
                    val o = dataStart + (f * channels + c) * 3
                    val v = (bytes[o].toInt() and 0xFF) or ((bytes[o + 1].toInt() and 0xFF) shl 8) or (bytes[o + 2].toInt() shl 16)
                    acc += v / 8388608f
                }
                mono[f] = acc / channels
            }
            format == 1 && bits == 32 -> for (f in 0 until frames) {
                var acc = 0f
                for (c in 0 until channels) acc += buf.getInt(dataStart + (f * channels + c) * 4) / 2147483648f
                mono[f] = acc / channels
            }
            format == 3 && bits == 32 -> for (f in 0 until frames) {
                var acc = 0f
                for (c in 0 until channels) acc += buf.getFloat(dataStart + (f * channels + c) * 4)
                mono[f] = acc / channels
            }
            else -> throw UnsupportedAudioFormatException(
                "Unsupported WAV encoding (format tag $format, $bits-bit). Only PCM 8/16/24/32-bit and 32-bit float are supported.",
            )
        }
        val audio = PcmAudio(mono, sampleRate).resampledTo(targetSampleRate)
        return DecodedWav(audio, sampleRate, channels, bits)
    }

    fun decode(stream: InputStream, targetSampleRate: Int = PcmAudio.TARGET_SAMPLE_RATE): DecodedWav =
        decode(stream.use { it.readBytes() }, targetSampleRate)

    /** 16-bit PCM mono WAV bytes. */
    fun encode(audio: PcmAudio): ByteArray {
        val pcm = audio.toPcm16Bytes()
        val out = ByteArrayOutputStream(44 + pcm.size)
        val header = ByteBuffer.allocate(44).order(ByteOrder.LITTLE_ENDIAN)
        header.put("RIFF".toByteArray(Charsets.US_ASCII))
        header.putInt(36 + pcm.size)
        header.put("WAVE".toByteArray(Charsets.US_ASCII))
        header.put("fmt ".toByteArray(Charsets.US_ASCII))
        header.putInt(16)
        header.putShort(1)
        header.putShort(1)
        header.putInt(audio.sampleRate)
        header.putInt(audio.sampleRate * 2)
        header.putShort(2)
        header.putShort(16)
        header.put("data".toByteArray(Charsets.US_ASCII))
        header.putInt(pcm.size)
        out.write(header.array())
        out.write(pcm)
        return out.toByteArray()
    }

    private fun ascii(bytes: ByteArray, at: Int): String =
        if (at + 4 <= bytes.size) String(bytes, at, 4, Charsets.US_ASCII) else ""

    /** A human-readable guess at what a non-WAV file is, for the error message. */
    private fun sniff(bytes: ByteArray): String {
        val head = ascii(bytes, 0)
        return when {
            head.startsWith("ID3") || (bytes[0].toInt() and 0xFF) == 0xFF && (bytes[1].toInt() and 0xE0) == 0xE0 -> "MP3 is not supported; record in the app or supply a WAV file"
            bytes.size > 8 && ascii(bytes, 4) == "ftyp" -> "MP4/M4A is not supported; record in the app or supply a WAV file"
            head == "OggS" -> "OGG/Opus is not supported; record in the app or supply a WAV file"
            head == "fLaC" -> "FLAC is not supported; record in the app or supply a WAV file"
            head == "FORM" -> "AIFF is not supported; record in the app or supply a WAV file"
            else -> "Not a WAV file (unrecognised container)"
        }
    }
}
