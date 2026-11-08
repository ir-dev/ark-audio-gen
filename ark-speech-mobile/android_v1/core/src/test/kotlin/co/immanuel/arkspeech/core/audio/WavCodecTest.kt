package co.immanuel.arkspeech.core.audio

import co.immanuel.arkspeech.core.support.TestSupport
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class WavCodecTest {
    @Test fun `stereo 44k wav is converted to 16k mono`() {
        val d = WavCodec.decode(TestSupport.bytes("edge_stereo_44k.wav"))
        assertEquals(44100, d.originalSampleRate)
        assertEquals(2, d.originalChannels)
        assertEquals(16000, d.audio.sampleRate)
        assertTrue(d.audio.durationMs in 8600..8900, "duration=${d.audio.durationMs}")
    }

    @Test fun `m4a is reported as unsupported with a useful message`() {
        val e = assertFailsWith<UnsupportedAudioFormatException> { WavCodec.decode(TestSupport.bytes("edge_unsupported.m4a")) }
        assertTrue(e.message!!.contains("M4A"), e.message)
    }

    @Test fun `random bytes are rejected`() {
        assertFailsWith<UnsupportedAudioFormatException> { WavCodec.decode(ByteArray(100) { (it * 7).toByte() }) }
        assertFailsWith<UnsupportedAudioFormatException> { WavCodec.decode(ByteArray(3)) }
    }

    @Test fun `encode decode round trip preserves samples`() {
        val src = TestSupport.syntheticSpeech(1000)
        val back = WavCodec.decode(WavCodec.encode(src)).audio
        assertEquals(src.sampleCount, back.sampleCount)
        var maxErr = 0f
        for (i in src.samples.indices) maxErr = maxOf(maxErr, kotlin.math.abs(src.samples[i] - back.samples[i]))
        assertTrue(maxErr < 1f / 32000, "max error $maxErr")
    }

    @Test fun `pcm16 bytes down-mix channels`() {
        val bytes = byteArrayOf(0, 0x40, 0, 0x40.toByte()) // one stereo frame, both channels 0x4000
        val a = PcmAudio.fromPcm16(bytes, 16000, 2)
        assertEquals(1, a.sampleCount)
        assertEquals(0.5f, a.samples[0], 0.001f)
    }
}
