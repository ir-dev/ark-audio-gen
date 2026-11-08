package co.immanuel.arkspeech.core.audio

import co.immanuel.arkspeech.core.support.TestSupport
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class AudioValidatorTest {
    private val validator = AudioValidator()

    @Test fun `empty recording is rejected`() {
        val r = validator.validate(PcmAudio.empty())
        assertEquals(AudioVerdict.REJECT, r.verdict)
        assertTrue(AudioIssue.EMPTY in r.issues)
    }

    @Test fun `digital silence is rejected as silent`() {
        val r = validator.validate(TestSupport.wav("edge_silence_4s.wav"))
        assertEquals(AudioVerdict.REJECT, r.verdict)
        assertTrue(AudioIssue.SILENT in r.issues, "issues=${r.issues}")
    }

    @Test fun `white noise is rejected as noise`() {
        val r = validator.validate(TestSupport.wav("edge_noise_4s.wav"))
        assertEquals(AudioVerdict.REJECT, r.verdict, "report=$r")
        assertTrue(AudioIssue.NOISY in r.issues, "issues=${r.issues}")
        assertTrue(r.estimatedSnrDb < 4.0, "snr=${r.estimatedSnrDb}")
    }

    @Test fun `too short recording is rejected`() {
        val r = validator.validate(TestSupport.wav("edge_too_short.wav"))
        assertEquals(AudioVerdict.REJECT, r.verdict)
        assertTrue(AudioIssue.TOO_SHORT in r.issues)
    }

    @Test fun `heavily clipped speech is flagged and rejected`() {
        val r = validator.validate(TestSupport.wav("edge_clipped.wav"))
        assertTrue(AudioIssue.CLIPPING in r.issues, "issues=${r.issues}")
        assertEquals(AudioVerdict.REJECT, r.verdict, "clipping ratio=${r.clippingRatio}")
    }

    @Test fun `very quiet speech is not silently accepted`() {
        val r = validator.validate(TestSupport.wav("edge_very_quiet.wav"))
        assertTrue(AudioIssue.TOO_QUIET in r.issues || AudioIssue.SILENT in r.issues, "issues=${r.issues} rms=${r.rmsDbfs}")
    }

    @Test fun `clean speech passes`() {
        for (name in listOf("en_appointment.wav", "hi_appointment.wav", "ta_appointment.wav", "bn_payment.wav")) {
            val r = validator.validate(TestSupport.wav(name))
            assertTrue(r.isUsable, "$name: $r")
            assertFalse(AudioIssue.SILENT in r.issues, name)
            assertFalse(AudioIssue.TOO_SHORT in r.issues, name)
            assertTrue(r.estimatedSnrDb > 12.0, "$name snr=${r.estimatedSnrDb}")
        }
    }

    @Test fun `noisy speech is usable but warns`() {
        val r = validator.validate(TestSupport.wav("hi_appointment_noisy.wav"))
        assertTrue(r.isUsable, "$r")
    }

    @Test fun `recording over the three minute cap is rejected`() {
        val long = TestSupport.syntheticSpeech(181_000)
        val r = validator.validate(long)
        assertTrue(AudioIssue.TOO_LONG in r.issues)
        assertEquals(AudioVerdict.REJECT, r.verdict)
    }

    @Test fun `synthetic speech helper passes validation`() {
        val r = validator.validate(TestSupport.syntheticSpeech(4000))
        assertTrue(r.isUsable, "$r")
    }
}
