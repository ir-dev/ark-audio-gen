package co.immanuel.arkspeech

import androidx.test.ext.junit.runners.AndroidJUnit4
import co.immanuel.arkspeech.core.model.ModelKind
import co.immanuel.arkspeech.engine.SherpaVad
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class SherpaVadTest {
    @Before fun models() = TestSupport.requireVad()

    private fun vad(): SherpaVad {
        val pack = TestSupport.graph.store.installed(ModelKind.VAD).first()
        return SherpaVad(TestSupport.graph.store.file(pack, pack.files.first().name))
    }

    @Test fun speechProducesSegmentsAndSilenceDoesNot() {
        vad().use { v ->
            val speech = v.segments(TestSupport.wav("en_appointment.wav"))
            assertTrue("expected speech segments, got $speech", speech.isNotEmpty())
            assertTrue("speech should cover > 3 s, got ${speech.sumOf { it.durationMs }} ms", speech.sumOf { it.durationMs } > 3000)
            val silence = v.segments(TestSupport.wav("edge_silence_4s.wav"))
            assertTrue("silence should have no segments, got $silence", silence.isEmpty())
        }
    }

    @Test fun trailingSilenceIsExcluded() {
        vad().use { v ->
            val audio = TestSupport.wav("edge_speech_then_silence.wav")
            val segs = v.segments(audio)
            assertTrue(segs.isNotEmpty())
            assertTrue("last segment ends at ${segs.last().endMs} of ${audio.durationMs}", segs.last().endMs < audio.durationMs - 2000)
        }
    }

    @Test fun segmentsNeverExceedTheCap() {
        vad().use { v ->
            val segs = v.segments(TestSupport.wav("hi_appointment.wav"))
            assertTrue(segs.all { it.durationMs <= 25_500 })
        }
    }
}
