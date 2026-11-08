package co.immanuel.arkspeech.core.audio

import co.immanuel.arkspeech.core.support.TestSupport
import co.immanuel.arkspeech.core.vad.EnergyVad
import kotlin.test.Test
import kotlin.test.assertTrue

class EnergyVadTest {
    private val vad = EnergyVad()

    @Test fun `silence yields no segments`() {
        assertTrue(vad.segments(TestSupport.wav("edge_silence_4s.wav")).isEmpty())
    }

    @Test fun `speech followed by silence ends before the silence`() {
        val audio = TestSupport.wav("edge_speech_then_silence.wav")
        val segs = vad.segments(audio)
        assertTrue(segs.isNotEmpty())
        assertTrue(segs.last().endMs < audio.durationMs - 2000, "segments=$segs duration=${audio.durationMs}")
        assertTrue(segs.sumOf { it.durationMs } > 4000)
    }

    @Test fun `segments never exceed the configured maximum`() {
        val segs = EnergyVad(maxSegmentMs = 3000).segments(TestSupport.wav("en_appointment.wav"))
        assertTrue(segs.all { it.durationMs <= 3000 }, "$segs")
    }
}
