package co.immanuel.arkspeech

import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import co.immanuel.arkspeech.core.lang.LanguagePolicy
import co.immanuel.arkspeech.core.lang.SegmentedLanguageIdentifier
import co.immanuel.arkspeech.core.model.ModelKind
import co.immanuel.arkspeech.core.vad.SpeechSegment
import co.immanuel.arkspeech.engine.AnuvaniLanguageDetector
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class AnuvaniLidTest {
    @Before fun models() = TestSupport.requireLid()

    private fun detector(): AnuvaniLanguageDetector {
        val pack = TestSupport.graph.store.installed(ModelKind.LID).first()
        return AnuvaniLanguageDetector(TestSupport.graph.store.file(pack, pack.files.first().name), threads = 2, disableKleidiAi = TestSupport.graph.settings.disableAcceleratedKernels)
    }

    @Test fun probabilitiesSumToOneAndCover23Languages(): Unit = runBlocking {
        detector().use { d ->
            val scores = d.classify(TestSupport.wav("hi_appointment.wav"))
            assertEquals(23, scores.size)
            assertEquals(1.0, scores.sumOf { it.probability.toDouble() }, 0.01)
            Log.i("AnuvaniLidTest", "hi_appointment top3=${scores.take(3)}")
        }
    }

    @Test fun syntheticClipsAreIdentified(): Unit = runBlocking {
        detector().use { d ->
            val id = SegmentedLanguageIdentifier(d, LanguagePolicy())
            val report = StringBuilder()
            var correct = 0
            val clips = TestSupport.speechClips.filterKeys { !it.contains("codemix") }
            for ((name, expected) in clips) {
                val audio = TestSupport.wav(name)
                val r = id.identify(audio, listOf(SpeechSegment(0, audio.durationMs)))
                val ok = r.code == expected.first
                if (ok) correct++
                report.append("$name -> ${r.code} ${"%.2f".format(r.confidence)} uncertain=${r.uncertain} alt=${r.alternatives.take(2)} ${if (ok) "OK" else "MISS"}\n")
            }
            Log.i("AnuvaniLidTest", "\n$report")
            // Synthetic TTS voices are out of domain for the model; most still have to land.
            assertTrue("only $correct/${clips.size} clips identified:\n$report", correct >= clips.size * 0.6)
        }
    }

    @Test fun noiseIsNotConfident(): Unit = runBlocking {
        detector().use { d ->
            val scores = d.classify(TestSupport.wav("edge_noise_4s.wav"))
            Log.i("AnuvaniLidTest", "noise top3=${scores.take(3)}")
            assertTrue("noise should not be a confident language: ${scores.first()}", scores.first().probability < 0.9f)
        }
    }
}
