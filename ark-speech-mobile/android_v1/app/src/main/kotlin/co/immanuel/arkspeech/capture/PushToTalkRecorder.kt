package co.immanuel.arkspeech.capture

import android.annotation.SuppressLint
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.media.audiofx.AutomaticGainControl
import android.media.audiofx.NoiseSuppressor
import co.immanuel.arkspeech.core.audio.PcmAudio
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.log10
import kotlin.math.sqrt

/**
 * Push-to-talk capture: records 16 kHz mono PCM only between [start] and [stop]/[cancel], stops by
 * itself at [maxDurationMs] (3 minutes), and keeps everything in memory. No service, no wake lock;
 * the screen is kept on by the UI only while this is recording.
 */
class PushToTalkRecorder(
    private val maxDurationMs: Long = DEFAULT_MAX_MS,
    private val sampleRate: Int = PcmAudio.TARGET_SAMPLE_RATE,
    private val useVoiceEffects: Boolean = true,
) {
    sealed class State {
        object Idle : State()
        data class Recording(val elapsedMs: Long, val levelDbfs: Float, val maxMs: Long) : State()
        data class Finished(val audio: PcmAudio, val autoStopped: Boolean) : State()
        object Cancelled : State()
        data class Error(val message: String) : State()
    }

    private val _state = MutableStateFlow<State>(State.Idle)
    val state: StateFlow<State> = _state

    private var job: Job? = null
    private val stopRequested = AtomicBoolean(false)
    private val cancelRequested = AtomicBoolean(false)

    val isRecording: Boolean get() = _state.value is State.Recording

    /** Requires RECORD_AUDIO; the caller checks the permission first. */
    @SuppressLint("MissingPermission")
    fun start(scope: CoroutineScope) {
        if (job?.isActive == true) return
        stopRequested.set(false); cancelRequested.set(false)
        job = scope.launch(Dispatchers.IO) {
            val minBuf = AudioRecord.getMinBufferSize(sampleRate, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT)
            if (minBuf <= 0) { _state.value = State.Error("Microphone does not support $sampleRate Hz mono"); return@launch }
            val bufferBytes = maxOf(minBuf, sampleRate * 2 / 5) // >= 200 ms
            val record = try {
                AudioRecord(MediaRecorder.AudioSource.VOICE_RECOGNITION, sampleRate, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT, bufferBytes)
            } catch (e: Exception) { _state.value = State.Error("Could not open microphone: ${e.javaClass.simpleName}"); return@launch }
            if (record.state != AudioRecord.STATE_INITIALIZED) { record.release(); _state.value = State.Error("Microphone unavailable"); return@launch }
            var ns: NoiseSuppressor? = null; var agc: AutomaticGainControl? = null
            if (useVoiceEffects) {
                if (NoiseSuppressor.isAvailable()) ns = runCatching { NoiseSuppressor.create(record.audioSessionId)?.apply { enabled = true } }.getOrNull()
                if (AutomaticGainControl.isAvailable()) agc = runCatching { AutomaticGainControl.create(record.audioSessionId)?.apply { enabled = true } }.getOrNull()
            }
            val maxSamples = (sampleRate * maxDurationMs / 1000).toInt()
            val buffer = ShortArray(maxSamples)
            val chunk = ShortArray(sampleRate / 10) // 100 ms
            var filled = 0
            var autoStopped = false
            try {
                record.startRecording()
                _state.value = State.Recording(0, -90f, maxDurationMs)
                while (isActive && !stopRequested.get() && !cancelRequested.get()) {
                    val want = minOf(chunk.size, maxSamples - filled)
                    if (want <= 0) { autoStopped = true; break }
                    val n = record.read(chunk, 0, want)
                    if (n < 0) { _state.value = State.Error("Microphone read failed ($n)"); cancelRequested.set(true); break }
                    System.arraycopy(chunk, 0, buffer, filled, n)
                    filled += n
                    _state.value = State.Recording(filled * 1000L / sampleRate, levelDb(chunk, n), maxDurationMs)
                }
            } finally {
                runCatching { record.stop() }
                ns?.release(); agc?.release()
                record.release()
            }
            if (cancelRequested.get()) {
                buffer.fill(0) // do not leave speech in memory longer than needed
                if (_state.value !is State.Error) _state.value = State.Cancelled
            } else {
                _state.value = State.Finished(PcmAudio.fromShorts(buffer, sampleRate, filled), autoStopped)
            }
        }
    }

    fun stop() { stopRequested.set(true) }

    fun cancel() { cancelRequested.set(true) }

    fun reset() { if (!isRecording) _state.value = State.Idle }

    private fun levelDb(s: ShortArray, n: Int): Float {
        if (n == 0) return -90f
        var acc = 0.0
        for (i in 0 until n) { val v = s[i] / 32768.0; acc += v * v }
        val rms = sqrt(acc / n)
        return if (rms < 1e-6) -90f else (20 * log10(rms)).toFloat()
    }

    companion object { const val DEFAULT_MAX_MS = 180_000L }
}
