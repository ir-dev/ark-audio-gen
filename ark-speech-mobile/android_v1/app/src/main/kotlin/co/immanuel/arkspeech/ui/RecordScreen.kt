package co.immanuel.arkspeech.ui

import android.Manifest
import android.content.pm.PackageManager
import android.view.WindowManager
import androidx.activity.compose.LocalActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import co.immanuel.arkspeech.capture.PushToTalkRecorder

/** Routes the record tab to the right screen for the current phase. */
@Composable
fun RecordFlow(vm: MainViewModel, state: UiState, modifier: Modifier) {
    when (state.phase) {
        Phase.IDLE, Phase.RECORDING, Phase.PROCESSING, Phase.ERROR -> RecordScreen(vm, state, modifier)
        Phase.REVIEW -> ReviewScreen(vm, state, modifier)
        Phase.RESULT -> ResultScreen(vm, state, modifier)
    }
}

@Composable
fun RecordScreen(vm: MainViewModel, state: UiState, modifier: Modifier) {
    val context = LocalContext.current
    val activity = LocalActivity.current
    var hasPermission by remember { mutableStateOf(ContextCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED) }
    val permission = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { hasPermission = it; if (it) vm.startRecording() }
    val picker = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri -> uri?.let { vm.processWav(it) } }
    val recording = state.recorder as? PushToTalkRecorder.State.Recording
    val holdToTalk = vm.settings.holdToTalk
    val summary = remember(state.modelsVersion) { vm.engineSummary() }

    // Keep the screen on only while recording: no wake lock, nothing in the background.
    DisposableEffect(recording != null) {
        val w = activity?.window
        if (recording != null) w?.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON) else w?.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        onDispose { w?.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON) }
    }

    fun begin() { if (hasPermission) vm.startRecording() else permission.launch(Manifest.permission.RECORD_AUDIO) }

    Column(modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(24.dp), horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.spacedBy(16.dp)) {
        Text("Ark Speech", style = MaterialTheme.typography.headlineMedium, fontWeight = FontWeight.Bold)
        Text("Push to talk · offline · ${summary.asrDefault ?: "no speech model installed"}", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant, textAlign = TextAlign.Center)
        Spacer(Modifier.height(8.dp))

        val stateText = when {
            state.phase == Phase.PROCESSING -> state.processingLabel
            recording != null -> "Recording"
            state.phase == Phase.ERROR -> "Error"
            else -> if (holdToTalk) "Hold to talk" else "Tap to start"
        }
        Text(stateText, style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.SemiBold)
        Text(if (recording != null) "${recording.elapsedMs.asMmSs()} / ${recording.maxMs.asMmSs()}" else "0:00 / 3:00", style = MaterialTheme.typography.displaySmall)
        if (recording != null) {
            LinearProgressIndicator(progress = { (recording.elapsedMs.toFloat() / recording.maxMs).coerceIn(0f, 1f) }, modifier = Modifier.fillMaxWidth())
            val level = ((recording.levelDbfs + 60f) / 60f).coerceIn(0f, 1f)
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(2.dp)) {
                for (i in 0 until 20) Box(Modifier.weight(1f).height(10.dp).background(if (i < (level * 20).toInt()) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.surfaceVariant))
            }
            if (recording.levelDbfs < -45f && recording.elapsedMs > 1500) Text("Very quiet — move closer to the microphone", color = MaterialTheme.colorScheme.error, style = MaterialTheme.typography.bodySmall)
        }

        if (state.phase == Phase.PROCESSING) {
            CircularProgressIndicator()
        } else {
            val active = recording != null
            Box(
                Modifier.size(140.dp).background(if (active) MaterialTheme.colorScheme.error else MaterialTheme.colorScheme.primary, CircleShape)
                    .pointerInput(holdToTalk, hasPermission) {
                        if (holdToTalk) {
                            detectTapGestures(onPress = {
                                begin()
                                try { awaitRelease() } finally { vm.stopRecording() }
                            })
                        } else {
                            detectTapGestures(onTap = { if (vm.recorder.isRecording) vm.stopRecording() else begin() })
                        }
                    },
                contentAlignment = Alignment.Center,
            ) { Text(if (active) "STOP" else "TALK", color = MaterialTheme.colorScheme.onPrimary, style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.Bold) }
            if (recording != null) TextButton(onClick = { vm.cancelRecording() }) { Text("Cancel recording") }
        }

        if (state.phase == Phase.ERROR) {
            Text(state.error ?: "Something went wrong", color = MaterialTheme.colorScheme.error, textAlign = TextAlign.Center)
            Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                Button(onClick = { vm.discard(); begin() }) { Text("Retry") }
                OutlinedButton(onClick = { vm.discard() }) { Text("Dismiss") }
            }
        }

        if (state.phase == Phase.IDLE) {
            Spacer(Modifier.height(8.dp))
            OutlinedButton(onClick = { picker.launch(arrayOf("audio/*", "application/octet-stream")) }) { Text("Process a WAV file instead") }
            val s = summary
            Text("VAD: ${s.vad}\nLanguage ID: ${s.lid ?: "none installed"}\nSpeech models: ${if (s.asrInstalled.isEmpty()) "none installed (see Models)" else s.asrInstalled.joinToString()}", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant, textAlign = TextAlign.Center)
            Text("Audio stays on this device. Nothing is saved unless you tick 'keep' on the result screen.", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant, textAlign = TextAlign.Center)
        }
    }
}
