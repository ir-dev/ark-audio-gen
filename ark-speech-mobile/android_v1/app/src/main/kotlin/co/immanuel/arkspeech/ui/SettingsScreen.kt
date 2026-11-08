package co.immanuel.arkspeech.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Slider
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.key
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp

@Composable
fun SettingsScreen(vm: MainViewModel, state: UiState, modifier: Modifier) {
    val s = vm.settings
    var tick by remember { mutableStateOf(0) }
    fun changed() { tick++; vm.bumpVersion() }
    Column(modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
        Text("Settings", style = MaterialTheme.typography.headlineSmall)
        key(tick) {
            SectionCard("Recording") {
                ToggleRow("Hold to talk (release to stop)", s.holdToTalk) { s.holdToTalk = it; changed() }
                Text("Recording stops by itself at 3 minutes. The microphone is only open while the button is held.", style = MaterialTheme.typography.bodySmall)
            }
            SectionCard("Privacy") {
                ToggleRow("Keep recordings by default", s.defaultRetainAudio) { s.defaultRetainAudio = it; changed() }
                ToggleRow("Keep transcripts by default", s.defaultRetainTranscript) { s.defaultRetainTranscript = it; changed() }
                val items = vm.retention.list()
                Text("${items.size} retained item(s), ${vm.retention.totalBytes().asMb()} in app-private storage", style = MaterialTheme.typography.bodySmall)
                items.take(20).forEach { it ->
                    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween, verticalAlignment = Alignment.CenterVertically) {
                        Text("${it.requestId.take(8)} · ${if (it.audio != null) "audio " else ""}${if (it.result != null) "result" else ""} · ${it.sizeBytes.asMb()}", style = MaterialTheme.typography.bodySmall)
                        TextButton(onClick = { vm.retention.delete(it.requestId); changed() }) { Text("Delete") }
                    }
                }
                if (items.isNotEmpty()) OutlinedButton(onClick = { vm.retention.deleteAll(); changed() }) { Text("Delete everything retained") }
            }
            SectionCard("Performance") {
                var threads by remember { mutableStateOf(s.inferenceThreads.toFloat()) }
                Text("Inference threads: ${threads.toInt()} (device has ${Runtime.getRuntime().availableProcessors()} cores)")
                Slider(value = threads, onValueChange = { threads = it }, onValueChangeFinished = { s.inferenceThreads = threads.toInt(); changed() }, valueRange = 1f..6f, steps = 4)
                ToggleRow("Keep the speech model loaded between recordings (faster, uses RAM)", s.keepModelsLoaded) { s.keepModelsLoaded = it; changed() }
                ToggleRow("Download models over Wi-Fi only", s.wifiOnlyDownloads) { s.wifiOnlyDownloads = it; changed() }
                ToggleRow("Disable accelerated CPU kernels for language ID (fixes crashes on emulators or misreported CPUs)", s.disableAcceleratedKernels) { s.disableAcceleratedKernels = it; changed() }
            }
            SectionCard("Administrator PIN") {
                var pin by remember { mutableStateOf("") }
                Text(if (s.hasAdminPin) "A PIN protects the Admin tab." else "No PIN set: the Admin tab is open. Set one for shared devices.", style = MaterialTheme.typography.bodySmall)
                OutlinedTextField(value = pin, onValueChange = { pin = it }, label = { Text("New PIN") }, singleLine = true)
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    Button(onClick = { s.setAdminPin(pin); pin = ""; vm.lockAdmin(); changed() }, enabled = pin.length >= 4) { Text("Set PIN") }
                    if (s.hasAdminPin) OutlinedButton(onClick = { s.setAdminPin(null); changed() }) { Text("Remove PIN") }
                }
            }
            SectionCard("About") {
                vm.deviceInfo.forEach { (k, v) -> KeyValue(k, v) }
                Text("Runtimes: sherpa-onnx 2.0.1 (Apache-2.0), ONNX Runtime 1.31.1 (MIT). Models: see Models tab for licences. Processing is entirely on-device.", style = MaterialTheme.typography.bodySmall)
            }
        }
    }
}

@Composable
private fun ToggleRow(label: String, checked: Boolean, onChange: (Boolean) -> Unit) {
    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween, verticalAlignment = Alignment.CenterVertically) {
        Text(label, modifier = Modifier.weight(1f))
        Switch(checked = checked, onCheckedChange = onChange)
    }
}
