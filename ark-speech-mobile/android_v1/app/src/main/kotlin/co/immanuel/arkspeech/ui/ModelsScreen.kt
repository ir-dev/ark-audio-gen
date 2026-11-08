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
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import co.immanuel.arkspeech.core.model.ModelKind
import co.immanuel.arkspeech.model.InstallState

/** Install, verify and remove model packs; shows which engine each stage will use. */
@Composable
fun ModelsScreen(vm: MainViewModel, state: UiState, modifier: Modifier) {
    val summary = remember(state.modelsVersion, state.downloads) { vm.engineSummary() }
    val totalRam = vm.deviceInfo["totalRamMb"]?.toIntOrNull() ?: 0
    Column(modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
        Text("Models", style = MaterialTheme.typography.headlineSmall)
        SectionCard("Active engines") {
            KeyValue("Voice activity", summary.vad)
            KeyValue("Language ID", summary.lid ?: "none — install Anuvani-LID")
            KeyValue("Default speech model", summary.asrDefault ?: "none installed")
            Text("Device: ${vm.deviceInfo["manufacturer"]} ${vm.deviceInfo["model"]}, ${totalRam} MB RAM, ${vm.deviceInfo["cores"]} cores, ${vm.deviceInfo["abi"]}", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
            Text("Model files live in\n${vm.store.root}\n(adb push works during development; see scripts/fetch-models.sh)", style = MaterialTheme.typography.bodySmall, fontFamily = FontFamily.Monospace)
        }
        listOf(ModelKind.VAD to "Voice activity detection", ModelKind.LID to "Language identification", ModelKind.ASR to "Speech recognition").forEach { (kind, title) ->
            SectionCard(title) {
                vm.catalog.byKind(kind).forEach { pack ->
                    val st = vm.store.state(pack)
                    val dl = state.downloads[pack.id]
                    val fits = pack.peakRamMb == 0 || totalRam == 0 || pack.peakRamMb < totalRam * 0.45
                    Column(Modifier.fillMaxWidth().padding(vertical = 6.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
                        Text(pack.displayName, fontWeight = FontWeight.Medium)
                        Text(pack.description, style = MaterialTheme.typography.bodySmall)
                        Text("${pack.installedSizeBytes.asMb()} · ${pack.licence} · ${pack.tier}${if (pack.peakRamMb > 0) " · ~${pack.peakRamMb} MB RAM" else ""}${if (pack.languages == listOf("*")) " · all languages" else " · ${pack.languages.size} languages"}", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                        if (!fits) Text("Probably too heavy for this device's RAM", color = MaterialTheme.colorScheme.error, style = MaterialTheme.typography.bodySmall)
                        if (dl?.running == true) {
                            LinearProgressIndicator(progress = { dl.fraction }, modifier = Modifier.fillMaxWidth())
                            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                                Text("${dl.downloadedBytes.asMb()} of ${dl.totalBytes.asMb()}", style = MaterialTheme.typography.bodySmall)
                                TextButton(onClick = { vm.cancelDownload(pack) }) { Text("Cancel") }
                            }
                        } else Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                            when (st) {
                                InstallState.INSTALLED -> {
                                    Text("Installed", color = MaterialTheme.colorScheme.tertiary, modifier = Modifier.padding(top = 12.dp))
                                    OutlinedButton(onClick = { vm.verifyPack(pack) }, enabled = !state.busy) { Text("Verify") }
                                    TextButton(onClick = { vm.deletePack(pack) }) { Text("Remove") }
                                }
                                InstallState.PARTIAL -> {
                                    Button(onClick = { vm.download(pack) }) { Text("Resume download") }
                                    TextButton(onClick = { vm.deletePack(pack) }) { Text("Remove") }
                                }
                                InstallState.NOT_INSTALLED -> Button(onClick = { vm.download(pack) }) { Text(if (vm.settings.wifiOnlyDownloads) "Download (Wi-Fi)" else "Download") }
                            }
                        }
                        if (dl?.failed == true) Text("Download failed (${dl.reason}). Check the connection and try again.", color = MaterialTheme.colorScheme.error, style = MaterialTheme.typography.bodySmall)
                    }
                }
            }
        }
        Text("Downloads use the system download manager over Wi-Fi only by default (Settings). Models are the only thing this app ever downloads; audio never leaves the device.", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
    }
}
