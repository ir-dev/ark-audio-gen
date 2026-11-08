package co.immanuel.arkspeech.ui

import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.material3.FilterChip
import co.immanuel.arkspeech.core.config.ConfigJson
import co.immanuel.arkspeech.core.config.ExtractionDefinition
import co.immanuel.arkspeech.core.lang.Languages

/**
 * Administrator console: list, enable/disable, edit (as JSON, with validation and a live test box),
 * version history with rollback, import/export of configuration bundles, and ASR routing.
 */
@Composable
fun AdminScreen(vm: MainViewModel, state: UiState, modifier: Modifier) {
    if (vm.settings.hasAdminPin && !state.adminUnlocked) { PinGate(vm, modifier); return }
    val editing = state.editing
    if (editing != null) { DefinitionEditor(vm, state, editing, modifier); return }

    val defs = remember(state.modelsVersion) { vm.definitions() }
    var historyFor by remember { mutableStateOf<String?>(null) }
    var confirmDelete by remember { mutableStateOf<String?>(null) }
    val exportLauncher = rememberLauncherForActivityResult(ActivityResultContracts.CreateDocument("application/json")) { uri -> uri?.let { vm.exportBundle(it) } }
    val importLauncher = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri -> uri?.let { vm.importBundle(it, replace = false) } }

    Column(modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween, verticalAlignment = Alignment.CenterVertically) {
            Text("Extraction definitions", style = MaterialTheme.typography.headlineSmall)
            if (vm.settings.hasAdminPin) TextButton(onClick = { vm.lockAdmin() }) { Text("Lock") }
        }
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Button(onClick = { vm.startEditing(null) }) { Text("New") }
            OutlinedButton(onClick = { importLauncher.launch(arrayOf("application/json", "text/plain", "*/*")) }) { Text("Import") }
            OutlinedButton(onClick = { exportLauncher.launch("ark-speech-config.json") }) { Text("Export") }
        }
        defs.forEach { d ->
            SectionCard("${d.intent.name}  ·  v${d.version}") {
                Text(d.intent.description.ifBlank { d.id }, style = MaterialTheme.typography.bodySmall)
                Text("id ${d.id} · ${if (d.languages.isEmpty()) "all languages" else d.languages.joinToString()} · ${d.fields.size} fields (${d.fields.count { it.required }} required) · threshold ${d.confidenceThreshold} · ${d.fallback}", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                if (d.updatedAt.isNotBlank()) Text("updated ${d.updatedAt.take(19).replace('T', ' ')} by ${d.updatedBy.ifBlank { "?" }} — ${d.changeNote}", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    Switch(checked = d.enabled, onCheckedChange = { vm.setDefinitionEnabled(d.id, it) })
                    Text(if (d.enabled) "Enabled" else "Disabled")
                    TextButton(onClick = { vm.startEditing(d) }) { Text("Edit") }
                    TextButton(onClick = { historyFor = if (historyFor == d.id) null else d.id }) { Text("History") }
                    TextButton(onClick = { confirmDelete = d.id }) { Text("Delete") }
                }
                if (historyFor == d.id) {
                    val history = vm.repository.history(d.id)
                    history.forEach { h ->
                        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween, verticalAlignment = Alignment.CenterVertically) {
                            Text("v${h.version} · ${h.updatedAt.take(16).replace('T', ' ')} · ${h.updatedBy.ifBlank { "?" }} · ${h.changeNote.ifBlank { if (h.enabled) "enabled" else "disabled" }}", style = MaterialTheme.typography.bodySmall, modifier = Modifier.weight(1f))
                            if (h.version != d.version) TextButton(onClick = { vm.rollback(d.id, h.version) }) { Text("Restore") }
                        }
                    }
                }
            }
        }
        RoutingCard(vm, state)
    }
    confirmDelete?.let { id ->
        AlertDialog(onDismissRequest = { confirmDelete = null }, title = { Text("Delete $id?") }, text = { Text("All versions of this definition are removed. Export first if you may need them.") },
            confirmButton = { TextButton(onClick = { vm.deleteDefinition(id); confirmDelete = null }) { Text("Delete") } }, dismissButton = { TextButton(onClick = { confirmDelete = null }) { Text("Cancel") } })
    }
}

@Composable
private fun PinGate(vm: MainViewModel, modifier: Modifier) {
    var pin by remember { mutableStateOf("") }
    var wrong by remember { mutableStateOf(false) }
    Column(modifier.fillMaxSize().padding(24.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
        Text("Administrator", style = MaterialTheme.typography.headlineSmall)
        OutlinedTextField(value = pin, onValueChange = { pin = it; wrong = false }, label = { Text("PIN") }, singleLine = true, isError = wrong)
        Button(onClick = { wrong = !vm.unlockAdmin(pin) }) { Text("Unlock") }
        if (wrong) Text("Wrong PIN", color = MaterialTheme.colorScheme.error)
    }
}

@Composable
private fun RoutingCard(vm: MainViewModel, state: UiState) {
    val routing = remember(state.modelsVersion) { vm.repository.routing() }
    var json by remember(routing) { mutableStateOf(routing?.let { ConfigJson.pretty.encodeToString(co.immanuel.arkspeech.core.asr.RoutingTable.serializer(), it) } ?: "") }
    var error by remember { mutableStateOf<String?>(null) }
    SectionCard("Speech model routing") {
        Text("Which installed model transcribes which language. byLanguage overrides win, then defaultModelId, then fallbackModelIds. Pack ids: " + vm.catalog.byKind(co.immanuel.arkspeech.core.model.ModelKind.ASR).joinToString { it.id }, style = MaterialTheme.typography.bodySmall)
        OutlinedTextField(value = json, onValueChange = { json = it; error = null }, modifier = Modifier.fillMaxWidth(), minLines = 4, textStyle = MaterialTheme.typography.bodySmall.copy(fontFamily = FontFamily.Monospace), isError = error != null)
        error?.let { Text(it, color = MaterialTheme.colorScheme.error, style = MaterialTheme.typography.bodySmall) }
        Button(onClick = {
            try { vm.saveRouting(ConfigJson.pretty.decodeFromString(co.immanuel.arkspeech.core.asr.RoutingTable.serializer(), json)) } catch (e: Exception) { error = e.message?.take(200) }
        }) { Text("Save routing") }
    }
}

@Composable
private fun DefinitionEditor(vm: MainViewModel, state: UiState, def: ExtractionDefinition, modifier: Modifier) {
    var json by remember(def.id, def.version) { mutableStateOf(ConfigJson.pretty.encodeToString(ExtractionDefinition.serializer(), def)) }
    var note by remember { mutableStateOf("") }
    var testText by remember { mutableStateOf(def.examples.firstOrNull()?.text ?: "") }
    var testLang by remember { mutableStateOf(def.examples.firstOrNull()?.language ?: def.languages.firstOrNull() ?: "en") }
    Column(modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween, verticalAlignment = Alignment.CenterVertically) {
            Text("Edit ${def.id}", style = MaterialTheme.typography.headlineSmall)
            TextButton(onClick = { vm.stopEditing() }) { Text("Close") }
        }
        Text("The definition is edited as JSON so every property of the schema (docs/ADMIN_CONFIG.md) is reachable without an app update: intent, languages, trigger phrases and patterns, fields with types, validation, anchors, examples, threshold and fallback. Saving creates a new version.", style = MaterialTheme.typography.bodySmall)
        OutlinedTextField(value = json, onValueChange = { json = it }, modifier = Modifier.fillMaxWidth(), minLines = 16, textStyle = MaterialTheme.typography.bodySmall.copy(fontFamily = FontFamily.Monospace))
        if (state.editorErrors.isNotEmpty()) SectionCard("Problems") { state.editorErrors.forEach { Text(it, color = MaterialTheme.colorScheme.error, style = MaterialTheme.typography.bodySmall) } }
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            OutlinedButton(onClick = { vm.validateDefinitionJson(json) }) { Text("Validate") }
            OutlinedButton(onClick = { vm.validateDefinitionJson(json)?.let { vm.testDefinition(it, testText, testLang) } }) { Text("Test") }
        }
        SectionCard("Try a sentence") {
            OutlinedTextField(value = testText, onValueChange = { testText = it }, modifier = Modifier.fillMaxWidth(), label = { Text("Transcript to test") })
            FlowRow(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                (def.languages.ifEmpty { listOf("en", "hi", "ta", "te", "kn", "bn") }).forEach { l -> FilterChip(selected = testLang == l, onClick = { testLang = l }, label = { Text(Languages.displayName(l)) }) }
            }
            if (state.testOutput.isNotBlank()) Text(state.testOutput, fontFamily = FontFamily.Monospace, style = MaterialTheme.typography.bodySmall)
        }
        HorizontalDivider()
        OutlinedTextField(value = note, onValueChange = { note = it }, modifier = Modifier.fillMaxWidth(), label = { Text("Change note (recorded in the version history)") })
        Button(onClick = { vm.validateDefinitionJson(json)?.let { vm.saveDefinition(it, note.ifBlank { "Edited" }) } }, modifier = Modifier.fillMaxWidth()) { Text("Save as new version") }
    }
}
