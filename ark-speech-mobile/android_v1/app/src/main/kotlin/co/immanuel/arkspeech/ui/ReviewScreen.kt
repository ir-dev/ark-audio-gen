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
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.FilterChip
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import co.immanuel.arkspeech.core.lang.Languages
import androidx.compose.foundation.layout.FlowRow

/** Language confirmation/override and transcript preview before extraction. */
@Composable
fun ReviewScreen(vm: MainViewModel, state: UiState, modifier: Modifier) {
    val result = state.result ?: return
    val lang = result.language
    val detectedCode = lang?.code
    val choice = state.languageChoice
    var showAll by remember { mutableStateOf(false) }

    Column(modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
        Text("Review", style = MaterialTheme.typography.headlineSmall)
        SectionCard("Language") {
            val code = lang?.code
            if (lang == null || code == null) {
                Text("No language detected. Pick the spoken language:")
            } else {
                val pct = (lang.confidence * 100).toInt()
                Text("${Languages.displayName(code)} · $pct% ${if (lang.uncertain) "· uncertain, please confirm" else ""}${if (lang.isCodeMixed) "\nMixed-language speech detected: " + lang.timeShare.entries.sortedByDescending { it.value }.joinToString { "${Languages.displayName(it.key)} ${(it.value * 100).toInt()}%" } else ""}",
                    color = if (lang.uncertain) MaterialTheme.colorScheme.error else MaterialTheme.colorScheme.onSurface)
                if (!lang.supported) Text("This language has no installed speech model; choose another or install one under Models.", color = MaterialTheme.colorScheme.error)
            }
            val options = (listOfNotNull(detectedCode) + (lang?.alternatives?.map { it.code } ?: emptyList())).distinct().take(4)
            FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                options.forEach { code -> FilterChip(selected = choice == code, onClick = { vm.chooseLanguage(code) }, label = { Text(Languages.displayName(code)) }) }
                FilterChip(selected = choice != null && choice !in options, onClick = { showAll = true }, label = { Text(if (choice != null && choice !in options) Languages.displayName(choice) else "Other…") })
                DropdownMenu(expanded = showAll, onDismissRequest = { showAll = false }) {
                    Languages.all.sortedBy { it.name }.forEach { l -> DropdownMenuItem(text = { Text("${l.name} · ${l.nativeName}") }, onClick = { vm.chooseLanguage(l.code); showAll = false }) }
                }
            }
            if (choice != null && choice != detectedCode) {
                OutlinedButton(onClick = { vm.retranscribe() }) { Text("Transcribe again in ${Languages.displayName(choice)}") }
            }
        }
        SectionCard("Transcript") {
            val t = result.transcript
            Text("${t?.modelId ?: ""} · confidence ${((t?.confidence ?: 0f) * 100).toInt()}% (${t?.confidenceSource ?: "n/a"}) · script ${t?.script ?: "?"}", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
            OutlinedTextField(value = state.transcriptDraft, onValueChange = { vm.setTranscriptDraft(it) }, modifier = Modifier.fillMaxWidth(), minLines = 4, label = { Text("Correct the text if needed") })
            if (result.warnings.isNotEmpty()) Text(result.warnings.joinToString("\n"), style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.error)
        }
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            Button(onClick = { vm.extract() }, enabled = state.transcriptDraft.isNotBlank() && !state.busy, modifier = Modifier.weight(1f)) { Text("Extract") }
            TextButton(onClick = { vm.discard() }) { Text("Discard") }
        }
    }
}
