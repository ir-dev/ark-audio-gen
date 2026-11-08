package co.immanuel.arkspeech.ui

import android.content.Intent
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Checkbox
import androidx.compose.material3.FilterChip
import androidx.compose.material3.HorizontalDivider
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
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalClipboardManager
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.foundation.layout.FlowRow
import co.immanuel.arkspeech.core.lang.Languages
import co.immanuel.arkspeech.core.nlu.FieldStatus
import co.immanuel.arkspeech.core.nlu.IntentStatus
import co.immanuel.arkspeech.core.pipeline.ClarificationKind
import co.immanuel.arkspeech.core.pipeline.PipelineStatus

/** Intent, fields, clarifications, consent and the structured JSON. */
@Composable
fun ResultScreen(vm: MainViewModel, state: UiState, modifier: Modifier) {
    val result = state.result ?: return
    val clipboard = LocalClipboardManager.current
    val context = LocalContext.current
    var showJson by remember { mutableStateOf(false) }

    Column(modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
        Text(result.status.humanLabel(), style = MaterialTheme.typography.headlineSmall, color = if (result.status == PipelineStatus.OK) MaterialTheme.colorScheme.tertiary else MaterialTheme.colorScheme.onSurface)

        if (result.status in setOf(PipelineStatus.AUDIO_REJECTED, PipelineStatus.NO_SPEECH, PipelineStatus.MODEL_UNAVAILABLE, PipelineStatus.FAILED, PipelineStatus.UNSUPPORTED_LANGUAGE)) {
            SectionCard("What happened") {
                result.audio?.let { a -> Text("Audio: ${a.durationMs.asMmSs()}, level ${a.rmsDbfs} dBFS, SNR ${a.estimatedSnrDb} dB${if (a.issues.isNotEmpty()) ", issues: " + a.issues.joinToString { it.name.lowercase().replace('_', ' ') } else ""}") }
                result.warnings.forEach { Text(it, color = MaterialTheme.colorScheme.error) }
                when (result.status) {
                    PipelineStatus.AUDIO_REJECTED -> Text("Try again closer to the microphone, in a quieter place, and speak for at least a second.")
                    PipelineStatus.NO_SPEECH -> Text("No speech was detected in the recording.")
                    PipelineStatus.MODEL_UNAVAILABLE -> Text("Install a speech model under Models, then record again.")
                    PipelineStatus.UNSUPPORTED_LANGUAGE -> Text("Detected ${result.language?.code?.let { Languages.displayName(it) } ?: "an unsupported language"}. Choose the language manually on the review screen.")
                    else -> {}
                }
            }
            Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                Button(onClick = { vm.discard(); vm.startRecording() }) { Text("Record again") }
                if (result.status == PipelineStatus.UNSUPPORTED_LANGUAGE) OutlinedButton(onClick = { vm.backToReview() }) { Text("Pick language") }
                TextButton(onClick = { vm.discard() }) { Text("Done") }
            }
            return@Column
        }

        SectionCard("Heard") {
            Text("${result.language?.code?.let { Languages.displayName(it) } ?: "?"} · ${((result.language?.confidence ?: 0f) * 100).toInt()}%${if (result.language?.source == co.immanuel.arkspeech.core.lang.LanguageSource.USER_OVERRIDE) " (chosen by you)" else ""}", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
            Text(result.transcript?.text ?: "", style = MaterialTheme.typography.bodyLarge)
            TextButton(onClick = { vm.backToReview() }) { Text("Edit transcript or language") }
        }

        SectionCard("Intent") {
            val intent = result.intent
            val best = intent?.best
            if (intent == null || best == null) {
                Text("No configured intent matched this transcript.")
                Text("Pick one to extract anyway:", style = MaterialTheme.typography.bodySmall)
                FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    vm.definitions().filter { it.enabled }.forEach { d -> FilterChip(selected = false, onClick = { vm.chooseIntent(d.id) }, label = { Text(d.intent.name) }) }
                }
            } else {
                Text("${best.intentName} · ${(intent.confidence * 100).toInt()}% · v${best.definitionVersion}", fontWeight = FontWeight.Medium)
                if (best.matchedPhrases.isNotEmpty()) Text("matched: " + best.matchedPhrases.joinToString(), style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                if (intent.status == IntentStatus.AMBIGUOUS || intent.status == IntentStatus.LOW_CONFIDENCE) {
                    Text(if (intent.status == IntentStatus.AMBIGUOUS) "Which did you mean?" else "Is this right?", color = MaterialTheme.colorScheme.error)
                    FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        (listOf(best) + intent.alternatives).forEach { c -> FilterChip(selected = false, onClick = { vm.chooseIntent(c.definitionId) }, label = { Text("${c.intentName} ${(c.score * 100).toInt()}%") }) }
                    }
                }
            }
        }

        if (result.fields.isNotEmpty()) SectionCard("Fields") {
            result.fields.values.forEach { f ->
                Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween, verticalAlignment = Alignment.CenterVertically) {
                    Column(Modifier.weight(1f)) {
                        Text(f.name + if (f.required) " *" else "", fontWeight = FontWeight.Medium)
                        Text(f.value?.let { v -> v + (f.extra["currency"]?.let { " $it" } ?: "") + (f.raw?.takeIf { it != v }?.let { " (\"$it\")" } ?: "") } ?: "—", style = MaterialTheme.typography.bodyMedium)
                        if (f.confidence > 0f && f.status != FieldStatus.USER_PROVIDED) Text("${(f.confidence * 100).toInt()}%${f.reason?.let { " · $it" } ?: ""}", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                    }
                    StatusChip(f.status)
                }
                HorizontalDivider()
            }
        }

        val fieldClarifications = result.clarifications.filter { it.kind != ClarificationKind.LANGUAGE && it.kind != ClarificationKind.INTENT }
        if (fieldClarifications.isNotEmpty()) SectionCard("Please clarify") {
            fieldClarifications.forEach { c -> ClarificationRow(c.field!!, c.prompt, c.options, c.kind, vm) }
        }

        SectionCard("Keep a copy?") {
            Text("Nothing is stored unless you tick these. Files stay in this app's private storage and can be deleted in Settings.", style = MaterialTheme.typography.bodySmall)
            Row(verticalAlignment = Alignment.CenterVertically) { Checkbox(checked = state.consent.retainAudio, onCheckedChange = { vm.setConsent(audio = it) }); Text("Keep the recording") }
            Row(verticalAlignment = Alignment.CenterVertically) { Checkbox(checked = state.consent.retainTranscript, onCheckedChange = { vm.setConsent(transcript = it) }); Text("Keep the transcript and result") }
        }

        SectionCard("Processing") {
            result.processing.stages.forEach { s -> KeyValue("${s.stage} ${s.detail?.let { "($it)" } ?: ""}", "${s.ms} ms · ${s.status}") }
            KeyValue("total", "${result.processing.totalMs} ms")
            KeyValue("models", listOfNotNull(result.processing.lidEngine, result.processing.asrModelId).joinToString(" + "))
        }

        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            OutlinedButton(onClick = { showJson = !showJson }) { Text(if (showJson) "Hide JSON" else "Show JSON") }
            OutlinedButton(onClick = { clipboard.setText(AnnotatedString(result.toJson())) }) { Text("Copy") }
            OutlinedButton(onClick = {
                val send = Intent(Intent.ACTION_SEND).apply { type = "application/json"; putExtra(Intent.EXTRA_TEXT, result.toJson()) }
                context.startActivity(Intent.createChooser(send, "Share result"))
            }) { Text("Share") }
        }
        if (showJson) Text(result.toJson(), fontFamily = FontFamily.Monospace, style = MaterialTheme.typography.bodySmall)

        Button(onClick = { vm.discard() }, modifier = Modifier.fillMaxWidth()) { Text("New recording") }
    }
}

@Composable
private fun ClarificationRow(field: String, prompt: String, options: List<String>, kind: ClarificationKind, vm: MainViewModel) {
    var text by remember(field, prompt) { mutableStateOf("") }
    Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
        Text(prompt, fontWeight = FontWeight.Medium)
        if (options.isNotEmpty()) FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            options.forEach { o -> FilterChip(selected = false, onClick = { if (kind == ClarificationKind.FIELD_LOW_CONFIDENCE && o == options.first()) vm.confirmField(field) else vm.provideField(field, o) }, label = { Text(o) }) }
        }
        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            OutlinedTextField(value = text, onValueChange = { text = it }, modifier = Modifier.weight(1f), singleLine = true, label = { Text("Type a value") })
            Button(onClick = { if (text.isNotBlank()) vm.provideField(field, text) }, enabled = text.isNotBlank()) { Text("Save") }
        }
        HorizontalDivider()
    }
}
