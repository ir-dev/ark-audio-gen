package co.immanuel.arkspeech.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.AssistChip
import androidx.compose.material3.Card
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import co.immanuel.arkspeech.core.nlu.FieldStatus
import co.immanuel.arkspeech.core.pipeline.PipelineStatus

@Composable
fun SectionCard(title: String, modifier: Modifier = Modifier, content: @Composable () -> Unit) {
    Card(modifier = modifier.fillMaxWidth()) {
        Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Text(title, style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
            content()
        }
    }
}

@Composable
fun KeyValue(key: String, value: String) {
    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
        Text(key, style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
        Text(value, style = MaterialTheme.typography.bodyMedium, fontWeight = FontWeight.Medium)
    }
}

@Composable
fun StatusChip(status: FieldStatus) {
    val (label, color) = when (status) {
        FieldStatus.EXTRACTED -> "extracted" to Color(0xFF2E7D32)
        FieldStatus.USER_PROVIDED -> "confirmed" to Color(0xFF2E7D32)
        FieldStatus.DEFAULTED -> "default" to Color(0xFF546E7A)
        FieldStatus.LOW_CONFIDENCE -> "confirm?" to Color(0xFFEF6C00)
        FieldStatus.AMBIGUOUS -> "ambiguous" to Color(0xFFEF6C00)
        FieldStatus.MISSING -> "missing" to Color(0xFF9E9E9E)
        FieldStatus.INVALID -> "invalid" to Color(0xFFC62828)
    }
    AssistChip(onClick = {}, label = { Text(label) }, colors = androidx.compose.material3.AssistChipDefaults.assistChipColors(labelColor = color))
}

fun PipelineStatus.humanLabel(): String = when (this) {
    PipelineStatus.OK -> "Complete"
    PipelineStatus.NEEDS_CLARIFICATION -> "Needs your confirmation"
    PipelineStatus.AUDIO_REJECTED -> "Recording unusable"
    PipelineStatus.NO_SPEECH -> "No speech found"
    PipelineStatus.UNSUPPORTED_LANGUAGE -> "Language not supported"
    PipelineStatus.MODEL_UNAVAILABLE -> "No speech model installed"
    PipelineStatus.NO_INTENT -> "Could not tell what you meant"
    PipelineStatus.FAILED -> "Processing failed"
    PipelineStatus.IN_PROGRESS -> "Working…"
}

fun Long.asMmSs(): String = "%d:%02d".format(this / 60000, (this / 1000) % 60)

fun Long.asMb(): String = "%.1f MB".format(this / 1048576.0)
