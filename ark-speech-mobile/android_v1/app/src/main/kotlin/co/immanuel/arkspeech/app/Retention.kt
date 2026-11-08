package co.immanuel.arkspeech.app

import android.content.Context
import co.immanuel.arkspeech.core.audio.PcmAudio
import co.immanuel.arkspeech.core.audio.WavCodec
import co.immanuel.arkspeech.core.pipeline.PipelineResult
import java.io.File

/**
 * The only place recordings or transcripts are ever written. Nothing is written unless the user
 * consented for that interaction; files stay in app-private storage and can be deleted from Settings.
 */
class Retention(context: Context) {
    private val dir = File(context.filesDir, "retained").apply { mkdirs() }

    data class Item(val requestId: String, val audio: File?, val result: File?) {
        val sizeBytes: Long get() = (audio?.length() ?: 0L) + (result?.length() ?: 0L)
    }

    fun saveAudio(requestId: String, audio: PcmAudio): File = File(dir, "$requestId.wav").also { it.writeBytes(WavCodec.encode(audio)) }

    fun saveResult(result: PipelineResult): File = File(dir, "${result.requestId}.json").also { it.writeText(result.toJson()) }

    fun list(): List<Item> {
        val files = dir.listFiles() ?: return emptyList()
        return files.groupBy { it.nameWithoutExtension }.map { (id, fs) -> Item(id, fs.firstOrNull { it.extension == "wav" }, fs.firstOrNull { it.extension == "json" }) }
            .sortedByDescending { (it.audio ?: it.result)?.lastModified() ?: 0L }
    }

    fun delete(requestId: String) { File(dir, "$requestId.wav").delete(); File(dir, "$requestId.json").delete() }

    fun deleteAll() { dir.listFiles()?.forEach { it.delete() } }

    fun totalBytes(): Long = dir.listFiles()?.sumOf { it.length() } ?: 0L
}
