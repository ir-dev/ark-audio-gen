package co.immanuel.arkspeech.model

import android.app.DownloadManager
import android.content.Context
import android.net.Uri
import co.immanuel.arkspeech.core.model.ModelPack
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.withContext
import java.io.File

data class DownloadProgress(val packId: String, val downloadedBytes: Long, val totalBytes: Long, val running: Boolean, val failed: Boolean, val reason: String? = null) {
    val fraction: Float get() = if (totalBytes <= 0) 0f else (downloadedBytes.toFloat() / totalBytes).coerceIn(0f, 1f)
}

/**
 * Downloads model packs with the system `DownloadManager`: resumable, survives process death, honours
 * the Wi-Fi-only choice, and needs no foreground service of our own. Files land directly in the
 * [ModelStore] directory. Model downloads are the only network use in the app, and only on request.
 */
class ModelDownloader(private val context: Context, private val store: ModelStore) {

    private val dm = context.getSystemService(Context.DOWNLOAD_SERVICE) as DownloadManager
    private val prefs = context.getSharedPreferences("model_downloads", Context.MODE_PRIVATE)

    fun enqueue(pack: ModelPack, wifiOnly: Boolean) {
        store.dir(pack).mkdirs()
        val ids = ArrayList<Long>()
        for (f in pack.files) {
            val target = store.file(pack, f.name)
            if (target.isFile && (f.sizeBytes <= 0 || target.length() == f.sizeBytes)) continue
            target.delete()
            val req = DownloadManager.Request(Uri.parse(f.url))
                .setTitle("${pack.displayName}: ${f.name}")
                .setDescription("Ark Speech model pack")
                .setNotificationVisibility(DownloadManager.Request.VISIBILITY_VISIBLE)
                .setAllowedOverMetered(!wifiOnly)
                .setAllowedOverRoaming(false)
                .setDestinationInExternalFilesDir(context, null, "models/${pack.id}/${f.name}")
            ids += dm.enqueue(req)
        }
        prefs.edit().putString(pack.id, ids.joinToString(",")).apply()
    }

    fun cancel(pack: ModelPack) {
        ids(pack).forEach { dm.remove(it) }
        prefs.edit().remove(pack.id).apply()
    }

    fun isActive(pack: ModelPack): Boolean = progress(pack)?.running == true

    fun progress(pack: ModelPack): DownloadProgress? {
        val ids = ids(pack)
        if (ids.isEmpty()) return null
        var done = 0L; var total = 0L; var running = false; var failed = false; var reason: String? = null
        dm.query(DownloadManager.Query().setFilterById(*ids.toLongArray())).use { c ->
            val iStatus = c.getColumnIndex(DownloadManager.COLUMN_STATUS)
            val iDone = c.getColumnIndex(DownloadManager.COLUMN_BYTES_DOWNLOADED_SO_FAR)
            val iTotal = c.getColumnIndex(DownloadManager.COLUMN_TOTAL_SIZE_BYTES)
            val iReason = c.getColumnIndex(DownloadManager.COLUMN_REASON)
            while (c.moveToNext()) {
                done += c.getLong(iDone).coerceAtLeast(0)
                total += c.getLong(iTotal).coerceAtLeast(0)
                when (c.getInt(iStatus)) {
                    DownloadManager.STATUS_RUNNING, DownloadManager.STATUS_PENDING, DownloadManager.STATUS_PAUSED -> running = true
                    DownloadManager.STATUS_FAILED -> { failed = true; reason = "error ${c.getInt(iReason)}" }
                }
            }
        }
        if (!running) {
            // Finished (or failed): forget the ids so the store's own file check becomes the truth.
            if (!failed || store.isInstalled(pack)) prefs.edit().remove(pack.id).apply()
        }
        val expected = pack.files.sumOf { it.sizeBytes }.takeIf { it > 0 } ?: total
        return DownloadProgress(pack.id, done, expected, running, failed, reason)
    }

    /** Emits progress for every active pack once a second until none is running. */
    fun watch(packs: List<ModelPack>): Flow<Map<String, DownloadProgress>> = flow {
        while (true) {
            val snapshot = packs.mapNotNull { p -> progress(p)?.let { p.id to it } }.toMap()
            emit(snapshot)
            if (snapshot.values.none { it.running }) break
            kotlinx.coroutines.delay(1000)
        }
    }

    private fun ids(pack: ModelPack): List<Long> = prefs.getString(pack.id, null)?.split(',')?.mapNotNull { it.toLongOrNull() } ?: emptyList()

    /** Computes checksums off the main thread after a download completes. */
    suspend fun verify(pack: ModelPack): Map<String, Boolean> = withContext(Dispatchers.IO) { store.verify(pack) }

    companion object {
        fun removePartial(dir: File) { dir.listFiles()?.filter { it.length() == 0L }?.forEach { it.delete() } }
    }
}
