package co.immanuel.arkspeech.model

import android.content.Context
import co.immanuel.arkspeech.core.config.ConfigJson
import co.immanuel.arkspeech.core.model.ModelCatalog
import co.immanuel.arkspeech.core.model.ModelKind
import co.immanuel.arkspeech.core.model.ModelPack
import java.io.File
import java.security.MessageDigest

enum class InstallState { NOT_INSTALLED, PARTIAL, INSTALLED }

/**
 * Where model packs live on the device and whether they are complete. Packs are plain files under
 * the app-specific external files directory (`Android/data/<pkg>/files/models/<packId>/`), which
 * `DownloadManager` can write to and `adb push` can fill during development, and which Android
 * removes on uninstall. Falls back to internal storage when external storage is unavailable.
 */
class ModelStore(context: Context, val catalog: ModelCatalog) {

    /** Where downloads go and where `adb push` is documented to put files. */
    val root: File = (context.getExternalFilesDir(null)?.let { File(it, "models") } ?: File(context.filesDir, "models")).apply { mkdirs() }

    /** Internal storage is also honoured (e.g. `run-as` copies on emulator images that block adb push). */
    private val roots: List<File> = listOf(root, File(context.filesDir, "models")).distinct()

    /** The directory holding the pack: the first root where its files are complete, else the download root. */
    fun dir(pack: ModelPack): File = roots.map { File(it, pack.id) }.firstOrNull { d -> complete(pack, d) } ?: File(root, pack.id)

    private fun complete(pack: ModelPack, d: File): Boolean = pack.files.isNotEmpty() && pack.files.all { f -> File(d, f.name).let { it.isFile && it.length() > 0 && (f.sizeBytes <= 0 || it.length() == f.sizeBytes) } }

    fun file(pack: ModelPack, name: String): File = File(dir(pack), name)

    fun state(pack: ModelPack): InstallState {
        if (pack.files.isEmpty()) return InstallState.NOT_INSTALLED
        val present = pack.files.count { f -> file(pack, f.name).let { it.isFile && it.length() > 0 && (f.sizeBytes <= 0 || it.length() == f.sizeBytes) } }
        return when (present) { 0 -> InstallState.NOT_INSTALLED; pack.files.size -> InstallState.INSTALLED; else -> InstallState.PARTIAL }
    }

    fun isInstalled(pack: ModelPack): Boolean = state(pack) == InstallState.INSTALLED

    fun installed(kind: ModelKind): List<ModelPack> = catalog.byKind(kind).filter { isInstalled(it) }

    fun installedBytes(pack: ModelPack): Long = pack.files.sumOf { file(pack, it.name).takeIf { f -> f.isFile }?.length() ?: 0L }

    fun delete(pack: ModelPack): Boolean = roots.map { File(it, pack.id) }.filter { it.exists() }.all { it.deleteRecursively() }

    /** Full SHA-256 check of every file that declares one. Slow for large packs; run off the main thread. */
    fun verify(pack: ModelPack): Map<String, Boolean> = pack.files.associate { f ->
        val file = file(pack, f.name)
        val ok = when {
            !file.isFile -> false
            f.sha256 == null -> f.sizeBytes <= 0 || file.length() == f.sizeBytes
            else -> sha256(file).equals(f.sha256, ignoreCase = true)
        }
        f.name to ok
    }

    companion object {
        fun loadCatalog(context: Context): ModelCatalog =
            context.assets.open("models/catalog.json").bufferedReader().use { ConfigJson.pretty.decodeFromString(ModelCatalog.serializer(), it.readText()) }

        fun sha256(file: File): String {
            val md = MessageDigest.getInstance("SHA-256")
            file.inputStream().use { input ->
                val buf = ByteArray(1 shl 16)
                while (true) {
                    val n = input.read(buf); if (n < 0) break
                    md.update(buf, 0, n)
                }
            }
            return md.digest().joinToString("") { "%02x".format(it) }
        }
    }
}
