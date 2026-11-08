package co.immanuel.arkspeech.core.model

import kotlinx.serialization.Serializable

@Serializable
enum class ModelKind { VAD, LID, ASR }

@Serializable
data class ModelFile(
    /** File name inside the pack directory on the device. */
    val name: String,
    val url: String,
    val sha256: String? = null,
    val sizeBytes: Long = 0,
    /** When the download is a .tar.bz2, the path of this file inside the archive. */
    val archivePath: String? = null,
)

/**
 * One installable model pack. [engine] names the adapter that can load it (e.g. "sherpa-whisper",
 * "sherpa-omnilingual-ctc", "ort-anuvani-lid"); [params] carries engine-specific settings such as the
 * Whisper encoder/decoder file names. Adding a pack is a catalogue entry, not a code change, as long
 * as an adapter for its engine exists.
 */
@Serializable
data class ModelPack(
    val id: String,
    val kind: ModelKind,
    val engine: String,
    val displayName: String,
    val description: String = "",
    /** Language codes the pack handles; ["*"] means any. */
    val languages: List<String> = listOf("*"),
    val files: List<ModelFile> = emptyList(),
    /** Download archive (tar.bz2) when files are packed; null when files download individually. */
    val archiveUrl: String? = null,
    val archiveSha256: String? = null,
    val installedSizeBytes: Long = 0,
    val licence: String = "",
    val source: String = "",
    val params: Map<String, String> = emptyMap(),
    /** default = installed by the setup flow; optional = offered; heavy = for high-end devices only. */
    val tier: String = "optional",
    /** Approximate peak RAM while running, from measurements, for the device-fit check. */
    val peakRamMb: Int = 0,
    val notes: String = "",
) {
    fun supports(language: String): Boolean = "*" in languages || language in languages
}

@Serializable
data class ModelCatalog(val version: Int = 1, val packs: List<ModelPack> = emptyList()) {
    fun get(id: String): ModelPack? = packs.firstOrNull { it.id == id }
    fun byKind(kind: ModelKind): List<ModelPack> = packs.filter { it.kind == kind }
}
