package co.immanuel.arkspeech.core.config

import co.immanuel.arkspeech.core.asr.RoutingTable
import co.immanuel.arkspeech.core.audio.AudioValidationPolicy
import co.immanuel.arkspeech.core.lang.LanguagePolicy
import co.immanuel.arkspeech.core.util.Clock
import kotlinx.serialization.Serializable
import java.io.File
import java.time.format.DateTimeFormatter

/** Summary line for the administrator's list and version history. */
@Serializable
data class DefinitionVersionInfo(
    val id: String,
    val version: Int,
    val intentName: String,
    val enabled: Boolean,
    val updatedAt: String,
    val updatedBy: String,
    val changeNote: String,
)

/**
 * Storage for extraction definitions with per-definition version history. Saving produces a new,
 * immutable version; enabling/disabling and rollback are themselves recorded as versions so every
 * change the administrator makes is attributable and reversible.
 */
interface DefinitionRepository {
    /** The latest version of every definition, enabled or not. */
    fun list(): List<ExtractionDefinition>
    fun get(id: String): ExtractionDefinition?
    fun getVersion(id: String, version: Int): ExtractionDefinition?
    fun history(id: String): List<DefinitionVersionInfo>

    /** Validates and stores [definition] as the next version of its id. Returns the stored copy. */
    fun save(definition: ExtractionDefinition, author: String = "", note: String = ""): ExtractionDefinition
    fun setEnabled(id: String, enabled: Boolean, author: String = ""): ExtractionDefinition?
    fun rollback(id: String, toVersion: Int, author: String = ""): ExtractionDefinition?
    fun delete(id: String): Boolean

    fun routing(): RoutingTable?
    fun saveRouting(routing: RoutingTable)
    fun languagePolicy(): LanguagePolicy?
    fun saveLanguagePolicy(policy: LanguagePolicy)
    fun audioPolicy(): AudioValidationPolicy?
    fun saveAudioPolicy(policy: AudioValidationPolicy)

    /** Enabled definitions that apply to [language] — what the pipeline actually uses. */
    fun activeFor(language: String?): List<ExtractionDefinition> =
        list().filter { it.enabled && it.appliesTo(language) }

    fun exportBundle(note: String = ""): ConfigBundle
    /** Imports every definition in the bundle as a new version. [replace] removes definitions not in the bundle. */
    fun importBundle(bundle: ConfigBundle, author: String = "", replace: Boolean = false): ImportReport
}

@Serializable
data class ImportReport(val imported: List<String>, val removed: List<String>, val errors: List<ConfigError>) {
    val ok: Boolean get() = errors.isEmpty()
}

/**
 * JSON-files-on-disk implementation. Layout under [root]:
 *
 *     definitions/<id>/v<N>.json     one file per version
 *     definitions/<id>/current       the active version number
 *     routing.json, language-policy.json, audio-policy.json
 *
 * Plain files keep the mechanism inspectable, trivially backed up and importable without the app.
 */
class FileDefinitionRepository(private val root: File, private val clock: Clock = Clock.System) : DefinitionRepository {

    private val json = ConfigJson.pretty
    private val defsDir = File(root, "definitions")
    private val lock = Any()

    init {
        defsDir.mkdirs()
    }

    override fun list(): List<ExtractionDefinition> = synchronized(lock) {
        (defsDir.listFiles { f -> f.isDirectory } ?: emptyArray())
            .mapNotNull { dir -> currentVersion(dir.name)?.let { readVersion(dir.name, it) } }
            .sortedBy { it.id }
    }

    override fun get(id: String): ExtractionDefinition? = synchronized(lock) {
        currentVersion(id)?.let { readVersion(id, it) }
    }

    override fun getVersion(id: String, version: Int): ExtractionDefinition? = synchronized(lock) { readVersion(id, version) }

    override fun history(id: String): List<DefinitionVersionInfo> = synchronized(lock) {
        val dir = File(defsDir, id)
        (dir.listFiles { f -> f.name.startsWith("v") && f.name.endsWith(".json") } ?: emptyArray())
            .mapNotNull { f -> f.name.removePrefix("v").removeSuffix(".json").toIntOrNull()?.let { readVersion(id, it) } }
            .sortedByDescending { it.version }
            .map { DefinitionVersionInfo(it.id, it.version, it.intent.name, it.enabled, it.updatedAt, it.updatedBy, it.changeNote) }
    }

    override fun save(definition: ExtractionDefinition, author: String, note: String): ExtractionDefinition = synchronized(lock) {
        val errors = DefinitionValidator.validate(definition)
        if (errors.isNotEmpty()) throw InvalidDefinitionException(errors)
        val next = (latestVersion(definition.id) ?: 0) + 1
        val stored = definition.copy(
            version = next,
            updatedAt = DateTimeFormatter.ISO_INSTANT.format(clock.now()),
            updatedBy = author.ifBlank { definition.updatedBy },
            changeNote = note.ifBlank { definition.changeNote },
        )
        writeVersion(stored)
        setCurrent(stored.id, next)
        stored
    }

    override fun setEnabled(id: String, enabled: Boolean, author: String): ExtractionDefinition? = synchronized(lock) {
        val cur = get(id) ?: return null
        if (cur.enabled == enabled) return cur
        save(cur.copy(enabled = enabled), author, if (enabled) "Enabled" else "Disabled")
    }

    override fun rollback(id: String, toVersion: Int, author: String): ExtractionDefinition? = synchronized(lock) {
        val target = readVersion(id, toVersion) ?: return null
        save(target, author, "Rolled back to version $toVersion")
    }

    override fun delete(id: String): Boolean = synchronized(lock) { File(defsDir, id).deleteRecursively() }

    override fun routing(): RoutingTable? = readSingle("routing.json")
    override fun saveRouting(routing: RoutingTable) = writeSingle("routing.json", json.encodeToString(RoutingTable.serializer(), routing))
    override fun languagePolicy(): LanguagePolicy? = readSingle("language-policy.json")
    override fun saveLanguagePolicy(policy: LanguagePolicy) = writeSingle("language-policy.json", json.encodeToString(LanguagePolicy.serializer(), policy))
    override fun audioPolicy(): AudioValidationPolicy? = readSingle("audio-policy.json")
    override fun saveAudioPolicy(policy: AudioValidationPolicy) = writeSingle("audio-policy.json", json.encodeToString(AudioValidationPolicy.serializer(), policy))

    override fun exportBundle(note: String): ConfigBundle = ConfigBundle(
        definitions = list(),
        routing = routing(),
        languagePolicy = languagePolicy(),
        audioPolicy = audioPolicy(),
        exportedAt = DateTimeFormatter.ISO_INSTANT.format(clock.now()),
        note = note,
    )

    override fun importBundle(bundle: ConfigBundle, author: String, replace: Boolean): ImportReport = synchronized(lock) {
        val errors = DefinitionValidator.validateBundle(bundle)
        if (errors.isNotEmpty()) return ImportReport(emptyList(), emptyList(), errors)
        val imported = bundle.definitions.map { save(it, author, "Imported").id }
        val removed = if (replace) list().map { it.id }.filter { it !in imported }.onEach { delete(it) } else emptyList()
        bundle.routing?.let { saveRouting(it) }
        bundle.languagePolicy?.let { saveLanguagePolicy(it) }
        bundle.audioPolicy?.let { saveAudioPolicy(it) }
        ImportReport(imported, removed, emptyList())
    }

    // --- files ---

    private fun currentVersion(id: String): Int? = File(defsDir, "$id/current").takeIf { it.exists() }?.readText()?.trim()?.toIntOrNull()

    private fun latestVersion(id: String): Int? = (File(defsDir, id).listFiles() ?: emptyArray())
        .mapNotNull { it.name.removePrefix("v").removeSuffix(".json").toIntOrNull().takeIf { _ -> it.name.startsWith("v") } }
        .maxOrNull()

    private fun readVersion(id: String, version: Int): ExtractionDefinition? {
        val f = File(defsDir, "$id/v$version.json")
        if (!f.exists()) return null
        return runCatching { json.decodeFromString(ExtractionDefinition.serializer(), f.readText()) }.getOrNull()
    }

    private fun writeVersion(def: ExtractionDefinition) {
        val dir = File(defsDir, def.id).apply { mkdirs() }
        atomicWrite(File(dir, "v${def.version}.json"), json.encodeToString(ExtractionDefinition.serializer(), def))
    }

    private fun setCurrent(id: String, version: Int) = atomicWrite(File(defsDir, "$id/current"), version.toString())

    private inline fun <reified T> readSingle(name: String): T? {
        val f = File(root, name)
        if (!f.exists()) return null
        return runCatching { json.decodeFromString<T>(f.readText()) }.getOrNull()
    }

    private fun writeSingle(name: String, content: String) = synchronized(lock) { atomicWrite(File(root, name), content) }

    private fun atomicWrite(target: File, content: String) {
        val tmp = File(target.parentFile, target.name + ".tmp")
        tmp.writeText(content)
        if (!tmp.renameTo(target)) {
            target.delete()
            if (!tmp.renameTo(target)) throw java.io.IOException("Could not write ${target.name}")
        }
    }
}
