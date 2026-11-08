package co.immanuel.arkspeech.core.asr

import kotlinx.serialization.Serializable

/**
 * Which ASR model pack to use for which language. Administrators edit this (it is part of the
 * configuration bundle); the default sends everything to one multilingual model and lets a deployment
 * add a per-language override, e.g. "hi" -> an IndicConformer Hindi pack.
 */
@Serializable
data class RoutingTable(
    val defaultModelId: String,
    val byLanguage: Map<String, String> = emptyMap(),
    /** Tried in order when the chosen model is not installed. */
    val fallbackModelIds: List<String> = emptyList(),
) {
    fun candidates(language: String?): List<String> {
        val out = ArrayList<String>()
        if (language != null) byLanguage[language]?.let { out += it }
        out += defaultModelId
        out += fallbackModelIds
        return out.distinct()
    }
}

class NoModelAvailableException(val language: String?, val tried: List<String>) :
    Exception("No installed ASR model for language '${language ?: "unknown"}'; tried $tried")

/** Resolves a language to the first installed model pack in the routing order. */
class ModelRouter(private val table: RoutingTable, private val factory: RecognizerFactory) {
    fun resolve(language: String?): String {
        val installed = factory.availableModelIds()
        val tried = table.candidates(language)
        return tried.firstOrNull { it in installed } ?: throw NoModelAvailableException(language, tried)
    }
}
