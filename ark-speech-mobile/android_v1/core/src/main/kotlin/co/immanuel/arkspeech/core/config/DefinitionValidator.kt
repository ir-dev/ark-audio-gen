package co.immanuel.arkspeech.core.config

import co.immanuel.arkspeech.core.lang.Languages
import kotlinx.serialization.Serializable
import java.util.regex.PatternSyntaxException

@Serializable
data class ConfigError(val path: String, val message: String)

class InvalidDefinitionException(val errors: List<ConfigError>) :
    Exception("Invalid definition: " + errors.joinToString("; ") { "${it.path}: ${it.message}" })

/** Structural validation of a definition before it is saved or imported. */
object DefinitionValidator {

    private val idPattern = Regex("^[a-z][a-z0-9_]{1,63}$")
    private val fieldNamePattern = Regex("^[a-z][a-z0-9_]{0,63}$")

    fun validate(def: ExtractionDefinition): List<ConfigError> {
        val errors = ArrayList<ConfigError>()
        if (!idPattern.matches(def.id)) errors += ConfigError("id", "must be snake_case, 2-64 chars, starting with a letter")
        if (def.intent.name.isBlank()) errors += ConfigError("intent.name", "must not be blank")
        if (def.version < 1) errors += ConfigError("version", "must be >= 1")
        if (def.confidenceThreshold !in 0f..1f) errors += ConfigError("confidenceThreshold", "must be between 0 and 1")
        def.languages.forEach { if (!Languages.isSupported(it)) errors += ConfigError("languages", "unknown language code '$it'") }
        checkLangMap(def.triggers.phrases, "triggers.phrases", errors)
        checkLangMap(def.triggers.negativePhrases, "triggers.negativePhrases", errors)
        def.triggers.patterns.forEachIndexed { i, p -> checkRegex(p, "triggers.patterns[$i]", errors) }
        val hasEvidence = def.triggers.phrases.values.any { it.isNotEmpty() } || def.triggers.patterns.isNotEmpty()
        if (!hasEvidence) errors += ConfigError("triggers", "at least one trigger phrase or pattern is required")

        val names = HashSet<String>()
        def.fields.forEachIndexed { i, f ->
            val p = "fields[$i]"
            if (!fieldNamePattern.matches(f.name)) errors += ConfigError("$p.name", "must be snake_case")
            if (!names.add(f.name)) errors += ConfigError("$p.name", "duplicate field name '${f.name}'")
            f.validation.regex?.let { checkRegex(it, "$p.validation.regex", errors) }
            f.extraction.patterns.forEachIndexed { j, pat -> checkRegex(pat, "$p.extraction.patterns[$j]", errors) }
            checkLangMap(f.extraction.anchors, "$p.extraction.anchors", errors)
            checkLangMap(f.extraction.anchorsBefore, "$p.extraction.anchorsBefore", errors)
            if (f.type == FieldType.ENUM && f.validation.allowedValues.isEmpty()) {
                errors += ConfigError("$p.validation.allowedValues", "ENUM fields need at least one allowed value")
            }
            if (f.type != FieldType.ENUM && f.validation.allowedValues.isNotEmpty()) {
                errors += ConfigError("$p.validation.allowedValues", "allowedValues is only meaningful for ENUM fields")
            }
            f.validation.allowedValues.forEachIndexed { j, av ->
                if (av.value.isBlank()) errors += ConfigError("$p.validation.allowedValues[$j].value", "must not be blank")
                checkLangMap(av.synonyms, "$p.validation.allowedValues[$j].synonyms", errors)
            }
            if (f.validation.min != null && f.validation.max != null && f.validation.min > f.validation.max) {
                errors += ConfigError("$p.validation", "min is greater than max")
            }
            f.confidenceThreshold?.let { if (it !in 0f..1f) errors += ConfigError("$p.confidenceThreshold", "must be between 0 and 1") }
            if (f.extraction.maxTokens < 1) errors += ConfigError("$p.extraction.maxTokens", "must be >= 1")
        }
        def.examples.forEachIndexed { i, ex ->
            if (!Languages.isSupported(ex.language)) errors += ConfigError("examples[$i].language", "unknown language code '${ex.language}'")
            ex.expected.keys.forEach { k -> if (k !in names) errors += ConfigError("examples[$i].expected", "unknown field '$k'") }
        }
        return errors
    }

    fun validateBundle(bundle: ConfigBundle): List<ConfigError> {
        val errors = ArrayList<ConfigError>()
        if (bundle.schemaVersion.substringBefore('.') != CONFIG_SCHEMA_VERSION.substringBefore('.')) {
            errors += ConfigError("schemaVersion", "unsupported schema version '${bundle.schemaVersion}' (expected $CONFIG_SCHEMA_VERSION)")
        }
        val ids = HashSet<String>()
        bundle.definitions.forEachIndexed { i, d ->
            if (!ids.add(d.id)) errors += ConfigError("definitions[$i].id", "duplicate definition id '${d.id}'")
            errors += validate(d).map { ConfigError("definitions[$i].${it.path}", it.message) }
        }
        bundle.routing?.let { r ->
            if (r.defaultModelId.isBlank()) errors += ConfigError("routing.defaultModelId", "must not be blank")
            r.byLanguage.keys.forEach { if (!Languages.isSupported(it)) errors += ConfigError("routing.byLanguage", "unknown language code '$it'") }
        }
        return errors
    }

    private fun checkLangMap(map: Map<String, List<String>>, path: String, errors: MutableList<ConfigError>) {
        map.forEach { (lang, values) ->
            if (lang != "*" && !Languages.isSupported(lang)) errors += ConfigError(path, "unknown language code '$lang'")
            if (values.any { it.isBlank() }) errors += ConfigError("$path.$lang", "blank entry")
        }
    }

    private fun checkRegex(pattern: String, path: String, errors: MutableList<ConfigError>) {
        try {
            Regex(pattern)
        } catch (e: PatternSyntaxException) {
            errors += ConfigError(path, "invalid regular expression: ${e.description}")
        } catch (e: IllegalArgumentException) {
            errors += ConfigError(path, "invalid regular expression")
        }
    }
}
