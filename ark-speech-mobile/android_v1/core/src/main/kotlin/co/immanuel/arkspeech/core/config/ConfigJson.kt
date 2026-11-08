package co.immanuel.arkspeech.core.config

import kotlinx.serialization.ExperimentalSerializationApi
import kotlinx.serialization.json.Json

/** The one JSON configuration used for config files and the output contract. */
@OptIn(ExperimentalSerializationApi::class)
object ConfigJson {
    val pretty: Json = Json {
        prettyPrint = true
        prettyPrintIndent = "  "
        encodeDefaults = true
        ignoreUnknownKeys = true
        explicitNulls = false
        isLenient = true
    }
    val compact: Json = Json {
        encodeDefaults = true
        ignoreUnknownKeys = true
        explicitNulls = false
        isLenient = true
    }
}
