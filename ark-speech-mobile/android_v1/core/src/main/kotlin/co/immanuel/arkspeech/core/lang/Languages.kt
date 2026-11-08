package co.immanuel.arkspeech.core.lang

import kotlinx.serialization.Serializable

/**
 * One supported language. [code] is the identifier used everywhere in this app and in the output
 * contract (ISO 639-1 where one exists, otherwise ISO 639-3). The engine-specific codes are what the
 * model adapters need to pass to their runtimes.
 */
@Serializable
data class Language(
    val code: String,
    val name: String,
    val nativeName: String,
    val script: String,
    /** Label used by CIIL Anuvani-LID (null when the model does not know the language). */
    val anuvaniLabel: String? = null,
    /** Whisper language code (null when Whisper has no token for it). */
    val whisperCode: String? = null,
    /** Omnilingual ASR `{lang}_{Script}` id. */
    val omnilingualCode: String? = null,
    /** Dolphin language code (null when Dolphin does not cover it). */
    val dolphinCode: String? = null,
)

/**
 * The 22 scheduled languages of India plus English: the label set of Anuvani-LID and the set this app
 * prioritises. Everything else is "unsupported" and handled gracefully (see LanguagePolicy).
 */
object Languages {
    val all: List<Language> = listOf(
        Language("as", "Assamese", "অসমীয়া", "Bengali", "assamese", "as", "asm_Beng", null),
        Language("bn", "Bengali", "বাংলা", "Bengali", "bengali", "bn", "ben_Beng", "bn"),
        Language("brx", "Bodo", "बड़ो", "Devanagari", "bodo", null, "brx_Deva", null),
        Language("doi", "Dogri", "डोगरी", "Devanagari", "dogri", null, "doi_Deva", null),
        Language("en", "English", "English", "Latin", "english", "en", "eng_Latn", null),
        Language("gu", "Gujarati", "ગુજરાતી", "Gujarati", "gujarati", "gu", "guj_Gujr", "gu"),
        Language("hi", "Hindi", "हिन्दी", "Devanagari", "hindi", "hi", "hin_Deva", "hi"),
        Language("kn", "Kannada", "ಕನ್ನಡ", "Kannada", "kannada", "kn", "kan_Knda", null),
        Language("ks", "Kashmiri", "کٲشُر", "Arabic", "kashmiri", null, "kas_Arab", "ks"),
        Language("kok", "Konkani", "कोंकणी", "Devanagari", "konkani", null, "gom_Deva", null),
        Language("mai", "Maithili", "मैथिली", "Devanagari", "maithili", null, "mai_Deva", null),
        Language("ml", "Malayalam", "മലയാളം", "Malayalam", "malayalam", "ml", "mal_Mlym", null),
        Language("mni", "Manipuri", "মৈতৈলোন্", "Bengali", "manipuri", null, "mni_Beng", null),
        Language("mr", "Marathi", "मराठी", "Devanagari", "marathi", "mr", "mar_Deva", "mr"),
        Language("ne", "Nepali", "नेपाली", "Devanagari", "nepali", "ne", "npi_Deva", "ne"),
        Language("or", "Odia", "ଓଡ଼ିଆ", "Odia", "odia", null, "ory_Orya", "or"),
        Language("pa", "Punjabi", "ਪੰਜਾਬੀ", "Gurmukhi", "punjabi", "pa", "pan_Guru", null),
        Language("sa", "Sanskrit", "संस्कृतम्", "Devanagari", "sanskrit", "sa", "san_Deva", null),
        Language("sat", "Santali", "ᱥᱟᱱᱛᱟᱲᱤ", "Ol Chiki", "santali", null, "sat_Olck", null),
        Language("sd", "Sindhi", "سنڌي", "Arabic", "sindhi", "sd", "snd_Arab", null),
        Language("ta", "Tamil", "தமிழ்", "Tamil", "tamil", "ta", "tam_Taml", "ta"),
        Language("te", "Telugu", "తెలుగు", "Telugu", "telugu", "te", "tel_Telu", "te"),
        Language("ur", "Urdu", "اردو", "Arabic", "urdu", "ur", "urd_Arab", "ur"),
    )

    private val byCode = all.associateBy { it.code }
    private val byAnuvani = all.filter { it.anuvaniLabel != null }.associateBy { it.anuvaniLabel!! }

    val codes: Set<String> = byCode.keys

    fun get(code: String): Language? = byCode[code]

    fun require(code: String): Language = byCode[code] ?: throw IllegalArgumentException("Unknown language code: $code")

    fun fromAnuvaniLabel(label: String): Language? = byAnuvani[label.lowercase()]

    fun fromWhisperCode(code: String): Language? = all.firstOrNull { it.whisperCode == code }

    fun isSupported(code: String): Boolean = code in byCode

    fun displayName(code: String): String = byCode[code]?.name ?: code

    /** Dominant Unicode script of a text, by letter count. Used to check transcripts against the language. */
    fun dominantScript(text: String): String? {
        val counts = HashMap<String, Int>()
        for (ch in text) {
            if (!Character.isLetter(ch)) continue
            val block = Character.UnicodeScript.of(ch.code)
            val name = when (block) {
                Character.UnicodeScript.LATIN -> "Latin"
                Character.UnicodeScript.DEVANAGARI -> "Devanagari"
                Character.UnicodeScript.BENGALI -> "Bengali"
                Character.UnicodeScript.GURMUKHI -> "Gurmukhi"
                Character.UnicodeScript.GUJARATI -> "Gujarati"
                Character.UnicodeScript.ORIYA -> "Odia"
                Character.UnicodeScript.TAMIL -> "Tamil"
                Character.UnicodeScript.TELUGU -> "Telugu"
                Character.UnicodeScript.KANNADA -> "Kannada"
                Character.UnicodeScript.MALAYALAM -> "Malayalam"
                Character.UnicodeScript.ARABIC -> "Arabic"
                Character.UnicodeScript.OL_CHIKI -> "Ol Chiki"
                Character.UnicodeScript.MEETEI_MAYEK -> "Meetei Mayek"
                else -> block.name
            }
            counts[name] = (counts[name] ?: 0) + 1
        }
        return counts.maxByOrNull { it.value }?.key
    }
}
