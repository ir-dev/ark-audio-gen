package co.immanuel.arkspeech.core.nlu

import java.text.Normalizer

/** One token of the transcript with its span in [NormalizedText.text]. */
data class Token(val text: String, val lower: String, val start: Int, val end: Int, val isPunct: Boolean)

/**
 * A transcript prepared for matching. [text] keeps the original case and script (so extracted values
 * read naturally) but is NFC-normalised with native digits converted to ASCII; [lower] is its
 * lower-cased twin used for phrase and pattern matching.
 */
class NormalizedText(val original: String) {
    val text: String = TextNormalizer.canonical(original)
    val lower: String = text.lowercase()
    val tokens: List<Token> = TextNormalizer.tokenize(text)
    val words: List<Token> get() = tokens.filter { !it.isPunct }

    /** Token-boundary-aware search for a phrase (already normalised by [TextNormalizer.phrase]). */
    fun indexOfPhrase(phrase: String, from: Int = 0): Int {
        if (phrase.isEmpty()) return -1
        var idx = lower.indexOf(phrase, from)
        while (idx >= 0) {
            if (TextNormalizer.isBoundary(lower, idx - 1) && TextNormalizer.isBoundary(lower, idx + phrase.length)) return idx
            idx = lower.indexOf(phrase, idx + 1)
        }
        return -1
    }

    fun containsPhrase(phrase: String): Boolean = indexOfPhrase(phrase) >= 0

    /** Index of the first token starting at or after [charIndex]. */
    fun tokenIndexAt(charIndex: Int): Int = tokens.indexOfFirst { it.start >= charIndex }
}

object TextNormalizer {

    /** NFC + native digits to ASCII + whitespace collapse. Case and punctuation are preserved. */
    fun canonical(s: String): String {
        val nfc = Normalizer.normalize(s, Normalizer.Form.NFC)
        val sb = StringBuilder(nfc.length)
        var lastSpace = false
        for (ch in nfc) {
            val c = asciiDigit(ch) ?: ch
            if (c.isWhitespace()) {
                if (!lastSpace) sb.append(' ')
                lastSpace = true
            } else {
                sb.append(c)
                lastSpace = false
            }
        }
        return sb.toString().trim()
    }

    /** Normalise a configured phrase the same way the text is matched: canonical, lower, no punctuation. */
    fun phrase(s: String): String = canonical(s).lowercase().map { if (isWordChar(it)) it else ' ' }.joinToString("")
        .replace(Regex("\\s+"), " ").trim()

    fun tokenize(text: String): List<Token> {
        val out = ArrayList<Token>()
        var i = 0
        while (i < text.length) {
            val c = text[i]
            when {
                c.isWhitespace() -> i++
                isWordChar(c) -> {
                    val start = i
                    while (i < text.length && isWordChar(text[i])) i++
                    val t = text.substring(start, i)
                    out += Token(t, t.lowercase(), start, i, false)
                }
                else -> {
                    out += Token(c.toString(), c.toString(), i, i + 1, true)
                    i++
                }
            }
        }
        return out
    }

    /** Letters, digits and combining marks (Indic vowel signs, nukta, virama) belong to words. */
    fun isWordChar(c: Char): Boolean {
        if (Character.isLetterOrDigit(c)) return true
        return when (Character.getType(c).toByte()) {
            Character.NON_SPACING_MARK, Character.COMBINING_SPACING_MARK, Character.ENCLOSING_MARK -> true
            else -> c == '\'' || c == '‍' || c == '‌' // ZWJ / ZWNJ inside Indic words
        }
    }

    fun isBoundary(s: String, idx: Int): Boolean = idx < 0 || idx >= s.length || !isWordChar(s[idx])

    /** Maps a digit in any Indic script (and Arabic-Indic for Urdu) to its ASCII digit. */
    fun asciiDigit(c: Char): Char? {
        val code = c.code
        for (zero in DIGIT_ZEROS) {
            if (code in zero..zero + 9) return ('0' + (code - zero))
        }
        return null
    }

    private val DIGIT_ZEROS = intArrayOf(
        0x0966, // Devanagari
        0x09E6, // Bengali
        0x0A66, // Gurmukhi
        0x0AE6, // Gujarati
        0x0B66, // Odia
        0x0BE6, // Tamil
        0x0C66, // Telugu
        0x0CE6, // Kannada
        0x0D66, // Malayalam
        0x0660, // Arabic-Indic
        0x06F0, // Extended Arabic-Indic (Urdu)
        0x1C50, // Ol Chiki
        0xABF0, // Meetei Mayek
    )
}
