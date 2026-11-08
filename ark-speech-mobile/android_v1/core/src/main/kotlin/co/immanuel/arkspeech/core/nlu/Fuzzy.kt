package co.immanuel.arkspeech.core.nlu

import kotlin.math.abs
import kotlin.math.max
import kotlin.math.roundToInt

/**
 * Bounded approximate matching for ASR misspellings ("अपौइंतमेंत" for "अपॉइंटमेंट", "பிறச்சனை" for
 * "பிரச்சனை"). Only words of at least [MIN_CHARS] code units take part, at most 25 % of the length may
 * differ, and callers give fuzzy hits a weight that cannot clear an acceptance threshold on its own:
 * a fuzzy match can make the app *ask*, never decide.
 */
object Fuzzy {
    const val MIN_CHARS = 5

    /** At most 20 % of the longer word may differ (after folding), and never less than one edit. */
    fun maxEdits(length: Int): Int = max(1, (length * 0.2).toInt())

    fun similar(a: String, b: String): Boolean {
        if (a.length < MIN_CHARS || b.length < MIN_CHARS) return false
        val fa = fold(a); val fb = fold(b)
        if (fa == fb) return true
        val limit = maxEdits(max(fa.length, fb.length))
        if (abs(fa.length - fb.length) > limit) return false
        return levenshtein(fa, fb, limit) <= limit
    }

    /**
     * Folds characters that speech recognisers confuse within a script: retroflex/dental pairs, long/short
     * vowel signs, sibilants, nukta and candrabindu. Used only for comparison, never shown to the user.
     */
    fun fold(s: String): String {
        val sb = StringBuilder(s.length)
        for (ch in s) {
            val f = FOLD[ch]
            if (f == null) sb.append(ch) else if (f != '\u0000') sb.append(f)
        }
        return sb.toString()
    }

    private val FOLD: Map<Char, Char> = buildMap {
        // Devanagari (hi, mr, ne, kok, mai, doi, brx, sa)
        put('ट', 'त'); put('ठ', 'थ'); put('ड', 'द'); put('ढ', 'ध'); put('ण', 'न'); put('ळ', 'ल'); put('ष', 'श'); put('ॉ', 'ौ'); put('ऑ', 'औ')
        put('ॅ', 'ै'); put('ऍ', 'ऐ'); put('ी', 'ि'); put('ू', 'ु'); put('ँ', 'ं'); put('\u093C', '\u0000')
        // Tamil
        put('ற', 'ர'); put('ன', 'ந'); put('ண', 'ந'); put('ள', 'ல'); put('ழ', 'ல'); put('ஸ', 'ச'); put('ஷ', 'ச'); put('ஜ', 'ச'); put('ஹ', 'க'); put('ீ', 'ி'); put('ூ', 'ு')
        // Bengali / Assamese / Manipuri
        put('ণ', 'ন'); put('ষ', 'শ'); put('স', 'শ'); put('\u09DC', 'ড'); put('\u09DD', 'ঢ'); put('\u09DF', 'য'); put('ী', 'ি'); put('ূ', 'ু'); put('\u09BC', '\u0000')
        // Telugu
        put('ణ', 'న'); put('ళ', 'ల'); put('శ', 'స'); put('ష', 'స'); put('ీ', 'ి'); put('ూ', 'ు'); put('ఠ', 'ట'); put('ఢ', 'డ'); put('థ', 'త'); put('ధ', 'ద')
        // Kannada
        put('ಣ', 'ನ'); put('ಳ', 'ಲ'); put('ಶ', 'ಸ'); put('ಷ', 'ಸ'); put('ೀ', 'ಿ'); put('ೂ', 'ು'); put('ಠ', 'ಟ'); put('ಢ', 'ಡ'); put('ಥ', 'ತ'); put('ಧ', 'ದ')
        // Gujarati
        put('ટ', 'ત'); put('ઠ', 'થ'); put('ડ', 'દ'); put('ઢ', 'ધ'); put('ણ', 'ન'); put('ષ', 'શ'); put('ી', 'િ'); put('ૂ', 'ુ')
        // Gurmukhi
        put('ਟ', 'ਤ'); put('ਠ', 'ਥ'); put('ਡ', 'ਦ'); put('ਢ', 'ਧ'); put('ਣ', 'ਨ'); put('ੀ', 'ਿ'); put('ੂ', 'ੁ')
        // Malayalam
        put('ണ', 'ന'); put('ള', 'ല'); put('ഴ', 'ല'); put('റ', 'ര'); put('ശ', 'സ'); put('ഷ', 'സ'); put('ീ', 'ി'); put('ൂ', 'ു')
        // Odia
        put('ଟ', 'ତ'); put('ଠ', 'ଥ'); put('ଡ', 'ଦ'); put('ଢ', 'ଧ'); put('ଣ', 'ନ'); put('ଷ', 'ଶ'); put('ସ', 'ଶ'); put('ୀ', 'ି'); put('ୂ', 'ୁ')
        // Arabic script (ur, ks, sd): common letter variants
        put('ي', 'ی'); put('ك', 'ک'); put('ہ', 'ه'); put('ۀ', 'ه'); put('ة', 'ه')
    }

    /** Levenshtein distance with early exit once every cell in a row exceeds [limit]. */
    fun levenshtein(a: String, b: String, limit: Int = Int.MAX_VALUE): Int {
        if (a == b) return 0
        if (a.isEmpty()) return b.length
        if (b.isEmpty()) return a.length
        var prev = IntArray(b.length + 1) { it }
        var cur = IntArray(b.length + 1)
        for (i in 1..a.length) {
            cur[0] = i
            var rowMin = cur[0]
            for (j in 1..b.length) {
                val cost = if (a[i - 1] == b[j - 1]) 0 else 1
                cur[j] = minOf(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
                if (cur[j] < rowMin) rowMin = cur[j]
            }
            if (rowMin > limit) return rowMin
            val t = prev; prev = cur; cur = t
        }
        return prev[b.length]
    }

    /** First word of [text] within edit distance of [word], or null. */
    fun findWord(text: NormalizedText, word: String): Token? = text.words.firstOrNull { similar(it.lower, word) }
}
