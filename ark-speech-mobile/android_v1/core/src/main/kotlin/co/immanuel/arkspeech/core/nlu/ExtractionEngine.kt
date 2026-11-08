package co.immanuel.arkspeech.core.nlu

import co.immanuel.arkspeech.core.config.ExtractionDefinition
import co.immanuel.arkspeech.core.config.FieldDefinition
import co.immanuel.arkspeech.core.config.FieldType
import co.immanuel.arkspeech.core.config.TemporalDirection
import co.immanuel.arkspeech.core.util.Clock
import kotlinx.serialization.Serializable
import java.time.LocalDate
import java.time.temporal.ChronoUnit
import kotlin.math.abs

@Serializable
enum class FieldStatus {
    /** A single confident, valid value. */
    EXTRACTED,
    /** Nothing found. */
    MISSING,
    /** One value found but below the confidence threshold: show it and ask for confirmation. */
    LOW_CONFIDENCE,
    /** Several distinct values found with similar confidence: the user must choose. */
    AMBIGUOUS,
    /** A value was found but failed the field's validation rules. */
    INVALID,
    /** The administrator's explicit default was applied because nothing was found. */
    DEFAULTED,
    /** Entered or confirmed by the user during clarification. */
    USER_PROVIDED,
}

@Serializable
data class FieldResult(
    val name: String,
    val type: FieldType,
    val required: Boolean,
    val status: FieldStatus,
    val value: String? = null,
    val raw: String? = null,
    val confidence: Float = 0f,
    val candidates: List<Candidate> = emptyList(),
    val reason: String? = null,
    val extra: Map<String, String> = emptyMap(),
) {
    val needsUser: Boolean
        get() = when (status) {
            FieldStatus.MISSING -> required
            FieldStatus.LOW_CONFIDENCE, FieldStatus.AMBIGUOUS, FieldStatus.INVALID -> true
            else -> false
        }
}

@Serializable
data class ExtractionResult(
    val definitionId: String,
    val definitionVersion: Int,
    val fields: Map<String, FieldResult>,
    val engine: String = "rules-v1",
) {
    val missingRequired: List<String> get() = fields.values.filter { it.required && it.status == FieldStatus.MISSING }.map { it.name }
    val needsClarification: Boolean get() = fields.values.any { it.needsUser }
}

/**
 * Rule-driven entity extraction over a definition's fields. For every field it gathers candidates
 * from (1) the administrator's regex patterns, (2) anchor phrases followed by a value, (3) the type's
 * own scanner (dates, times, phones, amounts, numbers, durations, enum synonyms), then decides:
 * one confident value → EXTRACTED; competing values → AMBIGUOUS; weak evidence → LOW_CONFIDENCE;
 * nothing → MISSING. Free-text STRING fields are only ever filled from patterns or anchors, so the
 * engine never invents a name or a place. Deterministic and fully testable on the JVM.
 */
class ExtractionEngine(private val clock: Clock = Clock.System, private val defaultThreshold: Float = 0.6f, private val ambiguityMargin: Float = 0.25f) {

    val engineId = "rules-v1"

    fun extract(text: NormalizedText, def: ExtractionDefinition, language: String?): ExtractionResult {
        val today = clock.today()
        val claimed = ArrayList<IntRange>()
        val results = LinkedHashMap<String, FieldResult>()
        // Words that express the intent itself ("appointment", "book") never belong to a name or place.
        val triggerWords = (def.triggers.phrases.values.flatten()).flatMap { TextNormalizer.phrase(it).split(' ') }.filter { it.length > 2 }.toSet()
        // Specific types first so their spans are not re-used by generic number fields.
        val ordered = def.fields.sortedBy { typeOrder(it.type) }
        for (f in ordered) {
            val candidates = candidatesFor(text, f, language, today, claimed, triggerWords)
            val result = decide(f, candidates, today)
            results[f.name] = result
            spanOf(result)?.let { claimed += it }
        }
        // Second pass: an ambiguous choice often has one option that is really part of another field
        // ("Road" inside the location "Gandhi Road bus stop"); drop those and decide again.
        for (f in ordered) {
            val r = results.getValue(f.name)
            if (r.status != FieldStatus.AMBIGUOUS) continue
            val others = results.filterKeys { it != f.name }.values.mapNotNull { spanOf(it) }
            val kept = r.candidates.filter { c -> others.none { c.overlaps(it.first, it.last + 1) } }
            if (kept.size != r.candidates.size) {
                val again = decide(f, kept, today)
                results[f.name] = again
                spanOf(again)?.let { claimed += it }
            }
        }
        // Preserve the administrator's field order in the output.
        val inOrder = LinkedHashMap<String, FieldResult>()
        def.fields.forEach { inOrder[it.name] = results.getValue(it.name) }
        return ExtractionResult(def.id, def.version, inOrder, engineId)
    }

    /** Parses and validates a value the user typed or chose for a field during clarification. */
    fun userValue(field: FieldDefinition, input: String, today: LocalDate = clock.today()): FieldResult {
        val raw = input.trim()
        if (raw.isEmpty()) return FieldResult(field.name, field.type, field.required, FieldStatus.MISSING, reason = "empty answer")
        val canonical: String? = when (field.type) {
            FieldType.STRING -> raw
            FieldType.INTEGER -> NumberParser.parseSingle(raw)?.takeIf { it % 1.0 == 0.0 }?.let { NumberParser.format(it) }
            FieldType.NUMBER -> NumberParser.parseSingle(raw)?.let { NumberParser.format(it) }
            FieldType.AMOUNT -> AmountParser.findAll(NormalizedText(raw), allowBare = true).maxByOrNull { it.confidence }?.value ?: NumberParser.parseSingle(raw)?.let { NumberParser.format(it) }
            FieldType.DATE -> DateParser.parseSingle(raw, today, field.extraction.temporalDirection)
            FieldType.TIME -> TimeParser.parseSingle(raw)
            FieldType.PHONE -> PhoneParser.parseSingle(raw)
            FieldType.DURATION -> DurationParser.findAll(NormalizedText(raw)).firstOrNull()?.value
            FieldType.BOOLEAN -> raw.lowercase().let { l -> if (l in Lexicons.yesWords || l == "true") "true" else if (l in Lexicons.noWords || l == "false") "false" else null }
            FieldType.ENUM -> enumCandidates(NormalizedText(raw), field, null).maxByOrNull { it.confidence }?.value
                ?: field.validation.allowedValues.firstOrNull { it.value.equals(raw, ignoreCase = true) }?.value
        }
        if (canonical == null) return FieldResult(field.name, field.type, field.required, FieldStatus.INVALID, raw = raw, reason = "could not interpret as ${field.type.name.lowercase()}")
        val problem = validate(field, canonical, today)
        return if (problem != null) FieldResult(field.name, field.type, field.required, FieldStatus.INVALID, canonical, raw, 1f, reason = problem)
        else FieldResult(field.name, field.type, field.required, FieldStatus.USER_PROVIDED, canonical, raw, 1f)
    }

    private fun spanOf(r: FieldResult): IntRange? {
        if (r.status != FieldStatus.EXTRACTED && r.status != FieldStatus.LOW_CONFIDENCE) return null
        val c = r.candidates.firstOrNull { it.value == r.value } ?: return null
        return if (c.start >= 0 && c.end > c.start) c.start until c.end else null
    }

    // ---- candidate generation ----

    private fun candidatesFor(text: NormalizedText, f: FieldDefinition, language: String?, today: LocalDate, claimed: List<IntRange>, triggerWords: Set<String> = emptySet()): List<Candidate> {
        val out = ArrayList<Candidate>()
        for (pat in f.extraction.patterns) {
            val re = runCatching { Regex(pat, RegexOption.IGNORE_CASE) }.getOrNull() ?: continue
            for (m in re.findAll(text.text)) {
                val g = m.groups["value"] ?: m.groups[1] ?: m.groups[0] ?: continue
                val raw = g.value.trim(); if (raw.isEmpty()) continue
                val start = m.range.first + m.value.indexOf(g.value); val end = start + g.value.length
                typed(f, raw, today, language, text, start, end)?.let { out += it.copy(confidence = 0.92f, source = "pattern") }
            }
        }
        for (anchor in forLanguage(f.extraction.anchors, language)) {
            val p = TextNormalizer.phrase(anchor); if (p.isEmpty()) continue
            var from = 0
            while (true) {
                val at = text.indexOfPhrase(p, from); if (at < 0) break
                from = at + p.length
                val startTok = text.tokenIndexAt(at + p.length); if (startTok < 0) continue
                anchored(text, f, language, today, startTok, forward = true, triggerWords)?.let { out += it.copy(source = "anchor") }
            }
        }
        for (anchor in forLanguage(f.extraction.anchorsBefore, language)) {
            val p = TextNormalizer.phrase(anchor); if (p.isEmpty()) continue
            var from = 0
            while (true) {
                val at = text.indexOfPhrase(p, from); if (at < 0) break
                from = at + p.length
                val endTok = text.tokens.indexOfLast { it.end <= at }; if (endTok < 0) continue
                anchored(text, f, language, today, endTok, forward = false, triggerWords)?.let { out += it.copy(source = "anchor") }
            }
        }
        if (f.type != FieldType.STRING && f.type != FieldType.BOOLEAN) {
            out += scan(text, f, language, today)
        }
        // Spans already explained by another field (a date, an amount) are not re-used.
        return out.filter { c -> c.source == "pattern" || claimed.none { r -> c.overlaps(r.first, r.last + 1) } }
    }

    /**
     * Value next to an anchor phrase: the following tokens when [forward], the preceding ones otherwise
     * (postpositions). Typed fields scan a small window; STRING fields take up to maxTokens words.
     */
    private fun anchored(text: NormalizedText, f: FieldDefinition, language: String?, today: LocalDate, edgeTok: Int, forward: Boolean, triggerWords: Set<String> = emptySet()): Candidate? {
        val toks = text.tokens
        val stops = forLanguage(f.extraction.stopWords, language).map { TextNormalizer.phrase(it) }.toSet() + triggerWords
        val step = if (forward) 1 else -1
        return when (f.type) {
            FieldType.STRING -> {
                val picked = ArrayList<Token>()
                var i = edgeTok
                while (i in toks.indices && picked.size < f.extraction.maxTokens) {
                    val tk = toks[i]
                    if (tk.isPunct) { if (picked.isEmpty()) { i += step; continue } else break }
                    if (tk.lower in stops || tk.lower in GENERIC_STOPS || triggerWords.any { Fuzzy.similar(tk.lower, it) }) break // incl. misspelled intent words
                    picked += tk; i += step
                }
                if (picked.isEmpty()) null else {
                    val s = picked.minOf { it.start }; val e = picked.maxOf { it.end }
                    val raw = text.text.substring(s, e)
                    Candidate(raw, raw, 0.8f, "anchor", s, e)
                }
            }
            FieldType.BOOLEAN -> {
                val range = if (forward) edgeTok until minOf(toks.size, edgeTok + 3) else maxOf(0, edgeTok - 2)..edgeTok
                range.map { toks[it] }.firstNotNullOfOrNull { tk ->
                    when (tk.lower) { in Lexicons.yesWords -> Candidate("true", tk.text, 0.9f, "anchor", tk.start, tk.end); in Lexicons.noWords -> Candidate("false", tk.text, 0.9f, "anchor", tk.start, tk.end); else -> null }
                }
            }
            else -> {
                val lo = if (forward) edgeTok else maxOf(0, edgeTok - 7)
                val hi = if (forward) minOf(toks.size - 1, edgeTok + 7) else edgeTok
                val from = toks.getOrNull(lo)?.start ?: return null
                val to = toks.getOrNull(hi)?.end ?: return null
                val inWindow = scan(text, f, language, today).filter { it.start >= from && it.end <= to }
                (if (forward) inWindow.minByOrNull { it.start } else inWindow.maxByOrNull { it.end })?.copy(confidence = 0.88f)
            }
        }
    }

    private fun scan(text: NormalizedText, f: FieldDefinition, language: String?, today: LocalDate): List<Candidate> = when (f.type) {
        FieldType.DATE -> DateParser.findAll(text, today, f.extraction.temporalDirection)
        FieldType.TIME -> TimeParser.findAll(text)
        FieldType.PHONE -> PhoneParser.findAll(text)
        FieldType.AMOUNT -> AmountParser.findAll(text, allowBare = true)
        FieldType.DURATION -> DurationParser.findAll(text)
        FieldType.ENUM -> enumCandidates(text, f, language)
        FieldType.INTEGER, FieldType.NUMBER -> NumberParser.findAll(text).filter { !it.isDigitSequence && it.value < 1e9 }
            .filter { f.type == FieldType.NUMBER || it.value % 1.0 == 0.0 }
            .filter { n -> !looksLikeTimeOrDate(text, n) }
            .map { Candidate(NumberParser.format(it.value), it.raw, 0.6f, "scanner", it.start, it.end) }
        FieldType.STRING, FieldType.BOOLEAN -> emptyList()
    }

    private fun enumCandidates(text: NormalizedText, f: FieldDefinition, language: String?): List<Candidate> {
        val out = ArrayList<Candidate>()
        for (av in f.validation.allowedValues) {
            val forms = (forLanguage(av.synonyms, language) + av.value).map { TextNormalizer.phrase(it) }.filter { it.isNotEmpty() }.distinct()
            var best: Candidate? = null
            for (form in forms) {
                var at = text.indexOfPhrase(form)
                var conf = if (form.contains(' ')) 0.95f else 0.9f
                if (at < 0 && form.length >= 5) { at = text.lower.indexOf(form); conf = 0.75f } // fused CTC output
                var len = form.length
                if (at < 0 && !form.contains(' ')) Fuzzy.findWord(text, form)?.let { tk -> at = tk.start; len = tk.end - tk.start; conf = 0.55f } // misspelled
                if (at < 0) continue
                val c = Candidate(av.value, text.text.substring(at, at + len), conf, "synonym", at, at + len)
                if (best == null || c.raw.length > best.raw.length) best = c
            }
            best?.let { out += it }
        }
        return out
    }

    /** Numbers that are really part of a time or date expression are not quantities. */
    private fun looksLikeTimeOrDate(text: NormalizedText, n: NumberParser.NumberMatch): Boolean {
        val toks = text.tokens
        for (k in listOf(n.startToken - 1, n.endToken, n.endToken + 1)) {
            val tk = toks.getOrNull(k) ?: continue
            if (tk.lower in Lexicons.hourMarkers || Lexicons.months.containsKey(tk.lower) || Lexicons.dayPeriods.containsKey(tk.lower) || tk.lower == "am" || tk.lower == "pm") return true
        }
        return false
    }

    /** Converts raw text captured by a pattern into a typed candidate. */
    private fun typed(f: FieldDefinition, raw: String, today: LocalDate, language: String?, text: NormalizedText, start: Int, end: Int): Candidate? {
        if (f.type == FieldType.STRING) return Candidate(raw, raw, 0.9f, "pattern", start, end)
        val sub = NormalizedText(raw)
        val c = when (f.type) {
            FieldType.ENUM -> enumCandidates(sub, f, language).maxByOrNull { it.confidence }
            else -> scan(sub, f.copy(extraction = f.extraction.copy(anchors = emptyMap(), patterns = emptyList())), language, today).maxByOrNull { it.confidence }
        } ?: return null
        return c.copy(raw = raw, start = start, end = end)
    }

    // ---- decision ----

    private fun decide(f: FieldDefinition, candidates: List<Candidate>, today: LocalDate): FieldResult {
        val threshold = f.confidenceThreshold ?: defaultThreshold
        // Merge duplicates by canonical value, keeping the best confidence.
        val merged = candidates.groupBy { it.value }.map { (_, cs) -> cs.maxByOrNull { it.confidence }!! }.sortedByDescending { it.confidence }
        if (merged.isEmpty()) {
            f.defaultValue?.let { return FieldResult(f.name, f.type, f.required, FieldStatus.DEFAULTED, it, null, 1f, reason = "administrator default applied") }
            return FieldResult(f.name, f.type, f.required, FieldStatus.MISSING, candidates = emptyList())
        }
        val top = merged.first()
        val second = merged.getOrNull(1)
        if (second != null && top.confidence - second.confidence < ambiguityMargin) {
            return FieldResult(f.name, f.type, f.required, FieldStatus.AMBIGUOUS, null, null, top.confidence, merged, "several possible values: " + merged.joinToString(", ") { "'${it.raw}'" })
        }
        val problem = validate(f, top.value, today)
        if (problem != null) return FieldResult(f.name, f.type, f.required, FieldStatus.INVALID, top.value, top.raw, top.confidence, merged, problem, top.extra)
        val status = if (top.confidence >= threshold) FieldStatus.EXTRACTED else FieldStatus.LOW_CONFIDENCE
        return FieldResult(f.name, f.type, f.required, status, top.value, top.raw, top.confidence, merged, top.note, top.extra)
    }

    /** Null when valid, else a reason. */
    fun validate(f: FieldDefinition, value: String, today: LocalDate): String? {
        val v = f.validation
        v.regex?.let { if (!Regex(it).matches(value)) return "does not match required pattern" }
        if (f.type == FieldType.ENUM && v.allowedValues.isNotEmpty() && v.allowedValues.none { it.value == value }) return "not one of the allowed values"
        if (f.type in setOf(FieldType.INTEGER, FieldType.NUMBER, FieldType.AMOUNT)) {
            val num = value.toDoubleOrNull() ?: return "not a number"
            v.min?.let { if (num < it) return "below minimum ${NumberParser.format(it)}" }
            v.max?.let { if (num > it) return "above maximum ${NumberParser.format(it)}" }
        }
        if (f.type == FieldType.STRING) {
            v.minLength?.let { if (value.length < it) return "shorter than $it characters" }
            v.maxLength?.let { if (value.length > it) return "longer than $it characters" }
        }
        if (f.type == FieldType.DATE) {
            val d = runCatching { LocalDate.parse(value) }.getOrNull() ?: return "not a date"
            v.maxDaysFromToday?.let { if (abs(ChronoUnit.DAYS.between(today, d)) > it) return "more than $it days from today" }
            if (f.extraction.temporalDirection == TemporalDirection.FUTURE && d.isBefore(today)) return "date is in the past"
            if (f.extraction.temporalDirection == TemporalDirection.PAST && d.isAfter(today)) return "date is in the future"
        }
        return null
    }

    private fun forLanguage(map: Map<String, List<String>>, language: String?): List<String> {
        val out = ArrayList<String>()
        map["*"]?.let { out += it }
        if (language != null) map[language]?.let { out += it } else map.filterKeys { it != "*" }.values.forEach { out += it }
        if (language != null && language != "en") map["en"]?.let { out += it }
        return out.distinct()
    }

    private fun typeOrder(t: FieldType) = when (t) {
        FieldType.PHONE -> 0; FieldType.AMOUNT -> 1; FieldType.DATE -> 2; FieldType.TIME -> 3; FieldType.DURATION -> 4
        FieldType.ENUM -> 5; FieldType.NUMBER, FieldType.INTEGER -> 6; FieldType.BOOLEAN -> 7; FieldType.STRING -> 8
    }

    companion object {
        /** Function words that end an anchored name/place in the supported languages. */
        private val GENERIC_STOPS = setOf(
            "for", "at", "on", "in", "to", "and", "with", "by", "the", "a", "an", "my", "is", "of", "ke", "ki", "ka", "ko", "se", "par", "aur",
            "के", "की", "का", "को", "से", "पर", "और", "साथ", "में", "है", "हैं", "ன்", "உடன்", "க்கு", "தோ", "కు", "తో", "ಗೆ", "ಜೊತೆ", "কে", "এর", "সাথে",
        )
    }
}
