package co.immanuel.arkspeech.core.nlu

import co.immanuel.arkspeech.core.config.TemporalDirection
import co.immanuel.arkspeech.core.nlu.Lexicons.NumKind
import java.time.DayOfWeek
import java.time.LocalDate
import java.time.temporal.TemporalAdjusters
import kotlin.math.abs
import kotlin.math.floor

/** Numbers written as digits (any script, already ASCII) or spoken as words, including Indian lakh/crore. */
object NumberParser {

    class NumberMatch(
        val value: Double,
        val startToken: Int,
        /** exclusive */
        val endToken: Int,
        val start: Int,
        val end: Int,
        val raw: String,
        val isDigitSequence: Boolean,
        /** digits as spoken, only for digit sequences ("nine eight seven…" → "987…") */
        val digits: String? = null,
    )

    private val ordinalSuffix = Regex("^(\\d+)(st|nd|rd|th|वीं|वां|वाँ|ஆம்|వ|ನೇ)$")

    fun findAll(t: NormalizedText, fromToken: Int = 0, toToken: Int = t.tokens.size): List<NumberMatch> {
        val toks = t.tokens
        val out = ArrayList<NumberMatch>()
        var i = fromToken
        while (i < toToken) {
            val tok = toks[i]
            if (tok.isPunct || (!isDigitToken(tok) && numberWord(tok) == null)) { i++; continue }
            // Collect a run of numeric tokens (digits, number words, joining punctuation that glues digits).
            val group = ArrayList<Int>()
            var j = i
            while (j < toToken) {
                val tj = toks[j]
                if (tj.isPunct) {
                    // "2,500" or "2.5": punctuation glued to digits on both sides stays in the group.
                    val prev = toks.getOrNull(j - 1); val next = toks.getOrNull(j + 1)
                    val glued = (tj.text == "," || tj.text == ".") && prev != null && next != null && prev.end == tj.start && tj.end == next.start &&
                        isDigitToken(prev) && isDigitToken(next) && (tj.text == "." || next.text.length == 3)
                    if (glued) { group += j; j++; continue } else break
                }
                if (isDigitToken(tj) || numberWord(tj) != null) { group += j; j++ } else break
            }
            if (group.isEmpty()) { i++; continue }
            val wordsOnly = group.filter { !toks[it].isPunct }
            val firstTok = toks[group.first()]; val lastTok = toks[group.last()]
            val raw = t.text.substring(firstTok.start, lastTok.end)

            // Spoken digit sequences: "nine eight seven six" → 9876 (three or more single digits in a row).
            val singleDigits = wordsOnly.all { idx -> val tk = toks[idx]; (isDigitToken(tk) && tk.text.length == 1) || (numberWord(tk)?.let { it.kind == NumKind.UNIT && it.value < 10 && it.value % 1.0 == 0.0 } == true) }
            if (singleDigits && wordsOnly.size >= 3) {
                val digits = wordsOnly.joinToString("") { idx -> val tk = toks[idx]; if (isDigitToken(tk)) tk.text else numberWord(tk)!!.value.toInt().toString() }
                out += NumberMatch(digits.toDouble(), group.first(), group.last() + 1, firstTok.start, lastTok.end, raw, true, digits)
                i = group.last() + 1
                continue
            }
            val value = compose(group.map { toks[it] })
            if (value != null) out += NumberMatch(value, group.first(), group.last() + 1, firstTok.start, lastTok.end, raw, false)
            i = group.last() + 1
        }
        return out
    }

    /** A number spoken or written on its own (e.g. a user's typed clarification). */
    fun parseSingle(text: String): Double? {
        val t = NormalizedText(text)
        val all = findAll(t)
        return if (all.size == 1 && all[0].startToken == 0 && all[0].endToken == t.tokens.size) all[0].value else text.trim().toDoubleOrNull()
    }

    fun isDigitToken(tok: Token): Boolean = !tok.isPunct && (tok.text.all { it in '0'..'9' } || ordinalSuffix.matches(tok.lower))

    fun numberWord(tok: Token): Lexicons.NumWord? = if (tok.isPunct) null else Lexicons.numberWords[tok.lower]

    private fun digitValue(tok: Token): Double? {
        if (tok.text.all { it in '0'..'9' }) return tok.text.toDoubleOrNull()
        return ordinalSuffix.find(tok.lower)?.groupValues?.get(1)?.toDoubleOrNull()
    }

    /** Standard additive/multiplicative composition shared by all the lexicons. */
    private fun compose(tokens: List<Token>): Double? {
        var total = 0.0
        var current = 0.0
        var any = false
        var i = 0
        while (i < tokens.size) {
            val tok = tokens[i]
            if (tok.isPunct) {
                // glued decimal or thousands separator: "2" "." "5" / "2" "," "500"
                val next = tokens.getOrNull(i + 1) ?: break
                val nv = digitValue(next) ?: break
                if (tok.text == ".") current = current + nv / Math.pow(10.0, next.text.length.toDouble())
                else current = current * 1000 + nv
                i += 2; any = true
                continue
            }
            val dv = digitValue(tok)
            if (dv != null) {
                if (any && current != 0.0) return null // "5 7" is not one number
                current += dv; any = true; i++; continue
            }
            val w = numberWord(tok) ?: return null
            any = true
            when (w.kind) {
                NumKind.UNIT -> current += w.value
                NumKind.FRACTION -> current += w.value
                NumKind.HUNDRED -> current = (if (current == 0.0) 1.0 else current) * 100
                NumKind.MULTIPLIER -> { total += (if (current == 0.0) 1.0 else current) * w.value; current = 0.0 }
            }
            i++
        }
        return if (any) total + current else null
    }

    fun format(v: Double): String = if (v == floor(v) && abs(v) < 1e15) v.toLong().toString() else v.toString()
}

/** Dates: numeric forms, "5 November [2026]", relative words and weekdays, across the lexicon's languages. */
object DateParser {

    private val isoDate = Regex("\\b(\\d{4})-(\\d{1,2})-(\\d{1,2})\\b")
    private val dmy = Regex("\\b(\\d{1,2})[/.-](\\d{1,2})(?:[/.-](\\d{2,4}))?\\b")

    fun findAll(t: NormalizedText, today: LocalDate, direction: TemporalDirection): List<Candidate> {
        val out = ArrayList<Candidate>()
        val isoSpans = ArrayList<IntRange>()
        for (m in isoDate.findAll(t.text)) {
            isoSpans += m.range
            safeDate(m.groupValues[1].toInt(), m.groupValues[2].toInt(), m.groupValues[3].toInt())?.let {
                out += Candidate(it.toString(), m.value, 0.95f, "scanner", m.range.first, m.range.last + 1)
            }
        }
        for (m in dmy.findAll(t.text)) {
            if (isoSpans.any { it.first <= m.range.last && m.range.first <= it.last }) continue // "11-12" inside "2026-11-12"
            val d = m.groupValues[1].toInt(); val mo = m.groupValues[2].toInt()
            val yRaw = m.groupValues[3]
            val year = when { yRaw.isEmpty() -> inferYear(mo, d, today, direction); yRaw.length == 2 -> 2000 + yRaw.toInt(); else -> yRaw.toInt() }
            safeDate(year, mo, d)?.let { out += Candidate(it.toString(), m.value, if (yRaw.isEmpty()) 0.85f else 0.95f, "scanner", m.range.first, m.range.last + 1) }
        }
        // Month names with a day number nearby.
        val toks = t.tokens
        for ((idx, tok) in toks.withIndex()) {
            val month = Lexicons.months[tok.lower] ?: continue
            var day: Int? = null; var dayTok: Token? = null
            for (k in listOf(idx - 1, idx - 2, idx + 1, idx + 2)) {
                val tk = toks.getOrNull(k) ?: continue
                if (tk.isPunct) continue
                val v = dayNumber(tk) ?: continue
                day = v; dayTok = tk; break
            }
            if (day == null || dayTok == null) continue
            var year: Int? = null; var yearTok: Token? = null
            for (k in listOf(idx + 1, idx + 2, idx + 3)) {
                val tk = toks.getOrNull(k) ?: continue
                if (tk === dayTok) continue
                val v = tk.text.toIntOrNull() ?: continue
                if (v in 1900..2100) { year = v; yearTok = tk; break }
            }
            val y = year ?: inferYear(month, day, today, direction)
            val date = safeDate(y, month, day) ?: continue
            val spanToks = listOfNotNull(tok, dayTok, yearTok)
            val s = spanToks.minOf { it.start }; val e = spanToks.maxOf { it.end }
            out += Candidate(date.toString(), t.text.substring(s, e), if (year != null) 0.95f else 0.85f, "scanner", s, e)
        }
        // Relative days (longest phrases first so "day after tomorrow" beats "tomorrow").
        val taken = ArrayList<IntRange>()
        for ((phrase, offsets) in Lexicons.relativeDays.entries.sortedByDescending { it.key.length }) {
            var from = 0
            while (true) {
                val at = t.indexOfPhrase(phrase, from); if (at < 0) break
                val range = at until at + phrase.length
                from = at + phrase.length
                if (taken.any { it.first <= range.last && range.first <= it.last }) continue
                taken += range
                val raw = t.text.substring(at, at + phrase.length)
                val chosen = when {
                    offsets.size == 1 -> offsets
                    direction == TemporalDirection.FUTURE -> offsets.filter { it > 0 }
                    direction == TemporalDirection.PAST -> offsets.filter { it < 0 }
                    else -> offsets
                }
                for (o in chosen) {
                    val conf = if (chosen.size == 1) 0.9f else 0.6f
                    val note = if (chosen.size > 1) "'$raw' can mean $o day(s) from today; confirm" else null
                    out += Candidate(today.plusDays(o.toLong()).toString(), raw, conf, "scanner", at, at + phrase.length, note = note)
                }
            }
        }
        // Weekdays: next occurrence (or previous when the field looks back).
        for (tok in toks) {
            val dow = Lexicons.weekdays[tok.lower] ?: continue
            val target = DayOfWeek.of(dow)
            val date = if (direction == TemporalDirection.PAST) today.with(TemporalAdjusters.previous(target)) else today.with(TemporalAdjusters.next(target))
            out += Candidate(date.toString(), tok.text, 0.75f, "scanner", tok.start, tok.end, note = "weekday resolved to the ${if (direction == TemporalDirection.PAST) "previous" else "next"} ${target.name.lowercase()}")
        }
        return out
    }

    /** Parse a typed or spoken answer to a clarification question. */
    fun parseSingle(text: String, today: LocalDate, direction: TemporalDirection): String? {
        val t = NormalizedText(text)
        val all = findAll(t, today, direction).sortedByDescending { it.confidence }
        return all.firstOrNull()?.value
    }

    private fun dayNumber(tk: Token): Int? {
        Lexicons.bengaliOrdinalDays[tk.lower]?.let { return it }
        Lexicons.ordinalWords[tk.lower]?.let { return it }
        val v = if (NumberParser.isDigitToken(tk)) Regex("\\d+").find(tk.text)?.value?.toIntOrNull()
        else NumberParser.numberWord(tk)?.takeIf { it.kind == NumKind.UNIT && it.value % 1.0 == 0.0 }?.value?.toInt()
        return v?.takeIf { it in 1..31 }
    }

    private fun inferYear(month: Int, day: Int, today: LocalDate, direction: TemporalDirection): Int {
        val thisYear = safeDate(today.year, month, day) ?: return today.year
        return when (direction) {
            TemporalDirection.FUTURE -> if (thisYear.isBefore(today)) today.year + 1 else today.year
            TemporalDirection.PAST -> if (thisYear.isAfter(today)) today.year - 1 else today.year
            TemporalDirection.UNSPECIFIED -> today.year
        }
    }

    private fun safeDate(y: Int, m: Int, d: Int): LocalDate? = runCatching { LocalDate.of(y, m, d) }.getOrNull()
}

/** Clock times: "10:30", "10 am", "ten in the morning", "सुबह दस बजे", "பத்து மணிக்கு", "साढ़े दस". */
object TimeParser {

    private val hhmm = Regex("\\b(\\d{1,2})[:.](\\d{2})(?:\\s*(am|pm|a\\.m\\.|p\\.m\\.))?\\b")
    private val hAmPm = Regex("\\b(\\d{1,2})\\s*(am|pm|a\\.m\\.|p\\.m\\.)(?![a-z])")

    fun findAll(t: NormalizedText): List<Candidate> {
        val out = ArrayList<Candidate>()
        val used = ArrayList<IntRange>()
        for (m in hhmm.findAll(t.lower)) {
            val h = m.groupValues[1].toInt(); val mi = m.groupValues[2].toInt()
            if (h > 23 || mi > 59) continue
            val ap = m.groupValues[3]
            val h24 = applyAmPm(h, ap)
            out += Candidate(fmt(h24, mi), t.text.substring(m.range.first, m.range.last + 1), if (ap.isEmpty() && h <= 12) 0.7f else 0.95f, "scanner", m.range.first, m.range.last + 1)
            used += m.range
        }
        for (m in hAmPm.findAll(t.lower)) {
            if (used.any { it.first <= m.range.last && m.range.first <= it.last }) continue
            val h = m.groupValues[1].toInt(); if (h > 12) continue
            out += Candidate(fmt(applyAmPm(h, m.groupValues[2]), 0), t.text.substring(m.range.first, m.range.last + 1), 0.95f, "scanner", m.range.first, m.range.last + 1)
            used += m.range
        }
        // Spoken hours: number + hour marker and/or a day-period word within a few tokens.
        val toks = t.tokens
        for ((idx, tok) in toks.withIndex()) {
            if (used.any { it.first <= tok.end - 1 && tok.start <= it.last }) continue
            val hour = hourValue(tok) ?: continue
            var marker = false; var period: Lexicons.DayPeriod? = null; var minutes = 0
            var s = tok.start; var e = tok.end
            for (k in 1..2) toks.getOrNull(idx + k)?.let { nt -> if (nt.lower in Lexicons.hourMarkers) { marker = true; e = maxOf(e, nt.end) } }
            for (k in -3..3) {
                if (k == 0) continue
                val nt = toks.getOrNull(idx + k) ?: continue
                Lexicons.dayPeriods[nt.lower]?.let { p -> if (period == null) { period = p; s = minOf(s, nt.start); e = maxOf(e, nt.end) } }
            }
            toks.getOrNull(idx - 1)?.let { pt -> Lexicons.hourFractionPrefixes[pt.lower]?.let { f -> minutes = f; s = minOf(s, pt.start) } }
            if (!marker && period == null && minutes == 0) continue
            var h = hour
            if (minutes < 0) { h -= 1; minutes = 60 + minutes; if (h <= 0) h = 12 }
            val h24 = when (period) {
                Lexicons.DayPeriod.MORNING -> if (h == 12) 0 else h
                Lexicons.DayPeriod.AFTERNOON -> if (h < 12) h + 12 else h
                Lexicons.DayPeriod.EVENING -> if (h < 12) h + 12 else h
                Lexicons.DayPeriod.NIGHT -> if (h in 7..11) h + 12 else if (h == 12) 0 else h
                null -> h
            }
            if (h24 !in 0..23) continue
            val raw = t.text.substring(s, e)
            val conf = if (period != null || hour > 12) 0.9f else 0.5f
            val note = if (period == null && hour <= 12) "no AM/PM given; confirm" else null
            out += Candidate(fmt(h24, minutes), raw, conf, "scanner", s, e, note = note)
        }
        return out
    }

    fun parseSingle(text: String): String? = findAll(NormalizedText(text)).maxByOrNull { it.confidence }?.value

    private fun hourValue(tok: Token): Int? {
        if (tok.isPunct) return null
        if (tok.text.all { it in '0'..'9' }) return tok.text.toIntOrNull()?.takeIf { it in 0..23 }
        return NumberParser.numberWord(tok)?.takeIf { it.kind == NumKind.UNIT && it.value % 1.0 == 0.0 && it.value in 1.0..12.0 }?.value?.toInt()
    }

    private fun applyAmPm(h: Int, ap: String): Int = when {
        ap.startsWith("a") -> if (h == 12) 0 else h
        ap.startsWith("p") -> if (h < 12) h + 12 else h
        else -> h
    }

    private fun fmt(h: Int, m: Int) = "%02d:%02d".format(h, m)
}

/** Indian phone numbers spoken as digits or written with separators; canonical form is the 10 digits. */
object PhoneParser {

    fun findAll(t: NormalizedText): List<Candidate> {
        val out = ArrayList<Candidate>()
        val toks = t.tokens
        var i = 0
        while (i < toks.size) {
            val sb = StringBuilder()
            var j = i
            var s = -1; var e = -1
            var pendingRepeat = 1
            while (j < toks.size) {
                val tk = toks[j]
                when {
                    tk.isPunct && (tk.text == "-" || tk.text == "+" || tk.text == "." || tk.text == "(" || tk.text == ")") -> { j++; continue }
                    tk.isPunct -> break
                    tk.text.all { it in '0'..'9' } -> { repeat(pendingRepeat) { sb.append(tk.text) }; pendingRepeat = 1 }
                    Lexicons.digitWords.containsKey(tk.lower) -> { repeat(pendingRepeat) { sb.append(Lexicons.digitWords[tk.lower]) }; pendingRepeat = 1 }
                    Lexicons.repeatWords.containsKey(tk.lower) -> { pendingRepeat = Lexicons.repeatWords[tk.lower]!!; j++; continue }
                    else -> break
                }
                if (s < 0) s = tk.start
                e = tk.end
                j++
            }
            val digits = sb.toString()
            if (digits.length >= 10) {
                val raw = t.text.substring(s, e)
                val local = when {
                    digits.length == 10 -> digits
                    digits.length == 11 && digits.startsWith("0") -> digits.substring(1)
                    digits.length == 12 && digits.startsWith("91") -> digits.substring(2)
                    digits.length == 13 && digits.startsWith("091") -> digits.substring(3)
                    else -> null
                }
                if (local != null) {
                    val mobile = local[0] in '6'..'9'
                    out += Candidate(local, raw, if (mobile) 0.9f else 0.6f, "scanner", s, e, note = if (mobile) null else "does not look like an Indian mobile number")
                } else {
                    out += Candidate(digits, raw, 0.3f, "scanner", s, e, note = "${digits.length} digits; expected 10")
                }
            }
            i = maxOf(j, i + 1)
        }
        return out
    }

    fun parseSingle(text: String): String? = findAll(NormalizedText(text)).maxByOrNull { it.confidence }?.value
}

/** Money: a number next to a currency word or symbol; canonical value is the plain number, currency in extra. */
object AmountParser {

    fun findAll(t: NormalizedText, allowBare: Boolean): List<Candidate> {
        val out = ArrayList<Candidate>()
        val toks = t.tokens
        val numbers = NumberParser.findAll(t).filter { !it.isDigitSequence }
        for (n in numbers) {
            var s = n.start; var e = n.end
            var currency: String? = null
            for (k in listOf(n.startToken - 1, n.startToken - 2, n.endToken, n.endToken + 1)) {
                val tk = toks.getOrNull(k) ?: continue
                if (tk.lower in Lexicons.currencyWords || tk.text == "₹") { currency = "INR"; s = minOf(s, tk.start); e = maxOf(e, tk.end); break }
            }
            if (currency == null && n.start > 0 && t.text[n.start - 1] == '₹') { currency = "INR"; s = n.start - 1 }
            if (currency == null && !allowBare) continue
            if (currency == null && (n.value >= 1e9 || numbers.size != 1)) continue
            out += Candidate(
                NumberParser.format(n.value), t.text.substring(s, e), if (currency != null) 0.9f else 0.5f, "scanner", s, e,
                extra = mapOf("currency" to (currency ?: "INR")), note = if (currency == null) "no currency word; assumed INR" else null,
            )
        }
        return out
    }
}

/** Durations → ISO-8601 ("PT2H30M", "P3D"). */
object DurationParser {

    fun findAll(t: NormalizedText): List<Candidate> {
        val out = ArrayList<Candidate>()
        val toks = t.tokens
        for (n in NumberParser.findAll(t).filter { !it.isDigitSequence }) {
            val unitTok = (n.endToken..n.endToken + 1).mapNotNull { toks.getOrNull(it) }.firstOrNull { Lexicons.durationUnits.containsKey(it.lower) } ?: continue
            val unit = Lexicons.durationUnits[unitTok.lower]!!
            val iso = toIso(n.value, unit) ?: continue
            out += Candidate(iso, t.text.substring(n.start, unitTok.end), 0.9f, "scanner", n.start, unitTok.end)
        }
        return out
    }

    fun toIso(value: Double, unit: Lexicons.DurationUnit): String? {
        if (value <= 0) return null
        return when (unit) {
            Lexicons.DurationUnit.MINUTE -> "PT${value.toInt()}M"
            Lexicons.DurationUnit.HOUR -> {
                val h = floor(value).toInt(); val m = ((value - h) * 60).toInt()
                if (m == 0) "PT${h}H" else if (h == 0) "PT${m}M" else "PT${h}H${m}M"
            }
            Lexicons.DurationUnit.DAY -> "P${value.toInt()}D"
            Lexicons.DurationUnit.WEEK -> "P${value.toInt()}W"
            Lexicons.DurationUnit.MONTH -> "P${value.toInt()}M"
        }
    }
}
