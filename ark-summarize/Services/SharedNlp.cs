using System.Text.RegularExpressions;
using ArkSummarize.Models;

namespace ArkSummarize.Services;

/// <summary>
/// Deterministic text-processing primitives shared by every engine: sentence splitting,
/// word tokenisation, frequency/keyword extraction and named-entity extraction. Entities
/// and keywords are rule-based and engine-independent, so both the lexical and the
/// embedding engines reuse exactly this code for them.
/// </summary>
public static partial class SharedNlp
{
    // ---- Regexes (compiled once) ------------------------------------------
    [GeneratedRegex(@"(?<=[.!?])\s+(?=[A-Z0-9""'(\[])", RegexOptions.Compiled)]
    private static partial Regex SentenceSplit();

    [GeneratedRegex(@"[A-Za-z][A-Za-z'\-]*", RegexOptions.Compiled)]
    private static partial Regex WordToken();

    [GeneratedRegex(@"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b", RegexOptions.Compiled)]
    private static partial Regex EmailRx();

    [GeneratedRegex(@"\bhttps?://[^\s)>\]]+", RegexOptions.Compiled)]
    private static partial Regex UrlRx();

    [GeneratedRegex(@"(?<!\w)(?:\+?\d{1,3}[\s.\-]?)?(?:\(\d{2,4}\)[\s.\-]?)?\d{3,4}[\s.\-]\d{3,4}(?:[\s.\-]\d{2,4})?(?!\w)", RegexOptions.Compiled)]
    private static partial Regex PhoneRx();

    [GeneratedRegex(@"(?<![\w.])(?:[$€£¥₹]\s?\d[\d,]*(?:\.\d+)?(?:\s?(?:k|m|bn|billion|million|thousand))?|\d[\d,]*(?:\.\d+)?\s?(?:dollars|usd|eur|euros|pounds|gbp|inr|rupees|yen))\b", RegexOptions.Compiled | RegexOptions.IgnoreCase)]
    private static partial Regex MoneyRx();

    [GeneratedRegex(@"\b\d+(?:\.\d+)?\s?%|\b\d+(?:\.\d+)?\s?percent\b", RegexOptions.Compiled | RegexOptions.IgnoreCase)]
    private static partial Regex PercentRx();

    [GeneratedRegex(@"\b(?:\d{4}-\d{2}-\d{2}|\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+\d{4})?|\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*(?:,?\s+\d{4})?)\b", RegexOptions.Compiled | RegexOptions.IgnoreCase)]
    private static partial Regex DateRx();

    [GeneratedRegex(@"\b(?:[01]?\d|2[0-3]):[0-5]\d(?:\s?[ap]\.?m\.?)?|\b\d{1,2}\s?[ap]\.?m\.?\b", RegexOptions.Compiled | RegexOptions.IgnoreCase)]
    private static partial Regex TimeRx();

    // Sequences of Capitalised words → candidate proper nouns.
    [GeneratedRegex(@"\b([A-Z][a-zA-Z'&.\-]+(?:\s+(?:of|the|and|for|de|van|von|al)\s+)?(?:\s+[A-Z][a-zA-Z'&.\-]+)*)\b", RegexOptions.Compiled)]
    private static partial Regex ProperNounRx();

    // ---- Sentences & tokens -----------------------------------------------
    public static List<string> SplitSentences(string text)
    {
        var normalized = Regex.Replace((text ?? "").Replace("\r", " ").Replace("\n", " "), @"\s+", " ").Trim();
        if (normalized.Length == 0) return new List<string>();

        return SentenceSplit().Split(normalized)
            .Select(s => s.Trim())
            .Where(s => s.Length > 0)
            .ToList();
    }

    public static List<string> WordTokens(string text) =>
        WordToken().Matches(text).Select(m => m.Value).ToList();

    public static List<string> LowerWordTokens(string text) =>
        WordToken().Matches(text).Select(m => m.Value.ToLowerInvariant()).ToList();

    /// <summary>Lower-cased word frequencies with stopwords and very short tokens removed.</summary>
    public static Dictionary<string, int> WordFrequencies(IEnumerable<string> words)
    {
        var freq = new Dictionary<string, int>(StringComparer.Ordinal);
        foreach (var w in words)
        {
            var key = w.ToLowerInvariant();
            if (key.Length < 3 || Stopwords.Contains(key)) continue;
            freq[key] = freq.GetValueOrDefault(key) + 1;
        }
        return freq;
    }

    public static List<string> TopKeywords(Dictionary<string, int> freq, int n = 8) =>
        freq.OrderByDescending(kv => kv.Value)
            .ThenBy(kv => kv.Key)
            .Take(n)
            .Select(kv => kv.Key)
            .ToList();

    // ---- Entities ---------------------------------------------------------
    public static IReadOnlyList<Entity> ExtractEntities(string text)
    {
        var found = new List<Entity>();
        var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

        void Add(string value, string type)
        {
            value = value.Trim().TrimEnd('.', ',', ';', ':');
            if (value.Length == 0) return;
            var dedupeKey = type + "" + value.ToLowerInvariant();
            if (seen.Add(dedupeKey))
                found.Add(new Entity(value, type));
        }

        foreach (Match m in EmailRx().Matches(text)) Add(m.Value, "email");
        foreach (Match m in UrlRx().Matches(text)) Add(m.Value, "url");
        foreach (Match m in MoneyRx().Matches(text)) Add(m.Value, "money");
        foreach (Match m in PercentRx().Matches(text)) Add(m.Value, "percentage");
        foreach (Match m in DateRx().Matches(text)) Add(m.Value, "date");
        foreach (Match m in TimeRx().Matches(text)) Add(m.Value, "time");
        foreach (Match m in PhoneRx().Matches(text)) Add(m.Value, "phone");

        // Capitalised phrases → people / organisations / places. The hard part is telling a
        // genuine proper noun from a common word that's merely capitalised because it starts a
        // sentence ("Please…", "Revenue…"). Without a dictionary we use two robust signals:
        //   1) a token is high-confidence if it appears capitalised *away from* a sentence start;
        //   2) calendar words (days/months) are dropped here — they're captured as dates instead.
        var sentenceStarts = ComputeSentenceStartOffsets(text);

        // A single regex match can run across a sentence boundary because abbreviations carry an
        // internal period ("…Contoso Ltd. Please review…"). Break each match on a period+space so
        // the text after the period is correctly treated as sentence-initial.
        var candidates = new List<(string Value, bool AtStart)>();
        foreach (Match m in ProperNounRx().Matches(text))
        {
            var segs = Regex.Split(m.Value, @"\.\s+");
            for (int si = 0; si < segs.Length; si++)
            {
                var seg = segs[si].Trim();
                if (seg.Length == 0) continue;
                bool atStart = si == 0 ? sentenceStarts.Contains(m.Index) : true;
                candidates.Add((seg, atStart));
            }
        }

        // Pass 1: collect tokens that appear capitalised in a non-sentence-initial position.
        var confirmedCaps = new HashSet<string>(StringComparer.Ordinal);
        foreach (var (value, atStart) in candidates)
        {
            var toks = value.Split(' ', StringSplitOptions.RemoveEmptyEntries);
            for (int wi = 0; wi < toks.Length; wi++)
            {
                bool sentenceInitial = wi == 0 && atStart;
                if (!sentenceInitial && toks[wi].Length > 0 && char.IsUpper(toks[wi][0]))
                    confirmedCaps.Add(toks[wi].TrimEnd('.', ',', ';', ':'));
            }
        }

        // Pass 2: build entities, dropping ambiguous sentence-initial / calendar tokens.
        foreach (var (value, atSentenceStart) in candidates)
        {
            var toks = value.Split(' ', StringSplitOptions.RemoveEmptyEntries);
            var kept = new List<string>();

            for (int wi = 0; wi < toks.Length; wi++)
            {
                var word = toks[wi];
                var lw = word.ToLowerInvariant().Trim('.', ',', ';', ':');
                if (lw.Length == 0) continue;

                // Calendar words belong to the date/time entities, not names.
                if (CalendarWords.Contains(lw)) continue;

                // A lone capitalised word at a sentence start ("Please…", "Revenue…") is the
                // ambiguous case — keep it only if the same token is confirmed elsewhere. A
                // *multi-word* capitalised run at a sentence start ("Acme Corp announced…") is
                // almost always a genuine proper noun, so we trust it.
                bool sentenceInitial = wi == 0 && atSentenceStart;
                if (sentenceInitial && toks.Length == 1 && !confirmedCaps.Contains(word.TrimEnd('.', ',', ';', ':')))
                    continue;

                // Connector words ("of", "and", "the"…) are kept only between real tokens.
                if (Stopwords.Contains(lw) || SentenceStarters.Contains(lw))
                {
                    if (kept.Count == 0) continue; // don't start a phrase with a connector
                    kept.Add(word);
                    continue;
                }

                kept.Add(word);
            }

            // Trim trailing connectors.
            while (kept.Count > 0 &&
                   (Stopwords.Contains(kept[^1].ToLowerInvariant()) || SentenceStarters.Contains(kept[^1].ToLowerInvariant())))
                kept.RemoveAt(kept.Count - 1);

            if (kept.Count == 0) continue;
            var phrase = string.Join(' ', kept).Trim();
            if (phrase.Length < 3) continue;

            Add(phrase, ClassifyProperNoun(phrase));
        }

        return found
            .OrderBy(e => TypeOrder(e.Type))
            .ThenBy(e => e.Text, StringComparer.OrdinalIgnoreCase)
            .ToList();
    }

    private static string ClassifyProperNoun(string phrase)
    {
        var lower = phrase.ToLowerInvariant();
        string[] orgSuffixes = { "inc", "inc.", "llc", "ltd", "ltd.", "corp", "corp.", "company", "co.", "group",
            "technologies", "systems", "solutions", "labs", "university", "institute", "bank", "foundation",
            "department", "ministry", "agency", "association", "committee", "studios", "media", "ventures", "partners" };
        string[] placeHints = { "street", "avenue", "road", "city", "county", "state", "river", "mountain", "lake",
            "island", "valley", "park", "bay", "ocean", "sea", "north", "south", "east", "west" };

        if (orgSuffixes.Any(s => lower.EndsWith(" " + s) || lower == s || lower.Contains(" " + s + " ")))
            return "organization";
        if (placeHints.Any(s => lower.Contains(s)))
            return "location";

        return "name";
    }

    private static int TypeOrder(string type) => type switch
    {
        "name" => 0,
        "organization" => 1,
        "location" => 2,
        "email" => 3,
        "url" => 4,
        "phone" => 5,
        "money" => 6,
        "percentage" => 7,
        "date" => 8,
        "time" => 9,
        _ => 10,
    };

    /// <summary>Character offsets at which a new sentence begins (start of text, or the first
    /// non-space character following a <c>. ! ?</c>).</summary>
    private static HashSet<int> ComputeSentenceStartOffsets(string text)
    {
        var starts = new HashSet<int>();
        bool expectStart = true;
        for (int i = 0; i < text.Length; i++)
        {
            char c = text[i];
            if (expectStart && !char.IsWhiteSpace(c))
            {
                starts.Add(i);
                expectStart = false;
            }
            if (c is '.' or '!' or '?' or '\n')
                expectStart = true;
        }
        return starts;
    }

    public static int CountOccurrences(string s, char c)
    {
        int n = 0;
        foreach (var ch in s) if (ch == c) n++;
        return n;
    }

    // ---- Lexicons ---------------------------------------------------------
    public static readonly HashSet<string> CalendarWords = new(StringComparer.OrdinalIgnoreCase)
    {
        "monday","tuesday","wednesday","thursday","friday","saturday","sunday",
        "mon","tue","tues","wed","thu","thur","thurs","fri","sat","sun",
        "january","february","march","april","may","june","july","august",
        "september","october","november","december",
        "jan","feb","mar","apr","jun","jul","aug","sep","sept","oct","nov","dec",
        "today","tomorrow","yesterday",
    };

    public static readonly HashSet<string> SentenceStarters = new(StringComparer.OrdinalIgnoreCase)
    {
        "the","a","an","this","that","these","those","it","he","she","they","we","you","i",
        "but","and","or","so","then","however","therefore","thus","also","meanwhile","yet",
        "in","on","at","for","with","as","if","when","while","after","before","because","since",
    };

    public static readonly HashSet<string> Stopwords = new(StringComparer.Ordinal)
    {
        "the","a","an","and","or","but","if","then","else","when","at","by","for","with","about",
        "against","between","into","through","during","before","after","above","below","to","from",
        "up","down","in","out","on","off","over","under","again","further","once","here","there",
        "all","any","both","each","few","more","most","other","some","such","no","nor","not","only",
        "own","same","so","than","too","very","can","will","just","should","now","of","is","are",
        "was","were","be","been","being","have","has","had","having","do","does","did","doing","this",
        "that","these","those","i","me","my","myself","we","our","ours","ourselves","you","your","yours",
        "he","him","his","she","her","hers","it","its","they","them","their","theirs","what","which",
        "who","whom","whose","where","why","how","as","because","while","also","into","upon","per",
    };
}
