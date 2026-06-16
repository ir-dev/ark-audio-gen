using System.Diagnostics;
using ArkSummarize.Models;

namespace ArkSummarize.Services;

/// <summary>
/// The default engine: a fully-managed, CPU-only analyser that needs no GPU, no model
/// download and no external service, so responses are effectively instant.
///
/// It combines three classic, lightweight NLP techniques:
///  • <b>Summary</b>  — extractive summarisation. Sentences are scored by the combined
///    frequency of their (non-stopword) terms, lightly biased toward earlier sentences,
///    and the top-ranked ones are returned in their original reading order.
///  • <b>Intent</b>   — a rule/cue based classifier that weighs lexical and structural
///    signals (question marks, imperatives, sentiment cues, transactional vocabulary…).
///  • <b>Entities</b> — shared deterministic extractors (see <see cref="SharedNlp"/>).
/// </summary>
public sealed class SummarizationService : ISummarizationService
{
    public string Key => "lexical";
    public string DisplayName => "Ark Lexical Engine";
    public string Description => "Frequency-based extractive summary + rule-based intent. Instant, fully offline, no model.";

    public Task<SummaryResult> SummarizeAsync(string text, int maxSentences = 3, CancellationToken cancellationToken = default)
        => Task.FromResult(Summarize(text, maxSentences));

    private static SummaryResult Summarize(string text, int maxSentences)
    {
        var sw = Stopwatch.StartNew();
        text = (text ?? "").Trim();
        maxSentences = Math.Clamp(maxSentences, 1, 10);

        var result = new SummaryResult { Summary = "", Intent = "statement", Engine = "Ark Lexical Engine" };
        if (text.Length == 0)
        {
            result.ProcessingMs = sw.Elapsed.TotalMilliseconds;
            return result;
        }

        var sentences = SharedNlp.SplitSentences(text);
        var allWords = SharedNlp.WordTokens(text);

        result.SentenceCount = sentences.Count;
        result.WordCount = allWords.Count;

        var freq = SharedNlp.WordFrequencies(allWords);
        var maxFreq = freq.Count > 0 ? freq.Values.Max() : 1;
        result.Keywords = SharedNlp.TopKeywords(freq);

        result.Summary = BuildSummary(sentences, freq, maxFreq, maxSentences);
        (result.Intent, result.IntentConfidence) = ClassifyIntent(text);
        result.Entities = SharedNlp.ExtractEntities(text);

        result.ProcessingMs = sw.Elapsed.TotalMilliseconds;
        return result;
    }

    // ---- Summary ----------------------------------------------------------
    private static string BuildSummary(
        List<string> sentences, Dictionary<string, int> freq, int maxFreq, int maxSentences)
    {
        if (sentences.Count == 0) return "";
        if (sentences.Count <= maxSentences) return string.Join(" ", sentences);

        var scored = new List<(int Index, double Score)>(sentences.Count);
        for (int i = 0; i < sentences.Count; i++)
        {
            var words = SharedNlp.LowerWordTokens(sentences[i]);
            if (words.Count == 0) { scored.Add((i, 0)); continue; }

            double sum = 0;
            foreach (var w in words)
                if (freq.TryGetValue(w, out var f))
                    sum += (double)f / maxFreq;

            // Normalise by length (avoid favouring very long sentences) and add a mild
            // positional prior — the opening sentences of a passage carry more signal.
            double score = sum / Math.Sqrt(words.Count);
            double positionBoost = 1.0 + 0.15 * (1.0 - (double)i / sentences.Count);
            score *= positionBoost;

            // Penalise extremely short or extremely long sentences.
            if (words.Count < 4) score *= 0.5;
            if (words.Count > 40) score *= 0.8;

            scored.Add((i, score));
        }

        var chosen = scored
            .OrderByDescending(s => s.Score)
            .Take(maxSentences)
            .Select(s => s.Index)
            .OrderBy(i => i) // restore reading order
            .ToList();

        return string.Join(" ", chosen.Select(i => sentences[i].Trim()));
    }

    // ---- Intent -----------------------------------------------------------
    private static readonly (string Intent, string[] Cues)[] IntentCues =
    {
        ("question",      new[] { "?", "what ", "why ", "how ", "when ", "where ", "who ", "which ", "could you tell", "do you", "does it", "is it", "are there", "can you explain" }),
        ("request",       new[] { "please", "could you", "can you", "would you", "i need", "i want", "we need", "kindly", "request", "let me know", "send me", "i would like" }),
        ("complaint",     new[] { "not working", "doesn't work", "does not work", "broken", "issue", "problem", "disappointed", "refund", "terrible", "worst", "angry", "unacceptable", "complaint", "failed", "error", "frustrat", "poor" }),
        ("praise",        new[] { "thank", "thanks", "great job", "well done", "love it", "love this", "excellent", "awesome", "amazing", "appreciate", "fantastic", "wonderful", "good job", "brilliant" }),
        ("transactional", new[] { "order", "invoice", "payment", "purchase", "buy ", "subscribe", "cancel", "booking", "reserve", "checkout", "shipment", "delivery", "refund", "billing" }),
        ("instruction",   new[] { "first,", "firstly", "step ", "then ", "next,", "finally", "follow these", "in order to", "make sure", "you should", "do not", "don't ", "ensure that" }),
        ("opinion",       new[] { "i think", "i believe", "in my opinion", "i feel", "personally", "it seems", "arguably", "i'd argue", "from my perspective" }),
        ("announcement",  new[] { "we are pleased", "announcing", "introducing", "we're excited", "launch", "release", "now available", "starting today", "effective immediately" }),
    };

    private static (string Intent, double Confidence) ClassifyIntent(string text)
    {
        var lower = " " + text.ToLowerInvariant() + " ";
        var scores = new Dictionary<string, double>();

        foreach (var (intent, cues) in IntentCues)
        {
            double s = 0;
            foreach (var cue in cues)
            {
                if (cue == "?")
                {
                    s += SharedNlp.CountOccurrences(text, '?') * 2.0;
                }
                else
                {
                    int idx = 0, hits = 0;
                    while ((idx = lower.IndexOf(cue, idx, StringComparison.Ordinal)) >= 0) { hits++; idx += cue.Length; }
                    s += hits;
                }
            }
            if (s > 0) scores[intent] = s;
        }

        // Structural prior: a question mark anywhere strongly suggests a question.
        if (SharedNlp.CountOccurrences(text, '?') > 0)
            scores["question"] = scores.GetValueOrDefault("question") + 1.5;

        if (scores.Count == 0)
            return ("statement", 0.5);

        var total = scores.Values.Sum();
        var best = scores.OrderByDescending(kv => kv.Value).First();
        double confidence = Math.Round(0.5 + 0.5 * (best.Value / total), 2);
        return (best.Key, Math.Clamp(confidence, 0.5, 0.99));
    }
}
