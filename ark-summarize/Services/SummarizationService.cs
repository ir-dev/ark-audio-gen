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
        (result.Intent, result.IntentConfidence) = SharedNlp.ClassifyIntent(text);
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
}
