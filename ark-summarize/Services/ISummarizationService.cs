using ArkSummarize.Models;

namespace ArkSummarize.Services;

public interface ISummarizationService
{
    /// <summary>Stable identifier used to select this engine via the UI / API (e.g. "lexical").</summary>
    string Key { get; }

    /// <summary>Human-readable name of the engine (shown in the UI).</summary>
    string DisplayName { get; }

    /// <summary>A one-line description of how the engine works (shown in the UI).</summary>
    string Description { get; }

    /// <summary>
    /// Analyses a block of text and returns its inferred intent, a summary and the named
    /// entities found within it.
    /// </summary>
    /// <param name="text">The raw text to analyse.</param>
    /// <param name="maxSentences">Maximum number of sentences in the summary (1–10).</param>
    Task<SummaryResult> SummarizeAsync(string text, int maxSentences = 3, CancellationToken cancellationToken = default);
}
