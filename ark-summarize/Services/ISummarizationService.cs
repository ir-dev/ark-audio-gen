using ArkSummarize.Models;

namespace ArkSummarize.Services;

public interface ISummarizationService
{
    /// <summary>Name of the engine powering analysis (shown in the UI).</summary>
    string EngineName { get; }

    /// <summary>
    /// Analyses a block of text and returns its inferred intent, an extractive summary
    /// and the named entities found within it.
    /// </summary>
    /// <param name="text">The raw text to analyse.</param>
    /// <param name="maxSentences">Maximum number of sentences in the summary (1–10).</param>
    SummaryResult Summarize(string text, int maxSentences = 3);
}
