using System.Text.Json.Serialization;

namespace ArkSummarize.Models;

/// <summary>A named entity discovered in the input text.</summary>
public record Entity(
    [property: JsonPropertyName("text")] string Text,
    [property: JsonPropertyName("type")] string Type);

/// <summary>
/// The structured analysis of a block of text. This is the shape returned by both the
/// web tool and the JSON API: an inferred <c>intent</c>, an extractive <c>summary</c>,
/// and the <c>entities</c> found in the text.
/// </summary>
public class SummaryResult
{
    /// <summary>Name of the engine that produced this result.</summary>
    [JsonPropertyName("engine")]
    public string Engine { get; set; } = "";

    /// <summary>The inferred high-level intent of the text (e.g. "question", "request").</summary>
    [JsonPropertyName("intent")]
    public string Intent { get; set; } = "statement";

    /// <summary>Confidence (0–1) of the inferred intent.</summary>
    [JsonPropertyName("intentConfidence")]
    public double IntentConfidence { get; set; }

    /// <summary>A concise extractive summary of the text.</summary>
    [JsonPropertyName("summary")]
    public string Summary { get; set; } = "";

    /// <summary>Named entities discovered in the text.</summary>
    [JsonPropertyName("entities")]
    public IReadOnlyList<Entity> Entities { get; set; } = Array.Empty<Entity>();

    // --- Diagnostics (handy in the UI, omitted from a minimal API response) ---

    [JsonPropertyName("keywords")]
    public IReadOnlyList<string> Keywords { get; set; } = Array.Empty<string>();

    [JsonPropertyName("sentenceCount")]
    public int SentenceCount { get; set; }

    [JsonPropertyName("wordCount")]
    public int WordCount { get; set; }

    [JsonPropertyName("processingMs")]
    public double ProcessingMs { get; set; }
}
