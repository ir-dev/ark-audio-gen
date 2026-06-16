using System.ComponentModel.DataAnnotations;

namespace ArkSummarize.Models;

/// <summary>An engine the user can pick on the summarize page.</summary>
public record EngineOption(string Key, string DisplayName, string Description);

/// <summary>The outcome of running one engine (success carries a result, failure an error).</summary>
public class EngineRunResult
{
    public string Key { get; set; } = "";
    public string DisplayName { get; set; } = "";
    public SummaryResult? Result { get; set; }
    public string? Error { get; set; }
}

public class SummarizeViewModel
{
    [Display(Name = "Text to summarize")]
    public string? Text { get; set; }

    [Display(Name = "Maximum summary sentences")]
    [Range(1, 10)]
    public int MaxSentences { get; set; } = 3;

    /// <summary>Engine keys the user selected (one or many — many ⇒ side-by-side compare).</summary>
    public List<string> SelectedEngines { get; set; } = new();

    /// <summary>All engines available to choose from.</summary>
    public IReadOnlyList<EngineOption> AvailableEngines { get; set; } = Array.Empty<EngineOption>();

    /// <summary>Per-engine results, in the order they were run.</summary>
    public List<EngineRunResult> Results { get; set; } = new();

    public string? ErrorMessage { get; set; }

    // --- API console (signed-in user's credentials) ---
    public string ApiBaseUrl { get; set; } = "";
    public string ApiKeyName { get; set; } = "X-Ark-Api-Key";
    public string ApiKey { get; set; } = "";
}
