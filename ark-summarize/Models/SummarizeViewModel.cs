using System.ComponentModel.DataAnnotations;

namespace ArkSummarize.Models;

public class SummarizeViewModel
{
    [Display(Name = "Text to summarize")]
    public string? Text { get; set; }

    [Display(Name = "Maximum summary sentences")]
    [Range(1, 10)]
    public int MaxSentences { get; set; } = 3;

    public SummaryResult? Result { get; set; }

    public string? ErrorMessage { get; set; }

    // --- API console (signed-in user's credentials) ---
    public string ApiBaseUrl { get; set; } = "";
    public string ApiKeyName { get; set; } = "X-Ark-Api-Key";
    public string ApiKey { get; set; } = "";
}
