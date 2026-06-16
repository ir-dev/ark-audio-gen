using System.ComponentModel.DataAnnotations;

namespace ArkTextTranslator.Models;

public class TranslateViewModel
{
    [Display(Name = "Text to translate")]
    [Required(ErrorMessage = "Enter some text to translate.")]
    public string? Text { get; set; }

    /// <summary>
    /// Source language as a FLORES-200 code, or "auto" to detect it.
    /// </summary>
    [Display(Name = "From")]
    public string SourceLanguage { get; set; } = "auto";

    /// <summary>Target language as a FLORES-200 code.</summary>
    [Display(Name = "To")]
    public string TargetLanguage { get; set; } = "eng_Latn";

    public TranslationResult? Result { get; set; }

    public string? ErrorMessage { get; set; }

    /// <summary>Engine in use, surfaced in the UI (e.g. "Onnx (NLLB-200)").</summary>
    public string Engine { get; set; } = "";

    public IReadOnlyList<Language> Languages { get; set; } = Models.Languages.All;
}
