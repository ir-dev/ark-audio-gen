using System.ComponentModel.DataAnnotations;

namespace ArkSpeechToText.Models;

public class TranscribeViewModel
{
    [Display(Name = "Audio file (.wav / .mp3 / .ogg / .opus)")]
    public IFormFile? File { get; set; }

    [Display(Name = "Language")]
    public string Language { get; set; } = "auto";

    public TranscriptionResult? Result { get; set; }

    public string? ErrorMessage { get; set; }

    public string ModelSize { get; set; } = "base";
}
