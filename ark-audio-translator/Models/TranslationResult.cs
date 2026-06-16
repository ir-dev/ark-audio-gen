namespace ArkTextTranslator.Models;

/// <summary>The full result of a translation run.</summary>
public record TranslationResult(
    string TranslatedText,
    string DetectedLanguage,
    double DetectionConfidence,
    string SourceLanguage,
    string TargetLanguage,
    TimeSpan ProcessingTime);
