using ArkTextTranslator.Models;

namespace ArkTextTranslator.Services;

/// <summary>Offline, CPU language identification.</summary>
public interface ILanguageDetector
{
    /// <summary>
    /// Detects the most likely language of <paramref name="text"/>.
    /// </summary>
    /// <returns>
    /// The matched <see cref="Language"/> (or <c>null</c> if it isn't in the registry)
    /// together with the detected ISO 639-3 code and a 0..1 confidence margin.
    /// </returns>
    (Language? Language, string Iso6393, double Confidence) Detect(string text);
}
