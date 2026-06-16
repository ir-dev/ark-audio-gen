using ArkTextTranslator.Models;

namespace ArkTextTranslator.Services;

public interface ITranslationService
{
    /// <summary>Human-readable name of the active engine (shown in the UI).</summary>
    string EngineName { get; }

    /// <summary>
    /// Translates <paramref name="text"/> into <paramref name="targetFlores"/>.
    /// </summary>
    /// <param name="text">The source text (any supported language).</param>
    /// <param name="sourceFlores">Source FLORES-200 code, or "auto" to detect it.</param>
    /// <param name="targetFlores">Target FLORES-200 code.</param>
    Task<TranslationResult> TranslateAsync(
        string text, string sourceFlores, string targetFlores, CancellationToken cancellationToken = default);
}
