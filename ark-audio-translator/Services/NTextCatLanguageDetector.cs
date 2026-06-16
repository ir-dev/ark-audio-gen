using ArkTextTranslator.Models;
using NTextCat;

namespace ArkTextTranslator.Services;

/// <summary>
/// Fully-managed, offline language detection backed by
/// <see href="https://github.com/ivanakcheurov/ntextcat">NTextCat</see>. It uses
/// character n-gram language profiles (the bundled <c>Core14</c> set covers ~280
/// languages), runs entirely on the CPU and needs no model download — detection is
/// effectively instant.
/// </summary>
public sealed class NTextCatLanguageDetector : ILanguageDetector
{
    private readonly RankedLanguageIdentifier _identifier;
    private readonly ILogger<NTextCatLanguageDetector> _logger;

    public NTextCatLanguageDetector(IWebHostEnvironment env, ILogger<NTextCatLanguageDetector> logger)
    {
        _logger = logger;
        var profilePath = Path.Combine(env.ContentRootPath, "LanguageModels", "Core14.profile.xml");
        if (!File.Exists(profilePath))
            throw new FileNotFoundException(
                $"Language profile not found at '{profilePath}'. It ships with the app and should be copied to the output directory.",
                profilePath);

        using var stream = File.OpenRead(profilePath);
        _identifier = new RankedLanguageIdentifierFactory().Load(stream);
        _logger.LogInformation("Loaded NTextCat language profile from {Path}", profilePath);
    }

    public (Language? Language, string Iso6393, double Confidence) Detect(string text)
    {
        if (string.IsNullOrWhiteSpace(text))
            return (null, "und", 0d);

        // Identify returns languages ranked best-first; the score is a distance, so
        // lower is a better match.
        var ranked = _identifier.Identify(text).Take(2).ToArray();
        if (ranked.Length == 0)
            return (null, "und", 0d);

        var bestCode = ranked[0].Item1.Iso639_3;
        var confidence = ComputeConfidence(ranked);

        return (Languages.FromIso6393(bestCode), bestCode, confidence);
    }

    /// <summary>
    /// Turns the n-gram distance scores into a rough 0..1 confidence based on the
    /// margin between the best and second-best candidate. A clear winner scores high.
    /// </summary>
    private static double ComputeConfidence(IReadOnlyList<Tuple<LanguageInfo, double>> ranked)
    {
        var best = ranked[0].Item2;
        if (ranked.Count < 2)
            return 1d;

        var second = ranked[1].Item2;
        if (second <= 0)
            return 1d;

        var margin = (second - best) / second;
        return Math.Clamp(margin, 0d, 1d);
    }
}
