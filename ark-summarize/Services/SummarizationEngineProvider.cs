namespace ArkSummarize.Services;

/// <summary>
/// Registry of the available summarization engines. The lexical engine is the default;
/// callers may request any engine by its <see cref="ISummarizationService.Key"/>.
/// </summary>
public sealed class SummarizationEngineProvider
{
    public const string DefaultKey = "lexical";

    private readonly Dictionary<string, ISummarizationService> _byKey;

    public SummarizationEngineProvider(IEnumerable<ISummarizationService> engines)
    {
        _byKey = engines.ToDictionary(e => e.Key, StringComparer.OrdinalIgnoreCase);
        // Order the public list with the default first, then the rest alphabetically.
        Engines = _byKey.Values
            .OrderByDescending(e => e.Key.Equals(DefaultKey, StringComparison.OrdinalIgnoreCase))
            .ThenBy(e => e.DisplayName, StringComparer.OrdinalIgnoreCase)
            .ToList();
    }

    /// <summary>All registered engines (default first).</summary>
    public IReadOnlyList<ISummarizationService> Engines { get; }

    /// <summary>The default engine (lexical).</summary>
    public ISummarizationService Default => _byKey[DefaultKey];

    public bool TryGet(string? key, out ISummarizationService engine)
    {
        if (!string.IsNullOrWhiteSpace(key) && _byKey.TryGetValue(key.Trim(), out var found))
        {
            engine = found;
            return true;
        }
        engine = Default;
        return false;
    }

    /// <summary>Resolves an engine by key, falling back to the default if unknown/empty.</summary>
    public ISummarizationService Resolve(string? key)
    {
        TryGet(key, out var engine);
        return engine;
    }

    public bool IsKnown(string? key) =>
        !string.IsNullOrWhiteSpace(key) && _byKey.ContainsKey(key.Trim());
}
