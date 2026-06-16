namespace ArkTextTranslator.Models;

/// <summary>
/// A supported language with the various codes the pipeline needs:
/// a display name, ISO 639-1 (used by LibreTranslate), ISO 639-3 (emitted by the
/// NTextCat detector) and the FLORES-200 code (required by the NLLB model).
/// </summary>
public record Language(string Name, string Iso6391, string Iso6393, string Flores);

/// <summary>
/// Central registry of languages the UI offers and that the engines can map between.
/// FLORES-200 codes follow the NLLB convention &lt;language&gt;_&lt;Script&gt;.
/// </summary>
public static class Languages
{
    public static readonly IReadOnlyList<Language> All = new[]
    {
        new Language("English",                "en",      "eng", "eng_Latn"),
        new Language("Spanish",                "es",      "spa", "spa_Latn"),
        new Language("French",                 "fr",      "fra", "fra_Latn"),
        new Language("German",                 "de",      "deu", "deu_Latn"),
        new Language("Italian",                "it",      "ita", "ita_Latn"),
        new Language("Portuguese",             "pt",      "por", "por_Latn"),
        new Language("Dutch",                  "nl",      "nld", "nld_Latn"),
        new Language("Russian",                "ru",      "rus", "rus_Cyrl"),
        new Language("Ukrainian",              "uk",      "ukr", "ukr_Cyrl"),
        new Language("Polish",                 "pl",      "pol", "pol_Latn"),
        new Language("Turkish",                "tr",      "tur", "tur_Latn"),
        new Language("Arabic",                 "ar",      "ara", "arb_Arab"),
        new Language("Hebrew",                 "he",      "heb", "heb_Hebr"),
        new Language("Hindi",                  "hi",      "hin", "hin_Deva"),
        new Language("Bengali",                "bn",      "ben", "ben_Beng"),
        new Language("Urdu",                   "ur",      "urd", "urd_Arab"),
        new Language("Tamil",                  "ta",      "tam", "tam_Taml"),
        new Language("Telugu",                 "te",      "tel", "tel_Telu"),
        new Language("Kannada",                "kn",      "kan", "kan_Knda"),
        new Language("Malayalam",              "ml",      "mal", "mal_Mlym"),
        new Language("Chinese (Simplified)",   "zh",      "zho", "zho_Hans"),
        new Language("Chinese (Traditional)",  "zh-Hant", "zht", "zho_Hant"),
        new Language("Japanese",               "ja",      "jpn", "jpn_Jpan"),
        new Language("Korean",                 "ko",      "kor", "kor_Hang"),
        new Language("Vietnamese",             "vi",      "vie", "vie_Latn"),
        new Language("Thai",                   "th",      "tha", "tha_Thai"),
        new Language("Indonesian",             "id",      "ind", "ind_Latn"),
        new Language("Swahili",                "sw",      "swa", "swh_Latn"),
        new Language("Greek",                  "el",      "ell", "ell_Grek"),
        new Language("Czech",                  "cs",      "ces", "ces_Latn"),
        new Language("Romanian",               "ro",      "ron", "ron_Latn"),
        new Language("Hungarian",              "hu",      "hun", "hun_Latn"),
        new Language("Swedish",                "sv",      "swe", "swe_Latn"),
        new Language("Finnish",                "fi",      "fin", "fin_Latn"),
        new Language("Danish",                 "da",      "dan", "dan_Latn"),
        new Language("Norwegian",              "no",      "nob", "nob_Latn"),
    };

    // Common alternate ISO 639-3 codes that detectors may emit, mapped to our canonical code.
    private static readonly Dictionary<string, string> Iso6393Aliases = new(StringComparer.OrdinalIgnoreCase)
    {
        ["arb"] = "ara", // Standard Arabic
        ["nor"] = "nob", // Norwegian macrolanguage -> Bokmål
        ["nno"] = "nob",
        ["cmn"] = "zho", // Mandarin -> Chinese
        ["pes"] = "fas", // (not registered, left for completeness)
    };

    private static readonly Dictionary<string, Language> ByIso6391 =
        All.ToDictionary(l => l.Iso6391, StringComparer.OrdinalIgnoreCase);

    private static readonly Dictionary<string, Language> ByIso6393 =
        All.ToDictionary(l => l.Iso6393, StringComparer.OrdinalIgnoreCase);

    private static readonly Dictionary<string, Language> ByFlores =
        All.ToDictionary(l => l.Flores, StringComparer.OrdinalIgnoreCase);

    public static Language? FromIso6391(string? code) =>
        code is not null && ByIso6391.TryGetValue(code, out var l) ? l : null;

    public static Language? FromIso6393(string? code)
    {
        if (string.IsNullOrWhiteSpace(code))
            return null;
        if (ByIso6393.TryGetValue(code, out var l))
            return l;
        if (Iso6393Aliases.TryGetValue(code, out var canonical) && ByIso6393.TryGetValue(canonical, out l))
            return l;
        return null;
    }

    public static Language? FromFlores(string? code) =>
        code is not null && ByFlores.TryGetValue(code, out var l) ? l : null;
}
