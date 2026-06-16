using System.Diagnostics;
using System.Text.Json;
using System.Text.Json.Serialization;
using ArkTextTranslator.Models;
using Microsoft.Extensions.Options;

namespace ArkTextTranslator.Services;

/// <summary>
/// Translation backed by a <see href="https://libretranslate.com">LibreTranslate</see>
/// HTTP server (self-hostable via Docker). Correctness is owned by the mature
/// LibreTranslate engine; this app only adapts FLORES-200 codes to the ISO 639-1
/// codes it expects. Language detection still runs locally via <see cref="ILanguageDetector"/>.
/// </summary>
public sealed class LibreTranslateService : ITranslationService
{
    private readonly HttpClient _http;
    private readonly LibreTranslateOptions _options;
    private readonly ILanguageDetector _detector;
    private readonly ILogger<LibreTranslateService> _logger;

    public LibreTranslateService(
        HttpClient http,
        IOptions<TranslationOptions> options,
        ILanguageDetector detector,
        ILogger<LibreTranslateService> logger)
    {
        _http = http;
        _options = options.Value.LibreTranslate;
        _detector = detector;
        _logger = logger;
    }

    public string EngineName => "LibreTranslate";

    public async Task<TranslationResult> TranslateAsync(
        string text, string sourceFlores, string targetFlores, CancellationToken cancellationToken = default)
    {
        var target = Languages.FromFlores(targetFlores)
            ?? throw new ArgumentException($"Unsupported target language '{targetFlores}'.", nameof(targetFlores));

        var stopwatch = Stopwatch.StartNew();

        // Resolve the source language: either the explicit choice or local detection.
        Language? source;
        string detectedName;
        double confidence;
        if (sourceFlores.Equals("auto", StringComparison.OrdinalIgnoreCase))
        {
            var (lang, iso, conf) = _detector.Detect(text);
            source = lang;
            detectedName = lang?.Name ?? iso;
            confidence = conf;
        }
        else
        {
            source = Languages.FromFlores(sourceFlores)
                ?? throw new ArgumentException($"Unsupported source language '{sourceFlores}'.", nameof(sourceFlores));
            detectedName = source.Name;
            confidence = 1d;
        }

        var sourceCode = source?.Iso6391 ?? "auto";

        var payload = new Dictionary<string, string>
        {
            ["q"] = text,
            ["source"] = sourceCode,
            ["target"] = target.Iso6391,
            ["format"] = "text",
        };
        if (!string.IsNullOrWhiteSpace(_options.ApiKey))
            payload["api_key"] = _options.ApiKey!;

        var url = _options.Endpoint.TrimEnd('/') + "/translate";
        using var response = await _http.PostAsync(url, new FormUrlEncodedContent(payload), cancellationToken);

        var body = await response.Content.ReadAsStringAsync(cancellationToken);
        if (!response.IsSuccessStatusCode)
        {
            _logger.LogError("LibreTranslate returned {Status}: {Body}", (int)response.StatusCode, body);
            throw new InvalidOperationException(
                $"LibreTranslate request failed ({(int)response.StatusCode}). Is the server running at {_options.Endpoint}?");
        }

        var parsed = JsonSerializer.Deserialize<LibreResponse>(body)
            ?? throw new InvalidOperationException("Empty response from LibreTranslate.");

        stopwatch.Stop();

        // LibreTranslate echoes its own detected language when source=auto.
        if (parsed.DetectedLanguage is { Language: { Length: > 0 } detectedIso })
        {
            var byIso = Languages.FromIso6391(detectedIso);
            detectedName = byIso?.Name ?? detectedIso;
            source ??= byIso;
        }

        return new TranslationResult(
            parsed.TranslatedText ?? string.Empty,
            detectedName,
            confidence,
            source?.Flores ?? sourceFlores,
            target.Flores,
            stopwatch.Elapsed);
    }

    private sealed class LibreResponse
    {
        [JsonPropertyName("translatedText")]
        public string? TranslatedText { get; set; }

        [JsonPropertyName("detectedLanguage")]
        public DetectedLanguage? DetectedLanguage { get; set; }
    }

    private sealed class DetectedLanguage
    {
        [JsonPropertyName("language")]
        public string? Language { get; set; }

        [JsonPropertyName("confidence")]
        public double Confidence { get; set; }
    }
}
