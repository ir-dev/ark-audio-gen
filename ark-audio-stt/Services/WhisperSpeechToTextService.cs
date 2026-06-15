using System.Diagnostics;
using System.Text;
using ArkSpeechToText.Models;
using Microsoft.Extensions.Options;
using Whisper.net;

namespace ArkSpeechToText.Services;

/// <summary>
/// Lightweight, CPU-hosted speech-to-text powered by Whisper.net (whisper.cpp).
/// The ggml model is downloaded once on first use and cached on disk. The
/// <see cref="WhisperFactory"/> is created lazily and reused for the lifetime
/// of the app; a fresh processor is built per request because processors are
/// single-use.
/// </summary>
public sealed class WhisperSpeechToTextService : ISpeechToTextService, IDisposable
{
    private const string ModelBaseUrl = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main";

    private readonly WhisperOptions _options;
    private readonly ILogger<WhisperSpeechToTextService> _logger;
    private readonly string _modelDirectory;
    private readonly SemaphoreSlim _initLock = new(1, 1);
    private WhisperFactory? _factory;

    public WhisperSpeechToTextService(
        IOptions<WhisperOptions> options,
        IWebHostEnvironment env,
        ILogger<WhisperSpeechToTextService> logger)
    {
        _options = options.Value;
        _logger = logger;
        _modelDirectory = Path.IsPathRooted(_options.ModelDirectory)
            ? _options.ModelDirectory
            : Path.Combine(env.ContentRootPath, _options.ModelDirectory);
    }

    public string ModelSize => _options.ModelSize;

    public async Task<TranscriptionResult> TranscribeAsync(
        string wavFilePath, string? language, CancellationToken cancellationToken = default)
    {
        var factory = await GetFactoryAsync(cancellationToken);

        float[] samples = AudioConverter.ConvertToWhisperFormat(wavFilePath);
        var audioDuration = TimeSpan.FromSeconds(samples.Length / 16000.0);

        var requestedLanguage = string.IsNullOrWhiteSpace(language) ? _options.DefaultLanguage : language;

        var builder = factory.CreateBuilder();
        if (_options.Threads > 0)
            builder = builder.WithThreads(_options.Threads);

        builder = requestedLanguage.Equals("auto", StringComparison.OrdinalIgnoreCase)
            ? builder.WithLanguageDetection()
            : builder.WithLanguage(requestedLanguage);

        await using var processor = builder.Build();

        var segments = new List<TranscriptionSegment>();
        var text = new StringBuilder();
        string detectedLanguage = requestedLanguage;

        var stopwatch = Stopwatch.StartNew();
        await foreach (var segment in processor.ProcessAsync(samples, cancellationToken))
        {
            detectedLanguage = segment.Language ?? detectedLanguage;
            segments.Add(new TranscriptionSegment(segment.Start, segment.End, segment.Text.Trim()));
            text.Append(segment.Text);
        }
        stopwatch.Stop();

        _logger.LogInformation(
            "Transcribed {Duration:g} of audio in {Elapsed:g} ({Segments} segments)",
            audioDuration, stopwatch.Elapsed, segments.Count);

        return new TranscriptionResult(
            text.ToString().Trim(),
            segments,
            detectedLanguage,
            audioDuration,
            stopwatch.Elapsed);
    }

    private async Task<WhisperFactory> GetFactoryAsync(CancellationToken cancellationToken)
    {
        if (_factory is not null)
            return _factory;

        await _initLock.WaitAsync(cancellationToken);
        try
        {
            if (_factory is null)
            {
                var modelPath = await EnsureModelAsync(cancellationToken);
                _logger.LogInformation("Loading Whisper model from {Path}", modelPath);
                _factory = WhisperFactory.FromPath(modelPath);
            }
        }
        finally
        {
            _initLock.Release();
        }

        return _factory;
    }

    private async Task<string> EnsureModelAsync(CancellationToken cancellationToken)
    {
        Directory.CreateDirectory(_modelDirectory);
        var modelPath = Path.Combine(_modelDirectory, $"ggml-{_options.ModelSize}.bin");

        if (File.Exists(modelPath))
            return modelPath;

        var url = $"{ModelBaseUrl}/ggml-{_options.ModelSize}.bin";
        _logger.LogInformation("Whisper model not found locally. Downloading {Url} ...", url);

        using var http = new HttpClient { Timeout = TimeSpan.FromMinutes(15) };
        using var response = await http.GetAsync(url, HttpCompletionOption.ResponseHeadersRead, cancellationToken);
        response.EnsureSuccessStatusCode();

        var tempPath = modelPath + ".download";
        await using (var fileStream = File.Create(tempPath))
        await using (var httpStream = await response.Content.ReadAsStreamAsync(cancellationToken))
        {
            await httpStream.CopyToAsync(fileStream, cancellationToken);
        }

        File.Move(tempPath, modelPath, overwrite: true);
        _logger.LogInformation("Model downloaded to {Path}", modelPath);
        return modelPath;
    }

    public void Dispose()
    {
        _factory?.Dispose();
        _initLock.Dispose();
    }
}
