using ArkTextTranslator.Models;
using ArkTextTranslator.Services;
using Microsoft.AspNetCore.Mvc;

namespace ArkTextTranslator.Controllers;

public class TranslationController : Controller
{
    private const int MaxInputChars = 20_000;

    private readonly ITranslationService _translator;
    private readonly ILogger<TranslationController> _logger;

    public TranslationController(ITranslationService translator, ILogger<TranslationController> logger)
    {
        _translator = translator;
        _logger = logger;
    }

    [HttpGet]
    public IActionResult Index() =>
        View(new TranslateViewModel { Engine = _translator.EngineName });

    [HttpPost]
    [ValidateAntiForgeryToken]
    public async Task<IActionResult> Index(TranslateViewModel model, CancellationToken cancellationToken)
    {
        model.Engine = _translator.EngineName;
        model.Languages = Languages.All;

        if (string.IsNullOrWhiteSpace(model.Text))
        {
            model.ErrorMessage = "Please enter some text to translate.";
            return View(model);
        }

        if (model.Text.Length > MaxInputChars)
        {
            model.ErrorMessage = $"Text is too long ({model.Text.Length:N0} chars). Limit is {MaxInputChars:N0}.";
            return View(model);
        }

        try
        {
            model.Result = await _translator.TranslateAsync(
                model.Text, model.SourceLanguage, model.TargetLanguage, cancellationToken);
        }
        catch (OperationCanceledException)
        {
            return new EmptyResult();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Translation failed");
            model.ErrorMessage = $"Translation failed: {ex.Message}";
        }

        return View(model);
    }

    /// <summary>
    /// JSON API: POST { text, source?, target } (form or JSON). Returns the translation.
    /// </summary>
    [HttpPost("api/translate")]
    public async Task<IActionResult> Api([FromBody] TranslateRequest request, CancellationToken cancellationToken)
    {
        if (request is null || string.IsNullOrWhiteSpace(request.Text))
            return BadRequest(new { error = "Field 'text' is required." });

        if (request.Text.Length > MaxInputChars)
            return BadRequest(new { error = $"Text too long. Limit is {MaxInputChars} characters." });

        var target = string.IsNullOrWhiteSpace(request.Target) ? "eng_Latn" : request.Target;
        var source = string.IsNullOrWhiteSpace(request.Source) ? "auto" : request.Source;

        try
        {
            var result = await _translator.TranslateAsync(request.Text, source, target, cancellationToken);
            return Ok(new
            {
                translatedText = result.TranslatedText,
                detectedLanguage = result.DetectedLanguage,
                detectionConfidence = Math.Round(result.DetectionConfidence, 3),
                sourceLanguage = result.SourceLanguage,
                targetLanguage = result.TargetLanguage,
                engine = _translator.EngineName,
                processingSeconds = Math.Round(result.ProcessingTime.TotalSeconds, 3),
            });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "API translation failed");
            return StatusCode(500, new { error = ex.Message });
        }
    }

    public record TranslateRequest(string? Text, string? Source, string? Target);
}
