using System.Security.Claims;
using ArkTextTranslator.Data;
using ArkTextTranslator.Models;
using ArkTextTranslator.Services;
using ArkTextTranslator.Services.Auth;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;

namespace ArkTextTranslator.Controllers;

[Authorize]
public class TranslationController : Controller
{
    private const int MaxInputChars = 20_000;

    private readonly ITranslationService _translator;
    private readonly AppDbContext _db;
    private readonly ILogger<TranslationController> _logger;

    public TranslationController(ITranslationService translator, AppDbContext db, ILogger<TranslationController> logger)
    {
        _translator = translator;
        _db = db;
        _logger = logger;
    }

    [HttpGet]
    public async Task<IActionResult> Index(CancellationToken cancellationToken)
    {
        var model = new TranslateViewModel { Engine = _translator.EngineName };
        await PopulateApiConsoleAsync(model, cancellationToken);
        return View(model);
    }

    [HttpPost]
    [ValidateAntiForgeryToken]
    public async Task<IActionResult> Index(TranslateViewModel model, CancellationToken cancellationToken)
    {
        model.Engine = _translator.EngineName;
        model.Languages = Languages.All;
        await PopulateApiConsoleAsync(model, cancellationToken);

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
    /// JSON API: POST { text, source?, target }. Authenticated by the caller's API key
    /// sent in their configured custom header (see the profile page).
    /// </summary>
    [HttpPost("api/translate")]
    [Authorize(AuthenticationSchemes = ApiKeyAuthenticationOptions.Scheme)]
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

    private async Task PopulateApiConsoleAsync(TranslateViewModel model, CancellationToken cancellationToken)
    {
        model.ApiBaseUrl = $"{Request.Scheme}://{Request.Host}";
        var email = User.FindFirstValue(ClaimTypes.Email);
        var user = await _db.Users.AsNoTracking().FirstOrDefaultAsync(u => u.Email == email, cancellationToken);
        if (user is not null)
        {
            model.ApiKeyName = user.ApiKeyName;
            model.ApiKey = user.ApiKey;
        }
    }

    public record TranslateRequest(string? Text, string? Source, string? Target);
}
