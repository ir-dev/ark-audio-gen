using System.Security.Claims;
using ArkSummarize.Data;
using ArkSummarize.Models;
using ArkSummarize.Services;
using ArkSummarize.Services.Auth;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;

namespace ArkSummarize.Controllers;

[Authorize]
public class SummarizeController : Controller
{
    private const int MaxChars = 100_000;

    private readonly SummarizationEngineProvider _engines;
    private readonly AppDbContext _db;
    private readonly ILogger<SummarizeController> _logger;

    public SummarizeController(SummarizationEngineProvider engines, AppDbContext db, ILogger<SummarizeController> logger)
    {
        _engines = engines;
        _db = db;
        _logger = logger;
    }

    [HttpGet]
    public async Task<IActionResult> Index(CancellationToken cancellationToken)
    {
        var model = new SummarizeViewModel { SelectedEngines = { SummarizationEngineProvider.DefaultKey } };
        PopulateEngines(model);
        await PopulateApiConsoleAsync(model, cancellationToken);
        return View(model);
    }

    [HttpPost]
    [ValidateAntiForgeryToken]
    public async Task<IActionResult> Index(SummarizeViewModel model, CancellationToken cancellationToken)
    {
        PopulateEngines(model);
        await PopulateApiConsoleAsync(model, cancellationToken);

        if (string.IsNullOrWhiteSpace(model.Text))
        {
            model.ErrorMessage = "Please paste some text to summarize.";
            return View(model);
        }

        if (model.Text.Length > MaxChars)
        {
            model.ErrorMessage = $"Text is too long ({model.Text.Length:N0} chars). The limit is {MaxChars:N0} characters.";
            return View(model);
        }

        // Resolve the chosen engines (default to lexical if none ticked); keep them in registry order.
        var requested = model.SelectedEngines
            .Where(_engines.IsKnown)
            .ToHashSet(StringComparer.OrdinalIgnoreCase);
        if (requested.Count == 0)
            requested.Add(SummarizationEngineProvider.DefaultKey);

        foreach (var engine in _engines.Engines.Where(e => requested.Contains(e.Key)))
        {
            var run = new EngineRunResult { Key = engine.Key, DisplayName = engine.DisplayName };
            try
            {
                run.Result = await engine.SummarizeAsync(model.Text, model.MaxSentences, cancellationToken);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Engine {Engine} failed", engine.Key);
                run.Error = ex.Message;
            }
            model.Results.Add(run);
        }

        return View(model);
    }

    /// <summary>
    /// JSON API: POST <c>{ "text": "...", "maxSentences": 3, "model": "lexical" }</c>.
    /// Authenticated by the caller's API key sent in their configured custom header.
    /// The <c>model</c> field is optional and defaults to the lexical engine.
    /// Returns <c>{ engine, intent, summary, entities, ... }</c>.
    /// </summary>
    [HttpPost("api/summarize")]
    [Authorize(AuthenticationSchemes = ApiKeyAuthenticationOptions.Scheme)]
    [Consumes("application/json")]
    public async Task<IActionResult> Api([FromBody] SummarizeApiRequest? request, CancellationToken cancellationToken)
    {
        if (request is null || string.IsNullOrWhiteSpace(request.Text))
            return BadRequest(new { error = "Field 'text' is required." });

        if (request.Text.Length > MaxChars)
            return BadRequest(new { error = $"Text exceeds the {MaxChars} character limit." });

        // Unknown model names fall back to the default rather than failing the request.
        if (!string.IsNullOrWhiteSpace(request.Model) && !_engines.IsKnown(request.Model))
            return BadRequest(new
            {
                error = $"Unknown model '{request.Model}'.",
                availableModels = _engines.Engines.Select(e => e.Key)
            });

        var engine = _engines.Resolve(request.Model);
        try
        {
            var result = await engine.SummarizeAsync(request.Text, request.MaxSentences ?? 3, cancellationToken);
            return Ok(result);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "API summarization failed for engine {Engine}", engine.Key);
            return StatusCode(500, new { error = ex.Message, engine = engine.Key });
        }
    }

    /// <summary>Lists the available engine keys (handy for API consumers).</summary>
    [HttpGet("api/models")]
    [Authorize(AuthenticationSchemes = ApiKeyAuthenticationOptions.Scheme)]
    public IActionResult Models() =>
        Ok(new
        {
            @default = SummarizationEngineProvider.DefaultKey,
            models = _engines.Engines.Select(e => new { key = e.Key, name = e.DisplayName, description = e.Description })
        });

    private void PopulateEngines(SummarizeViewModel model) =>
        model.AvailableEngines = _engines.Engines
            .Select(e => new EngineOption(e.Key, e.DisplayName, e.Description))
            .ToList();

    private async Task PopulateApiConsoleAsync(SummarizeViewModel model, CancellationToken cancellationToken)
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
}

/// <summary>Request body for <c>POST /api/summarize</c>.</summary>
public class SummarizeApiRequest
{
    public string? Text { get; set; }
    public int? MaxSentences { get; set; }

    /// <summary>Engine key (e.g. "lexical" or "minilm"). Optional; defaults to the lexical engine.</summary>
    public string? Model { get; set; }
}
