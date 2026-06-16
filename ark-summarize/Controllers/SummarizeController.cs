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

    private readonly ISummarizationService _summarizer;
    private readonly AppDbContext _db;
    private readonly ILogger<SummarizeController> _logger;

    public SummarizeController(ISummarizationService summarizer, AppDbContext db, ILogger<SummarizeController> logger)
    {
        _summarizer = summarizer;
        _db = db;
        _logger = logger;
    }

    [HttpGet]
    public async Task<IActionResult> Index(CancellationToken cancellationToken)
    {
        var model = new SummarizeViewModel();
        await PopulateApiConsoleAsync(model, cancellationToken);
        return View(model);
    }

    [HttpPost]
    [ValidateAntiForgeryToken]
    public async Task<IActionResult> Index(SummarizeViewModel model, CancellationToken cancellationToken)
    {
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

        try
        {
            model.Result = _summarizer.Summarize(model.Text, model.MaxSentences);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Summarization failed");
            model.ErrorMessage = $"Summarization failed: {ex.Message}";
        }

        return View(model);
    }

    /// <summary>
    /// JSON API: POST <c>{ "text": "...", "maxSentences": 3 }</c>. Authenticated by the
    /// caller's API key sent in their configured custom header (see the profile page).
    /// Returns <c>{ intent, summary, entities, ... }</c>.
    /// </summary>
    [HttpPost("api/summarize")]
    [Authorize(AuthenticationSchemes = ApiKeyAuthenticationOptions.Scheme)]
    [Consumes("application/json")]
    public IActionResult Api([FromBody] SummarizeApiRequest? request)
    {
        if (request is null || string.IsNullOrWhiteSpace(request.Text))
            return BadRequest(new { error = "Field 'text' is required." });

        if (request.Text.Length > MaxChars)
            return BadRequest(new { error = $"Text exceeds the {MaxChars} character limit." });

        try
        {
            var result = _summarizer.Summarize(request.Text, request.MaxSentences ?? 3);
            return Ok(result);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "API summarization failed");
            return StatusCode(500, new { error = ex.Message });
        }
    }

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
}
