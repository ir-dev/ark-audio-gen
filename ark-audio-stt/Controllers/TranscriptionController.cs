using System.Security.Claims;
using ArkSpeechToText.Data;
using ArkSpeechToText.Models;
using ArkSpeechToText.Services;
using ArkSpeechToText.Services.Auth;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;

namespace ArkSpeechToText.Controllers;

[Authorize]
public class TranscriptionController : Controller
{
    private const long MaxUploadBytes = 200L * 1024 * 1024; // 200 MB

    private readonly ISpeechToTextService _speechToText;
    private readonly AppDbContext _db;
    private readonly ILogger<TranscriptionController> _logger;

    public TranscriptionController(ISpeechToTextService speechToText, AppDbContext db, ILogger<TranscriptionController> logger)
    {
        _speechToText = speechToText;
        _db = db;
        _logger = logger;
    }

    [HttpGet]
    public async Task<IActionResult> Index(CancellationToken cancellationToken)
    {
        var model = new TranscribeViewModel { ModelSize = _speechToText.ModelSize };
        await PopulateApiConsoleAsync(model, cancellationToken);
        return View(model);
    }

    [HttpPost]
    [ValidateAntiForgeryToken]
    [RequestSizeLimit(MaxUploadBytes)]
    [RequestFormLimits(MultipartBodyLengthLimit = MaxUploadBytes)]
    public async Task<IActionResult> Index(TranscribeViewModel model, CancellationToken cancellationToken)
    {
        model.ModelSize = _speechToText.ModelSize;
        await PopulateApiConsoleAsync(model, cancellationToken);

        if (model.File is null || model.File.Length == 0)
        {
            model.ErrorMessage = "Please choose a non-empty audio file.";
            return View(model);
        }

        if (!AudioConverter.IsSupported(model.File.FileName))
        {
            model.ErrorMessage = $"Unsupported file type. Supported: {string.Join(", ", AudioConverter.SupportedExtensions)}.";
            return View(model);
        }

        var tempPath = NewTempPath(model.File.FileName);
        try
        {
            await using (var stream = System.IO.File.Create(tempPath))
            {
                await model.File.CopyToAsync(stream, cancellationToken);
            }

            model.Result = await _speechToText.TranscribeAsync(tempPath, model.Language, cancellationToken);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Transcription failed for {File}", model.File.FileName);
            model.ErrorMessage = $"Transcription failed: {ex.Message}";
        }
        finally
        {
            TryDelete(tempPath);
        }

        return View(model);
    }

    /// <summary>
    /// JSON API: POST a multipart form with a "file" field (and optional "language").
    /// Authenticated by the caller's API key sent in their configured custom header
    /// (see the profile page).
    /// </summary>
    [HttpPost("api/transcribe")]
    [Authorize(AuthenticationSchemes = ApiKeyAuthenticationOptions.Scheme)]
    [RequestSizeLimit(MaxUploadBytes)]
    [RequestFormLimits(MultipartBodyLengthLimit = MaxUploadBytes)]
    public async Task<IActionResult> Api(IFormFile? file, [FromForm] string? language, CancellationToken cancellationToken)
    {
        if (file is null || file.Length == 0)
            return BadRequest(new { error = "No file uploaded." });

        if (!AudioConverter.IsSupported(file.FileName))
            return BadRequest(new { error = $"Unsupported file type. Supported: {string.Join(", ", AudioConverter.SupportedExtensions)}." });

        var tempPath = NewTempPath(file.FileName);
        try
        {
            await using (var stream = System.IO.File.Create(tempPath))
            {
                await file.CopyToAsync(stream, cancellationToken);
            }

            var result = await _speechToText.TranscribeAsync(tempPath, language, cancellationToken);
            return Ok(new
            {
                text = result.Text,
                language = result.Language,
                durationSeconds = result.Duration.TotalSeconds,
                processingSeconds = result.ProcessingTime.TotalSeconds,
                segments = result.Segments.Select(s => new
                {
                    start = s.Start.TotalSeconds,
                    end = s.End.TotalSeconds,
                    text = s.Text
                })
            });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "API transcription failed for {File}", file.FileName);
            return StatusCode(500, new { error = ex.Message });
        }
        finally
        {
            TryDelete(tempPath);
        }
    }

    private async Task PopulateApiConsoleAsync(TranscribeViewModel model, CancellationToken cancellationToken)
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

    private static string NewTempPath(string originalFileName)
    {
        var ext = Path.GetExtension(originalFileName).ToLowerInvariant();
        return Path.Combine(Path.GetTempPath(), $"ark-stt-{Guid.NewGuid():N}{ext}");
    }

    private static void TryDelete(string path)
    {
        try
        {
            if (System.IO.File.Exists(path))
                System.IO.File.Delete(path);
        }
        catch
        {
            // best effort cleanup
        }
    }
}
