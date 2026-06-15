using ArkSpeechToText.Models;
using ArkSpeechToText.Services;
using Microsoft.AspNetCore.Mvc;

namespace ArkSpeechToText.Controllers;

public class TranscriptionController : Controller
{
    private const long MaxUploadBytes = 200L * 1024 * 1024; // 200 MB

    private readonly ISpeechToTextService _speechToText;
    private readonly ILogger<TranscriptionController> _logger;

    public TranscriptionController(ISpeechToTextService speechToText, ILogger<TranscriptionController> logger)
    {
        _speechToText = speechToText;
        _logger = logger;
    }

    [HttpGet]
    public IActionResult Index() => View(new TranscribeViewModel { ModelSize = _speechToText.ModelSize });

    [HttpPost]
    [ValidateAntiForgeryToken]
    [RequestSizeLimit(MaxUploadBytes)]
    [RequestFormLimits(MultipartBodyLengthLimit = MaxUploadBytes)]
    public async Task<IActionResult> Index(TranscribeViewModel model, CancellationToken cancellationToken)
    {
        model.ModelSize = _speechToText.ModelSize;

        if (model.File is null || model.File.Length == 0)
        {
            model.ErrorMessage = "Please choose a non-empty .wav file.";
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

    /// <summary>JSON API: POST a multipart form with a "file" field. Returns the transcription.</summary>
    [HttpPost("api/transcribe")]
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
