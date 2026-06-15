using ArkSpeechToText.Models;

namespace ArkSpeechToText.Services;

public interface ISpeechToTextService
{
    /// <summary>The model size currently in use (e.g. "base").</summary>
    string ModelSize { get; }

    /// <summary>Transcribes a WAV file to text on the CPU.</summary>
    /// <param name="wavFilePath">Path to a .wav file on disk.</param>
    /// <param name="language">BCP-47 code (e.g. "en") or "auto" for detection.</param>
    Task<TranscriptionResult> TranscribeAsync(string wavFilePath, string? language, CancellationToken cancellationToken = default);
}
