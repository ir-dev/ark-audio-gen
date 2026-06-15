namespace ArkSpeechToText.Models;

/// <summary>
/// Configuration for the CPU-hosted Whisper speech-to-text engine.
/// Bound from the "Whisper" section of appsettings.json.
/// </summary>
public class WhisperOptions
{
    public const string SectionName = "Whisper";

    /// <summary>
    /// ggml model size. Lighter = faster but less accurate.
    /// Valid values: tiny, tiny.en, base, base.en, small, small.en, medium, large-v3.
    /// "base" is a good lightweight CPU default (~150 MB).
    /// </summary>
    public string ModelSize { get; set; } = "base";

    /// <summary>
    /// Directory (relative to content root, or absolute) where the ggml model is cached.
    /// </summary>
    public string ModelDirectory { get; set; } = "App_Data";

    /// <summary>
    /// Default transcription language. "auto" enables language detection.
    /// </summary>
    public string DefaultLanguage { get; set; } = "auto";

    /// <summary>
    /// Number of CPU threads used for inference. 0 = let Whisper decide.
    /// </summary>
    public int Threads { get; set; } = 0;
}
