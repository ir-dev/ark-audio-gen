namespace ArkSpeechToText.Models;

/// <summary>A single time-stamped chunk of recognized speech.</summary>
public record TranscriptionSegment(TimeSpan Start, TimeSpan End, string Text);

/// <summary>The full result of a transcription run.</summary>
public record TranscriptionResult(
    string Text,
    IReadOnlyList<TranscriptionSegment> Segments,
    string Language,
    TimeSpan Duration,
    TimeSpan ProcessingTime);
