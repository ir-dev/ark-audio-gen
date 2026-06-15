using Concentus;
using Concentus.Oggfile;
using NAudio.Vorbis;
using NAudio.Wave;
using NAudio.Wave.SampleProviders;
using NLayer.NAudioSupport;

namespace ArkSpeechToText.Services;

/// <summary>
/// Converts WAV/MP3/OGG/Opus audio into the 16 kHz, mono, 32-bit float PCM
/// format that Whisper expects. Uses NAudio's fully-managed resampler plus
/// managed decoders (NLayer for MP3, NVorbis for OGG Vorbis, Concentus for
/// Opus), so it runs on any platform (Windows / Linux / macOS) without native
/// codecs.
/// </summary>
public static class AudioConverter
{
    private const int WhisperSampleRate = 16000;

    public static readonly string[] SupportedExtensions = { ".wav", ".mp3", ".ogg", ".opus" };

    public static bool IsSupported(string fileName) =>
        SupportedExtensions.Contains(Path.GetExtension(fileName).ToLowerInvariant());

    public static float[] ConvertToWhisperFormat(string filePath)
    {
        var extension = Path.GetExtension(filePath).ToLowerInvariant();
        switch (extension)
        {
            case ".wav":
                using (var reader = new WaveFileReader(filePath))
                    return ToWhisperSamples(reader);

            case ".mp3":
                // NLayer's managed decoder works cross-platform (NAudio's built-in
                // MP3 reader relies on Windows-only ACM/DMO codecs). Mp3FileReaderBase
                // does not own the stream, so dispose the FileStream ourselves.
                using (var fileStream = File.OpenRead(filePath))
                using (var reader = new Mp3FileReaderBase(fileStream, wf => new Mp3FrameDecompressor(wf)))
                    return ToWhisperSamples(reader);

            case ".opus":
                // Always Ogg-encapsulated Opus.
                return DecodeOggOpus(filePath);

            case ".ogg":
                // An .ogg container may hold Vorbis or Opus; sniff the codec.
                if (IsOggOpus(filePath))
                    return DecodeOggOpus(filePath);
                using (var reader = new VorbisWaveReader(filePath)) // NVorbis (managed)
                    return ToWhisperSamples(reader);

            default:
                throw new NotSupportedException(
                    $"Unsupported audio format '{extension}'. Supported: {string.Join(", ", SupportedExtensions)}");
        }
    }

    /// <summary>True if the Ogg stream carries Opus (vs. Vorbis).</summary>
    private static bool IsOggOpus(string filePath)
    {
        using var fs = File.OpenRead(filePath);
        Span<byte> header = stackalloc byte[256];
        int read = fs.Read(header);
        // The Opus identification header begins with the magic "OpusHead".
        return header[..read].IndexOf("OpusHead"u8) >= 0;
    }

    /// <summary>
    /// Decodes Ogg Opus to 16 kHz mono float samples using Concentus (managed).
    /// The Opus decoder resamples and down-mixes natively, so no NAudio stage is
    /// needed.
    /// </summary>
    private static float[] DecodeOggOpus(string filePath)
    {
        using var fileStream = File.OpenRead(filePath);
        var decoder = OpusCodecFactory.CreateDecoder(WhisperSampleRate, 1);
        var oggStream = new OpusOggReadStream(decoder, fileStream);

        var samples = new List<float>();
        while (oggStream.HasNextPacket)
        {
            short[]? packet = oggStream.DecodeNextPacket();
            if (packet is null)
                continue;
            foreach (var sample in packet)
                samples.Add(sample / 32768f);
        }

        return samples.ToArray();
    }

    private static float[] ToWhisperSamples(WaveStream reader)
    {
        ISampleProvider sampleProvider = reader.ToSampleProvider();

        // Down-mix to a single channel by averaging.
        if (sampleProvider.WaveFormat.Channels > 1)
            sampleProvider = new MonoDownmixSampleProvider(sampleProvider);

        // Resample to 16 kHz if needed.
        if (sampleProvider.WaveFormat.SampleRate != WhisperSampleRate)
            sampleProvider = new WdlResamplingSampleProvider(sampleProvider, WhisperSampleRate);

        var samples = new List<float>();
        var buffer = new float[WhisperSampleRate]; // 1 second at a time
        int read;
        while ((read = sampleProvider.Read(buffer, 0, buffer.Length)) > 0)
        {
            for (int i = 0; i < read; i++)
                samples.Add(buffer[i]);
        }

        return samples.ToArray();
    }

    /// <summary>Averages all source channels into a single mono channel.</summary>
    private sealed class MonoDownmixSampleProvider : ISampleProvider
    {
        private readonly ISampleProvider _source;
        private readonly int _channels;
        private float[] _sourceBuffer = Array.Empty<float>();

        public WaveFormat WaveFormat { get; }

        public MonoDownmixSampleProvider(ISampleProvider source)
        {
            _source = source;
            _channels = source.WaveFormat.Channels;
            WaveFormat = WaveFormat.CreateIeeeFloatWaveFormat(source.WaveFormat.SampleRate, 1);
        }

        public int Read(float[] buffer, int offset, int count)
        {
            int sourceSamplesRequired = count * _channels;
            if (_sourceBuffer.Length < sourceSamplesRequired)
                _sourceBuffer = new float[sourceSamplesRequired];

            int sourceRead = _source.Read(_sourceBuffer, 0, sourceSamplesRequired);

            int written = 0;
            for (int n = 0; n + _channels <= sourceRead; n += _channels)
            {
                float sum = 0f;
                for (int ch = 0; ch < _channels; ch++)
                    sum += _sourceBuffer[n + ch];
                buffer[offset + written++] = sum / _channels;
            }

            return written;
        }
    }
}
