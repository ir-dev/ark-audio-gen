namespace ArkTextTranslator.Models;

/// <summary>
/// Configuration for the translation pipeline. Bound from the "Translation"
/// section of appsettings.json.
/// </summary>
public class TranslationOptions
{
    public const string SectionName = "Translation";

    /// <summary>
    /// Which translation engine to use:
    /// <list type="bullet">
    /// <item><c>Onnx</c> — offline, CPU-hosted NLLB-200 model via ONNX Runtime (default).</item>
    /// <item><c>LibreTranslate</c> — calls a LibreTranslate HTTP endpoint.</item>
    /// </list>
    /// </summary>
    public string Provider { get; set; } = "Onnx";

    /// <summary>Offline ONNX NLLB engine settings.</summary>
    public OnnxOptions Onnx { get; set; } = new();

    /// <summary>LibreTranslate HTTP engine settings.</summary>
    public LibreTranslateOptions LibreTranslate { get; set; } = new();
}

/// <summary>Settings for the CPU-hosted ONNX NLLB translator.</summary>
public class OnnxOptions
{
    /// <summary>
    /// Hugging Face repo (Xenova ONNX export) the model + tokenizer are pulled from
    /// on first run. The default is the distilled 600M model — a good CPU/quality balance.
    /// </summary>
    public string ModelRepo { get; set; } = "Xenova/nllb-200-distilled-600M";

    /// <summary>
    /// Quantization variant of the ONNX weights to download.
    /// "" = full fp32, "_quantized" = int8 (smaller + faster on CPU, slightly lower quality).
    /// </summary>
    public string QuantizationSuffix { get; set; } = "_quantized";

    /// <summary>Directory (relative to content root, or absolute) where model files are cached.</summary>
    public string ModelDirectory { get; set; } = "App_Data";

    /// <summary>Number of CPU threads used for inference. 0 = let ONNX Runtime decide.</summary>
    public int Threads { get; set; } = 0;

    /// <summary>Hard cap on generated tokens per sentence (safety against runaway decoding).</summary>
    public int MaxOutputTokens { get; set; } = 256;
}

/// <summary>Settings for the LibreTranslate HTTP engine.</summary>
public class LibreTranslateOptions
{
    /// <summary>Base URL of the LibreTranslate server, e.g. http://localhost:5000.</summary>
    public string Endpoint { get; set; } = "http://localhost:5000";

    /// <summary>Optional API key (only required by some hosted instances).</summary>
    public string? ApiKey { get; set; }
}
