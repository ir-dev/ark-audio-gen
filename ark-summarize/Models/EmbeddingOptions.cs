namespace ArkSummarize.Models;

/// <summary>
/// Configuration for the MiniLM embedding engine (bound from the "Embedding" section).
/// The ONNX model + vocabulary are downloaded once from Hugging Face and cached on disk;
/// everything thereafter runs locally on the CPU.
/// </summary>
public class EmbeddingOptions
{
    public const string SectionName = "Embedding";

    /// <summary>Hugging Face repo to pull the model + vocab from.</summary>
    public string ModelRepo { get; set; } = "Xenova/all-MiniLM-L6-v2";

    /// <summary>Relative path of the ONNX model file within the repo. Quantized (int8) by default.</summary>
    public string ModelFile { get; set; } = "onnx/model_quantized.onnx";

    /// <summary>Relative path of the WordPiece vocabulary within the repo.</summary>
    public string VocabFile { get; set; } = "vocab.txt";

    /// <summary>Where downloaded model files are cached (relative paths are under the content root).</summary>
    public string ModelDirectory { get; set; } = "App_Data/models";

    /// <summary>Maximum tokens fed to the encoder per text (MiniLM supports up to 256/512).</summary>
    public int MaxTokens { get; set; } = 256;

    /// <summary>ONNX Runtime intra/inter-op thread count (0 = let the runtime decide).</summary>
    public int Threads { get; set; } = 0;
}
