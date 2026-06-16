namespace ArkSummarize.Models;

/// <summary>
/// Configuration for the abstractive (seq2seq) summarization engine, bound from the
/// "Abstractive" section. Defaults target <c>distilbart-cnn-6-6</c> (a distilled BART
/// summarizer with a byte-level BPE tokenizer) whose ONNX encoder/decoder + vocabulary are
/// downloaded once from Hugging Face and cached on disk; everything thereafter runs locally
/// on the CPU. Swap <see cref="ModelRepo"/> to e.g. <c>Xenova/distilbart-xsum-6-6</c> for
/// more aggressive (single-sentence) abstraction.
/// </summary>
public class AbstractiveOptions
{
    public const string SectionName = "Abstractive";

    public string ModelRepo { get; set; } = "Xenova/distilbart-cnn-6-6";
    public string EncoderFile { get; set; } = "onnx/encoder_model_quantized.onnx";
    public string DecoderFile { get; set; } = "onnx/decoder_model_quantized.onnx";
    public string VocabFile { get; set; } = "vocab.json";
    public string MergesFile { get; set; } = "merges.txt";
    public string ModelDirectory { get; set; } = "App_Data/models";

    /// <summary>Optional text prefix (BART needs none; T5-style models use "summarize: ").</summary>
    public string Prefix { get; set; } = "";

    public int MaxInputTokens { get; set; } = 512;
    public int MaxOutputTokens { get; set; } = 120;

    // BART special tokens: bos=0 <s>, pad=1, eos=2 </s>, unk=3.
    public int BosTokenId { get; set; } = 0;
    public int EosTokenId { get; set; } = 2;

    /// <summary>Token the decoder is seeded with (BART uses eos=2).</summary>
    public int DecoderStartTokenId { get; set; } = 2;

    /// <summary>Token forced as the first generated token (BART forces bos=0).</summary>
    public int ForcedBosTokenId { get; set; } = 0;

    public string UnknownToken { get; set; } = "<unk>";

    /// <summary>Blocks repeating n-grams of this size during greedy decoding (0 = off).</summary>
    public int NoRepeatNgramSize { get; set; } = 3;

    public int Threads { get; set; } = 0;
}
