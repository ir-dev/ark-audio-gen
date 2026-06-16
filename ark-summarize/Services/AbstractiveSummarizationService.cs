using System.Diagnostics;
using ArkSummarize.Models;
using Microsoft.Extensions.Options;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace ArkSummarize.Services;

/// <summary>
/// An <b>abstractive</b> engine: it actually <i>generates</i> a new, shorter summary instead
/// of selecting existing sentences. This is what lets it condense even a single long sentence
/// (which the extractive engines can only echo back).
///
/// <para>
/// It runs a distilled BART summarizer (<c>distilbart-cnn-6-6</c> by default) through ONNX
/// Runtime on the CPU — encoder + greedy decoder, with repeated-n-gram blocking to avoid
/// loops. The model + byte-level BPE vocabulary are downloaded once from Hugging Face and
/// cached on disk; everything thereafter runs locally — no GPU, no cloud. Intent, entities and
/// keywords reuse the shared deterministic extractors (see <see cref="SharedNlp"/>).
/// </para>
/// </summary>
public sealed class AbstractiveSummarizationService : ISummarizationService, IDisposable
{
    public string Key => "abstractive";
    public string DisplayName => "Abstractive (DistilBART, ONNX)";
    public string Description => "Distilled BART seq2seq that generates a new, condensed summary (not just selected sentences). CPU, offline after first download.";

    private readonly AbstractiveOptions _options;
    private readonly ILogger<AbstractiveSummarizationService> _logger;
    private readonly string _modelDirectory;
    private readonly SemaphoreSlim _initLock = new(1, 1);

    private ByteLevelBpeTokenizer? _tokenizer;
    private HashSet<int>? _specialIds;
    private InferenceSession? _encoder;
    private InferenceSession? _decoder;

    private string _encIdsName = "input_ids";
    private string _encMaskName = "attention_mask";
    private string _encHiddenOut = "last_hidden_state";
    private string _decIdsName = "input_ids";
    private string _decEncMaskName = "encoder_attention_mask";
    private string _decEncHiddenName = "encoder_hidden_states";
    private string _decLogitsOut = "logits";

    public AbstractiveSummarizationService(
        IOptions<AbstractiveOptions> options,
        IWebHostEnvironment env,
        ILogger<AbstractiveSummarizationService> logger)
    {
        _options = options.Value;
        _logger = logger;

        var baseDir = Path.IsPathRooted(_options.ModelDirectory)
            ? _options.ModelDirectory
            : Path.Combine(env.ContentRootPath, _options.ModelDirectory);
        _modelDirectory = Path.Combine(baseDir, _options.ModelRepo.Replace('/', '_'));
    }

    public async Task<SummaryResult> SummarizeAsync(string text, int maxSentences = 3, CancellationToken cancellationToken = default)
    {
        var sw = Stopwatch.StartNew();
        text = (text ?? "").Trim();
        maxSentences = Math.Clamp(maxSentences, 1, 10);

        var result = new SummaryResult { Summary = "", Intent = "statement", Engine = DisplayName };
        if (text.Length == 0)
        {
            result.ProcessingMs = sw.Elapsed.TotalMilliseconds;
            return result;
        }

        await EnsureLoadedAsync(cancellationToken);

        var allWords = SharedNlp.WordTokens(text);
        result.SentenceCount = SharedNlp.SplitSentences(text).Count;
        result.WordCount = allWords.Count;
        result.Keywords = SharedNlp.TopKeywords(SharedNlp.WordFrequencies(allWords));
        result.Entities = SharedNlp.ExtractEntities(text);
        (result.Intent, result.IntentConfidence) = SharedNlp.ClassifyIntent(text);

        // Scale the generated length with the requested sentence count.
        int maxOut = Math.Clamp(maxSentences * 32, 24, _options.MaxOutputTokens);
        result.Summary = Generate(text, maxOut, cancellationToken);

        result.ProcessingMs = sw.Elapsed.TotalMilliseconds;
        return result;
    }

    // ---- Generation --------------------------------------------------------
    private string Generate(string text, int maxOutputTokens, CancellationToken cancellationToken)
    {
        // Encoder input: <s> + byte-level BPE token ids + </s>
        var pieceIds = _tokenizer!.Encode(_options.Prefix + text);
        if (pieceIds.Count > _options.MaxInputTokens - 2)
            pieceIds = pieceIds.Take(_options.MaxInputTokens - 2).ToList();

        var inputIds = new List<long>(pieceIds.Count + 2) { _options.BosTokenId };
        foreach (var id in pieceIds) inputIds.Add(id);
        inputIds.Add(_options.EosTokenId);

        var encHidden = RunEncoder(inputIds, out int encLen);

        // Decoder is seeded with the start token and the forced BOS token.
        var decoderIds = new List<long> { _options.DecoderStartTokenId, _options.ForcedBosTokenId };
        var generated = new List<int>();

        for (int step = 0; step < maxOutputTokens; step++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var banned = BannedByNoRepeat(generated);
            int nextId = RunDecoderStep(decoderIds, encHidden, encLen, banned);
            if (nextId == _options.EosTokenId)
                break;
            decoderIds.Add(nextId);
            generated.Add(nextId);
        }

        return DecodeTokens(generated);
    }

    /// <summary>Decodes generated ids → text using the GPT-2/BART byte-level reverse mapping.</summary>
    private string DecodeTokens(List<int> ids)
    {
        var summary = _tokenizer!.Decode(ids, _specialIds!).Trim();

        // Tidy spacing: BART emits punctuation as separate tokens (" .", " ,").
        summary = System.Text.RegularExpressions.Regex.Replace(summary, @"\s+([.,;:!?%])", "$1");
        summary = System.Text.RegularExpressions.Regex.Replace(summary, @"\(\s+", "(");
        summary = System.Text.RegularExpressions.Regex.Replace(summary, @"\s+\)", ")");
        summary = System.Text.RegularExpressions.Regex.Replace(summary, @"\s{2,}", " ").Trim();

        if (summary.Length > 0 && char.IsLower(summary[0]))
            summary = char.ToUpperInvariant(summary[0]) + summary[1..];
        return summary;
    }

    /// <summary>Tokens that would complete an already-seen n-gram (prevents decode loops).</summary>
    private HashSet<int> BannedByNoRepeat(List<int> generated)
    {
        var banned = new HashSet<int>();
        int n = _options.NoRepeatNgramSize;
        if (n <= 0 || generated.Count < n - 1) return banned;

        int k = n - 1;
        var prefix = generated.GetRange(generated.Count - k, k);
        for (int i = 0; i + k < generated.Count; i++)
        {
            bool match = true;
            for (int j = 0; j < k; j++)
                if (generated[i + j] != prefix[j]) { match = false; break; }
            if (match) banned.Add(generated[i + k]);
        }
        return banned;
    }

    private DenseTensor<float> RunEncoder(IReadOnlyList<long> inputIds, out int seqLen)
    {
        seqLen = inputIds.Count;
        var ids = new DenseTensor<long>(inputIds.ToArray(), new[] { 1, seqLen });
        var mask = new DenseTensor<long>(Enumerable.Repeat(1L, seqLen).ToArray(), new[] { 1, seqLen });

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(_encIdsName, ids),
            NamedOnnxValue.CreateFromTensor(_encMaskName, mask),
        };

        using var results = _encoder!.Run(inputs);
        var hidden = results.First(r => r.Name == _encHiddenOut).AsTensor<float>();
        var dims = hidden.Dimensions.ToArray();
        return new DenseTensor<float>(hidden.ToArray(), dims);
    }

    private int RunDecoderStep(IReadOnlyList<long> decoderIds, DenseTensor<float> encHidden, int encLen, HashSet<int> banned)
    {
        int decLen = decoderIds.Count;
        var ids = new DenseTensor<long>(decoderIds.ToArray(), new[] { 1, decLen });
        var encMask = new DenseTensor<long>(Enumerable.Repeat(1L, encLen).ToArray(), new[] { 1, encLen });

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(_decIdsName, ids),
            NamedOnnxValue.CreateFromTensor(_decEncMaskName, encMask),
            NamedOnnxValue.CreateFromTensor(_decEncHiddenName, encHidden),
        };

        using var results = _decoder!.Run(inputs);
        var logits = results.First(r => r.Name == _decLogitsOut).AsTensor<float>();

        int vocab = logits.Dimensions[^1];
        int last = decLen - 1;

        int bestId = 0;
        float bestVal = float.NegativeInfinity;
        for (int v = 0; v < vocab; v++)
        {
            if (banned.Contains(v)) continue;
            float val = logits[0, last, v];
            if (val > bestVal) { bestVal = val; bestId = v; }
        }
        return bestId;
    }

    // ---- Initialisation ----------------------------------------------------
    private async Task EnsureLoadedAsync(CancellationToken cancellationToken)
    {
        if (_encoder is not null) return;

        await _initLock.WaitAsync(cancellationToken);
        try
        {
            if (_encoder is not null) return;

            Directory.CreateDirectory(_modelDirectory);
            var vocabPath = await EnsureFileAsync(_options.VocabFile, cancellationToken);
            var mergesPath = await EnsureFileAsync(_options.MergesFile, cancellationToken);
            var encPath = await EnsureFileAsync(_options.EncoderFile, cancellationToken);
            var decPath = await EnsureFileAsync(_options.DecoderFile, cancellationToken);

            _logger.LogInformation("Loading byte-level BPE tokenizer from {Path}", vocabPath);
            _tokenizer = ByteLevelBpeTokenizer.FromFiles(vocabPath, mergesPath, _options.UnknownToken);

            _specialIds = new HashSet<int> { _options.BosTokenId, _options.EosTokenId };
            foreach (var t in new[] { "<pad>", _options.UnknownToken, "<mask>", "<s>", "</s>" })
                if (_tokenizer.TryGetId(t, out var id)) _specialIds.Add(id);

            var sessionOptions = new Microsoft.ML.OnnxRuntime.SessionOptions();
            if (_options.Threads > 0)
            {
                sessionOptions.IntraOpNumThreads = _options.Threads;
                sessionOptions.InterOpNumThreads = _options.Threads;
            }

            _logger.LogInformation("Loading DistilBART ONNX encoder/decoder (this can take a moment) ...");
            _encoder = new InferenceSession(encPath, sessionOptions);
            _decoder = new InferenceSession(decPath, sessionOptions);
            ResolveTensorNames();
        }
        finally
        {
            _initLock.Release();
        }
    }

    private void ResolveTensorNames()
    {
        var encIn = _encoder!.InputMetadata.Keys.ToList();
        _encIdsName = encIn.FirstOrDefault(n => n.Contains("input_ids", StringComparison.OrdinalIgnoreCase)) ?? _encIdsName;
        _encMaskName = encIn.FirstOrDefault(n => n.Contains("attention_mask", StringComparison.OrdinalIgnoreCase)) ?? _encMaskName;
        _encHiddenOut = _encoder.OutputMetadata.Keys.First();

        var decIn = _decoder!.InputMetadata.Keys.ToList();
        _decEncHiddenName = decIn.FirstOrDefault(n => n.Contains("encoder_hidden", StringComparison.OrdinalIgnoreCase)) ?? _decEncHiddenName;
        _decEncMaskName = decIn.FirstOrDefault(n => n.Contains("encoder_attention", StringComparison.OrdinalIgnoreCase)) ?? _decEncMaskName;
        _decIdsName = decIn.FirstOrDefault(n =>
                          n.Contains("input_ids", StringComparison.OrdinalIgnoreCase) &&
                          !n.Contains("encoder", StringComparison.OrdinalIgnoreCase))
                      ?? decIn.First(n => !n.Contains("encoder", StringComparison.OrdinalIgnoreCase));
        _decLogitsOut = _decoder.OutputMetadata.Keys.FirstOrDefault(n => n.Contains("logits", StringComparison.OrdinalIgnoreCase))
                        ?? _decoder.OutputMetadata.Keys.First();

        _logger.LogInformation(
            "Abstractive I/O resolved — encoder({Ids},{Mask})->{Hidden}; decoder({DIds},{EMask},{EHidden})->{Logits}",
            _encIdsName, _encMaskName, _encHiddenOut, _decIdsName, _decEncMaskName, _decEncHiddenName, _decLogitsOut);
    }

    private async Task<string> EnsureFileAsync(string relativePath, CancellationToken cancellationToken)
    {
        var localPath = Path.Combine(_modelDirectory, relativePath.Replace('/', Path.DirectorySeparatorChar));
        if (File.Exists(localPath))
            return localPath;

        Directory.CreateDirectory(Path.GetDirectoryName(localPath)!);
        var url = $"https://huggingface.co/{_options.ModelRepo}/resolve/main/{relativePath}";
        _logger.LogInformation("Downloading {Url} ...", url);

        using var http = new HttpClient { Timeout = TimeSpan.FromMinutes(30) };
        using var response = await http.GetAsync(url, HttpCompletionOption.ResponseHeadersRead, cancellationToken);
        response.EnsureSuccessStatusCode();

        var tempPath = localPath + ".download";
        await using (var fileStream = File.Create(tempPath))
        await using (var httpStream = await response.Content.ReadAsStreamAsync(cancellationToken))
        {
            await httpStream.CopyToAsync(fileStream, cancellationToken);
        }

        File.Move(tempPath, localPath, overwrite: true);
        _logger.LogInformation("Saved {Path}", localPath);
        return localPath;
    }

    public void Dispose()
    {
        _encoder?.Dispose();
        _decoder?.Dispose();
        _initLock.Dispose();
    }
}
