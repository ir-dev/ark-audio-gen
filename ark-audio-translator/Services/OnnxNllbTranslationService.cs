using System.Diagnostics;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using ArkTextTranslator.Models;
using Microsoft.Extensions.Options;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.Tokenizers;

namespace ArkTextTranslator.Services;

/// <summary>
/// Fully-offline, CPU-hosted translation using Meta's <b>NLLB-200</b> (No Language
/// Left Behind) seq2seq model run through ONNX Runtime. The model + tokenizer are
/// downloaded once from Hugging Face and cached on disk; everything thereafter runs
/// locally on the CPU — no GPU, no cloud API.
///
/// <para>
/// Tokenization reuses the original SentencePiece model via
/// <see cref="SentencePieceTokenizer"/>. NLLB ids are the SentencePiece ids plus the
/// fairseq offset of 1 (verified empirically against the model's <c>tokenizer.json</c>),
/// with the source-language token prepended and <c>&lt;/s&gt;</c> appended. Decoding is a
/// greedy argmax loop that forces the target-language token first, matching NLLB's
/// <c>decoder_start_token_id = 2</c> / <c>forced_bos_token_id = target</c> scheme.
/// </para>
/// </summary>
public sealed class OnnxNllbTranslationService : ITranslationService, IDisposable
{
    private const int FairseqOffset = 1;       // NLLB id = SentencePiece id + 1
    private const int EosTokenId = 2;          // "</s>" — also the decoder start token
    private const int FirstLangTokenId = 256001;

    private readonly OnnxOptions _options;
    private readonly ILanguageDetector _detector;
    private readonly ILogger<OnnxNllbTranslationService> _logger;
    private readonly string _modelDirectory;
    private readonly SemaphoreSlim _initLock = new(1, 1);

    private SentencePieceTokenizer? _tokenizer;
    private IReadOnlyDictionary<string, int>? _langTokenIds;
    private InferenceSession? _encoder;
    private InferenceSession? _decoder;

    // Resolved (name-adaptive) tensor names for the two graphs.
    private string _encIdsName = "input_ids";
    private string _encMaskName = "attention_mask";
    private string _encHiddenOut = "last_hidden_state";
    private string _decIdsName = "input_ids";
    private string _decEncMaskName = "encoder_attention_mask";
    private string _decEncHiddenName = "encoder_hidden_states";
    private string _decLogitsOut = "logits";

    public OnnxNllbTranslationService(
        IOptions<TranslationOptions> options,
        IWebHostEnvironment env,
        ILanguageDetector detector,
        ILogger<OnnxNllbTranslationService> logger)
    {
        _options = options.Value.Onnx;
        _detector = detector;
        _logger = logger;

        var baseDir = Path.IsPathRooted(_options.ModelDirectory)
            ? _options.ModelDirectory
            : Path.Combine(env.ContentRootPath, _options.ModelDirectory);
        _modelDirectory = Path.Combine(baseDir, Sanitize(_options.ModelRepo));
    }

    public string EngineName => $"NLLB-200 (ONNX, {(_options.QuantizationSuffix == "" ? "fp32" : "int8")})";

    public async Task<TranslationResult> TranslateAsync(
        string text, string sourceFlores, string targetFlores, CancellationToken cancellationToken = default)
    {
        await EnsureLoadedAsync(cancellationToken);

        var target = Languages.FromFlores(targetFlores)
            ?? throw new ArgumentException($"Unsupported target language '{targetFlores}'.", nameof(targetFlores));

        var stopwatch = Stopwatch.StartNew();

        // Resolve the source language (explicit or auto-detected).
        Language? source;
        string detectedName;
        double confidence;
        string sourceCode;
        if (sourceFlores.Equals("auto", StringComparison.OrdinalIgnoreCase))
        {
            var (lang, iso, conf) = _detector.Detect(text);
            source = lang;
            detectedName = lang?.Name ?? iso;
            confidence = conf;
            // Fall back to English source token if the language isn't one NLLB knows.
            sourceCode = lang?.Flores ?? "eng_Latn";
        }
        else
        {
            source = Languages.FromFlores(sourceFlores)
                ?? throw new ArgumentException($"Unsupported source language '{sourceFlores}'.", nameof(sourceFlores));
            detectedName = source.Name;
            confidence = 1d;
            sourceCode = source.Flores;
        }

        var srcLangId = ResolveLangId(sourceCode);
        var tgtLangId = ResolveLangId(target.Flores);

        // Translate sentence-by-sentence so long inputs stay responsive and within
        // the model's context window.
        var output = new StringBuilder();
        foreach (var sentence in SplitSentences(text))
        {
            cancellationToken.ThrowIfCancellationRequested();
            var translated = TranslateSentence(sentence, srcLangId, tgtLangId, cancellationToken);
            if (output.Length > 0 && translated.Length > 0)
                output.Append(' ');
            output.Append(translated);
        }

        stopwatch.Stop();

        return new TranslationResult(
            output.ToString().Trim(),
            detectedName,
            confidence,
            source?.Flores ?? sourceCode,
            target.Flores,
            stopwatch.Elapsed);
    }

    private string TranslateSentence(string sentence, int srcLangId, int tgtLangId, CancellationToken cancellationToken)
    {
        if (string.IsNullOrWhiteSpace(sentence))
            return string.Empty;

        // --- Encode: [srcLang] + pieces(+offset) + </s> ---
        var pieceIds = _tokenizer!.EncodeToIds(sentence, addBeginningOfSentence: false, addEndOfSentence: false);
        var inputIds = new List<long>(pieceIds.Count + 2) { srcLangId };
        foreach (var id in pieceIds)
            inputIds.Add(id + FairseqOffset);
        inputIds.Add(EosTokenId);

        var encHidden = RunEncoder(inputIds, out int encLen);

        // --- Greedy decode, forcing the target-language token first ---
        var decoderIds = new List<long> { EosTokenId, tgtLangId };
        for (int step = 0; step < _options.MaxOutputTokens; step++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            int nextId = RunDecoderStep(decoderIds, encHidden, encLen);
            if (nextId == EosTokenId)
                break;
            decoderIds.Add(nextId);
        }

        // Drop the two seed tokens (</s>, tgtLang) and any specials/lang tokens, then
        // map NLLB ids back to SentencePiece ids and decode.
        var spIds = decoderIds
            .Skip(2)
            .Where(id => id >= FairseqOffset + 1 && id < FirstLangTokenId)
            .Select(id => (int)(id - FairseqOffset));

        return _tokenizer.Decode(spIds).Trim();
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
        // Copy the data out before the results are disposed so we can reuse the
        // encoder state across every decode step.
        var dims = hidden.Dimensions.ToArray();
        return new DenseTensor<float>(hidden.ToArray(), dims);
    }

    private int RunDecoderStep(IReadOnlyList<long> decoderIds, DenseTensor<float> encHidden, int encLen)
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

        // Argmax over the vocabulary at the final decoder position.
        int vocab = logits.Dimensions[^1];
        int last = decLen - 1;

        int bestId = 0;
        float bestVal = float.NegativeInfinity;
        for (int v = 0; v < vocab; v++)
        {
            float val = logits[0, last, v];
            if (val > bestVal)
            {
                bestVal = val;
                bestId = v;
            }
        }
        return bestId;
    }

    private int ResolveLangId(string flores)
    {
        if (_langTokenIds!.TryGetValue(flores, out var id))
            return id;
        throw new InvalidOperationException($"Language token '{flores}' not found in the NLLB tokenizer.");
    }

    // ---- Initialisation -----------------------------------------------------

    private async Task EnsureLoadedAsync(CancellationToken cancellationToken)
    {
        if (_encoder is not null)
            return;

        await _initLock.WaitAsync(cancellationToken);
        try
        {
            if (_encoder is not null)
                return;

            Directory.CreateDirectory(_modelDirectory);

            var spmPath = await EnsureFileAsync("sentencepiece.bpe.model", cancellationToken);
            var tokJsonPath = await EnsureFileAsync("tokenizer.json", cancellationToken);
            var encPath = await EnsureFileAsync($"onnx/encoder_model{_options.QuantizationSuffix}.onnx", cancellationToken);
            var decPath = await EnsureFileAsync($"onnx/decoder_model{_options.QuantizationSuffix}.onnx", cancellationToken);

            _logger.LogInformation("Loading SentencePiece tokenizer from {Path}", spmPath);
            await using (var spmStream = File.OpenRead(spmPath))
                _tokenizer = SentencePieceTokenizer.Create(spmStream, addBeginningOfSentence: false, addEndOfSentence: false);

            _langTokenIds = LoadLanguageTokenIds(tokJsonPath);
            _logger.LogInformation("Loaded {Count} NLLB language tokens", _langTokenIds.Count);

            var sessionOptions = new Microsoft.ML.OnnxRuntime.SessionOptions();
            if (_options.Threads > 0)
            {
                sessionOptions.IntraOpNumThreads = _options.Threads;
                sessionOptions.InterOpNumThreads = _options.Threads;
            }

            _logger.LogInformation("Loading NLLB ONNX encoder/decoder (this can take a moment) ...");
            _encoder = new InferenceSession(encPath, sessionOptions);
            _decoder = new InferenceSession(decPath, sessionOptions);

            ResolveTensorNames();
        }
        finally
        {
            _initLock.Release();
        }
    }

    /// <summary>
    /// Match the encoder/decoder tensor names at runtime so the code is robust to
    /// minor export-naming differences.
    /// </summary>
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
            "ONNX I/O resolved — encoder({Ids}, {Mask})->{Hidden}; decoder({DIds}, {EMask}, {EHidden})->{Logits}",
            _encIdsName, _encMaskName, _encHiddenOut, _decIdsName, _decEncMaskName, _decEncHiddenName, _decLogitsOut);
    }

    /// <summary>Reads the FLORES-200 language token ids from the model's tokenizer.json.</summary>
    private static IReadOnlyDictionary<string, int> LoadLanguageTokenIds(string tokenizerJsonPath)
    {
        using var stream = File.OpenRead(tokenizerJsonPath);
        using var doc = JsonDocument.Parse(stream);
        var map = new Dictionary<string, int>(StringComparer.Ordinal);
        foreach (var token in doc.RootElement.GetProperty("added_tokens").EnumerateArray())
        {
            var content = token.GetProperty("content").GetString();
            // FLORES-200 codes look like "eng_Latn"; skip <s>, <pad>, <mask>, ...
            if (content is not null && content.Length == 8 && content[3] == '_')
                map[content] = token.GetProperty("id").GetInt32();
        }
        return map;
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

    private static readonly Regex SentenceSplitter =
        new(@"(?<=[.!?။。！？؟])\s+", RegexOptions.Compiled);

    private static IEnumerable<string> SplitSentences(string text)
    {
        var trimmed = text.Trim();
        if (trimmed.Length == 0)
            yield break;

        foreach (var part in SentenceSplitter.Split(trimmed))
        {
            var s = part.Trim();
            if (s.Length > 0)
                yield return s;
        }
    }

    private static string Sanitize(string repo) => repo.Replace('/', '_');

    public void Dispose()
    {
        _encoder?.Dispose();
        _decoder?.Dispose();
        _initLock.Dispose();
    }
}
