using System.Diagnostics;
using ArkSummarize.Models;
using Microsoft.Extensions.Options;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.Tokenizers;

namespace ArkSummarize.Services;

/// <summary>
/// A semantic engine powered by the <b>all-MiniLM-L6-v2</b> sentence-embedding model run
/// through ONNX Runtime on the CPU. The model + vocabulary are downloaded once from
/// Hugging Face and cached on disk; everything thereafter runs locally — no GPU, no cloud.
///
/// <para>
/// Embeddings drive two of the three outputs:
///  • <b>Summary</b> — every sentence is embedded; the sentences whose vectors are closest
///    to the document centroid (its "gist") are selected. This is semantic, so it captures
///    meaning rather than just word overlap.
///  • <b>Intent</b>  — zero-shot classification: the document embedding is compared (cosine)
///    against the embeddings of a set of intent hypotheses; the closest label wins.
/// </para>
/// Entities and keywords use the same deterministic extractors as the lexical engine
/// (see <see cref="SharedNlp"/>), so those stay consistent across engines.
/// </summary>
public sealed class MiniLmSummarizationService : ISummarizationService, IDisposable
{
    public string Key => "minilm";
    public string DisplayName => "MiniLM Semantic (ONNX)";
    public string Description => "all-MiniLM-L6-v2 sentence embeddings: centroid-based summary + zero-shot intent. CPU, offline after first download.";

    private const int Hidden = 384;
    private const int ClsId = 101;
    private const int SepId = 102;

    // Intent label -> a short natural-language hypothesis that describes it. The hypothesis is
    // embedded once and compared against the document embedding (zero-shot classification).
    // Example-style hypotheses embed more distinctively than abstract definitions, which
    // sharpens the cosine separation between labels for short inputs.
    private static readonly (string Label, string Hypothesis)[] IntentHypotheses =
    {
        ("question",      "What is this and how does it work? Can you tell me where to find it?"),
        ("request",       "Please could you do this for me. I would like you to help with it."),
        ("complaint",     "This is broken and not working. I am very unhappy and want a refund."),
        ("praise",        "Thank you so much, this is excellent and amazing. Great job, I love it."),
        ("transactional", "I want to place an order and make a payment for my purchase or invoice."),
        ("instruction",   "First do this, then do that, and finally follow these steps in order."),
        ("opinion",       "I think and I believe that, in my opinion, this is probably the case."),
        ("announcement",  "We are excited to announce and introduce our new release, available now."),
        ("statement",     "This is a neutral factual statement describing general information."),
    };

    private readonly EmbeddingOptions _options;
    private readonly ILogger<MiniLmSummarizationService> _logger;
    private readonly string _modelDirectory;
    private readonly SemaphoreSlim _initLock = new(1, 1);

    private InferenceSession? _session;
    private BertTokenizer? _tokenizer;
    private float[][]? _labelEmbeddings;
    private string _idsName = "input_ids";
    private string _maskName = "attention_mask";
    private string? _typeName = "token_type_ids";
    private string _outName = "last_hidden_state";

    public MiniLmSummarizationService(
        IOptions<EmbeddingOptions> options,
        IWebHostEnvironment env,
        ILogger<MiniLmSummarizationService> logger)
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

        var sentences = SharedNlp.SplitSentences(text);
        var allWords = SharedNlp.WordTokens(text);
        result.SentenceCount = sentences.Count;
        result.WordCount = allWords.Count;
        result.Keywords = SharedNlp.TopKeywords(SharedNlp.WordFrequencies(allWords));
        result.Entities = SharedNlp.ExtractEntities(text);

        result.Summary = BuildSemanticSummary(sentences, maxSentences, cancellationToken);

        var docVec = Embed(text, cancellationToken);
        (result.Intent, result.IntentConfidence) = ClassifyIntent(docVec, text);

        result.ProcessingMs = sw.Elapsed.TotalMilliseconds;
        return result;
    }

    // ---- Summary: pick sentences closest to the document centroid ----------
    private string BuildSemanticSummary(List<string> sentences, int maxSentences, CancellationToken ct)
    {
        if (sentences.Count == 0) return "";
        if (sentences.Count <= maxSentences) return string.Join(" ", sentences);

        var vecs = new float[sentences.Count][];
        for (int i = 0; i < sentences.Count; i++)
        {
            ct.ThrowIfCancellationRequested();
            vecs[i] = Embed(sentences[i], ct);
        }

        // Centroid of the (already L2-normalised) sentence vectors, renormalised.
        var centroid = new float[Hidden];
        foreach (var v in vecs)
            for (int h = 0; h < Hidden; h++) centroid[h] += v[h];
        Normalize(centroid);

        var chosen = Enumerable.Range(0, sentences.Count)
            .Select(i => (Index: i, Score: Dot(vecs[i], centroid)
                          * (1.0 + 0.10 * (1.0 - (double)i / sentences.Count)))) // mild positional prior
            .OrderByDescending(s => s.Score)
            .Take(maxSentences)
            .Select(s => s.Index)
            .OrderBy(i => i)
            .ToList();

        return string.Join(" ", chosen.Select(i => sentences[i].Trim()));
    }

    // ---- Intent: zero-shot cosine against label hypotheses -----------------
    private (string Intent, double Confidence) ClassifyIntent(float[] docVec, string text)
    {
        var sims = new double[IntentHypotheses.Length];
        for (int i = 0; i < IntentHypotheses.Length; i++)
            sims[i] = Dot(docVec, _labelEmbeddings![i]);

        // Cheap, robust structural prior: a question mark is a strong question signal that the
        // embedding model alone confuses with how-to "instructions".
        if (SharedNlp.CountOccurrences(text, '?') > 0)
        {
            int qi = Array.FindIndex(IntentHypotheses, h => h.Label == "question");
            if (qi >= 0) sims[qi] += 0.15;
        }

        // Zero-shot cosine similarities cluster in a narrow band, so centre them on the mean
        // (which sharpens the contrast between labels) before a temperature-scaled softmax.
        double mean = sims.Average();
        const double temperature = 25.0;
        double sumExp = 0;
        var exp = new double[sims.Length];
        for (int i = 0; i < sims.Length; i++) { exp[i] = Math.Exp((sims[i] - mean) * temperature); sumExp += exp[i]; }

        int best = 0;
        for (int i = 1; i < sims.Length; i++) if (sims[i] > sims[best]) best = i;
        double confidence = Math.Round(exp[best] / sumExp, 2);
        return (IntentHypotheses[best].Label, Math.Clamp(confidence, 0.3, 0.99));
    }

    // ---- Embedding ---------------------------------------------------------
    private float[] Embed(string text, CancellationToken ct)
    {
        var ids = _tokenizer!.EncodeToIds(text, addSpecialTokens: true,
            considerPreTokenization: true, considerNormalization: true).ToList();

        // Truncate to the model's window, keeping a trailing [SEP].
        if (ids.Count > _options.MaxTokens)
        {
            ids = ids.Take(_options.MaxTokens).ToList();
            ids[^1] = SepId;
        }
        if (ids.Count == 0) ids = new List<int> { ClsId, SepId };

        int n = ids.Count;
        var idTensor = new DenseTensor<long>(new[] { 1, n });
        var maskTensor = new DenseTensor<long>(new[] { 1, n });
        for (int i = 0; i < n; i++) { idTensor[0, i] = ids[i]; maskTensor[0, i] = 1; }

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(_idsName, idTensor),
            NamedOnnxValue.CreateFromTensor(_maskName, maskTensor),
        };
        if (_typeName is not null)
        {
            var typeTensor = new DenseTensor<long>(new[] { 1, n }); // zeros
            inputs.Add(NamedOnnxValue.CreateFromTensor(_typeName, typeTensor));
        }

        using var results = _session!.Run(inputs);
        var hidden = results.First(r => r.Name == _outName).AsTensor<float>();

        // Mean-pool token embeddings (all mask=1 here) then L2-normalise.
        var pooled = new float[Hidden];
        for (int t = 0; t < n; t++)
            for (int h = 0; h < Hidden; h++)
                pooled[h] += hidden[0, t, h];
        for (int h = 0; h < Hidden; h++) pooled[h] /= n;
        Normalize(pooled);
        return pooled;
    }

    private static double Dot(float[] a, float[] b)
    {
        double s = 0;
        for (int i = 0; i < a.Length; i++) s += a[i] * b[i];
        return s;
    }

    private static void Normalize(float[] v)
    {
        double norm = 0;
        for (int i = 0; i < v.Length; i++) norm += v[i] * (double)v[i];
        norm = Math.Sqrt(norm);
        if (norm < 1e-9) return;
        for (int i = 0; i < v.Length; i++) v[i] = (float)(v[i] / norm);
    }

    // ---- Initialisation ----------------------------------------------------
    private async Task EnsureLoadedAsync(CancellationToken cancellationToken)
    {
        if (_session is not null) return;

        await _initLock.WaitAsync(cancellationToken);
        try
        {
            if (_session is not null) return;

            Directory.CreateDirectory(_modelDirectory);
            var vocabPath = await EnsureFileAsync(_options.VocabFile, cancellationToken);
            var modelPath = await EnsureFileAsync(_options.ModelFile, cancellationToken);

            _logger.LogInformation("Loading WordPiece vocabulary from {Path}", vocabPath);
            await using (var vocab = File.OpenRead(vocabPath))
                _tokenizer = BertTokenizer.Create(vocab, new BertOptions());

            var sessionOptions = new Microsoft.ML.OnnxRuntime.SessionOptions();
            if (_options.Threads > 0)
            {
                sessionOptions.IntraOpNumThreads = _options.Threads;
                sessionOptions.InterOpNumThreads = _options.Threads;
            }

            _logger.LogInformation("Loading MiniLM ONNX model from {Path}", modelPath);
            _session = new InferenceSession(modelPath, sessionOptions);
            ResolveTensorNames();

            // Pre-embed the intent hypotheses once.
            _labelEmbeddings = IntentHypotheses
                .Select(h => Embed(h.Hypothesis, cancellationToken))
                .ToArray();
            _logger.LogInformation("MiniLM engine ready ({Labels} intent hypotheses embedded)", _labelEmbeddings.Length);
        }
        finally
        {
            _initLock.Release();
        }
    }

    private void ResolveTensorNames()
    {
        var inputs = _session!.InputMetadata.Keys.ToList();
        _idsName = inputs.FirstOrDefault(n => n.Contains("input_ids", StringComparison.OrdinalIgnoreCase)) ?? _idsName;
        _maskName = inputs.FirstOrDefault(n => n.Contains("attention_mask", StringComparison.OrdinalIgnoreCase)) ?? _maskName;
        _typeName = inputs.FirstOrDefault(n => n.Contains("token_type", StringComparison.OrdinalIgnoreCase));

        // Prefer the rank-3 token-embeddings output (batch, seq, hidden); fall back to the first.
        _outName = _session.OutputMetadata
                       .FirstOrDefault(kv => kv.Value.Dimensions.Length == 3).Key
                   ?? _session.OutputMetadata.Keys.First();

        _logger.LogInformation("MiniLM I/O resolved — ids={Ids}, mask={Mask}, type={Type}, out={Out}",
            _idsName, _maskName, _typeName ?? "(none)", _outName);
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
        _session?.Dispose();
        _initLock.Dispose();
    }
}
