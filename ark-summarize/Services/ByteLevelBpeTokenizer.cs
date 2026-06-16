using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace ArkSummarize.Services;

/// <summary>
/// A self-contained GPT-2 / RoBERTa / BART style <b>byte-level BPE</b> tokenizer built from a
/// Hugging Face <c>vocab.json</c> + <c>merges.txt</c>. It reproduces the reference behaviour
/// (spaces encoded as <c>Ġ</c>, GPT-2 pre-tokenization regex, rank-ordered merges) which the
/// stock <c>BpeTokenizer</c> byte-level mode does not, so the seq2seq model receives exactly
/// the token ids it was trained on and decoding restores spacing correctly.
/// </summary>
public sealed partial class ByteLevelBpeTokenizer
{
    [GeneratedRegex(@"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+", RegexOptions.Compiled)]
    private static partial Regex Pattern();

    private readonly Dictionary<string, int> _vocab;
    private readonly Dictionary<int, string> _idToToken;
    private readonly Dictionary<(string, string), int> _ranks;
    private readonly char[] _byteToChar = new char[256];
    private readonly Dictionary<char, byte> _charToByte = new();
    private readonly int _unkId;

    public ByteLevelBpeTokenizer(Dictionary<string, int> vocab, IReadOnlyList<string> merges, string unknownToken)
    {
        _vocab = vocab;
        _idToToken = vocab.ToDictionary(kv => kv.Value, kv => kv.Key);
        _unkId = vocab.GetValueOrDefault(unknownToken, 0);

        _ranks = new Dictionary<(string, string), int>(merges.Count);
        for (int i = 0; i < merges.Count; i++)
        {
            var sp = merges[i].Split(' ');
            if (sp.Length == 2) _ranks[(sp[0], sp[1])] = i;
        }

        BuildByteMaps();
    }

    public static ByteLevelBpeTokenizer FromFiles(string vocabPath, string mergesPath, string unknownToken)
    {
        var vocab = JsonSerializer.Deserialize<Dictionary<string, int>>(File.ReadAllText(vocabPath))!;
        var merges = File.ReadAllLines(mergesPath).Where(l => l.Length > 0 && !l.StartsWith('#')).ToList();
        return new ByteLevelBpeTokenizer(vocab, merges, unknownToken);
    }

    public bool TryGetId(string token, out int id) => _vocab.TryGetValue(token, out id);

    public List<int> Encode(string text)
    {
        var ids = new List<int>();
        foreach (Match m in Pattern().Matches(text))
        {
            var sb = new StringBuilder();
            foreach (var b in Encoding.UTF8.GetBytes(m.Value))
                sb.Append(_byteToChar[b]);

            foreach (var sub in ApplyBpe(sb.ToString()))
                ids.Add(_vocab.TryGetValue(sub, out var id) ? id : _unkId);
        }
        return ids;
    }

    /// <summary>Maps ids back to text (skipping any ids in <paramref name="skip"/>).</summary>
    public string Decode(IEnumerable<int> ids, ISet<int> skip)
    {
        var sb = new StringBuilder();
        foreach (var id in ids)
        {
            if (skip.Contains(id)) continue;
            if (_idToToken.TryGetValue(id, out var tok)) sb.Append(tok);
        }

        var bytes = new List<byte>(sb.Length);
        foreach (var c in sb.ToString())
            if (_charToByte.TryGetValue(c, out var b)) bytes.Add(b);
        return Encoding.UTF8.GetString(bytes.ToArray());
    }

    private List<string> ApplyBpe(string token)
    {
        var word = token.Select(c => c.ToString()).ToList();
        while (word.Count > 1)
        {
            int best = int.MaxValue, bi = -1;
            for (int i = 0; i < word.Count - 1; i++)
                if (_ranks.TryGetValue((word[i], word[i + 1]), out var r) && r < best) { best = r; bi = i; }
            if (bi < 0) break;

            var merged = word[bi] + word[bi + 1];
            var next = new List<string>(word.Count);
            for (int i = 0; i < word.Count;)
            {
                if (i == bi) { next.Add(merged); i += 2; }
                else { next.Add(word[i]); i++; }
            }
            word = next;
        }
        return word;
    }

    private void BuildByteMaps()
    {
        var bs = new List<int>();
        for (int i = '!'; i <= '~'; i++) bs.Add(i);
        for (int i = '¡'; i <= '¬'; i++) bs.Add(i);
        for (int i = '®'; i <= 'ÿ'; i++) bs.Add(i);
        var cs = new List<int>(bs);
        int n = 0;
        for (int b = 0; b < 256; b++)
            if (!bs.Contains(b)) { bs.Add(b); cs.Add(256 + n); n++; }

        for (int i = 0; i < bs.Count; i++)
        {
            _byteToChar[bs[i]] = (char)cs[i];
            _charToByte[(char)cs[i]] = (byte)bs[i];
        }
    }
}
