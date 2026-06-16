using System.Security.Cryptography;
using System.Text;

namespace ArkSpeechToText.Services.Auth;

/// <summary>
/// A self-contained, cross-platform CAPTCHA: it renders a short distorted code as an
/// inline SVG (no System.Drawing / native deps) and keeps the expected answer in the
/// user's session. Used to throttle sign-in / OTP requests.
/// </summary>
public sealed class CaptchaService
{
    private const string SessionKey = "captcha.answer";
    // Avoid visually ambiguous characters (0/O, 1/I/L).
    private const string Alphabet = "ABCDEFGHJKMNPQRSTUVWXYZ23456789";

    private readonly IHttpContextAccessor _http;

    public CaptchaService(IHttpContextAccessor http) => _http = http;

    /// <summary>Generates a new challenge, stores the answer in session and returns the SVG markup.</summary>
    public string Generate(int length = 5)
    {
        var sb = new StringBuilder(length);
        for (int i = 0; i < length; i++)
            sb.Append(Alphabet[RandomNumberGenerator.GetInt32(Alphabet.Length)]);
        var code = sb.ToString();

        _http.HttpContext?.Session.SetString(SessionKey, code);
        return Render(code);
    }

    /// <summary>Validates the user's input against the stored answer (single-use).</summary>
    public bool Validate(string? input)
    {
        var session = _http.HttpContext?.Session;
        var expected = session?.GetString(SessionKey);
        session?.Remove(SessionKey); // one attempt per challenge
        return !string.IsNullOrWhiteSpace(expected)
            && string.Equals(expected, input?.Trim(), StringComparison.OrdinalIgnoreCase);
    }

    private static string Render(string code)
    {
        const int w = 180, h = 60;
        var svg = new StringBuilder();
        svg.Append($"<svg xmlns='http://www.w3.org/2000/svg' width='{w}' height='{h}' viewBox='0 0 {w} {h}' role='img' aria-label='captcha'>");
        svg.Append($"<rect width='{w}' height='{h}' fill='#0b1220' rx='8'/>");

        // Noise lines.
        for (int i = 0; i < 6; i++)
        {
            int x1 = Rand(w), y1 = Rand(h), x2 = Rand(w), y2 = Rand(h);
            svg.Append($"<line x1='{x1}' y1='{y1}' x2='{x2}' y2='{y2}' stroke='hsl({Rand(360)},70%,55%)' stroke-width='1' opacity='0.5'/>");
        }

        // Characters, each rotated/offset.
        for (int i = 0; i < code.Length; i++)
        {
            int x = 22 + i * 30;
            int y = 40 + Rand(8) - 4;
            int rot = Rand(40) - 20;
            svg.Append($"<text x='{x}' y='{y}' font-family='monospace' font-size='30' font-weight='700' fill='hsl({Rand(360)},80%,70%)' transform='rotate({rot} {x} {y})'>{code[i]}</text>");
        }

        svg.Append("</svg>");
        return svg.ToString();
    }

    private static int Rand(int maxExclusive) => RandomNumberGenerator.GetInt32(maxExclusive);
}
