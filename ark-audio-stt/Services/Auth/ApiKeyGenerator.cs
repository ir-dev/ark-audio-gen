using System.Security.Cryptography;

namespace ArkSpeechToText.Services.Auth;

/// <summary>Generates high-entropy, URL-safe API keys.</summary>
public static class ApiKeyGenerator
{
    public static string New(int byteLength = 32)
    {
        var bytes = RandomNumberGenerator.GetBytes(byteLength);
        var token = Convert.ToBase64String(bytes)
            .Replace('+', '-')
            .Replace('/', '_')
            .TrimEnd('=');
        return $"ark_{token}";
    }
}
