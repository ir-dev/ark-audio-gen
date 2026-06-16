using System.Security.Claims;
using System.Text.Encodings.Web;
using ArkSpeechToText.Data;
using Microsoft.AspNetCore.Authentication;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Options;

namespace ArkSpeechToText.Services.Auth;

public sealed class ApiKeyAuthenticationOptions : AuthenticationSchemeOptions
{
    public const string Scheme = "ApiKey";
}

/// <summary>
/// Authenticates API requests by an API key sent in a per-user custom HTTP header.
/// Because both the header <em>name</em> and <em>value</em> are user-defined, the
/// handler looks up the key value (which is unique + high-entropy) and then confirms
/// it arrived under that user's chosen header name.
/// </summary>
public sealed class ApiKeyAuthenticationHandler : AuthenticationHandler<ApiKeyAuthenticationOptions>
{
    private readonly AppDbContext _db;

    public ApiKeyAuthenticationHandler(
        IOptionsMonitor<ApiKeyAuthenticationOptions> options,
        ILoggerFactory logger,
        UrlEncoder encoder,
        AppDbContext db)
        : base(options, logger, encoder)
    {
        _db = db;
    }

    protected override async Task<AuthenticateResult> HandleAuthenticateAsync()
    {
        // Collect candidate (name -> value) pairs from request headers. Keys are
        // prefixed "ark_", which keeps the candidate set tiny.
        var candidates = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        foreach (var header in Request.Headers)
        {
            var value = header.Value.ToString();
            if (!string.IsNullOrEmpty(value) && value.StartsWith("ark_", StringComparison.Ordinal))
                candidates[header.Key] = value;
        }

        if (candidates.Count == 0)
            return AuthenticateResult.NoResult();

        var values = candidates.Values.ToArray();
        var user = await _db.Users.AsNoTracking()
            .FirstOrDefaultAsync(u => values.Contains(u.ApiKey));

        // Confirm the key arrived under the user's configured header name.
        if (user is null || !candidates.TryGetValue(user.ApiKeyName, out var sent) || sent != user.ApiKey)
            return AuthenticateResult.Fail("Invalid API key.");

        var claims = new[]
        {
            new Claim(ClaimTypes.NameIdentifier, user.Id.ToString()),
            new Claim(ClaimTypes.Email, user.Email),
            new Claim(ClaimTypes.Name, user.Email),
        };
        var identity = new ClaimsIdentity(claims, ApiKeyAuthenticationOptions.Scheme);
        var ticket = new AuthenticationTicket(new ClaimsPrincipal(identity), ApiKeyAuthenticationOptions.Scheme);
        return AuthenticateResult.Success(ticket);
    }
}
