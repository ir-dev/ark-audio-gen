using System.Security.Cryptography;
using ArkSpeechToText.Data;
using ArkSpeechToText.Models;
using Microsoft.EntityFrameworkCore;

namespace ArkSpeechToText.Services.Auth;

public enum OtpVerifyStatus { Success, Invalid, Expired, NotFound }

/// <summary>Issues and verifies email one-time passcodes.</summary>
public sealed class OtpService
{
    private static readonly TimeSpan Lifetime = TimeSpan.FromMinutes(10);
    private const int MaxAttempts = 5;

    private readonly AppDbContext _db;

    public OtpService(AppDbContext db) => _db = db;

    /// <summary>Generates a fresh 6-digit code for the email, invalidating older ones.</summary>
    public async Task<string> IssueAsync(string email, CancellationToken cancellationToken = default)
    {
        email = Normalize(email);

        // Drop any prior unconsumed codes for this address.
        var stale = await _db.OtpRequests
            .Where(o => o.Email == email && o.ConsumedAtUtc == null)
            .ToListAsync(cancellationToken);
        _db.OtpRequests.RemoveRange(stale);

        var code = RandomNumberGenerator.GetInt32(0, 1_000_000).ToString("D6");
        _db.OtpRequests.Add(new OtpRequest
        {
            Email = email,
            Code = code,
            ExpiresAtUtc = DateTime.UtcNow.Add(Lifetime),
        });
        await _db.SaveChangesAsync(cancellationToken);
        return code;
    }

    /// <summary>
    /// Verifies a code. On success the matching <see cref="User"/> is returned
    /// (created on first successful sign-in — i.e. registration).
    /// </summary>
    public async Task<(OtpVerifyStatus Status, User? User)> VerifyAsync(
        string email, string code, CancellationToken cancellationToken = default)
    {
        email = Normalize(email);

        var otp = await _db.OtpRequests
            .Where(o => o.Email == email && o.ConsumedAtUtc == null)
            .OrderByDescending(o => o.Id)
            .FirstOrDefaultAsync(cancellationToken);

        if (otp is null)
            return (OtpVerifyStatus.NotFound, null);

        if (otp.ExpiresAtUtc < DateTime.UtcNow || otp.Attempts >= MaxAttempts)
            return (OtpVerifyStatus.Expired, null);

        if (!CryptographicOperations.FixedTimeEquals(
                System.Text.Encoding.UTF8.GetBytes(otp.Code),
                System.Text.Encoding.UTF8.GetBytes(code?.Trim() ?? "")))
        {
            otp.Attempts++;
            await _db.SaveChangesAsync(cancellationToken);
            return (OtpVerifyStatus.Invalid, null);
        }

        otp.ConsumedAtUtc = DateTime.UtcNow;

        var user = await _db.Users.FirstOrDefaultAsync(u => u.Email == email, cancellationToken);
        if (user is null)
        {
            user = new User
            {
                Email = email,
                ApiKeyName = "X-Ark-Api-Key",
                ApiKey = ApiKeyGenerator.New(),
            };
            _db.Users.Add(user);
        }
        user.LastLoginAtUtc = DateTime.UtcNow;

        await _db.SaveChangesAsync(cancellationToken);
        return (OtpVerifyStatus.Success, user);
    }

    private static string Normalize(string email) => email.Trim().ToLowerInvariant();
}
