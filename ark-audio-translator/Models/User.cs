using System.ComponentModel.DataAnnotations;

namespace ArkTextTranslator.Models;

/// <summary>A registered user. Authentication is passwordless (email OTP).</summary>
public class User
{
    public int Id { get; set; }

    [Required, EmailAddress, MaxLength(256)]
    public string Email { get; set; } = "";

    /// <summary>HTTP header name the user sends their API key in (customisable).</summary>
    [MaxLength(64)]
    public string ApiKeyName { get; set; } = "X-Ark-Api-Key";

    /// <summary>The secret API key value. High-entropy, unique per user.</summary>
    [MaxLength(128)]
    public string ApiKey { get; set; } = "";

    public DateTime CreatedAtUtc { get; set; } = DateTime.UtcNow;

    public DateTime? LastLoginAtUtc { get; set; }
}
