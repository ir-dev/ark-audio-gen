namespace ArkSummarize.Models;

/// <summary>A one-time passcode issued for an email sign-in attempt.</summary>
public class OtpRequest
{
    public int Id { get; set; }

    public string Email { get; set; } = "";

    public string Code { get; set; } = "";

    public DateTime ExpiresAtUtc { get; set; }

    public DateTime? ConsumedAtUtc { get; set; }

    public int Attempts { get; set; }
}
