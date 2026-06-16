namespace ArkTextTranslator.Services.Auth;

public interface IEmailSender
{
    /// <summary>True when SMTP is configured and emails can actually be sent.</summary>
    bool IsConfigured { get; }

    /// <summary>
    /// Sends an email. Returns <c>true</c> if it was dispatched, <c>false</c> if SMTP
    /// isn't configured (the caller should then surface the OTP on screen).
    /// </summary>
    Task<bool> SendAsync(string toEmail, string subject, string body, CancellationToken cancellationToken = default);
}
