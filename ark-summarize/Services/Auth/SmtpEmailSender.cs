using System.Net;
using System.Net.Mail;
using Microsoft.Extensions.Options;

namespace ArkSummarize.Services.Auth;

/// <summary>
/// Sends mail over SMTP using the "Smtp" configuration. If no host is configured the
/// sender reports <see cref="IsConfigured"/> = false so callers can fall back to
/// showing the OTP on screen.
/// </summary>
public sealed class SmtpEmailSender : IEmailSender
{
    private readonly EmailOptions _options;
    private readonly ILogger<SmtpEmailSender> _logger;

    public SmtpEmailSender(IOptions<EmailOptions> options, ILogger<SmtpEmailSender> logger)
    {
        _options = options.Value;
        _logger = logger;
    }

    public bool IsConfigured => _options.IsConfigured;

    public async Task<bool> SendAsync(string toEmail, string subject, string body, CancellationToken cancellationToken = default)
    {
        if (!_options.IsConfigured)
            return false;

        using var message = new MailMessage
        {
            From = new MailAddress(_options.FromAddress, _options.FromName),
            Subject = subject,
            Body = body,
            IsBodyHtml = true,
        };
        message.To.Add(toEmail);

        using var client = new SmtpClient(_options.Host, _options.Port)
        {
            EnableSsl = _options.EnableSsl,
            DeliveryMethod = SmtpDeliveryMethod.Network,
        };
        if (!string.IsNullOrWhiteSpace(_options.User))
            client.Credentials = new NetworkCredential(_options.User, _options.Password);

        await client.SendMailAsync(message, cancellationToken);
        _logger.LogInformation("Sent email to {Email}", toEmail);
        return true;
    }
}
