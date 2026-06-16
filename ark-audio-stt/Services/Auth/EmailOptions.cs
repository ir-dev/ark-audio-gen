namespace ArkSpeechToText.Services.Auth;

/// <summary>
/// SMTP settings (bound from the "Smtp" section). When <see cref="Host"/> is empty
/// the app runs in "no-email" mode and shows the OTP on screen instead of mailing it.
/// </summary>
public class EmailOptions
{
    public const string SectionName = "Smtp";

    public string Host { get; set; } = "";
    public int Port { get; set; } = 587;
    public bool EnableSsl { get; set; } = true;
    public string User { get; set; } = "";
    public string Password { get; set; } = "";
    public string FromAddress { get; set; } = "no-reply@ark-transcribe.immanuel.co";
    public string FromName { get; set; } = "Ark Transcribe";

    public bool IsConfigured => !string.IsNullOrWhiteSpace(Host);
}
