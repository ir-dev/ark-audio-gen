using System.ComponentModel.DataAnnotations;

namespace ArkTextTranslator.Models;

public class LoginViewModel
{
    [Required, EmailAddress]
    [Display(Name = "Email address")]
    public string? Email { get; set; }

    [Required(ErrorMessage = "Enter the characters shown.")]
    [Display(Name = "CAPTCHA")]
    public string? Captcha { get; set; }

    public string? ReturnUrl { get; set; }

    public string? ErrorMessage { get; set; }
}

public class VerifyOtpViewModel
{
    [Required, EmailAddress]
    public string? Email { get; set; }

    [Required(ErrorMessage = "Enter the 6-digit code.")]
    [RegularExpression(@"^\d{6}$", ErrorMessage = "The code is 6 digits.")]
    [Display(Name = "One-time code")]
    public string? Code { get; set; }

    public string? ReturnUrl { get; set; }

    /// <summary>When SMTP isn't configured, the code is shown here instead of emailed.</summary>
    public string? OnScreenOtp { get; set; }

    public string? InfoMessage { get; set; }

    public string? ErrorMessage { get; set; }
}

public class ProfileViewModel
{
    public string Email { get; set; } = "";

    [Required, MaxLength(64)]
    [RegularExpression(@"^[A-Za-z0-9\-]+$", ErrorMessage = "Use letters, numbers and hyphens only.")]
    [Display(Name = "API key header name")]
    public string ApiKeyName { get; set; } = "X-Ark-Api-Key";

    public string ApiKey { get; set; } = "";

    public DateTime CreatedAtUtc { get; set; }
    public DateTime? LastLoginAtUtc { get; set; }

    public string? StatusMessage { get; set; }

    /// <summary>Absolute base URL of this site, for the sample API call.</summary>
    public string ApiBaseUrl { get; set; } = "";
}
