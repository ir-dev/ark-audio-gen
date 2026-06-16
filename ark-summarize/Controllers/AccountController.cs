using System.Security.Claims;
using ArkSummarize.Data;
using ArkSummarize.Models;
using ArkSummarize.Services.Auth;
using Microsoft.AspNetCore.Authentication;
using Microsoft.AspNetCore.Authentication.Cookies;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;

namespace ArkSummarize.Controllers;

public class AccountController : Controller
{
    private readonly AppDbContext _db;
    private readonly OtpService _otp;
    private readonly CaptchaService _captcha;
    private readonly IEmailSender _email;
    private readonly ILogger<AccountController> _logger;

    public AccountController(
        AppDbContext db, OtpService otp, CaptchaService captcha,
        IEmailSender email, ILogger<AccountController> logger)
    {
        _db = db;
        _otp = otp;
        _captcha = captcha;
        _email = email;
        _logger = logger;
    }

    /// <summary>Serves a fresh CAPTCHA image (also used by the "refresh" button).</summary>
    [HttpGet]
    public IActionResult Captcha()
    {
        var svg = _captcha.Generate();
        Response.Headers.CacheControl = "no-store, no-cache, must-revalidate";
        return Content(svg, "image/svg+xml");
    }

    [HttpGet]
    public IActionResult Login(string? returnUrl = null)
    {
        if (User.Identity?.IsAuthenticated == true)
            return RedirectToLocal(returnUrl);
        return View(new LoginViewModel { ReturnUrl = returnUrl });
    }

    [HttpPost]
    [ValidateAntiForgeryToken]
    public async Task<IActionResult> Login(LoginViewModel model, CancellationToken cancellationToken)
    {
        if (!ModelState.IsValid)
            return View(model);

        if (!_captcha.Validate(model.Captcha))
        {
            model.ErrorMessage = "CAPTCHA was incorrect. Please try again.";
            return View(model);
        }

        var code = await _otp.IssueAsync(model.Email!, cancellationToken);

        var sent = await _email.SendAsync(
            model.Email!,
            "Your Ark Summarize sign-in code",
            $"<p>Your one-time sign-in code is:</p><h2 style='letter-spacing:4px'>{code}</h2><p>It expires in 10 minutes.</p>",
            cancellationToken);

        TempData["Email"] = model.Email;
        TempData["ReturnUrl"] = model.ReturnUrl;
        if (!sent)
        {
            // SMTP not configured — surface the OTP on screen.
            TempData["OnScreenOtp"] = code;
            _logger.LogInformation("SMTP not configured; showing OTP on screen for {Email}", model.Email);
        }

        return RedirectToAction(nameof(Verify));
    }

    [HttpGet]
    public IActionResult Verify()
    {
        var email = TempData["Email"] as string;
        if (string.IsNullOrEmpty(email))
            return RedirectToAction(nameof(Login));

        var onScreen = TempData["OnScreenOtp"] as string;
        var model = new VerifyOtpViewModel
        {
            Email = email,
            ReturnUrl = TempData["ReturnUrl"] as string,
            OnScreenOtp = onScreen,
            InfoMessage = onScreen is null
                ? $"We emailed a 6-digit code to {email}. Enter it below."
                : "Email delivery isn't configured on this server, so your code is shown below.",
        };
        // Preserve across the upcoming POST round-trip if the user reloads.
        TempData.Keep();
        return View(model);
    }

    [HttpPost]
    [ValidateAntiForgeryToken]
    public async Task<IActionResult> Verify(VerifyOtpViewModel model, CancellationToken cancellationToken)
    {
        if (!ModelState.IsValid)
            return View(model);

        var (status, user) = await _otp.VerifyAsync(model.Email!, model.Code!, cancellationToken);
        if (status != OtpVerifyStatus.Success || user is null)
        {
            model.ErrorMessage = status switch
            {
                OtpVerifyStatus.Expired => "That code has expired or had too many attempts. Request a new one.",
                OtpVerifyStatus.NotFound => "No active code for this email. Request a new one.",
                _ => "Incorrect code. Please try again.",
            };
            return View(model);
        }

        var claims = new List<Claim>
        {
            new(ClaimTypes.NameIdentifier, user.Id.ToString()),
            new(ClaimTypes.Email, user.Email),
            new(ClaimTypes.Name, user.Email),
        };
        var identity = new ClaimsIdentity(claims, CookieAuthenticationDefaults.AuthenticationScheme);
        await HttpContext.SignInAsync(
            CookieAuthenticationDefaults.AuthenticationScheme,
            new ClaimsPrincipal(identity),
            new AuthenticationProperties { IsPersistent = true });

        return RedirectToLocal(model.ReturnUrl);
    }

    [HttpPost]
    [ValidateAntiForgeryToken]
    [Authorize]
    public async Task<IActionResult> Logout()
    {
        await HttpContext.SignOutAsync(CookieAuthenticationDefaults.AuthenticationScheme);
        return RedirectToAction("Index", "Home");
    }

    [HttpGet]
    [Authorize]
    public async Task<IActionResult> Profile(CancellationToken cancellationToken)
    {
        var user = await CurrentUserAsync(cancellationToken);
        if (user is null)
            return RedirectToAction(nameof(Login));

        return View(ToProfileViewModel(user));
    }

    [HttpPost]
    [ValidateAntiForgeryToken]
    [Authorize]
    public async Task<IActionResult> RegenerateKey(CancellationToken cancellationToken)
    {
        var user = await CurrentUserAsync(cancellationToken);
        if (user is null)
            return RedirectToAction(nameof(Login));

        user.ApiKey = ApiKeyGenerator.New();
        await _db.SaveChangesAsync(cancellationToken);

        var vm = ToProfileViewModel(user);
        vm.StatusMessage = "A new API key was generated. Your previous key no longer works.";
        return View(nameof(Profile), vm);
    }

    [HttpPost]
    [ValidateAntiForgeryToken]
    [Authorize]
    public async Task<IActionResult> UpdateHeaderName(ProfileViewModel model, CancellationToken cancellationToken)
    {
        var user = await CurrentUserAsync(cancellationToken);
        if (user is null)
            return RedirectToAction(nameof(Login));

        var name = (model.ApiKeyName ?? "").Trim();
        if (!System.Text.RegularExpressions.Regex.IsMatch(name, @"^[A-Za-z0-9\-]{1,64}$"))
        {
            var invalid = ToProfileViewModel(user);
            invalid.StatusMessage = "Header name must use letters, numbers and hyphens only (1–64 chars).";
            return View(nameof(Profile), invalid);
        }

        user.ApiKeyName = name;
        await _db.SaveChangesAsync(cancellationToken);

        var vm = ToProfileViewModel(user);
        vm.StatusMessage = "API key header name updated.";
        return View(nameof(Profile), vm);
    }

    private ProfileViewModel ToProfileViewModel(User user) => new()
    {
        Email = user.Email,
        ApiKeyName = user.ApiKeyName,
        ApiKey = user.ApiKey,
        CreatedAtUtc = user.CreatedAtUtc,
        LastLoginAtUtc = user.LastLoginAtUtc,
        ApiBaseUrl = $"{Request.Scheme}://{Request.Host}",
    };

    private Task<User?> CurrentUserAsync(CancellationToken cancellationToken)
    {
        var email = User.FindFirstValue(ClaimTypes.Email);
        return _db.Users.FirstOrDefaultAsync(u => u.Email == email, cancellationToken);
    }

    private IActionResult RedirectToLocal(string? returnUrl) =>
        Url.IsLocalUrl(returnUrl) ? Redirect(returnUrl!) : RedirectToAction("Index", "Summarize");
}
