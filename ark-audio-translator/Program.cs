using ArkTextTranslator.Data;
using ArkTextTranslator.Models;
using ArkTextTranslator.Services;
using ArkTextTranslator.Services.Auth;
using Microsoft.AspNetCore.Authentication.Cookies;
using Microsoft.EntityFrameworkCore;

var builder = WebApplication.CreateBuilder(args);

// Add services to the container.
builder.Services.AddControllersWithViews();
builder.Services.AddHttpContextAccessor();
builder.Services.AddDistributedMemoryCache();
builder.Services.AddSession(o =>
{
    o.Cookie.Name = ".Ark.Session";
    o.Cookie.HttpOnly = true;
    o.Cookie.IsEssential = true;
    o.IdleTimeout = TimeSpan.FromMinutes(30);
});

// --- Persistence (SQLite) ---------------------------------------------------
Directory.CreateDirectory(Path.Combine(builder.Environment.ContentRootPath, "App_Data"));
var connectionString = builder.Configuration.GetConnectionString("Default")
    ?? "Data Source=App_Data/ark-translator.db";
builder.Services.AddDbContext<AppDbContext>(o => o.UseSqlite(connectionString));

// --- Auth services ----------------------------------------------------------
builder.Services.Configure<EmailOptions>(builder.Configuration.GetSection(EmailOptions.SectionName));
builder.Services.AddSingleton<IEmailSender, SmtpEmailSender>();
builder.Services.AddScoped<OtpService>();
builder.Services.AddScoped<CaptchaService>();

builder.Services.AddAuthentication(options =>
    {
        options.DefaultScheme = CookieAuthenticationDefaults.AuthenticationScheme;
    })
    .AddCookie(options =>
    {
        options.LoginPath = "/Account/Login";
        options.LogoutPath = "/Account/Logout";
        options.AccessDeniedPath = "/Account/Login";
        options.ExpireTimeSpan = TimeSpan.FromDays(14);
        options.SlidingExpiration = true;
        options.Cookie.Name = ".Ark.Auth";
    })
    .AddScheme<ApiKeyAuthenticationOptions, ApiKeyAuthenticationHandler>(
        ApiKeyAuthenticationOptions.Scheme, _ => { });

builder.Services.AddAuthorization();

// --- Translation pipeline ---------------------------------------------------
builder.Services.Configure<TranslationOptions>(builder.Configuration.GetSection(TranslationOptions.SectionName));

// Offline, CPU language detection (NTextCat) — shared singleton.
builder.Services.AddSingleton<ILanguageDetector, NTextCatLanguageDetector>();

// Translation engine, selected by the "Translation:Provider" setting.
var provider = builder.Configuration[$"{TranslationOptions.SectionName}:Provider"] ?? "Onnx";
if (provider.Equals("LibreTranslate", StringComparison.OrdinalIgnoreCase))
{
    builder.Services.AddHttpClient<ITranslationService, LibreTranslateService>();
}
else
{
    // Default: fully-offline CPU NLLB-200 via ONNX Runtime.
    builder.Services.AddSingleton<ITranslationService, OnnxNllbTranslationService>();
}

var app = builder.Build();

// Create the database on first run.
using (var scope = app.Services.CreateScope())
{
    var db = scope.ServiceProvider.GetRequiredService<AppDbContext>();
    db.Database.EnsureCreated();
}

// Configure the HTTP request pipeline.
if (!app.Environment.IsDevelopment())
{
    app.UseExceptionHandler("/Home/Error");
    // The default HSTS value is 30 days. You may want to change this for production scenarios, see https://aka.ms/aspnetcore-hsts.
    app.UseHsts();
}

app.UseHttpsRedirection();
app.UseStaticFiles();

app.UseRouting();

app.UseSession();
app.UseAuthentication();
app.UseAuthorization();

app.MapControllers(); // attribute-routed endpoints (e.g. /api/translate)

app.MapControllerRoute(
    name: "default",
    pattern: "{controller=Home}/{action=Index}/{id?}");

app.Run();
