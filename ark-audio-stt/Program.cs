using ArkSpeechToText.Data;
using ArkSpeechToText.Models;
using ArkSpeechToText.Services;
using ArkSpeechToText.Services.Auth;
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
    ?? "Data Source=App_Data/ark-stt.db";
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

// --- Speech-to-text pipeline ------------------------------------------------
builder.Services.Configure<WhisperOptions>(builder.Configuration.GetSection(WhisperOptions.SectionName));
builder.Services.AddSingleton<ISpeechToTextService, WhisperSpeechToTextService>();

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

app.MapControllers(); // attribute-routed endpoints (e.g. /api/transcribe)

app.MapControllerRoute(
    name: "default",
    pattern: "{controller=Home}/{action=Index}/{id?}");

app.Run();
