using ArkSummarize.Data;
using ArkSummarize.Models;
using ArkSummarize.Services;
using ArkSummarize.Services.Auth;
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
    ?? "Data Source=App_Data/ark-summarize.db";
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

// --- Summarization engines (CPU-only) ---------------------------------------
// Default engine: fully-managed lexical analyser (no model, instant).
builder.Services.AddSingleton<ISummarizationService, SummarizationService>();
// Optional engine: MiniLM sentence embeddings via ONNX (model downloaded on first use).
builder.Services.Configure<EmbeddingOptions>(builder.Configuration.GetSection(EmbeddingOptions.SectionName));
builder.Services.AddSingleton<ISummarizationService, MiniLmSummarizationService>();
// Optional engine: abstractive seq2seq (DistilBART) via ONNX — generates a condensed summary.
builder.Services.Configure<AbstractiveOptions>(builder.Configuration.GetSection(AbstractiveOptions.SectionName));
builder.Services.AddSingleton<ISummarizationService, AbstractiveSummarizationService>();
builder.Services.AddSingleton<SummarizationEngineProvider>();

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
    app.UseHsts();
}

app.UseHttpsRedirection();
app.UseStaticFiles();

app.UseRouting();

app.UseSession();
app.UseAuthentication();
app.UseAuthorization();

app.MapControllers(); // attribute-routed endpoints (e.g. /api/summarize)

app.MapControllerRoute(
    name: "default",
    pattern: "{controller=Home}/{action=Index}/{id?}");

app.Run();
