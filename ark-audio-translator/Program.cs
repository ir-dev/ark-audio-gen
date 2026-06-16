using ArkTextTranslator.Models;
using ArkTextTranslator.Services;

var builder = WebApplication.CreateBuilder(args);

// Add services to the container.
builder.Services.AddControllersWithViews();

// Translation pipeline configuration.
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

app.UseAuthorization();

app.MapControllers(); // attribute-routed endpoints (e.g. /api/translate)

app.MapControllerRoute(
    name: "default",
    pattern: "{controller=Translation}/{action=Index}/{id?}");

app.Run();
