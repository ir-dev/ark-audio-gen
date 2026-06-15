using ArkSpeechToText.Models;
using ArkSpeechToText.Services;

var builder = WebApplication.CreateBuilder(args);

// Add services to the container.
builder.Services.AddControllersWithViews();

// Speech-to-text engine configuration + service.
builder.Services.Configure<WhisperOptions>(builder.Configuration.GetSection(WhisperOptions.SectionName));
builder.Services.AddSingleton<ISpeechToTextService, WhisperSpeechToTextService>();

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

app.MapControllers(); // attribute-routed endpoints (e.g. /api/transcribe)

app.MapControllerRoute(
    name: "default",
    pattern: "{controller=Transcription}/{action=Index}/{id?}");

app.Run();
