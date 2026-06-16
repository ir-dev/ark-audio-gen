using ArkTextTranslator.Models;
using Microsoft.EntityFrameworkCore;

namespace ArkTextTranslator.Data;

public class AppDbContext : DbContext
{
    public AppDbContext(DbContextOptions<AppDbContext> options) : base(options) { }

    public DbSet<User> Users => Set<User>();
    public DbSet<OtpRequest> OtpRequests => Set<OtpRequest>();

    protected override void OnModelCreating(ModelBuilder modelBuilder)
    {
        modelBuilder.Entity<User>(e =>
        {
            e.HasIndex(u => u.Email).IsUnique();
            e.HasIndex(u => u.ApiKey).IsUnique();
        });

        modelBuilder.Entity<OtpRequest>(e =>
        {
            e.HasIndex(o => o.Email);
        });
    }
}
