# Gator Database Migration Scripts

This directory contains database migration scripts for adding new fields, tables, and updating schema for the Gator platform.

## Migration Scripts

| Script | Description |
|--------|-------------|
| `add_domain_fields_migration.py` | Adds domain-related fields to the database |
| `migrate_add_appearance_locking.py` | Adds appearance locking fields for personas |
| `migrate_add_base_image_status.py` | Adds base image status tracking |
| `migrate_add_base_images.py` | Adds base images support |
| `migrate_add_branding.py` | Adds branding configuration fields |
| `migrate_add_content_generation_prefs.py` | Adds content generation preferences |
| `migrate_add_content_triggers.py` | Adds content trigger fields |
| `migrate_add_image_style.py` | Adds image style fields |
| `migrate_add_negative_prompt.py` | Adds negative prompt fields |
| `migrate_add_platform_policies.py` | Adds platform policy fields |
| `migrate_add_settings.py` | Adds settings fields |
| `migrate_add_social_media_posts.py` | Adds social media posts support |

## Running Migrations

Run migrations from the repository root:

```bash
# Run a specific migration
python scripts/migrations/migrate_add_appearance_locking.py

# Run all migrations (recommended to run in order)
for script in scripts/migrations/migrate_*.py; do
    echo "Running $script..."
    python "$script"
done
```

## Notes

- Always back up your database before running migrations
- Migrations are idempotent where possible (can be run multiple times safely)
- Run migrations in order based on their dependencies
