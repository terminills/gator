# Platform Policies Migration Fix

## Issue Summary

The `migrate_add_platform_policies.py` script was failing with an `ImportError`:

```
ImportError: cannot import name 'engine' from 'backend.database.connection'
```

This occurred on line 12 of the migration script when attempting to import `engine` directly.

## Root Cause

The `backend.database.connection` module does not export `engine` at the module level. Instead:
- The module exports a `database_manager` singleton instance
- The `engine` is a property of `database_manager` that is only initialized after calling `database_manager.connect()`
- The script was using an outdated import pattern that no longer matched the current database connection architecture

## Solution

Updated `migrate_add_platform_policies.py` to follow the standard pattern used by other migration scripts in the repository:

### Changes Made

1. **Fixed Imports**
   ```python
   # Before:
   from backend.database.connection import get_db_session, engine
   
   # After:
   from backend.database.connection import database_manager
   ```

2. **Added Database Connection Lifecycle**
   ```python
   # Connect to database before using engine
   await database_manager.connect()
   
   # Use the engine
   async with database_manager.engine.begin() as conn:
       await conn.run_sync(Base.metadata.create_all)
   
   # Clean up in finally block
   finally:
       await database_manager.disconnect()
   ```

3. **Updated Session Management**
   ```python
   # Before:
   async with get_db_session() as db:
   
   # After:
   async with database_manager.get_session() as db:
   ```

## Verification

The migration now works correctly and:
- ✅ Creates the `platform_policies` table
- ✅ Seeds 10 default platform policies (Instagram, Facebook, Twitter, OnlyFans, Patreon, Discord, Reddit, TikTok, YouTube, Twitch)
- ✅ Is idempotent (can be run multiple times without creating duplicates)
- ✅ Exits with status code 0 on success
- ✅ Properly cleans up database connections

## Usage

```bash
# Run the migration
python migrate_add_platform_policies.py

# Verify the migration
python -c "
import sqlite3
conn = sqlite3.connect('gator.db')
cursor = conn.cursor()
cursor.execute('SELECT COUNT(*) FROM platform_policies')
print(f'Platform policies: {cursor.fetchone()[0]}')
conn.close()
"
```

## Similar Patterns

This fix aligns with other migration scripts in the repository:
- `migrate_add_branding.py` - Uses `database_manager` pattern
- `migrate_add_settings.py` - Uses `database_manager.engine` after connect

## Related Files

- `migrate_add_platform_policies.py` - The fixed migration script
- `src/backend/database/connection.py` - Database connection management
- `src/backend/models/platform_policy.py` - Platform policy models and defaults
- `src/backend/services/platform_policy_service.py` - Platform policy service
