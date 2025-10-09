# Persona Update Session Cache Fix

## Problem
Personas were not updating correctly in production with multiple gunicorn workers. The logs showed successful UPDATE and COMMIT operations, but subsequent GET requests from other workers might not see the updated data.

## Root Cause
The issue was caused by SQLAlchemy session cache management when using Core `update()` statements:

1. **Core update() doesn't update session identity map**: When using SQLAlchemy Core's `update()` statement (as opposed to ORM-level updates), the session's identity map is not automatically updated with the new values.

2. **expire_on_commit=False**: The database connection is configured with `expire_on_commit=False` to keep objects valid after commit, which is good for performance but means objects won't automatically refresh.

3. **Stale cache after commit**: After the Core `update()` commits, when `get_persona()` is called to return the updated data, it may fetch from the session's stale cache instead of the database.

4. **Cross-session consistency**: Even though each worker has its own session, the pattern of "update with Core, commit, then immediately fetch" was returning stale data from the session that performed the update.

## Solution
Added `self.db.expire_all()` immediately after commit operations that use Core `update()` statements. This marks all objects in the session as stale, forcing the next query to fetch fresh data from the database.

### Code Changes

#### Before (Broken):
```python
await self.db.execute(stmt)
await self.db.commit()

logger.info(f"Updated persona {persona_id}: {list(update_data.keys())}")
return await self.get_persona(persona_id)  # May return stale data!
```

#### After (Fixed):
```python
await self.db.execute(stmt)
await self.db.commit()

# Expire session cache to ensure fresh data on next query
# This is necessary because Core update() doesn't update the session identity map
self.db.expire_all()

logger.info(f"Updated persona {persona_id}: {list(update_data.keys())}")
return await self.get_persona(persona_id)  # Now returns fresh data!
```

## Why This Works
1. Core `update()` executes SQL directly: `UPDATE personas SET name=? WHERE id=?`
2. The UPDATE is committed to the SQLite database with WAL mode
3. **Without expire_all()**: The session still has cached objects in its identity map
4. **With expire_all()**: All cached objects are marked as stale
5. Next `SELECT` query must fetch from database, seeing the committed changes
6. All workers (separate sessions) see the committed data

## Files Modified
1. `src/backend/services/persona_service.py` - Added `self.db.expire_all()` after update_persona commit
2. `src/backend/services/user_service.py` - Added `self.db.expire_all()` after update_user commit

## Testing
✅ **Unit Tests**: All critical tests pass, including `test_update_persona_success`
✅ **Integration Test**: Custom test verifies updates persist across sessions
✅ **Concurrent Test**: Multiple "workers" updating and reading show consistent data
✅ **Demo Script**: `demo.py` runs successfully with working persona updates

## When to Use expire_all()
Use `session.expire_all()` after commit when:
- Using SQLAlchemy Core `update()` or `delete()` statements
- Immediately fetching the updated/deleted records after commit
- Working with `expire_on_commit=False` (our configuration)
- Need to ensure cross-session consistency

Don't need `expire_all()` when:
- Using ORM-level updates (e.g., `persona.name = "new name"`)
- Not fetching the records after update
- Using `session.refresh(obj)` on specific objects

## Alternative Solutions Considered
1. **Remove expire_on_commit=False**: Would auto-expire objects but hurt performance
2. **Use ORM updates**: More code changes, slower for bulk updates
3. **Refresh specific objects**: Works but requires tracking which objects to refresh
4. **expire_all()**: ✅ Minimal change, ensures consistency, best solution

## Production Impact
- **Performance**: Negligible - expire_all() just marks objects as stale, no DB queries
- **Compatibility**: Fully backward compatible, no API changes
- **Reliability**: Fixes the data consistency issue without side effects

## Related Documentation
- SQLAlchemy Session Basics: https://docs.sqlalchemy.org/en/20/orm/session_basics.html
- expire_on_commit: https://docs.sqlalchemy.org/en/20/orm/session_api.html#sqlalchemy.orm.Session.params.expire_on_commit
- Core vs ORM: https://docs.sqlalchemy.org/en/20/core/tutorial.html
