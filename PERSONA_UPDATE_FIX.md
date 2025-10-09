# Persona Update Fix - SQLite WAL Mode Implementation

## Issue
Persona updates were not persisting correctly when using SQLite with multiple gunicorn workers. The logs showed successful UPDATE and COMMIT operations, but subsequent queries from other workers might not see the changes.

## Root Cause
The application uses SQLite with multiple gunicorn workers (separate processes). By default, SQLite uses rollback journal mode, which has poor concurrent write performance and can cause:

1. **Write Conflicts**: Multiple workers trying to update the database simultaneously
2. **Stale Reads**: Workers not seeing recently committed changes from other workers  
3. **Lost Updates**: Last-write-wins scenarios without proper coordination
4. **Lock Timeouts**: Workers timing out when trying to acquire database locks

## Solution
Enable SQLite's Write-Ahead Logging (WAL) mode for better concurrency support:

### Changes Made

#### 1. Database Connection Configuration (`src/backend/database/connection.py`)

**Added SQLite-specific connect_args:**
```python
connect_args = {}
if "sqlite" in database_url:
    connect_args = {
        "check_same_thread": False,  # Allow multi-threaded access
        "timeout": 30,  # 30 second timeout for database locks
    }
```

**Enable WAL mode at startup:**
```python
if "sqlite" in database_url:
    async with self.engine.begin() as conn:
        await conn.execute(text("PRAGMA journal_mode=WAL"))
        await conn.execute(text("PRAGMA synchronous=NORMAL"))
        await conn.execute(text("PRAGMA busy_timeout=30000"))  # 30 second busy timeout
        logger.info("SQLite WAL mode enabled for improved concurrency")
```

### Benefits of WAL Mode

1. **Better Concurrency**: Readers don't block writers, writers don't block readers
2. **Atomic Commits**: All changes in a transaction are applied atomically
3. **Better Performance**: Reduced I/O operations, improved throughput
4. **Crash Recovery**: Better data integrity and faster recovery

### Testing

Comprehensive tests verified:
- ✅ Single session updates work correctly
- ✅ Concurrent updates from multiple "workers" complete successfully
- ✅ All workers see consistent data after updates
- ✅ No data loss or corruption with concurrent access
- ✅ API endpoints (CREATE, UPDATE, GET) work as expected

### Verification

To verify WAL mode is enabled, check the logs at startup:
```
INFO - SQLite WAL mode enabled for improved concurrency
```

Or query the database directly:
```sql
PRAGMA journal_mode;
-- Should return: wal
```

### Production Recommendations

While this fix enables SQLite to work with multiple workers, for production deployments with high concurrency, consider:

1. **Use PostgreSQL**: Better multi-process concurrency, connection pooling, and scalability
2. **Limit Worker Count**: If staying with SQLite, limit gunicorn workers to 1-4
3. **Enable WAL Checkpoint**: Configure `wal_autocheckpoint` for optimal performance
4. **Monitor Lock Contention**: Watch for `busy_timeout` errors in logs

### Migration Path

For existing deployments, the WAL mode is enabled automatically on startup. No manual migration is required. The database file will create `-wal` and `-shm` files in the same directory.

## Files Modified

- `src/backend/database/connection.py` - Added WAL mode configuration and connect_args

## Related Issues

This fix resolves concurrent access issues that can manifest as:
- Updates not saving
- Stale data being returned
- Database lock timeouts
- Inconsistent read behavior across requests
