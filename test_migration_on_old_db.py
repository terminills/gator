#!/usr/bin/env python3
"""
Test that the migration correctly adds ai_domain and ai_subdomain to an old database.
"""

import asyncio
import sqlite3
import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from backend.database.migrations import run_migrations
from backend.config.logging import setup_logging, get_logger
from sqlalchemy.ext.asyncio import create_async_engine

setup_logging()
logger = get_logger(__name__)


def check_columns_exist(db_path: str) -> dict:
    """Check if ai_domain and ai_subdomain columns exist in acd_contexts table."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("PRAGMA table_info(acd_contexts)")
    columns = [row[1] for row in cursor.fetchall()]
    
    conn.close()
    
    return {
        "ai_domain": "ai_domain" in columns,
        "ai_subdomain": "ai_subdomain" in columns,
        "column_count": len(columns)
    }


async def test_migration_on_old_database():
    """Test migration on database with old schema."""
    
    db_path = "gator_old_schema.db"
    
    if not os.path.exists(db_path):
        print("❌ Old schema database not found. Run test_old_schema.py first.")
        return False
    
    print("=" * 70)
    print("TESTING MIGRATION ON OLD DATABASE SCHEMA")
    print("=" * 70)
    print()
    
    # Check initial state (should NOT have domain columns)
    print("1. Checking initial database state (OLD SCHEMA)...")
    initial_state = check_columns_exist(db_path)
    print(f"   Columns in table: {initial_state['column_count']}")
    print(f"   ai_domain exists: {initial_state['ai_domain']}")
    print(f"   ai_subdomain exists: {initial_state['ai_subdomain']}")
    
    if initial_state['ai_domain'] or initial_state['ai_subdomain']:
        print("\n   ⚠️  WARNING: Columns already exist! This is not an old schema.")
        return False
    
    print("\n   ✅ Confirmed: Old schema without domain columns")
    
    # Run migration
    print("\n2. Running migration on old database...")
    try:
        # Create engine for the old database
        engine = create_async_engine(
            f"sqlite+aiosqlite:///{db_path}",
            echo=False,
            future=True,
        )
        
        # Run migrations
        migration_results = await run_migrations(engine)
        
        print(f"   Migrations run: {migration_results['migrations_run']}")
        print(f"   Columns added: {migration_results['columns_added']}")
        print(f"   Success: {migration_results['success']}")
        
        if migration_results.get('error'):
            print(f"   ❌ Error: {migration_results['error']}")
            await engine.dispose()
            return False
        
        # Clean up engine
        await engine.dispose()
        
        print("\n   ✅ Migration completed")
        
    except Exception as e:
        print(f"\n   ❌ Migration failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Verify columns were added
    print("\n3. Verifying columns after migration...")
    final_state = check_columns_exist(db_path)
    print(f"   Columns in table: {final_state['column_count']}")
    print(f"   ai_domain exists: {final_state['ai_domain']}")
    print(f"   ai_subdomain exists: {final_state['ai_subdomain']}")
    
    if final_state['ai_domain'] and final_state['ai_subdomain']:
        print("\n   ✅ All required columns successfully added!")
        
        # Verify we added exactly 2 columns
        columns_added = final_state['column_count'] - initial_state['column_count']
        print(f"\n   Columns added: {columns_added} (expected: 2)")
        
        if columns_added == 2:
            success = True
        else:
            print("   ⚠️  Unexpected number of columns added")
            success = False
    else:
        print("\n   ❌ Some columns are still missing!")
        success = False
    
    # Check indexes were created
    print("\n4. Verifying indexes...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='index' AND name LIKE '%ai_domain%' OR name LIKE '%ai_subdomain%'")
    indexes = [row[0] for row in cursor.fetchall()]
    conn.close()
    
    print(f"   Indexes found: {indexes}")
    
    expected_indexes = ['ix_acd_contexts_ai_domain', 'ix_acd_contexts_ai_subdomain']
    indexes_ok = all(idx in indexes for idx in expected_indexes)
    
    if indexes_ok:
        print("   ✅ All indexes created successfully")
    else:
        print(f"   ⚠️  Some indexes missing. Expected: {expected_indexes}")
    
    print("\n" + "=" * 70)
    if success and indexes_ok:
        print("✅ MIGRATION TEST PASSED - Old database successfully migrated!")
    else:
        print("❌ MIGRATION TEST FAILED")
    print("=" * 70)
    
    return success and indexes_ok


if __name__ == "__main__":
    success = asyncio.run(test_migration_on_old_database())
    sys.exit(0 if success else 1)
