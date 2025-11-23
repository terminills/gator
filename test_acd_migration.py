#!/usr/bin/env python3
"""
Test script to verify ACD domain fields migration works correctly.
"""

import asyncio
import sqlite3
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from backend.database.connection import database_manager
from backend.database.migrations import run_migrations
from backend.config.logging import setup_logging, get_logger

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
        "all_columns": columns
    }


async def test_migration():
    """Test that the migration adds missing columns."""
    
    db_path = "gator.db"
    
    print("=" * 70)
    print("ACD DOMAIN FIELDS MIGRATION TEST")
    print("=" * 70)
    print()
    
    # Check current state
    print("1. Checking current database state...")
    try:
        initial_state = check_columns_exist(db_path)
        print(f"   ai_domain exists: {initial_state['ai_domain']}")
        print(f"   ai_subdomain exists: {initial_state['ai_subdomain']}")
    except Exception as e:
        print(f"   ❌ Could not check database: {e}")
        return False
    
    # Connect to database (this should run migrations automatically)
    print("\n2. Connecting to database (migrations will run automatically)...")
    try:
        await database_manager.connect()
        print("   ✅ Database connected")
    except Exception as e:
        print(f"   ❌ Connection failed: {e}")
        return False
    
    # Check if columns were added
    print("\n3. Verifying columns after migration...")
    try:
        final_state = check_columns_exist(db_path)
        print(f"   ai_domain exists: {final_state['ai_domain']}")
        print(f"   ai_subdomain exists: {final_state['ai_subdomain']}")
        
        if final_state['ai_domain'] and final_state['ai_subdomain']:
            print("\n   ✅ All required columns present!")
            success = True
        else:
            print("\n   ❌ Some columns are missing!")
            success = False
    except Exception as e:
        print(f"   ❌ Verification failed: {e}")
        success = False
    
    # Clean up
    print("\n4. Cleaning up...")
    await database_manager.disconnect()
    print("   ✅ Database disconnected")
    
    print("\n" + "=" * 70)
    if success:
        print("✅ MIGRATION TEST PASSED")
    else:
        print("❌ MIGRATION TEST FAILED")
    print("=" * 70)
    
    return success


if __name__ == "__main__":
    success = asyncio.run(test_migration())
    sys.exit(0 if success else 1)
