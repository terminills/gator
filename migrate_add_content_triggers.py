#!/usr/bin/env python3
"""
Database Migration: Add content_triggers JSON field

Adds the content_triggers JSON column to the personas table for storing
trigger-based model orchestration configuration. This enables:
- Trigger words that route to specific models/LoRAs
- Per-trigger positive and negative prompts
- Weight preferences for each model/LoRA
- View-type and category-based routing

Usage:
    python migrate_add_content_triggers.py
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from backend.database.connection import database_manager
from backend.config.logging import setup_logging, get_logger

# Setup logging
setup_logging()
logger = get_logger(__name__)


async def migrate_database():
    """Add content_triggers column to personas table."""
    print("🔄 Migrating database for trigger-based model orchestration...")
    
    try:
        # Connect to database
        await database_manager.connect()
        
        # Check if running on SQLite or PostgreSQL
        db_url = str(database_manager.engine.url)
        is_sqlite = "sqlite" in db_url
        
        print(f"   Database type: {'SQLite' if is_sqlite else 'PostgreSQL'}")
        
        # Add columns using raw SQL
        async with database_manager.engine.begin() as conn:
            from sqlalchemy import text
            
            # Check if columns already exist
            if is_sqlite:
                result = await conn.execute(
                    text("PRAGMA table_info(personas)")
                )
                columns = [row[1] for row in result.fetchall()]
                
                if "content_triggers" not in columns:
                    print("   Adding content_triggers column...")
                    await conn.execute(
                        text("ALTER TABLE personas ADD COLUMN content_triggers TEXT DEFAULT '{}'")
                    )
                    print("   ✅ Added content_triggers column")
                else:
                    print("   ℹ️ content_triggers column already exists")
                    
            else:
                # PostgreSQL
                result = await conn.execute(
                    text("""
                        SELECT column_name 
                        FROM information_schema.columns 
                        WHERE table_name = 'personas'
                    """)
                )
                columns = [row[0] for row in result.fetchall()]
                
                if "content_triggers" not in columns:
                    print("   Adding content_triggers column...")
                    await conn.execute(
                        text("ALTER TABLE personas ADD COLUMN content_triggers JSONB DEFAULT '{}'::jsonb")
                    )
                    print("   ✅ Added content_triggers column")
                else:
                    print("   ℹ️ content_triggers column already exists")
        
        print("✅ Migration complete!")
        
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        print(f"❌ Migration failed: {e}")
        raise
        
    finally:
        await database_manager.disconnect()


if __name__ == "__main__":
    asyncio.run(migrate_database())
