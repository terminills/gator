#!/usr/bin/env python3
"""
Database Migration: Add Visual Consistency Fields

Adds base_appearance_description, base_image_path, and appearance_locked
fields to the personas table for visual consistency features.

Usage:
    python migrate_add_appearance_locking.py
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
    """Add new columns to personas table."""
    print("🔄 Migrating database for appearance locking features...")
    
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
            else:
                result = await conn.execute(
                    text("""
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_name='personas'
                    """)
                )
                columns = [row[0] for row in result.fetchall()]
            
            # Add base_appearance_description if it doesn't exist
            if "base_appearance_description" not in columns:
                print("   Adding base_appearance_description column...")
                await conn.execute(
                    text("ALTER TABLE personas ADD COLUMN base_appearance_description TEXT")
                )
                print("   ✅ Added base_appearance_description")
            else:
                print("   ⏭️  base_appearance_description already exists")
            
            # Add base_image_path if it doesn't exist
            if "base_image_path" not in columns:
                print("   Adding base_image_path column...")
                await conn.execute(
                    text("ALTER TABLE personas ADD COLUMN base_image_path VARCHAR(500)")
                )
                print("   ✅ Added base_image_path")
            else:
                print("   ⏭️  base_image_path already exists")
            
            # Add appearance_locked if it doesn't exist
            if "appearance_locked" not in columns:
                print("   Adding appearance_locked column...")
                if is_sqlite:
                    await conn.execute(
                        text("ALTER TABLE personas ADD COLUMN appearance_locked BOOLEAN DEFAULT 0")
                    )
                else:
                    await conn.execute(
                        text("ALTER TABLE personas ADD COLUMN appearance_locked BOOLEAN DEFAULT FALSE")
                    )
                print("   ✅ Added appearance_locked")
                
                # Create index for appearance_locked
                print("   Creating index on appearance_locked...")
                await conn.execute(
                    text("CREATE INDEX ix_personas_appearance_locked ON personas (appearance_locked)")
                )
                print("   ✅ Created index")
            else:
                print("   ⏭️  appearance_locked already exists")
        
        print("\n✅ Migration completed successfully!")
        print("   New columns added to personas table:")
        print("   • base_appearance_description (TEXT) - Detailed baseline appearance")
        print("   • base_image_path (VARCHAR(500)) - Reference image path")
        print("   • appearance_locked (BOOLEAN) - Consistency lock flag")
        
        # Disconnect
        await database_manager.disconnect()
        
        return True
        
    except Exception as e:
        logger.error("Database migration failed", error=str(e))
        print(f"❌ Migration failed: {e}")
        return False


async def main():
    """Run database migration."""
    success = await migrate_database()
    if success:
        print("\n🎉 Database migration complete!")
        print("   You can now use the new appearance locking features.")
    else:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
