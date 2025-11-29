#!/usr/bin/env python3
"""
Database Migration: Add base_images JSON field

Adds the base_images JSON column to the personas table for storing
4 base images (face_shot, bikini_front, bikini_side, bikini_rear)
for complete physical appearance locking.

Usage:
    python migrate_add_base_images.py
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
    """Add base_images column to personas table."""
    print("🔄 Migrating database for 4 base images feature...")
    
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
            
            # Add base_images if it doesn't exist
            if "base_images" not in columns:
                print("   Adding base_images column...")
                if is_sqlite:
                    # SQLite uses JSON type (stored as TEXT internally)
                    await conn.execute(
                        text("ALTER TABLE personas ADD COLUMN base_images JSON DEFAULT '{}'")
                    )
                else:
                    # PostgreSQL supports native JSON/JSONB
                    await conn.execute(
                        text("ALTER TABLE personas ADD COLUMN base_images JSONB DEFAULT '{}'")
                    )
                print("   ✅ Added base_images column")
            else:
                print("   ⏭️  base_images column already exists")
        
        print("\n✅ Migration completed successfully!")
        print("   New column added to personas table:")
        print("   • base_images (JSON) - 4 base images for physical appearance:")
        print("     - face_shot: Portrait/headshot image")
        print("     - bikini_front: Front view bikini image")
        print("     - bikini_side: Side view bikini image")
        print("     - bikini_rear: Rear view bikini image")
        
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
        print("   You can now store 4 base images per persona.")
    else:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
