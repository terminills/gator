#!/usr/bin/env python3
"""
Database Migration: Add Base Image Status Field

Adds base_image_status field to the personas table for the approval workflow.

Usage:
    python migrate_add_base_image_status.py
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
    """Add base_image_status column to personas table."""
    print("🔄 Migrating database for base image status field...")
    
    try:
        # Connect to database
        await database_manager.connect()
        
        # Check if running on SQLite or PostgreSQL
        db_url = str(database_manager.engine.url)
        is_sqlite = "sqlite" in db_url
        
        print(f"   Database type: {'SQLite' if is_sqlite else 'PostgreSQL'}")
        
        # Add column using raw SQL
        async with database_manager.engine.begin() as conn:
            from sqlalchemy import text
            
            # Check if column already exists
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
                        WHERE table_name = 'personas'
                    """)
                )
                columns = [row[0] for row in result.fetchall()]
            
            if "base_image_status" in columns:
                print("✅ Column 'base_image_status' already exists")
            else:
                print("   Adding 'base_image_status' column...")
                
                if is_sqlite:
                    # SQLite doesn't support adding NOT NULL with default in one step
                    await conn.execute(
                        text("""
                            ALTER TABLE personas 
                            ADD COLUMN base_image_status VARCHAR(20) DEFAULT 'pending_upload'
                        """)
                    )
                    # Update existing rows
                    await conn.execute(
                        text("""
                            UPDATE personas 
                            SET base_image_status = 'pending_upload' 
                            WHERE base_image_status IS NULL
                        """)
                    )
                else:
                    # PostgreSQL
                    await conn.execute(
                        text("""
                            ALTER TABLE personas 
                            ADD COLUMN base_image_status VARCHAR(20) 
                            NOT NULL DEFAULT 'pending_upload'
                        """)
                    )
                
                print("✅ Added 'base_image_status' column")
                
                # Add index for better query performance
                try:
                    await conn.execute(
                        text("""
                            CREATE INDEX IF NOT EXISTS ix_personas_base_image_status 
                            ON personas (base_image_status)
                        """)
                    )
                    print("✅ Added index on 'base_image_status'")
                except Exception as e:
                    logger.warning(f"Could not add index (may already exist): {e}")
        
        await database_manager.disconnect()
        
        print("✅ Migration completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Migration failed: {str(e)}")
        print(f"❌ Migration failed: {str(e)}")
        try:
            await database_manager.disconnect()
        except:
            pass
        return False


async def main():
    """Main migration entry point."""
    success = await migrate_database()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())
