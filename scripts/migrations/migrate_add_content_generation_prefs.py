#!/usr/bin/env python3
"""
Database Migration: Add Content Generation Preferences to Personas

Adds new columns to the personas table for content generation preferences:
- default_image_resolution
- default_video_resolution
- post_style
- video_types
- nsfw_model_preference
- generation_quality
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from sqlalchemy import text
from backend.database.connection import database_manager
from backend.config.logging import get_logger

logger = get_logger(__name__)


async def migrate_add_content_generation_prefs():
    """Add content generation preference columns to personas table."""
    logger.info("Starting migration: Add content generation preferences")

    try:
        await database_manager.connect()
        
        async with database_manager.get_session() as session:
            # Check if columns already exist
            result = await session.execute(text("PRAGMA table_info(personas)"))
            columns = [row[1] for row in result.fetchall()]
            
            migrations_needed = []
            
            if "default_image_resolution" not in columns:
                migrations_needed.append(
                    "ALTER TABLE personas ADD COLUMN default_image_resolution VARCHAR(20) DEFAULT '1024x1024' NOT NULL"
                )
            
            if "default_video_resolution" not in columns:
                migrations_needed.append(
                    "ALTER TABLE personas ADD COLUMN default_video_resolution VARCHAR(20) DEFAULT '1920x1080' NOT NULL"
                )
            
            if "post_style" not in columns:
                migrations_needed.append(
                    "ALTER TABLE personas ADD COLUMN post_style VARCHAR(50) DEFAULT 'casual' NOT NULL"
                )
            
            if "video_types" not in columns:
                migrations_needed.append(
                    "ALTER TABLE personas ADD COLUMN video_types JSON DEFAULT '[]' NOT NULL"
                )
            
            if "nsfw_model_preference" not in columns:
                migrations_needed.append(
                    "ALTER TABLE personas ADD COLUMN nsfw_model_preference VARCHAR(100)"
                )
            
            if "generation_quality" not in columns:
                migrations_needed.append(
                    "ALTER TABLE personas ADD COLUMN generation_quality VARCHAR(20) DEFAULT 'standard' NOT NULL"
                )
            
            if not migrations_needed:
                logger.info("✅ All columns already exist, no migration needed")
                return
            
            # Execute migrations
            for migration_sql in migrations_needed:
                logger.info(f"Executing: {migration_sql}")
                await session.execute(text(migration_sql))
            
            await session.commit()
            logger.info(f"✅ Added {len(migrations_needed)} new columns to personas table")
            
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        raise
    finally:
        await database_manager.disconnect()


if __name__ == "__main__":
    asyncio.run(migrate_add_content_generation_prefs())
