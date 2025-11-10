#!/usr/bin/env python3
"""
Migration: Add social_media_posts table for engagement tracking

Creates the social_media_posts table to track published content and
real-time engagement metrics from social media platforms.
"""

import asyncio
from sqlalchemy import text

from backend.database.connection import get_async_session, engine
from backend.models.social_media_post import SocialMediaPostModel
from backend.database.connection import Base


async def migrate():
    """Run the migration to add social_media_posts table."""
    print("Starting migration: Add social_media_posts table")

    async with engine.begin() as conn:
        # Create the table
        await conn.run_sync(Base.metadata.create_all, tables=[SocialMediaPostModel.__table__])
        print("✓ Created social_media_posts table")

        # Add indexes for performance
        indexes = [
            """
            CREATE INDEX IF NOT EXISTS idx_smp_persona_platform 
            ON social_media_posts(persona_id, platform);
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_smp_published_status 
            ON social_media_posts(published_at, status);
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_smp_engagement_rate 
            ON social_media_posts(engagement_rate) 
            WHERE engagement_rate IS NOT NULL;
            """,
        ]

        for idx_sql in indexes:
            await conn.execute(text(idx_sql))

        print("✓ Created performance indexes")

    print("\nMigration completed successfully!")
    print("\nNew table: social_media_posts")
    print("  - Tracks published social media posts")
    print("  - Links to content, persona, and ACD contexts")
    print("  - Stores engagement metrics (likes, comments, shares, etc.)")
    print("  - Filters bot and AI persona interactions")
    print("  - Enables learning from real-time social signals")


if __name__ == "__main__":
    asyncio.run(migrate())
