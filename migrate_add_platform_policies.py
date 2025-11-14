#!/usr/bin/env python3
"""
Migration: Add Platform Policies Table

Creates the platform_policies table and seeds it with default platform policies.
This allows platform content rating rules to be managed dynamically without code changes.
"""

import asyncio
from sqlalchemy import text

from backend.database.connection import database_manager
from backend.models.platform_policy import PlatformPolicyModel, DEFAULT_PLATFORM_POLICIES
from backend.services.platform_policy_service import PlatformPolicyService
from backend.config.logging import get_logger

logger = get_logger(__name__)


async def migrate():
    """Add platform_policies table and seed with defaults."""
    logger.info("=" * 80)
    logger.info("MIGRATION: Add Platform Policies Table")
    logger.info("=" * 80)
    
    try:
        # Connect to database
        await database_manager.connect()
        logger.info("✓ Connected to database")
        
        # Create tables
        logger.info("Creating platform_policies table...")
        from backend.database.connection import Base
        async with database_manager.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("✓ Table created successfully")
        
        # Seed with default policies
        logger.info("\nSeeding default platform policies...")
        async with database_manager.get_session() as db:
            policy_service = PlatformPolicyService(db)
            created_count = await policy_service.initialize_default_policies()
            
            if created_count > 0:
                logger.info(f"✓ Created {created_count} default platform policies")
            else:
                logger.info("✓ Platform policies already exist, skipping seed")
        
        # List all policies
        logger.info("\nVerifying platform policies...")
        async with database_manager.get_session() as db:
            policy_service = PlatformPolicyService(db)
            policies = await policy_service.list_all_policies()
            
            logger.info(f"\nTotal platform policies: {len(policies)}")
            for policy in policies:
                logger.info(f"  - {policy.platform_display_name} ({policy.platform_name})")
                logger.info(f"    Allowed ratings: {', '.join(policy.allowed_content_ratings)}")
                logger.info(f"    Requires age verification: {policy.requires_age_verification}")
        
        logger.info("\n" + "=" * 80)
        logger.info("✅ MIGRATION COMPLETE")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"❌ Migration failed: {str(e)}")
        raise
    finally:
        await database_manager.disconnect()


if __name__ == "__main__":
    asyncio.run(migrate())
