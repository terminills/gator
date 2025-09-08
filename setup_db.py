#!/usr/bin/env python3
"""
Database Setup Script

Creates database tables for the Gator AI Influencer Platform.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from backend.database.connection import database_manager, Base
from backend.models.persona import PersonaModel
from backend.models.content import ContentModel
from backend.models.feed import RSSFeedModel, FeedItemModel
from backend.config.logging import setup_logging, get_logger

# Setup logging
setup_logging()
logger = get_logger(__name__)


async def create_tables():
    """Create all database tables."""
    print("🗄️  Setting up Gator database...")
    
    try:
        # Connect to database
        await database_manager.connect()
        
        # Create all tables
        async with database_manager.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        
        print("✅ Database tables created successfully!")
        print("   Tables:")
        print("   • personas - AI persona configurations")
        print("   • generated_content - Generated content tracking") 
        print("   • rss_feeds - RSS feed sources")
        print("   • feed_items - RSS feed item data")
        
        # Disconnect
        await database_manager.disconnect()
        
        return True
        
    except Exception as e:
        logger.error("Database setup failed", error=str(e))
        print(f"❌ Database setup failed: {e}")
        return False


async def main():
    """Run database setup."""
    success = await create_tables()
    if success:
        print("\n🎉 Database ready! You can now run the demo.")
        print("   Run: python demo.py")
    else:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())