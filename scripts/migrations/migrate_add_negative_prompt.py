"""
Database migration to add default_negative_prompt column to personas table.

This migration decouples negative prompts from hardcoded style defaults,
allowing each persona to have their own customized "don't do this" list.

Benefits:
- Total Control: A "Goth Girl" persona can have "bright colors, happy, sunshine" in her negative prompt
- Style Integrity: Anime personas won't fight against hardcoded "negative: anime" from photorealistic defaults
- Simplified Code: Configuration moves to database where it belongs

Usage:
    python migrate_add_negative_prompt.py
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from backend.database.connection import database_manager
from backend.config.logging import setup_logging, get_logger
from sqlalchemy import text

setup_logging()
logger = get_logger(__name__)


async def migrate_database():
    """Add default_negative_prompt column to personas table."""
    print("🔄 Migrating database for negative prompt decoupling...")

    try:
        await database_manager.connect()
        db_url = str(database_manager.engine.url)
        is_sqlite = "sqlite" in db_url

        async with database_manager.engine.begin() as conn:
            # Check existing columns
            if is_sqlite:
                result = await conn.execute(text("PRAGMA table_info(personas)"))
                columns = [row[1] for row in result.fetchall()]
            else:
                result = await conn.execute(
                    text(
                        "SELECT column_name FROM information_schema.columns "
                        "WHERE table_name = 'personas'"
                    )
                )
                columns = [row[0] for row in result.fetchall()]

            if "default_negative_prompt" not in columns:
                print("   Adding 'default_negative_prompt' column...")
                # Default generic negative prompt for backward compatibility
                default_neg = (
                    "ugly, blurry, low quality, distorted, deformed, bad anatomy"
                )

                if is_sqlite:
                    await conn.execute(
                        text(
                            f"ALTER TABLE personas ADD COLUMN default_negative_prompt "
                            f"TEXT DEFAULT '{default_neg}'"
                        )
                    )
                else:
                    await conn.execute(
                        text(
                            f"ALTER TABLE personas ADD COLUMN default_negative_prompt "
                            f"TEXT NOT NULL DEFAULT '{default_neg}'"
                        )
                    )
                print("✅ Added 'default_negative_prompt' column")
            else:
                print("✅ Column 'default_negative_prompt' already exists")

        await database_manager.disconnect()
        return True

    except Exception as e:
        logger.error(f"Migration failed: {str(e)}")
        print(f"❌ Migration failed: {str(e)}")
        return False


if __name__ == "__main__":
    success = asyncio.run(migrate_database())
    if success:
        print("\n🎉 Migration complete! Negative prompt decoupling is now available.")
    else:
        print("\n❌ Migration failed. Check logs for details.")
        sys.exit(1)
