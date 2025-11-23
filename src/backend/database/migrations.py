"""
Database Migrations

Automatic schema migration system that ensures the database schema
matches the current models. Handles missing columns safely.
"""

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine
from typing import List, Dict, Any

from backend.config.logging import get_logger

logger = get_logger(__name__)


async def check_column_exists(
    conn, table_name: str, column_name: str, is_sqlite: bool
) -> bool:
    """
    Check if a column exists in a table.

    Args:
        conn: Database connection
        table_name: Name of the table
        column_name: Name of the column
        is_sqlite: Whether the database is SQLite

    Returns:
        bool: True if column exists, False otherwise
    """
    if is_sqlite:
        result = await conn.execute(text(f"PRAGMA table_info({table_name})"))
        columns = [row[1] for row in result.fetchall()]
    else:
        result = await conn.execute(
            text(
                """
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = :table_name
            """
            ),
            {"table_name": table_name},
        )
        columns = [row[0] for row in result.fetchall()]

    return column_name in columns


async def table_exists(conn, table_name: str, is_sqlite: bool) -> bool:
    """
    Check if a table exists in the database.

    Args:
        conn: Database connection
        table_name: Name of the table
        is_sqlite: Whether the database is SQLite

    Returns:
        bool: True if table exists, False otherwise
    """
    if is_sqlite:
        result = await conn.execute(
            text("SELECT name FROM sqlite_master WHERE type='table' AND name = :table_name"),
            {"table_name": table_name}
        )
        tables = [row[0] for row in result.fetchall()]
    else:
        result = await conn.execute(
            text(
                """
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_name = :table_name
            """
            ),
            {"table_name": table_name},
        )
        tables = [row[0] for row in result.fetchall()]

    return table_name in tables


async def add_personas_appearance_columns(conn, is_sqlite: bool) -> List[str]:
    """
    Add appearance locking columns to personas table if they don't exist.

    Args:
        conn: Database connection
        is_sqlite: Whether the database is SQLite

    Returns:
        List of columns that were added
    """
    added_columns = []

    # Check if table exists first
    if not await table_exists(conn, "personas", is_sqlite):
        logger.debug("Personas table does not exist, skipping migration")
        return added_columns

    # Check and add base_appearance_description
    if not await check_column_exists(
        conn, "personas", "base_appearance_description", is_sqlite
    ):
        logger.info("Adding base_appearance_description column to personas table")
        await conn.execute(
            text("ALTER TABLE personas ADD COLUMN base_appearance_description TEXT")
        )
        added_columns.append("base_appearance_description")

    # Check and add base_image_path
    if not await check_column_exists(conn, "personas", "base_image_path", is_sqlite):
        logger.info("Adding base_image_path column to personas table")
        await conn.execute(
            text("ALTER TABLE personas ADD COLUMN base_image_path VARCHAR(500)")
        )
        added_columns.append("base_image_path")

    # Check and add appearance_locked
    if not await check_column_exists(conn, "personas", "appearance_locked", is_sqlite):
        logger.info("Adding appearance_locked column to personas table")
        if is_sqlite:
            await conn.execute(
                text(
                    "ALTER TABLE personas ADD COLUMN appearance_locked BOOLEAN DEFAULT 0"
                )
            )
        else:
            await conn.execute(
                text(
                    "ALTER TABLE personas ADD COLUMN appearance_locked BOOLEAN DEFAULT FALSE"
                )
            )
        added_columns.append("appearance_locked")

        # Create index for appearance_locked
        try:
            logger.info("Creating index on appearance_locked column")
            await conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS ix_personas_appearance_locked ON personas (appearance_locked)"
                )
            )
        except Exception as e:
            logger.warning(f"Could not create index on appearance_locked: {e}")

    # Check and add base_image_status
    if not await check_column_exists(conn, "personas", "base_image_status", is_sqlite):
        logger.info("Adding base_image_status column to personas table")
        if is_sqlite:
            await conn.execute(
                text(
                    "ALTER TABLE personas ADD COLUMN base_image_status VARCHAR(20) DEFAULT 'pending_upload'"
                )
            )
            # Update existing rows to have the default value
            await conn.execute(
                text(
                    "UPDATE personas SET base_image_status = 'pending_upload' WHERE base_image_status IS NULL"
                )
            )
        else:
            await conn.execute(
                text(
                    "ALTER TABLE personas ADD COLUMN base_image_status VARCHAR(20) NOT NULL DEFAULT 'pending_upload'"
                )
            )
        added_columns.append("base_image_status")

        # Create index for base_image_status
        try:
            logger.info("Creating index on base_image_status column")
            await conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS ix_personas_base_image_status ON personas (base_image_status)"
                )
            )
        except Exception as e:
            logger.warning(f"Could not create index on base_image_status: {e}")

    return added_columns


async def add_acd_domain_columns(conn, is_sqlite: bool) -> List[str]:
    """
    Add domain classification columns to acd_contexts table if they don't exist.

    Args:
        conn: Database connection
        is_sqlite: Whether the database is SQLite

    Returns:
        List of columns that were added
    """
    added_columns = []

    # Check if table exists first
    if not await table_exists(conn, "acd_contexts", is_sqlite):
        logger.debug("acd_contexts table does not exist, skipping migration")
        return added_columns

    # Check and add ai_domain
    if not await check_column_exists(conn, "acd_contexts", "ai_domain", is_sqlite):
        logger.info("Adding ai_domain column to acd_contexts table")
        await conn.execute(
            text("ALTER TABLE acd_contexts ADD COLUMN ai_domain VARCHAR(50)")
        )
        added_columns.append("ai_domain")

        # Create index for ai_domain
        try:
            logger.info("Creating index on ai_domain column")
            await conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS ix_acd_contexts_ai_domain ON acd_contexts (ai_domain)"
                )
            )
        except Exception as e:
            logger.warning(f"Could not create index on ai_domain: {e}")

    # Check and add ai_subdomain
    if not await check_column_exists(conn, "acd_contexts", "ai_subdomain", is_sqlite):
        logger.info("Adding ai_subdomain column to acd_contexts table")
        await conn.execute(
            text("ALTER TABLE acd_contexts ADD COLUMN ai_subdomain VARCHAR(50)")
        )
        added_columns.append("ai_subdomain")

        # Create index for ai_subdomain
        try:
            logger.info("Creating index on ai_subdomain column")
            await conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS ix_acd_contexts_ai_subdomain ON acd_contexts (ai_subdomain)"
                )
            )
        except Exception as e:
            logger.warning(f"Could not create index on ai_subdomain: {e}")

    return added_columns


async def run_migrations(engine: AsyncEngine) -> Dict[str, Any]:
    """
    Run all pending migrations on the database.
    
    Automatic migrations can be controlled via the AUTO_MIGRATE environment variable.
    Set AUTO_MIGRATE=false in production to disable automatic migrations.

    Args:
        engine: SQLAlchemy async engine

    Returns:
        Dict with migration results
    """
    import os
    
    # Check if auto-migration is enabled
    auto_migrate = os.getenv("AUTO_MIGRATE", "true").lower() in ("true", "1", "yes")
    env = os.getenv("GATOR_ENV", "development").lower()
    
    if not auto_migrate:
        logger.info("Automatic migrations disabled (AUTO_MIGRATE=false)")
        return {
            "migrations_run": [],
            "columns_added": [],
            "success": True,
            "skipped": True,
            "reason": "AUTO_MIGRATE disabled"
        }
    
    if env == "production":
        logger.warning(
            "Running automatic migrations in production. "
            "Set AUTO_MIGRATE=false to disable this in production."
        )
    
    db_url = str(engine.url)
    is_sqlite = "sqlite" in db_url

    logger.info("Checking for pending database migrations")

    results = {
        "migrations_run": [],
        "columns_added": [],
        "success": True,
        "error": None,
    }

    try:
        async with engine.begin() as conn:
            # Run personas table migrations
            columns_added = await add_personas_appearance_columns(conn, is_sqlite)

            if columns_added:
                results["migrations_run"].append("personas_appearance_locking")
                results["columns_added"].extend(columns_added)
                logger.info(
                    f"Added {len(columns_added)} column(s) to personas table: {', '.join(columns_added)}"
                )
            else:
                logger.info("All personas table columns are up to date")

            # Run ACD contexts table migrations
            acd_columns_added = await add_acd_domain_columns(conn, is_sqlite)

            if acd_columns_added:
                results["migrations_run"].append("acd_contexts_domain_fields")
                results["columns_added"].extend(acd_columns_added)
                logger.info(
                    f"Added {len(acd_columns_added)} column(s) to acd_contexts table: {', '.join(acd_columns_added)}"
                )
            else:
                logger.info("All acd_contexts table columns are up to date")

        return results

    except Exception as e:
        logger.error(f"Migration failed: {str(e)}")
        results["success"] = False
        results["error"] = str(e)
        return results
