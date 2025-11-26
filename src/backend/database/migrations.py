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

    # Check and add base_images (JSON field for 4 base images: face_shot, bikini_front, bikini_side, bikini_rear)
    if not await check_column_exists(conn, "personas", "base_images", is_sqlite):
        logger.info("Adding base_images column to personas table")
        if is_sqlite:
            # SQLite uses JSON type (stored as TEXT internally)
            await conn.execute(
                text("ALTER TABLE personas ADD COLUMN base_images JSON DEFAULT '{}'")
            )
            # Update existing rows to have the default value
            await conn.execute(
                text("UPDATE personas SET base_images = '{}' WHERE base_images IS NULL")
            )
        else:
            # PostgreSQL supports native JSON/JSONB
            await conn.execute(
                text("ALTER TABLE personas ADD COLUMN base_images JSONB DEFAULT '{}'")
            )
        added_columns.append("base_images")

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


async def add_persona_soul_columns(conn, is_sqlite: bool) -> List[str]:
    """
    Add persona soul fields for human-like response generation.
    
    These fields capture the "soul" of a persona:
    1. Origin & Demographics (The "Roots")
    2. Psychological Profile (The "Engine")
    3. Voice & Speech Patterns (The "Interface")
    4. Backstory & Lore (The "Context")
    5. Anti-Pattern (What they are NOT)

    Args:
        conn: Database connection
        is_sqlite: Whether the database is SQLite

    Returns:
        List of columns that were added
    """
    added_columns = []

    # Check if table exists first
    if not await table_exists(conn, "personas", is_sqlite):
        logger.debug("Personas table does not exist, skipping soul fields migration")
        return added_columns

    # ==================== ORIGIN & DEMOGRAPHICS ====================
    
    if not await check_column_exists(conn, "personas", "hometown", is_sqlite):
        logger.info("Adding hometown column to personas table")
        await conn.execute(
            text("ALTER TABLE personas ADD COLUMN hometown VARCHAR(200)")
        )
        added_columns.append("hometown")

    if not await check_column_exists(conn, "personas", "current_location", is_sqlite):
        logger.info("Adding current_location column to personas table")
        await conn.execute(
            text("ALTER TABLE personas ADD COLUMN current_location VARCHAR(200)")
        )
        added_columns.append("current_location")

    if not await check_column_exists(conn, "personas", "generation_age", is_sqlite):
        logger.info("Adding generation_age column to personas table")
        await conn.execute(
            text("ALTER TABLE personas ADD COLUMN generation_age VARCHAR(100)")
        )
        added_columns.append("generation_age")

    if not await check_column_exists(conn, "personas", "education_level", is_sqlite):
        logger.info("Adding education_level column to personas table")
        await conn.execute(
            text("ALTER TABLE personas ADD COLUMN education_level VARCHAR(200)")
        )
        added_columns.append("education_level")

    # ==================== PSYCHOLOGICAL PROFILE ====================
    
    if not await check_column_exists(conn, "personas", "mbti_type", is_sqlite):
        logger.info("Adding mbti_type column to personas table")
        await conn.execute(
            text("ALTER TABLE personas ADD COLUMN mbti_type VARCHAR(50)")
        )
        added_columns.append("mbti_type")

    if not await check_column_exists(conn, "personas", "enneagram_type", is_sqlite):
        logger.info("Adding enneagram_type column to personas table")
        await conn.execute(
            text("ALTER TABLE personas ADD COLUMN enneagram_type VARCHAR(50)")
        )
        added_columns.append("enneagram_type")

    if not await check_column_exists(conn, "personas", "political_alignment", is_sqlite):
        logger.info("Adding political_alignment column to personas table")
        await conn.execute(
            text("ALTER TABLE personas ADD COLUMN political_alignment VARCHAR(100)")
        )
        added_columns.append("political_alignment")

    if not await check_column_exists(conn, "personas", "risk_tolerance", is_sqlite):
        logger.info("Adding risk_tolerance column to personas table")
        await conn.execute(
            text("ALTER TABLE personas ADD COLUMN risk_tolerance VARCHAR(100)")
        )
        added_columns.append("risk_tolerance")

    if not await check_column_exists(conn, "personas", "optimism_cynicism_scale", is_sqlite):
        logger.info("Adding optimism_cynicism_scale column to personas table")
        await conn.execute(
            text("ALTER TABLE personas ADD COLUMN optimism_cynicism_scale INTEGER")
        )
        added_columns.append("optimism_cynicism_scale")

    # ==================== VOICE & SPEECH PATTERNS ====================
    
    if not await check_column_exists(conn, "personas", "linguistic_register", is_sqlite):
        logger.info("Adding linguistic_register column to personas table")
        if is_sqlite:
            await conn.execute(
                text("ALTER TABLE personas ADD COLUMN linguistic_register VARCHAR(50) DEFAULT 'blue_collar'")
            )
            await conn.execute(
                text("UPDATE personas SET linguistic_register = 'blue_collar' WHERE linguistic_register IS NULL")
            )
        else:
            await conn.execute(
                text("ALTER TABLE personas ADD COLUMN linguistic_register VARCHAR(50) NOT NULL DEFAULT 'blue_collar'")
            )
        added_columns.append("linguistic_register")

    if not await check_column_exists(conn, "personas", "typing_quirks", is_sqlite):
        logger.info("Adding typing_quirks column to personas table")
        if is_sqlite:
            await conn.execute(
                text("ALTER TABLE personas ADD COLUMN typing_quirks JSON DEFAULT '{}'")
            )
            await conn.execute(
                text("UPDATE personas SET typing_quirks = '{}' WHERE typing_quirks IS NULL")
            )
        else:
            await conn.execute(
                text("ALTER TABLE personas ADD COLUMN typing_quirks JSON NOT NULL DEFAULT '{}'")
            )
        added_columns.append("typing_quirks")

    if not await check_column_exists(conn, "personas", "signature_phrases", is_sqlite):
        logger.info("Adding signature_phrases column to personas table")
        if is_sqlite:
            await conn.execute(
                text("ALTER TABLE personas ADD COLUMN signature_phrases JSON DEFAULT '[]'")
            )
            await conn.execute(
                text("UPDATE personas SET signature_phrases = '[]' WHERE signature_phrases IS NULL")
            )
        else:
            await conn.execute(
                text("ALTER TABLE personas ADD COLUMN signature_phrases JSON NOT NULL DEFAULT '[]'")
            )
        added_columns.append("signature_phrases")

    if not await check_column_exists(conn, "personas", "trigger_topics", is_sqlite):
        logger.info("Adding trigger_topics column to personas table")
        if is_sqlite:
            await conn.execute(
                text("ALTER TABLE personas ADD COLUMN trigger_topics JSON DEFAULT '[]'")
            )
            await conn.execute(
                text("UPDATE personas SET trigger_topics = '[]' WHERE trigger_topics IS NULL")
            )
        else:
            await conn.execute(
                text("ALTER TABLE personas ADD COLUMN trigger_topics JSON NOT NULL DEFAULT '[]'")
            )
        added_columns.append("trigger_topics")

    # ==================== BACKSTORY & LORE ====================
    
    if not await check_column_exists(conn, "personas", "day_job", is_sqlite):
        logger.info("Adding day_job column to personas table")
        await conn.execute(
            text("ALTER TABLE personas ADD COLUMN day_job VARCHAR(200)")
        )
        added_columns.append("day_job")

    if not await check_column_exists(conn, "personas", "war_story", is_sqlite):
        logger.info("Adding war_story column to personas table")
        await conn.execute(
            text("ALTER TABLE personas ADD COLUMN war_story TEXT")
        )
        added_columns.append("war_story")

    if not await check_column_exists(conn, "personas", "vices_hobbies", is_sqlite):
        logger.info("Adding vices_hobbies column to personas table")
        if is_sqlite:
            await conn.execute(
                text("ALTER TABLE personas ADD COLUMN vices_hobbies JSON DEFAULT '[]'")
            )
            await conn.execute(
                text("UPDATE personas SET vices_hobbies = '[]' WHERE vices_hobbies IS NULL")
            )
        else:
            await conn.execute(
                text("ALTER TABLE personas ADD COLUMN vices_hobbies JSON NOT NULL DEFAULT '[]'")
            )
        added_columns.append("vices_hobbies")

    # ==================== ANTI-PATTERN ====================
    
    if not await check_column_exists(conn, "personas", "forbidden_phrases", is_sqlite):
        logger.info("Adding forbidden_phrases column to personas table")
        if is_sqlite:
            await conn.execute(
                text("ALTER TABLE personas ADD COLUMN forbidden_phrases JSON DEFAULT '[]'")
            )
            await conn.execute(
                text("UPDATE personas SET forbidden_phrases = '[]' WHERE forbidden_phrases IS NULL")
            )
        else:
            await conn.execute(
                text("ALTER TABLE personas ADD COLUMN forbidden_phrases JSON NOT NULL DEFAULT '[]'")
            )
        added_columns.append("forbidden_phrases")

    if not await check_column_exists(conn, "personas", "warmth_level", is_sqlite):
        logger.info("Adding warmth_level column to personas table")
        if is_sqlite:
            await conn.execute(
                text("ALTER TABLE personas ADD COLUMN warmth_level VARCHAR(20) DEFAULT 'warm'")
            )
            await conn.execute(
                text("UPDATE personas SET warmth_level = 'warm' WHERE warmth_level IS NULL")
            )
        else:
            await conn.execute(
                text("ALTER TABLE personas ADD COLUMN warmth_level VARCHAR(20) NOT NULL DEFAULT 'warm'")
            )
        added_columns.append("warmth_level")

    if not await check_column_exists(conn, "personas", "patience_level", is_sqlite):
        logger.info("Adding patience_level column to personas table")
        if is_sqlite:
            await conn.execute(
                text("ALTER TABLE personas ADD COLUMN patience_level VARCHAR(20) DEFAULT 'normal'")
            )
            await conn.execute(
                text("UPDATE personas SET patience_level = 'normal' WHERE patience_level IS NULL")
            )
        else:
            await conn.execute(
                text("ALTER TABLE personas ADD COLUMN patience_level VARCHAR(20) NOT NULL DEFAULT 'normal'")
            )
        added_columns.append("patience_level")

    if added_columns:
        logger.info(f"✓ Added {len(added_columns)} persona soul columns for human-like responses")

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
                logger.info("All personas appearance columns are up to date")

            # Run persona soul fields migration (for human-like responses)
            soul_columns_added = await add_persona_soul_columns(conn, is_sqlite)

            if soul_columns_added:
                results["migrations_run"].append("personas_soul_fields")
                results["columns_added"].extend(soul_columns_added)
                logger.info(
                    f"Added {len(soul_columns_added)} persona soul column(s): {', '.join(soul_columns_added)}"
                )
            else:
                logger.info("All persona soul columns are up to date")

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
