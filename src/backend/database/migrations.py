"""
Database Migrations

Automatic schema migration system that ensures the database schema
matches the current models. Handles missing columns safely.
"""

from typing import Any, Dict, List

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

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
            text(
                "SELECT name FROM sqlite_master WHERE type='table' AND name = :table_name"
            ),
            {"table_name": table_name},
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
            # PostgreSQL supports native JSON type
            await conn.execute(
                text(
                    "ALTER TABLE personas ADD COLUMN base_images JSON NOT NULL DEFAULT '{}'"
                )
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

    if not await check_column_exists(
        conn, "personas", "political_alignment", is_sqlite
    ):
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

    if not await check_column_exists(
        conn, "personas", "optimism_cynicism_scale", is_sqlite
    ):
        logger.info("Adding optimism_cynicism_scale column to personas table")
        await conn.execute(
            text("ALTER TABLE personas ADD COLUMN optimism_cynicism_scale INTEGER")
        )
        added_columns.append("optimism_cynicism_scale")

    # ==================== VOICE & SPEECH PATTERNS ====================

    if not await check_column_exists(
        conn, "personas", "linguistic_register", is_sqlite
    ):
        logger.info("Adding linguistic_register column to personas table")
        if is_sqlite:
            await conn.execute(
                text(
                    "ALTER TABLE personas ADD COLUMN linguistic_register VARCHAR(50) DEFAULT 'blue_collar'"
                )
            )
            await conn.execute(
                text(
                    "UPDATE personas SET linguistic_register = 'blue_collar' WHERE linguistic_register IS NULL"
                )
            )
        else:
            await conn.execute(
                text(
                    "ALTER TABLE personas ADD COLUMN linguistic_register VARCHAR(50) NOT NULL DEFAULT 'blue_collar'"
                )
            )
        added_columns.append("linguistic_register")

    if not await check_column_exists(conn, "personas", "typing_quirks", is_sqlite):
        logger.info("Adding typing_quirks column to personas table")
        if is_sqlite:
            await conn.execute(
                text("ALTER TABLE personas ADD COLUMN typing_quirks JSON DEFAULT '{}'")
            )
            await conn.execute(
                text(
                    "UPDATE personas SET typing_quirks = '{}' WHERE typing_quirks IS NULL"
                )
            )
        else:
            await conn.execute(
                text(
                    "ALTER TABLE personas ADD COLUMN typing_quirks JSON NOT NULL DEFAULT '{}'"
                )
            )
        added_columns.append("typing_quirks")

    if not await check_column_exists(conn, "personas", "signature_phrases", is_sqlite):
        logger.info("Adding signature_phrases column to personas table")
        if is_sqlite:
            await conn.execute(
                text(
                    "ALTER TABLE personas ADD COLUMN signature_phrases JSON DEFAULT '[]'"
                )
            )
            await conn.execute(
                text(
                    "UPDATE personas SET signature_phrases = '[]' WHERE signature_phrases IS NULL"
                )
            )
        else:
            await conn.execute(
                text(
                    "ALTER TABLE personas ADD COLUMN signature_phrases JSON NOT NULL DEFAULT '[]'"
                )
            )
        added_columns.append("signature_phrases")

    if not await check_column_exists(conn, "personas", "trigger_topics", is_sqlite):
        logger.info("Adding trigger_topics column to personas table")
        if is_sqlite:
            await conn.execute(
                text("ALTER TABLE personas ADD COLUMN trigger_topics JSON DEFAULT '[]'")
            )
            await conn.execute(
                text(
                    "UPDATE personas SET trigger_topics = '[]' WHERE trigger_topics IS NULL"
                )
            )
        else:
            await conn.execute(
                text(
                    "ALTER TABLE personas ADD COLUMN trigger_topics JSON NOT NULL DEFAULT '[]'"
                )
            )
        added_columns.append("trigger_topics")

    # ==================== BACKSTORY & LORE ====================

    if not await check_column_exists(conn, "personas", "day_job", is_sqlite):
        logger.info("Adding day_job column to personas table")
        await conn.execute(text("ALTER TABLE personas ADD COLUMN day_job VARCHAR(200)"))
        added_columns.append("day_job")

    if not await check_column_exists(conn, "personas", "war_story", is_sqlite):
        logger.info("Adding war_story column to personas table")
        await conn.execute(text("ALTER TABLE personas ADD COLUMN war_story TEXT"))
        added_columns.append("war_story")

    if not await check_column_exists(conn, "personas", "vices_hobbies", is_sqlite):
        logger.info("Adding vices_hobbies column to personas table")
        if is_sqlite:
            await conn.execute(
                text("ALTER TABLE personas ADD COLUMN vices_hobbies JSON DEFAULT '[]'")
            )
            await conn.execute(
                text(
                    "UPDATE personas SET vices_hobbies = '[]' WHERE vices_hobbies IS NULL"
                )
            )
        else:
            await conn.execute(
                text(
                    "ALTER TABLE personas ADD COLUMN vices_hobbies JSON NOT NULL DEFAULT '[]'"
                )
            )
        added_columns.append("vices_hobbies")

    # ==================== ANTI-PATTERN ====================

    if not await check_column_exists(conn, "personas", "forbidden_phrases", is_sqlite):
        logger.info("Adding forbidden_phrases column to personas table")
        if is_sqlite:
            await conn.execute(
                text(
                    "ALTER TABLE personas ADD COLUMN forbidden_phrases JSON DEFAULT '[]'"
                )
            )
            await conn.execute(
                text(
                    "UPDATE personas SET forbidden_phrases = '[]' WHERE forbidden_phrases IS NULL"
                )
            )
        else:
            await conn.execute(
                text(
                    "ALTER TABLE personas ADD COLUMN forbidden_phrases JSON NOT NULL DEFAULT '[]'"
                )
            )
        added_columns.append("forbidden_phrases")

    if not await check_column_exists(conn, "personas", "warmth_level", is_sqlite):
        logger.info("Adding warmth_level column to personas table")
        if is_sqlite:
            await conn.execute(
                text(
                    "ALTER TABLE personas ADD COLUMN warmth_level VARCHAR(20) DEFAULT 'warm'"
                )
            )
            await conn.execute(
                text(
                    "UPDATE personas SET warmth_level = 'warm' WHERE warmth_level IS NULL"
                )
            )
        else:
            await conn.execute(
                text(
                    "ALTER TABLE personas ADD COLUMN warmth_level VARCHAR(20) NOT NULL DEFAULT 'warm'"
                )
            )
        added_columns.append("warmth_level")

    if not await check_column_exists(conn, "personas", "patience_level", is_sqlite):
        logger.info("Adding patience_level column to personas table")
        if is_sqlite:
            await conn.execute(
                text(
                    "ALTER TABLE personas ADD COLUMN patience_level VARCHAR(20) DEFAULT 'normal'"
                )
            )
            await conn.execute(
                text(
                    "UPDATE personas SET patience_level = 'normal' WHERE patience_level IS NULL"
                )
            )
        else:
            await conn.execute(
                text(
                    "ALTER TABLE personas ADD COLUMN patience_level VARCHAR(20) NOT NULL DEFAULT 'normal'"
                )
            )
        added_columns.append("patience_level")

    if added_columns:
        logger.info(
            f"✓ Added {len(added_columns)} persona soul columns for human-like responses"
        )

    return added_columns


async def add_persona_physical_appearance_columns(conn, is_sqlite: bool) -> List[str]:
    """
    Add detailed physical appearance fields to personas table for fine-grained
    control over persona appearance generation.

    Added in PR #402: 18 new fields for physical characteristics like height,
    weight, hair color, eye color, body type, etc.

    Args:
        conn: Database connection
        is_sqlite: Whether the database is SQLite

    Returns:
        List of columns that were added
    """
    added_columns = []

    # Check if table exists first
    if not await table_exists(conn, "personas", is_sqlite):
        logger.debug(
            "Personas table does not exist, skipping physical appearance migration"
        )
        return added_columns

    # ==================== BASIC PHYSICAL ATTRIBUTES ====================

    if not await check_column_exists(conn, "personas", "height", is_sqlite):
        logger.info("Adding height column to personas table")
        await conn.execute(text("ALTER TABLE personas ADD COLUMN height VARCHAR(50)"))
        added_columns.append("height")

    if not await check_column_exists(conn, "personas", "weight", is_sqlite):
        logger.info("Adding weight column to personas table")
        await conn.execute(text("ALTER TABLE personas ADD COLUMN weight VARCHAR(50)"))
        added_columns.append("weight")

    if not await check_column_exists(conn, "personas", "hair_color", is_sqlite):
        logger.info("Adding hair_color column to personas table")
        await conn.execute(
            text("ALTER TABLE personas ADD COLUMN hair_color VARCHAR(50)")
        )
        added_columns.append("hair_color")

    if not await check_column_exists(conn, "personas", "hair_style", is_sqlite):
        logger.info("Adding hair_style column to personas table")
        await conn.execute(
            text("ALTER TABLE personas ADD COLUMN hair_style VARCHAR(100)")
        )
        added_columns.append("hair_style")

    if not await check_column_exists(conn, "personas", "eye_color", is_sqlite):
        logger.info("Adding eye_color column to personas table")
        await conn.execute(
            text("ALTER TABLE personas ADD COLUMN eye_color VARCHAR(50)")
        )
        added_columns.append("eye_color")

    if not await check_column_exists(conn, "personas", "skin_tone", is_sqlite):
        logger.info("Adding skin_tone column to personas table")
        await conn.execute(
            text("ALTER TABLE personas ADD COLUMN skin_tone VARCHAR(50)")
        )
        added_columns.append("skin_tone")

    # ==================== BODY MEASUREMENTS ====================

    if not await check_column_exists(conn, "personas", "measurements", is_sqlite):
        logger.info("Adding measurements column to personas table")
        await conn.execute(
            text("ALTER TABLE personas ADD COLUMN measurements VARCHAR(100)")
        )
        added_columns.append("measurements")

    if not await check_column_exists(conn, "personas", "cup_size", is_sqlite):
        logger.info("Adding cup_size column to personas table")
        await conn.execute(text("ALTER TABLE personas ADD COLUMN cup_size VARCHAR(20)"))
        added_columns.append("cup_size")

    if not await check_column_exists(conn, "personas", "muscle_tone", is_sqlite):
        logger.info("Adding muscle_tone column to personas table")
        await conn.execute(
            text("ALTER TABLE personas ADD COLUMN muscle_tone VARCHAR(50)")
        )
        added_columns.append("muscle_tone")

    if not await check_column_exists(conn, "personas", "build_type", is_sqlite):
        logger.info("Adding build_type column to personas table")
        await conn.execute(
            text("ALTER TABLE personas ADD COLUMN build_type VARCHAR(50)")
        )
        added_columns.append("build_type")

    # ==================== IDENTITY & SEXUALITY ====================

    if not await check_column_exists(conn, "personas", "sex", is_sqlite):
        logger.info("Adding sex column to personas table")
        await conn.execute(text("ALTER TABLE personas ADD COLUMN sex VARCHAR(20)"))
        added_columns.append("sex")

    if not await check_column_exists(conn, "personas", "sexual_orientation", is_sqlite):
        logger.info("Adding sexual_orientation column to personas table")
        await conn.execute(
            text("ALTER TABLE personas ADD COLUMN sexual_orientation VARCHAR(50)")
        )
        added_columns.append("sexual_orientation")

    if not await check_column_exists(conn, "personas", "turn_ons", is_sqlite):
        logger.info("Adding turn_ons column to personas table")
        if is_sqlite:
            await conn.execute(
                text("ALTER TABLE personas ADD COLUMN turn_ons JSON DEFAULT '[]'")
            )
            await conn.execute(
                text("UPDATE personas SET turn_ons = '[]' WHERE turn_ons IS NULL")
            )
        else:
            await conn.execute(
                text(
                    "ALTER TABLE personas ADD COLUMN turn_ons JSON NOT NULL DEFAULT '[]'"
                )
            )
        added_columns.append("turn_ons")

    if not await check_column_exists(conn, "personas", "turn_offs", is_sqlite):
        logger.info("Adding turn_offs column to personas table")
        if is_sqlite:
            await conn.execute(
                text("ALTER TABLE personas ADD COLUMN turn_offs JSON DEFAULT '[]'")
            )
            await conn.execute(
                text("UPDATE personas SET turn_offs = '[]' WHERE turn_offs IS NULL")
            )
        else:
            await conn.execute(
                text(
                    "ALTER TABLE personas ADD COLUMN turn_offs JSON NOT NULL DEFAULT '[]'"
                )
            )
        added_columns.append("turn_offs")

    # ==================== DISTINCTIVE FEATURES ====================

    if not await check_column_exists(
        conn, "personas", "distinctive_features", is_sqlite
    ):
        logger.info("Adding distinctive_features column to personas table")
        await conn.execute(
            text("ALTER TABLE personas ADD COLUMN distinctive_features TEXT")
        )
        added_columns.append("distinctive_features")

    if not await check_column_exists(conn, "personas", "age_appearance", is_sqlite):
        logger.info("Adding age_appearance column to personas table")
        await conn.execute(
            text("ALTER TABLE personas ADD COLUMN age_appearance VARCHAR(50)")
        )
        added_columns.append("age_appearance")

    if not await check_column_exists(conn, "personas", "ethnicity", is_sqlite):
        logger.info("Adding ethnicity column to personas table")
        await conn.execute(
            text("ALTER TABLE personas ADD COLUMN ethnicity VARCHAR(100)")
        )
        added_columns.append("ethnicity")

    if not await check_column_exists(conn, "personas", "body_modifications", is_sqlite):
        logger.info("Adding body_modifications column to personas table")
        if is_sqlite:
            await conn.execute(
                text(
                    "ALTER TABLE personas ADD COLUMN body_modifications JSON DEFAULT '[]'"
                )
            )
            await conn.execute(
                text(
                    "UPDATE personas SET body_modifications = '[]' WHERE body_modifications IS NULL"
                )
            )
        else:
            await conn.execute(
                text(
                    "ALTER TABLE personas ADD COLUMN body_modifications JSON NOT NULL DEFAULT '[]'"
                )
            )
        added_columns.append("body_modifications")

    if added_columns:
        logger.info(
            f"✓ Added {len(added_columns)} physical appearance columns for detailed persona control"
        )

    return added_columns


async def add_persona_ai_model_preference_columns(conn, is_sqlite: bool) -> List[str]:
    """
    Add AI model preference fields to personas table for per-persona
    model selection across different content generation types.

    Added in PR #402: 4 new fields for text, image, video, and voice model preferences.

    Args:
        conn: Database connection
        is_sqlite: Whether the database is SQLite

    Returns:
        List of columns that were added
    """
    added_columns = []

    # Check if table exists first
    if not await table_exists(conn, "personas", is_sqlite):
        logger.debug(
            "Personas table does not exist, skipping AI model preference migration"
        )
        return added_columns

    # ==================== AI MODEL PREFERENCES ====================

    if not await check_column_exists(
        conn, "personas", "text_model_preference", is_sqlite
    ):
        logger.info("Adding text_model_preference column to personas table")
        await conn.execute(
            text("ALTER TABLE personas ADD COLUMN text_model_preference VARCHAR(200)")
        )
        added_columns.append("text_model_preference")

    if not await check_column_exists(
        conn, "personas", "image_model_preference", is_sqlite
    ):
        logger.info("Adding image_model_preference column to personas table")
        await conn.execute(
            text("ALTER TABLE personas ADD COLUMN image_model_preference VARCHAR(200)")
        )
        added_columns.append("image_model_preference")

    if not await check_column_exists(
        conn, "personas", "video_model_preference", is_sqlite
    ):
        logger.info("Adding video_model_preference column to personas table")
        await conn.execute(
            text("ALTER TABLE personas ADD COLUMN video_model_preference VARCHAR(200)")
        )
        added_columns.append("video_model_preference")

    if not await check_column_exists(
        conn, "personas", "voice_model_preference", is_sqlite
    ):
        logger.info("Adding voice_model_preference column to personas table")
        await conn.execute(
            text("ALTER TABLE personas ADD COLUMN voice_model_preference VARCHAR(200)")
        )
        added_columns.append("voice_model_preference")

    if added_columns:
        logger.info(
            f"✓ Added {len(added_columns)} AI model preference columns for per-persona model selection"
        )

    return added_columns


async def add_content_triggers_column(conn, is_sqlite: bool) -> List[str]:
    """
    Add content_triggers JSON column to personas table for trigger-based
    model orchestration with LoRA stacking and dynamic prompts.

    Added in PR #428: Enables configurable trigger words that route to
    specific models/LoRAs with per-trigger positive/negative prompts
    and weight preferences.

    Args:
        conn: Database connection
        is_sqlite: Whether the database is SQLite

    Returns:
        List of columns that were added
    """
    added_columns = []

    # Check if table exists first
    if not await table_exists(conn, "personas", is_sqlite):
        logger.debug(
            "Personas table does not exist, skipping content_triggers migration"
        )
        return added_columns

    # Check and add content_triggers
    if not await check_column_exists(conn, "personas", "content_triggers", is_sqlite):
        logger.info("Adding content_triggers column to personas table")
        if is_sqlite:
            # SQLite uses TEXT for JSON storage
            # DEFAULT value is automatically applied to existing rows
            await conn.execute(
                text(
                    "ALTER TABLE personas ADD COLUMN content_triggers TEXT DEFAULT '{}'"
                )
            )
        else:
            # PostgreSQL supports native JSONB type
            # DEFAULT value is automatically applied to existing rows
            await conn.execute(
                text(
                    "ALTER TABLE personas ADD COLUMN content_triggers JSONB DEFAULT '{}'::jsonb"
                )
            )
        added_columns.append("content_triggers")

    if added_columns:
        logger.info(
            "✓ Added content_triggers column for trigger-based model orchestration"
        )

    return added_columns


async def add_persona_negative_prompt_column(conn, is_sqlite: bool) -> List[str]:
    """
    Add default_negative_prompt column to personas table for decoupling
    negative prompts from hardcoded style defaults.

    Added in PR #445: Allows each persona to have their own customized
    negative prompt for image generation, instead of relying on style-based
    defaults. This enables:
    - Anime personas to NOT have "anime" in their negative prompt
    - Custom negative prompts per persona (e.g., "bright colors" for goth)
    - Moving configuration from code to database

    Args:
        conn: Database connection
        is_sqlite: Whether the database is SQLite

    Returns:
        List of columns that were added
    """
    added_columns = []

    # Check if table exists first
    if not await table_exists(conn, "personas", is_sqlite):
        logger.debug(
            "Personas table does not exist, skipping negative prompt migration"
        )
        return added_columns

    # Check and add default_negative_prompt
    if not await check_column_exists(
        conn, "personas", "default_negative_prompt", is_sqlite
    ):
        logger.info("Adding default_negative_prompt column to personas table")
        # Default generic negative prompt to ensure backward compatibility
        default_neg = "ugly, blurry, low quality, distorted, deformed, bad anatomy"

        if is_sqlite:
            await conn.execute(
                text(
                    f"ALTER TABLE personas ADD COLUMN default_negative_prompt TEXT DEFAULT '{default_neg}'"
                )
            )
            # Update existing rows to have the default value
            await conn.execute(
                text(
                    f"UPDATE personas SET default_negative_prompt = '{default_neg}' WHERE default_negative_prompt IS NULL"
                )
            )
        else:
            await conn.execute(
                text(
                    f"ALTER TABLE personas ADD COLUMN default_negative_prompt TEXT NOT NULL DEFAULT '{default_neg}'"
                )
            )
        added_columns.append("default_negative_prompt")

    if added_columns:
        logger.info(
            "✓ Added default_negative_prompt column for persona-specific negative prompts"
        )

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
            "reason": "AUTO_MIGRATE disabled",
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

            # Run physical appearance fields migration (PR #402)
            physical_columns_added = await add_persona_physical_appearance_columns(
                conn, is_sqlite
            )

            if physical_columns_added:
                results["migrations_run"].append("personas_physical_appearance")
                results["columns_added"].extend(physical_columns_added)
                logger.info(
                    f"Added {len(physical_columns_added)} physical appearance column(s): {', '.join(physical_columns_added)}"
                )
            else:
                logger.info("All physical appearance columns are up to date")

            # Run AI model preference fields migration (PR #402)
            ai_model_columns_added = await add_persona_ai_model_preference_columns(
                conn, is_sqlite
            )

            if ai_model_columns_added:
                results["migrations_run"].append("personas_ai_model_preferences")
                results["columns_added"].extend(ai_model_columns_added)
                logger.info(
                    f"Added {len(ai_model_columns_added)} AI model preference column(s): {', '.join(ai_model_columns_added)}"
                )
            else:
                logger.info("All AI model preference columns are up to date")

            # Run content triggers migration (PR #428)
            content_triggers_added = await add_content_triggers_column(conn, is_sqlite)

            if content_triggers_added:
                results["migrations_run"].append("personas_content_triggers")
                results["columns_added"].extend(content_triggers_added)
                logger.info(
                    f"Added content_triggers column for trigger-based model orchestration"
                )
            else:
                logger.info("Content triggers column is up to date")

            # Run negative prompt decoupling migration (PR #445)
            negative_prompt_added = await add_persona_negative_prompt_column(
                conn, is_sqlite
            )

            if negative_prompt_added:
                results["migrations_run"].append("personas_negative_prompt")
                results["columns_added"].extend(negative_prompt_added)
                logger.info(
                    f"Added default_negative_prompt column for persona-specific negative prompts"
                )
            else:
                logger.info("Negative prompt column is up to date")

            # Run ACD scheduling columns migration (PR #433)
            acd_scheduling_added = await add_acd_scheduling_columns(conn, is_sqlite)

            if acd_scheduling_added:
                results["migrations_run"].append("acd_contexts_scheduling")
                results["columns_added"].extend(acd_scheduling_added)
                logger.info(
                    f"Added {len(acd_scheduling_added)} ACD scheduling column(s) for LLM-driven scheduler"
                )
            else:
                logger.info("All ACD scheduling columns are up to date")

            # Create business intelligence tables (PR #433)
            bi_tables_created = await create_business_intelligence_tables(
                conn, is_sqlite
            )

            if bi_tables_created:
                results["migrations_run"].append("business_intelligence_tables")
                results["columns_added"].extend(
                    [f"table:{t}" for t in bi_tables_created]
                )
                logger.info(
                    f"Created {len(bi_tables_created)} business intelligence table(s): {', '.join(bi_tables_created)}"
                )
            else:
                logger.info("All business intelligence tables are up to date")

            # Create installed_models table for AI model metadata
            installed_models_created = await create_installed_models_table(
                conn, is_sqlite
            )

            if installed_models_created:
                results["migrations_run"].append("installed_models_table")
                results["columns_added"].extend(
                    [f"table:{t}" for t in installed_models_created]
                )
                logger.info(
                    f"Created installed_models table for AI model metadata and triggers"
                )
            else:
                logger.info("Installed models table is up to date")

        return results

    except Exception as e:
        logger.error(f"Migration failed: {str(e)}")
        results["success"] = False
        results["error"] = str(e)
        return results


async def add_acd_scheduling_columns(conn, is_sqlite: bool) -> List[str]:
    """
    Add scheduling-related columns to acd_contexts table for LLM-driven
    scheduler integration.

    Added in PR #433: Enables LLM to reason about scheduling, timing optimization,
    dependencies, orchestration groups, and resource planning.

    Args:
        conn: Database connection
        is_sqlite: Whether the database is SQLite

    Returns:
        List of columns that were added
    """
    added_columns = []

    # Check if table exists first
    if not await table_exists(conn, "acd_contexts", is_sqlite):
        logger.debug("acd_contexts table does not exist, skipping scheduling migration")
        return added_columns

    # ==================== SCHEDULE CONFIGURATION ====================

    if not await check_column_exists(conn, "acd_contexts", "schedule_type", is_sqlite):
        logger.info("Adding schedule_type column to acd_contexts table")
        await conn.execute(
            text("ALTER TABLE acd_contexts ADD COLUMN schedule_type VARCHAR(30)")
        )
        added_columns.append("schedule_type")
        # Create index
        try:
            await conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS ix_acd_contexts_schedule_type ON acd_contexts (schedule_type)"
                )
            )
        except Exception as e:
            logger.warning(f"Could not create index on schedule_type: {e}")

    if not await check_column_exists(conn, "acd_contexts", "scheduled_for", is_sqlite):
        logger.info("Adding scheduled_for column to acd_contexts table")
        await conn.execute(
            text("ALTER TABLE acd_contexts ADD COLUMN scheduled_for TIMESTAMP")
        )
        added_columns.append("scheduled_for")
        # Create index
        try:
            await conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS ix_acd_contexts_scheduled_for ON acd_contexts (scheduled_for)"
                )
            )
        except Exception as e:
            logger.warning(f"Could not create index on scheduled_for: {e}")

    if not await check_column_exists(
        conn, "acd_contexts", "schedule_window_start", is_sqlite
    ):
        logger.info("Adding schedule_window_start column to acd_contexts table")
        await conn.execute(
            text("ALTER TABLE acd_contexts ADD COLUMN schedule_window_start TIMESTAMP")
        )
        added_columns.append("schedule_window_start")

    if not await check_column_exists(
        conn, "acd_contexts", "schedule_window_end", is_sqlite
    ):
        logger.info("Adding schedule_window_end column to acd_contexts table")
        await conn.execute(
            text("ALTER TABLE acd_contexts ADD COLUMN schedule_window_end TIMESTAMP")
        )
        added_columns.append("schedule_window_end")

    if not await check_column_exists(
        conn, "acd_contexts", "recurring_pattern", is_sqlite
    ):
        logger.info("Adding recurring_pattern column to acd_contexts table")
        if is_sqlite:
            await conn.execute(
                text("ALTER TABLE acd_contexts ADD COLUMN recurring_pattern JSON")
            )
        else:
            await conn.execute(
                text("ALTER TABLE acd_contexts ADD COLUMN recurring_pattern JSONB")
            )
        added_columns.append("recurring_pattern")

    # ==================== LLM SCHEDULING DECISION CONTEXT ====================

    if not await check_column_exists(
        conn, "acd_contexts", "schedule_decision_source", is_sqlite
    ):
        logger.info("Adding schedule_decision_source column to acd_contexts table")
        await conn.execute(
            text(
                "ALTER TABLE acd_contexts ADD COLUMN schedule_decision_source VARCHAR(30)"
            )
        )
        added_columns.append("schedule_decision_source")

    if not await check_column_exists(
        conn, "acd_contexts", "schedule_optimization_goal", is_sqlite
    ):
        logger.info("Adding schedule_optimization_goal column to acd_contexts table")
        await conn.execute(
            text(
                "ALTER TABLE acd_contexts ADD COLUMN schedule_optimization_goal VARCHAR(30)"
            )
        )
        added_columns.append("schedule_optimization_goal")

    if not await check_column_exists(
        conn, "acd_contexts", "schedule_reasoning", is_sqlite
    ):
        logger.info("Adding schedule_reasoning column to acd_contexts table")
        await conn.execute(
            text("ALTER TABLE acd_contexts ADD COLUMN schedule_reasoning TEXT")
        )
        added_columns.append("schedule_reasoning")

    if not await check_column_exists(
        conn, "acd_contexts", "schedule_constraints", is_sqlite
    ):
        logger.info("Adding schedule_constraints column to acd_contexts table")
        if is_sqlite:
            await conn.execute(
                text("ALTER TABLE acd_contexts ADD COLUMN schedule_constraints JSON")
            )
        else:
            await conn.execute(
                text("ALTER TABLE acd_contexts ADD COLUMN schedule_constraints JSONB")
            )
        added_columns.append("schedule_constraints")

    if not await check_column_exists(
        conn, "acd_contexts", "schedule_preferences", is_sqlite
    ):
        logger.info("Adding schedule_preferences column to acd_contexts table")
        if is_sqlite:
            await conn.execute(
                text("ALTER TABLE acd_contexts ADD COLUMN schedule_preferences JSON")
            )
        else:
            await conn.execute(
                text("ALTER TABLE acd_contexts ADD COLUMN schedule_preferences JSONB")
            )
        added_columns.append("schedule_preferences")

    # ==================== HISTORICAL CONTEXT FOR LLM LEARNING ====================

    if not await check_column_exists(
        conn, "acd_contexts", "historical_performance", is_sqlite
    ):
        logger.info("Adding historical_performance column to acd_contexts table")
        if is_sqlite:
            await conn.execute(
                text("ALTER TABLE acd_contexts ADD COLUMN historical_performance JSON")
            )
        else:
            await conn.execute(
                text("ALTER TABLE acd_contexts ADD COLUMN historical_performance JSONB")
            )
        added_columns.append("historical_performance")

    if not await check_column_exists(
        conn, "acd_contexts", "optimal_timing_learned", is_sqlite
    ):
        logger.info("Adding optimal_timing_learned column to acd_contexts table")
        if is_sqlite:
            await conn.execute(
                text("ALTER TABLE acd_contexts ADD COLUMN optimal_timing_learned JSON")
            )
        else:
            await conn.execute(
                text("ALTER TABLE acd_contexts ADD COLUMN optimal_timing_learned JSONB")
            )
        added_columns.append("optimal_timing_learned")

    if not await check_column_exists(
        conn, "acd_contexts", "audience_activity_pattern", is_sqlite
    ):
        logger.info("Adding audience_activity_pattern column to acd_contexts table")
        if is_sqlite:
            await conn.execute(
                text(
                    "ALTER TABLE acd_contexts ADD COLUMN audience_activity_pattern JSON"
                )
            )
        else:
            await conn.execute(
                text(
                    "ALTER TABLE acd_contexts ADD COLUMN audience_activity_pattern JSONB"
                )
            )
        added_columns.append("audience_activity_pattern")

    # ==================== FEEDBACK LOOP FOR SCHEDULER LEARNING ====================

    if not await check_column_exists(
        conn, "acd_contexts", "schedule_feedback_type", is_sqlite
    ):
        logger.info("Adding schedule_feedback_type column to acd_contexts table")
        await conn.execute(
            text(
                "ALTER TABLE acd_contexts ADD COLUMN schedule_feedback_type VARCHAR(30)"
            )
        )
        added_columns.append("schedule_feedback_type")

    if not await check_column_exists(
        conn, "acd_contexts", "schedule_feedback_data", is_sqlite
    ):
        logger.info("Adding schedule_feedback_data column to acd_contexts table")
        if is_sqlite:
            await conn.execute(
                text("ALTER TABLE acd_contexts ADD COLUMN schedule_feedback_data JSON")
            )
        else:
            await conn.execute(
                text("ALTER TABLE acd_contexts ADD COLUMN schedule_feedback_data JSONB")
            )
        added_columns.append("schedule_feedback_data")

    if not await check_column_exists(
        conn, "acd_contexts", "schedule_effectiveness_score", is_sqlite
    ):
        logger.info("Adding schedule_effectiveness_score column to acd_contexts table")
        await conn.execute(
            text(
                "ALTER TABLE acd_contexts ADD COLUMN schedule_effectiveness_score REAL"
            )
        )
        added_columns.append("schedule_effectiveness_score")

    # ==================== DEPENDENCIES AND ORCHESTRATION ====================

    if not await check_column_exists(
        conn, "acd_contexts", "depends_on_contexts", is_sqlite
    ):
        logger.info("Adding depends_on_contexts column to acd_contexts table")
        if is_sqlite:
            await conn.execute(
                text("ALTER TABLE acd_contexts ADD COLUMN depends_on_contexts JSON")
            )
        else:
            await conn.execute(
                text("ALTER TABLE acd_contexts ADD COLUMN depends_on_contexts JSONB")
            )
        added_columns.append("depends_on_contexts")

    if not await check_column_exists(
        conn, "acd_contexts", "blocks_contexts", is_sqlite
    ):
        logger.info("Adding blocks_contexts column to acd_contexts table")
        if is_sqlite:
            await conn.execute(
                text("ALTER TABLE acd_contexts ADD COLUMN blocks_contexts JSON")
            )
        else:
            await conn.execute(
                text("ALTER TABLE acd_contexts ADD COLUMN blocks_contexts JSONB")
            )
        added_columns.append("blocks_contexts")

    if not await check_column_exists(
        conn, "acd_contexts", "orchestration_group", is_sqlite
    ):
        logger.info("Adding orchestration_group column to acd_contexts table")
        await conn.execute(
            text("ALTER TABLE acd_contexts ADD COLUMN orchestration_group VARCHAR(100)")
        )
        added_columns.append("orchestration_group")
        # Create index
        try:
            await conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS ix_acd_contexts_orchestration_group ON acd_contexts (orchestration_group)"
                )
            )
        except Exception as e:
            logger.warning(f"Could not create index on orchestration_group: {e}")

    if not await check_column_exists(
        conn, "acd_contexts", "execution_order", is_sqlite
    ):
        logger.info("Adding execution_order column to acd_contexts table")
        await conn.execute(
            text("ALTER TABLE acd_contexts ADD COLUMN execution_order INTEGER")
        )
        added_columns.append("execution_order")

    # ==================== RESOURCE PLANNING ====================

    if not await check_column_exists(
        conn, "acd_contexts", "estimated_resources", is_sqlite
    ):
        logger.info("Adding estimated_resources column to acd_contexts table")
        if is_sqlite:
            await conn.execute(
                text("ALTER TABLE acd_contexts ADD COLUMN estimated_resources JSON")
            )
        else:
            await conn.execute(
                text("ALTER TABLE acd_contexts ADD COLUMN estimated_resources JSONB")
            )
        added_columns.append("estimated_resources")

    if not await check_column_exists(
        conn, "acd_contexts", "actual_resources", is_sqlite
    ):
        logger.info("Adding actual_resources column to acd_contexts table")
        if is_sqlite:
            await conn.execute(
                text("ALTER TABLE acd_contexts ADD COLUMN actual_resources JSON")
            )
        else:
            await conn.execute(
                text("ALTER TABLE acd_contexts ADD COLUMN actual_resources JSONB")
            )
        added_columns.append("actual_resources")

    if not await check_column_exists(
        conn, "acd_contexts", "resource_efficiency", is_sqlite
    ):
        logger.info("Adding resource_efficiency column to acd_contexts table")
        await conn.execute(
            text("ALTER TABLE acd_contexts ADD COLUMN resource_efficiency REAL")
        )
        added_columns.append("resource_efficiency")

    if added_columns:
        logger.info(
            f"✓ Added {len(added_columns)} ACD scheduling columns for LLM-driven scheduler"
        )

    return added_columns


async def create_business_intelligence_tables(conn, is_sqlite: bool) -> List[str]:
    """
    Create business intelligence tables for traffic, retention, revenue tracking,
    and content scheduling.

    Added in PR #433: Enables LLM reasoning about business metrics including
    traffic funnels, user churn, revenue trends, and scheduling feedback loops.

    Args:
        conn: Database connection
        is_sqlite: Whether the database is SQLite

    Returns:
        List of tables that were created
    """
    created_tables = []

    # ==================== TRAFFIC METRICS TABLE ====================

    if not await table_exists(conn, "traffic_metrics", is_sqlite):
        logger.info("Creating traffic_metrics table for conversion funnel analysis")
        if is_sqlite:
            await conn.execute(
                text(
                    """
                CREATE TABLE traffic_metrics (
                    id TEXT PRIMARY KEY,
                    user_id TEXT,
                    persona_id TEXT,
                    session_id VARCHAR(100) NOT NULL,
                    source_platform VARCHAR(50) NOT NULL,
                    referrer_url VARCHAR(1000),
                    utm_source VARCHAR(100),
                    utm_medium VARCHAR(100),
                    utm_campaign VARCHAR(100),
                    entry_page VARCHAR(500),
                    exit_page VARCHAR(500),
                    pages_visited JSON,
                    session_duration INTEGER,
                    page_views INTEGER DEFAULT 1,
                    device_type VARCHAR(50),
                    browser VARCHAR(100),
                    os VARCHAR(100),
                    converted_to_dm BOOLEAN DEFAULT 0,
                    converted_to_ppv BOOLEAN DEFAULT 0,
                    converted_to_subscription BOOLEAN DEFAULT 0,
                    revenue_generated DECIMAL(10, 2) DEFAULT 0,
                    messages_sent INTEGER DEFAULT 0,
                    content_viewed INTEGER DEFAULT 0,
                    ppv_offers_seen INTEGER DEFAULT 0,
                    ppv_offers_accepted INTEGER DEFAULT 0,
                    session_start TIMESTAMP NOT NULL,
                    session_end TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                    FOREIGN KEY (persona_id) REFERENCES personas(id) ON DELETE CASCADE
                )
            """
                )
            )
        else:
            await conn.execute(
                text(
                    """
                CREATE TABLE traffic_metrics (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
                    persona_id UUID REFERENCES personas(id) ON DELETE CASCADE,
                    session_id VARCHAR(100) NOT NULL,
                    source_platform VARCHAR(50) NOT NULL,
                    referrer_url VARCHAR(1000),
                    utm_source VARCHAR(100),
                    utm_medium VARCHAR(100),
                    utm_campaign VARCHAR(100),
                    entry_page VARCHAR(500),
                    exit_page VARCHAR(500),
                    pages_visited JSONB,
                    session_duration INTEGER,
                    page_views INTEGER DEFAULT 1,
                    device_type VARCHAR(50),
                    browser VARCHAR(100),
                    os VARCHAR(100),
                    converted_to_dm BOOLEAN DEFAULT FALSE,
                    converted_to_ppv BOOLEAN DEFAULT FALSE,
                    converted_to_subscription BOOLEAN DEFAULT FALSE,
                    revenue_generated DECIMAL(10, 2) DEFAULT 0,
                    messages_sent INTEGER DEFAULT 0,
                    content_viewed INTEGER DEFAULT 0,
                    ppv_offers_seen INTEGER DEFAULT 0,
                    ppv_offers_accepted INTEGER DEFAULT 0,
                    session_start TIMESTAMP WITH TIME ZONE NOT NULL,
                    session_end TIMESTAMP WITH TIME ZONE,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                )
            """
                )
            )
        created_tables.append("traffic_metrics")
        # Create indexes
        try:
            await conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS ix_traffic_metrics_session_id ON traffic_metrics (session_id)"
                )
            )
            await conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS ix_traffic_metrics_user_id ON traffic_metrics (user_id)"
                )
            )
            await conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS ix_traffic_metrics_persona_id ON traffic_metrics (persona_id)"
                )
            )
            await conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS ix_traffic_metrics_source_platform ON traffic_metrics (source_platform)"
                )
            )
            await conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS ix_traffic_metrics_created_at ON traffic_metrics (created_at)"
                )
            )
        except Exception as e:
            logger.warning(f"Could not create indexes on traffic_metrics: {e}")

    # ==================== USER RETENTION TABLE ====================

    if not await table_exists(conn, "user_retention", is_sqlite):
        logger.info("Creating user_retention table for churn prediction")
        if is_sqlite:
            await conn.execute(
                text(
                    """
                CREATE TABLE user_retention (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL UNIQUE,
                    first_interaction TIMESTAMP NOT NULL,
                    last_interaction TIMESTAMP NOT NULL,
                    total_sessions INTEGER DEFAULT 1,
                    avg_session_duration INTEGER,
                    total_time_spent INTEGER DEFAULT 0,
                    active_days_last_7 INTEGER DEFAULT 0,
                    active_days_last_30 INTEGER DEFAULT 0,
                    peak_activity_hour INTEGER,
                    preferred_platform VARCHAR(50),
                    days_since_last_active INTEGER DEFAULT 0,
                    churn_risk_score REAL DEFAULT 0.0,
                    churn_risk_level VARCHAR(20) DEFAULT 'low',
                    churn_predicted BOOLEAN DEFAULT 0,
                    churn_predicted_date TIMESTAMP,
                    lifetime_value DECIMAL(10, 2) DEFAULT 0,
                    ppv_purchases INTEGER DEFAULT 0,
                    total_spent DECIMAL(10, 2) DEFAULT 0,
                    avg_purchase_value DECIMAL(10, 2),
                    total_messages_sent INTEGER DEFAULT 0,
                    total_messages_received INTEGER DEFAULT 0,
                    response_rate REAL,
                    avg_response_time INTEGER,
                    favorite_personas JSON,
                    preferred_content_type VARCHAR(50),
                    re_engagement_attempts INTEGER DEFAULT 0,
                    last_re_engagement_attempt TIMESTAMP,
                    re_engagement_success BOOLEAN,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                )
            """
                )
            )
        else:
            await conn.execute(
                text(
                    """
                CREATE TABLE user_retention (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    user_id UUID NOT NULL UNIQUE REFERENCES users(id) ON DELETE CASCADE,
                    first_interaction TIMESTAMP WITH TIME ZONE NOT NULL,
                    last_interaction TIMESTAMP WITH TIME ZONE NOT NULL,
                    total_sessions INTEGER DEFAULT 1,
                    avg_session_duration INTEGER,
                    total_time_spent INTEGER DEFAULT 0,
                    active_days_last_7 INTEGER DEFAULT 0,
                    active_days_last_30 INTEGER DEFAULT 0,
                    peak_activity_hour INTEGER,
                    preferred_platform VARCHAR(50),
                    days_since_last_active INTEGER DEFAULT 0,
                    churn_risk_score REAL DEFAULT 0.0,
                    churn_risk_level VARCHAR(20) DEFAULT 'low',
                    churn_predicted BOOLEAN DEFAULT FALSE,
                    churn_predicted_date TIMESTAMP WITH TIME ZONE,
                    lifetime_value DECIMAL(10, 2) DEFAULT 0,
                    ppv_purchases INTEGER DEFAULT 0,
                    total_spent DECIMAL(10, 2) DEFAULT 0,
                    avg_purchase_value DECIMAL(10, 2),
                    total_messages_sent INTEGER DEFAULT 0,
                    total_messages_received INTEGER DEFAULT 0,
                    response_rate REAL,
                    avg_response_time INTEGER,
                    favorite_personas JSONB,
                    preferred_content_type VARCHAR(50),
                    re_engagement_attempts INTEGER DEFAULT 0,
                    last_re_engagement_attempt TIMESTAMP WITH TIME ZONE,
                    re_engagement_success BOOLEAN,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                )
            """
                )
            )
        created_tables.append("user_retention")
        # Create indexes
        try:
            await conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS ix_user_retention_user_id ON user_retention (user_id)"
                )
            )
            await conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS ix_user_retention_last_interaction ON user_retention (last_interaction)"
                )
            )
            await conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS ix_user_retention_churn_predicted ON user_retention (churn_predicted)"
                )
            )
            await conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS ix_user_retention_days_since_last_active ON user_retention (days_since_last_active)"
                )
            )
        except Exception as e:
            logger.warning(f"Could not create indexes on user_retention: {e}")

    # ==================== REVENUE INSIGHTS TABLE ====================

    if not await table_exists(conn, "revenue_insights", is_sqlite):
        logger.info("Creating revenue_insights table for revenue trend analysis")
        if is_sqlite:
            await conn.execute(
                text(
                    """
                CREATE TABLE revenue_insights (
                    id TEXT PRIMARY KEY,
                    persona_id TEXT,
                    period_type VARCHAR(20) NOT NULL,
                    period_start TIMESTAMP NOT NULL,
                    period_end TIMESTAMP NOT NULL,
                    ppv_revenue DECIMAL(10, 2) DEFAULT 0,
                    subscription_revenue DECIMAL(10, 2) DEFAULT 0,
                    tips_revenue DECIMAL(10, 2) DEFAULT 0,
                    total_revenue DECIMAL(10, 2) DEFAULT 0,
                    ppv_transactions INTEGER DEFAULT 0,
                    subscription_transactions INTEGER DEFAULT 0,
                    tip_transactions INTEGER DEFAULT 0,
                    total_transactions INTEGER DEFAULT 0,
                    ppv_conversion_rate REAL,
                    avg_ppv_price DECIMAL(10, 2),
                    revenue_per_user DECIMAL(10, 2),
                    revenue_per_message DECIMAL(10, 2),
                    revenue_change_percent REAL,
                    conversion_change_percent REAL,
                    active_users INTEGER DEFAULT 0,
                    new_users INTEGER DEFAULT 0,
                    paying_users INTEGER DEFAULT 0,
                    top_performing_content_types JSON,
                    top_performing_ppv_types JSON,
                    optimization_suggestions JSON,
                    predicted_next_period_revenue DECIMAL(10, 2),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (persona_id) REFERENCES personas(id) ON DELETE CASCADE
                )
            """
                )
            )
        else:
            await conn.execute(
                text(
                    """
                CREATE TABLE revenue_insights (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    persona_id UUID REFERENCES personas(id) ON DELETE CASCADE,
                    period_type VARCHAR(20) NOT NULL,
                    period_start TIMESTAMP WITH TIME ZONE NOT NULL,
                    period_end TIMESTAMP WITH TIME ZONE NOT NULL,
                    ppv_revenue DECIMAL(10, 2) DEFAULT 0,
                    subscription_revenue DECIMAL(10, 2) DEFAULT 0,
                    tips_revenue DECIMAL(10, 2) DEFAULT 0,
                    total_revenue DECIMAL(10, 2) DEFAULT 0,
                    ppv_transactions INTEGER DEFAULT 0,
                    subscription_transactions INTEGER DEFAULT 0,
                    tip_transactions INTEGER DEFAULT 0,
                    total_transactions INTEGER DEFAULT 0,
                    ppv_conversion_rate REAL,
                    avg_ppv_price DECIMAL(10, 2),
                    revenue_per_user DECIMAL(10, 2),
                    revenue_per_message DECIMAL(10, 2),
                    revenue_change_percent REAL,
                    conversion_change_percent REAL,
                    active_users INTEGER DEFAULT 0,
                    new_users INTEGER DEFAULT 0,
                    paying_users INTEGER DEFAULT 0,
                    top_performing_content_types JSONB,
                    top_performing_ppv_types JSONB,
                    optimization_suggestions JSONB,
                    predicted_next_period_revenue DECIMAL(10, 2),
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                )
            """
                )
            )
        created_tables.append("revenue_insights")
        # Create indexes
        try:
            await conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS ix_revenue_insights_persona_id ON revenue_insights (persona_id)"
                )
            )
            await conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS ix_revenue_insights_period_type ON revenue_insights (period_type)"
                )
            )
            await conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS ix_revenue_insights_period_start ON revenue_insights (period_start)"
                )
            )
        except Exception as e:
            logger.warning(f"Could not create indexes on revenue_insights: {e}")

    # ==================== CONTENT SCHEDULES TABLE ====================

    if not await table_exists(conn, "content_schedules", is_sqlite):
        logger.info("Creating content_schedules table for LLM-driven scheduling")
        if is_sqlite:
            await conn.execute(
                text(
                    """
                CREATE TABLE content_schedules (
                    id TEXT PRIMARY KEY,
                    persona_id TEXT NOT NULL,
                    acd_context_id TEXT,
                    schedule_type VARCHAR(50) NOT NULL,
                    scheduled_at TIMESTAMP NOT NULL,
                    content_type VARCHAR(50) NOT NULL,
                    content_data JSON NOT NULL,
                    platform VARCHAR(50) NOT NULL,
                    optimization_goal VARCHAR(50),
                    decision_source VARCHAR(50) NOT NULL,
                    decision_reasoning TEXT,
                    predicted_engagement REAL,
                    predicted_reach INTEGER,
                    predicted_conversions INTEGER,
                    confidence_score REAL,
                    constraints JSON,
                    competing_schedules JSON,
                    status VARCHAR(20) DEFAULT 'scheduled',
                    error_message TEXT,
                    retry_count INTEGER DEFAULT 0,
                    posted_at TIMESTAMP,
                    result_post_id TEXT,
                    result_content_id TEXT,
                    actual_engagement REAL,
                    actual_reach INTEGER,
                    actual_conversions INTEGER,
                    performance_vs_predicted REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (persona_id) REFERENCES personas(id) ON DELETE CASCADE,
                    FOREIGN KEY (acd_context_id) REFERENCES acd_contexts(id) ON DELETE SET NULL
                )
            """
                )
            )
        else:
            await conn.execute(
                text(
                    """
                CREATE TABLE content_schedules (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    persona_id UUID NOT NULL REFERENCES personas(id) ON DELETE CASCADE,
                    acd_context_id UUID REFERENCES acd_contexts(id) ON DELETE SET NULL,
                    schedule_type VARCHAR(50) NOT NULL,
                    scheduled_at TIMESTAMP WITH TIME ZONE NOT NULL,
                    content_type VARCHAR(50) NOT NULL,
                    content_data JSONB NOT NULL,
                    platform VARCHAR(50) NOT NULL,
                    optimization_goal VARCHAR(50),
                    decision_source VARCHAR(50) NOT NULL,
                    decision_reasoning TEXT,
                    predicted_engagement REAL,
                    predicted_reach INTEGER,
                    predicted_conversions INTEGER,
                    confidence_score REAL,
                    constraints JSONB,
                    competing_schedules JSONB,
                    status VARCHAR(20) DEFAULT 'scheduled',
                    error_message TEXT,
                    retry_count INTEGER DEFAULT 0,
                    posted_at TIMESTAMP WITH TIME ZONE,
                    result_post_id UUID,
                    result_content_id UUID,
                    actual_engagement REAL,
                    actual_reach INTEGER,
                    actual_conversions INTEGER,
                    performance_vs_predicted REAL,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                )
            """
                )
            )
        created_tables.append("content_schedules")
        # Create indexes
        try:
            await conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS ix_content_schedules_persona_id ON content_schedules (persona_id)"
                )
            )
            await conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS ix_content_schedules_scheduled_at ON content_schedules (scheduled_at)"
                )
            )
            await conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS ix_content_schedules_status ON content_schedules (status)"
                )
            )
            await conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS ix_content_schedules_schedule_type ON content_schedules (schedule_type)"
                )
            )
            await conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS ix_content_schedules_platform ON content_schedules (platform)"
                )
            )
        except Exception as e:
            logger.warning(f"Could not create indexes on content_schedules: {e}")

    # ==================== SCHEDULING FEEDBACK TABLE ====================

    if not await table_exists(conn, "scheduling_feedback", is_sqlite):
        logger.info("Creating scheduling_feedback table for learning loop")
        if is_sqlite:
            await conn.execute(
                text(
                    """
                CREATE TABLE scheduling_feedback (
                    id TEXT PRIMARY KEY,
                    schedule_id TEXT NOT NULL,
                    persona_id TEXT,
                    schedule_type VARCHAR(50) NOT NULL,
                    scheduled_time TIMESTAMP NOT NULL,
                    actual_post_time TIMESTAMP,
                    time_drift_seconds INTEGER,
                    platform VARCHAR(50) NOT NULL,
                    content_type VARCHAR(50) NOT NULL,
                    predicted_engagement REAL,
                    actual_engagement REAL,
                    engagement_prediction_error REAL,
                    predicted_reach INTEGER,
                    actual_reach INTEGER,
                    reach_prediction_error REAL,
                    prediction_accuracy REAL,
                    system_state_snapshot JSON,
                    audience_state_snapshot JSON,
                    competing_posts JSON,
                    time_of_day INTEGER,
                    day_of_week INTEGER,
                    what_worked JSON,
                    what_failed JSON,
                    recommendations JSON,
                    decision_source VARCHAR(50),
                    decision_source_score REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (schedule_id) REFERENCES content_schedules(id) ON DELETE CASCADE,
                    FOREIGN KEY (persona_id) REFERENCES personas(id) ON DELETE CASCADE
                )
            """
                )
            )
        else:
            await conn.execute(
                text(
                    """
                CREATE TABLE scheduling_feedback (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    schedule_id UUID NOT NULL REFERENCES content_schedules(id) ON DELETE CASCADE,
                    persona_id UUID REFERENCES personas(id) ON DELETE CASCADE,
                    schedule_type VARCHAR(50) NOT NULL,
                    scheduled_time TIMESTAMP WITH TIME ZONE NOT NULL,
                    actual_post_time TIMESTAMP WITH TIME ZONE,
                    time_drift_seconds INTEGER,
                    platform VARCHAR(50) NOT NULL,
                    content_type VARCHAR(50) NOT NULL,
                    predicted_engagement REAL,
                    actual_engagement REAL,
                    engagement_prediction_error REAL,
                    predicted_reach INTEGER,
                    actual_reach INTEGER,
                    reach_prediction_error REAL,
                    prediction_accuracy REAL,
                    system_state_snapshot JSONB,
                    audience_state_snapshot JSONB,
                    competing_posts JSONB,
                    time_of_day INTEGER,
                    day_of_week INTEGER,
                    what_worked JSONB,
                    what_failed JSONB,
                    recommendations JSONB,
                    decision_source VARCHAR(50),
                    decision_source_score REAL,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                )
            """
                )
            )
        created_tables.append("scheduling_feedback")
        # Create indexes
        try:
            await conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS ix_scheduling_feedback_schedule_id ON scheduling_feedback (schedule_id)"
                )
            )
            await conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS ix_scheduling_feedback_persona_id ON scheduling_feedback (persona_id)"
                )
            )
        except Exception as e:
            logger.warning(f"Could not create indexes on scheduling_feedback: {e}")

    if created_tables:
        logger.info(
            f"✓ Created {len(created_tables)} business intelligence tables for LLM reasoning"
        )

    return created_tables


async def create_installed_models_table(conn, is_sqlite: bool) -> List[str]:
    """
    Create installed_models table for tracking AI model metadata including
    CivitAI details, trigger words, and usage statistics.

    This enables:
    - Storing detailed metadata from CivitAI API
    - Managing trigger words for models
    - Matching models to triggers in workflows
    - Tracking model usage across the platform

    Args:
        conn: Database connection
        is_sqlite: Whether the database is SQLite

    Returns:
        List of tables that were created
    """
    created_tables = []

    if not await table_exists(conn, "installed_models", is_sqlite):
        logger.info(
            "Creating installed_models table for AI model metadata and trigger words"
        )
        if is_sqlite:
            await conn.execute(
                text(
                    """
                CREATE TABLE installed_models (
                    id TEXT PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    display_name VARCHAR(255),
                    model_type VARCHAR(50) NOT NULL,
                    source VARCHAR(50) NOT NULL DEFAULT 'local',
                    file_path VARCHAR(1000) NOT NULL UNIQUE,
                    file_name VARCHAR(255) NOT NULL,
                    file_size_mb REAL,
                    file_hash VARCHAR(128),
                    civitai_model_id INTEGER,
                    civitai_version_id INTEGER,
                    civitai_version_name VARCHAR(255),
                    civitai_url VARCHAR(500),
                    huggingface_repo_id VARCHAR(255),
                    huggingface_revision VARCHAR(100),
                    huggingface_filename VARCHAR(255),
                    huggingface_url VARCHAR(500),
                    description TEXT,
                    base_model VARCHAR(100),
                    trigger_words JSON DEFAULT '[]',
                    trained_words JSON DEFAULT '[]',
                    recommended_weight REAL DEFAULT 1.0,
                    recommended_steps INTEGER,
                    recommended_sampler VARCHAR(100),
                    recommended_cfg_scale REAL,
                    default_positive_prompt TEXT,
                    default_negative_prompt TEXT,
                    is_nsfw BOOLEAN DEFAULT 0,
                    is_active BOOLEAN DEFAULT 1,
                    usage_count INTEGER DEFAULT 0,
                    last_used_at TIMESTAMP,
                    extra_metadata JSON DEFAULT '{}',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
                )
            )
        else:
            await conn.execute(
                text(
                    """
                CREATE TABLE installed_models (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    name VARCHAR(255) NOT NULL,
                    display_name VARCHAR(255),
                    model_type VARCHAR(50) NOT NULL,
                    source VARCHAR(50) NOT NULL DEFAULT 'local',
                    file_path VARCHAR(1000) NOT NULL UNIQUE,
                    file_name VARCHAR(255) NOT NULL,
                    file_size_mb REAL,
                    file_hash VARCHAR(128),
                    civitai_model_id INTEGER,
                    civitai_version_id INTEGER,
                    civitai_version_name VARCHAR(255),
                    civitai_url VARCHAR(500),
                    huggingface_repo_id VARCHAR(255),
                    huggingface_revision VARCHAR(100),
                    huggingface_filename VARCHAR(255),
                    huggingface_url VARCHAR(500),
                    description TEXT,
                    base_model VARCHAR(100),
                    trigger_words JSONB DEFAULT '[]',
                    trained_words JSONB DEFAULT '[]',
                    recommended_weight REAL DEFAULT 1.0,
                    recommended_steps INTEGER,
                    recommended_sampler VARCHAR(100),
                    recommended_cfg_scale REAL,
                    default_positive_prompt TEXT,
                    default_negative_prompt TEXT,
                    is_nsfw BOOLEAN DEFAULT FALSE,
                    is_active BOOLEAN DEFAULT TRUE,
                    usage_count INTEGER DEFAULT 0,
                    last_used_at TIMESTAMP WITH TIME ZONE,
                    extra_metadata JSONB DEFAULT '{}',
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                )
            """
                )
            )
        created_tables.append("installed_models")

        # Create indexes for common queries
        try:
            await conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS ix_installed_models_name ON installed_models (name)"
                )
            )
            await conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS ix_installed_models_model_type ON installed_models (model_type)"
                )
            )
            await conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS ix_installed_models_source ON installed_models (source)"
                )
            )
            await conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS ix_installed_models_base_model ON installed_models (base_model)"
                )
            )
            await conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS ix_installed_models_is_active ON installed_models (is_active)"
                )
            )
            await conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS ix_installed_models_civitai_model_id ON installed_models (civitai_model_id)"
                )
            )
            await conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS ix_installed_models_civitai_version_id ON installed_models (civitai_version_id)"
                )
            )
            await conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS ix_installed_models_huggingface_repo_id ON installed_models (huggingface_repo_id)"
                )
            )
        except Exception as e:
            logger.warning(f"Could not create indexes on installed_models: {e}")

    if created_tables:
        logger.info(
            f"✓ Created installed_models table for AI model metadata and triggers"
        )

    return created_tables
