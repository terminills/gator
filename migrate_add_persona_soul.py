"""
Database Migration: Add Persona Soul Fields

This migration adds the persona soul fields to capture the full character
of each persona for human-like response generation:

1. Origin & Demographics (The "Roots")
   - hometown, current_location, generation_age, education_level

2. Psychological Profile (The "Engine")
   - mbti_type, enneagram_type, political_alignment, risk_tolerance, optimism_cynicism_scale

3. Voice & Speech Patterns (The "Interface")
   - linguistic_register, typing_quirks, signature_phrases, trigger_topics

4. Backstory & Lore (The "Context")
   - day_job, war_story, vices_hobbies

5. Anti-Pattern (What they are NOT)
   - forbidden_phrases, warmth_level, patience_level
"""

import asyncio
from sqlalchemy import text
from backend.database.connection import async_engine, get_async_session


async def migrate():
    """Add persona soul fields to the personas table."""
    
    # SQL statements to add new columns
    alter_statements = [
        # Origin & Demographics
        "ALTER TABLE personas ADD COLUMN IF NOT EXISTS hometown VARCHAR(200)",
        "ALTER TABLE personas ADD COLUMN IF NOT EXISTS current_location VARCHAR(200)",
        "ALTER TABLE personas ADD COLUMN IF NOT EXISTS generation_age VARCHAR(100)",
        "ALTER TABLE personas ADD COLUMN IF NOT EXISTS education_level VARCHAR(200)",
        
        # Psychological Profile
        "ALTER TABLE personas ADD COLUMN IF NOT EXISTS mbti_type VARCHAR(50)",
        "ALTER TABLE personas ADD COLUMN IF NOT EXISTS enneagram_type VARCHAR(50)",
        "ALTER TABLE personas ADD COLUMN IF NOT EXISTS political_alignment VARCHAR(100)",
        "ALTER TABLE personas ADD COLUMN IF NOT EXISTS risk_tolerance VARCHAR(100)",
        "ALTER TABLE personas ADD COLUMN IF NOT EXISTS optimism_cynicism_scale INTEGER",
        
        # Voice & Speech Patterns
        "ALTER TABLE personas ADD COLUMN IF NOT EXISTS linguistic_register VARCHAR(50) DEFAULT 'blue_collar' NOT NULL",
        "ALTER TABLE personas ADD COLUMN IF NOT EXISTS typing_quirks JSON DEFAULT '{}' NOT NULL",
        "ALTER TABLE personas ADD COLUMN IF NOT EXISTS signature_phrases JSON DEFAULT '[]' NOT NULL",
        "ALTER TABLE personas ADD COLUMN IF NOT EXISTS trigger_topics JSON DEFAULT '[]' NOT NULL",
        
        # Backstory & Lore
        "ALTER TABLE personas ADD COLUMN IF NOT EXISTS day_job VARCHAR(200)",
        "ALTER TABLE personas ADD COLUMN IF NOT EXISTS war_story TEXT",
        "ALTER TABLE personas ADD COLUMN IF NOT EXISTS vices_hobbies JSON DEFAULT '[]' NOT NULL",
        
        # Anti-Pattern
        "ALTER TABLE personas ADD COLUMN IF NOT EXISTS forbidden_phrases JSON DEFAULT '[]' NOT NULL",
        "ALTER TABLE personas ADD COLUMN IF NOT EXISTS warmth_level VARCHAR(20) DEFAULT 'warm' NOT NULL",
        "ALTER TABLE personas ADD COLUMN IF NOT EXISTS patience_level VARCHAR(20) DEFAULT 'normal' NOT NULL",
    ]
    
    async with async_engine.begin() as conn:
        for statement in alter_statements:
            try:
                await conn.execute(text(statement))
                print(f"✓ Executed: {statement[:60]}...")
            except Exception as e:
                # Column might already exist, that's ok
                if "duplicate column" in str(e).lower() or "already exists" in str(e).lower():
                    print(f"  Skipped (already exists): {statement[:60]}...")
                else:
                    print(f"✗ Error: {e}")
    
    print("\n✅ Migration complete: Persona soul fields added")


async def rollback():
    """Remove persona soul fields from the personas table."""
    
    drop_statements = [
        # Origin & Demographics
        "ALTER TABLE personas DROP COLUMN IF EXISTS hometown",
        "ALTER TABLE personas DROP COLUMN IF EXISTS current_location",
        "ALTER TABLE personas DROP COLUMN IF EXISTS generation_age",
        "ALTER TABLE personas DROP COLUMN IF EXISTS education_level",
        
        # Psychological Profile
        "ALTER TABLE personas DROP COLUMN IF EXISTS mbti_type",
        "ALTER TABLE personas DROP COLUMN IF EXISTS enneagram_type",
        "ALTER TABLE personas DROP COLUMN IF EXISTS political_alignment",
        "ALTER TABLE personas DROP COLUMN IF EXISTS risk_tolerance",
        "ALTER TABLE personas DROP COLUMN IF EXISTS optimism_cynicism_scale",
        
        # Voice & Speech Patterns
        "ALTER TABLE personas DROP COLUMN IF EXISTS linguistic_register",
        "ALTER TABLE personas DROP COLUMN IF EXISTS typing_quirks",
        "ALTER TABLE personas DROP COLUMN IF EXISTS signature_phrases",
        "ALTER TABLE personas DROP COLUMN IF EXISTS trigger_topics",
        
        # Backstory & Lore
        "ALTER TABLE personas DROP COLUMN IF EXISTS day_job",
        "ALTER TABLE personas DROP COLUMN IF EXISTS war_story",
        "ALTER TABLE personas DROP COLUMN IF EXISTS vices_hobbies",
        
        # Anti-Pattern
        "ALTER TABLE personas DROP COLUMN IF EXISTS forbidden_phrases",
        "ALTER TABLE personas DROP COLUMN IF EXISTS warmth_level",
        "ALTER TABLE personas DROP COLUMN IF EXISTS patience_level",
    ]
    
    async with async_engine.begin() as conn:
        for statement in drop_statements:
            try:
                await conn.execute(text(statement))
                print(f"✓ Executed: {statement[:60]}...")
            except Exception as e:
                print(f"✗ Error: {e}")
    
    print("\n✅ Rollback complete: Persona soul fields removed")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "rollback":
        asyncio.run(rollback())
    else:
        asyncio.run(migrate())
