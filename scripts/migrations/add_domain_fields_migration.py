"""
Database migration to add AI_DOMAIN and AI_SUBDOMAIN fields to acd_contexts table.

Run this script to update the database schema with domain classification fields.
"""

import asyncio
from sqlalchemy import text

from backend.database.connection import database_manager


async def migrate_add_domain_fields():
    """Add ai_domain and ai_subdomain columns to acd_contexts table."""
    
    await database_manager.connect()
    
    try:
        async with database_manager.session_factory() as session:
            # Check if columns already exist
            check_query = """
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'acd_contexts' 
            AND column_name IN ('ai_domain', 'ai_subdomain');
            """
            
            try:
                result = await session.execute(text(check_query))
                existing_columns = [row[0] for row in result.fetchall()]
            except Exception:
                # SQLite doesn't support information_schema, try pragma
                try:
                    result = await session.execute(text("PRAGMA table_info(acd_contexts);"))
                    columns = result.fetchall()
                    existing_columns = [col[1] for col in columns if col[1] in ['ai_domain', 'ai_subdomain']]
                except Exception as e:
                    print(f"Could not check existing columns: {e}")
                    print("Proceeding with migration...")
                    existing_columns = []
            
            # Add ai_domain column if it doesn't exist
            if 'ai_domain' not in existing_columns:
                print("Adding ai_domain column...")
                try:
                    await session.execute(text("""
                        ALTER TABLE acd_contexts 
                        ADD COLUMN ai_domain VARCHAR(50);
                    """))
                    print("✅ ai_domain column added")
                except Exception as e:
                    print(f"❌ Error adding ai_domain: {e}")
            else:
                print("✓ ai_domain column already exists")
            
            # Add ai_subdomain column if it doesn't exist
            if 'ai_subdomain' not in existing_columns:
                print("Adding ai_subdomain column...")
                try:
                    await session.execute(text("""
                        ALTER TABLE acd_contexts 
                        ADD COLUMN ai_subdomain VARCHAR(50);
                    """))
                    print("✅ ai_subdomain column added")
                except Exception as e:
                    print(f"❌ Error adding ai_subdomain: {e}")
            else:
                print("✓ ai_subdomain column already exists")
            
            # Create indexes for the new columns
            if 'ai_domain' not in existing_columns:
                print("Creating index on ai_domain...")
                try:
                    await session.execute(text("""
                        CREATE INDEX IF NOT EXISTS ix_acd_contexts_ai_domain 
                        ON acd_contexts (ai_domain);
                    """))
                    print("✅ Index on ai_domain created")
                except Exception as e:
                    print(f"Warning: Could not create index on ai_domain: {e}")
            
            if 'ai_subdomain' not in existing_columns:
                print("Creating index on ai_subdomain...")
                try:
                    await session.execute(text("""
                        CREATE INDEX IF NOT EXISTS ix_acd_contexts_ai_subdomain 
                        ON acd_contexts (ai_subdomain);
                    """))
                    print("✅ Index on ai_subdomain created")
                except Exception as e:
                    print(f"Warning: Could not create index on ai_subdomain: {e}")
            
            await session.commit()
            print("\n✅ Migration completed successfully!")
            print("\nNew fields added:")
            print("  - ai_domain: Top-level domain classification (cortical region)")
            print("  - ai_subdomain: Fine-grained subdomain specialization")
            
    finally:
        await database_manager.disconnect()


if __name__ == "__main__":
    print("=" * 70)
    print("DATABASE MIGRATION: Add AI_DOMAIN and AI_SUBDOMAIN fields")
    print("=" * 70)
    print()
    asyncio.run(migrate_add_domain_fields())
