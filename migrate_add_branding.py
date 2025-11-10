#!/usr/bin/env python3
"""
Database Migration: Add Branding Table

Adds the branding table to store site customization in the database.
This replaces environment variable configuration for branding.
"""

import asyncio
import sys
from sqlalchemy import text

from backend.database.connection import database_manager
from backend.models.branding import BrandingModel


async def migrate():
    """Run the migration to add branding table."""
    print("🔄 Starting branding table migration...")
    
    try:
        # Connect to database
        await database_manager.connect()
        print("✅ Connected to database")
        
        # Create tables
        from backend.database.connection import Base, engine
        async with engine.begin() as conn:
            # Create branding table
            await conn.run_sync(Base.metadata.create_all)
            print("✅ Branding table created")
            
            # Check if default branding exists
            result = await conn.execute(text("SELECT COUNT(*) FROM branding"))
            count = result.scalar()
            
            if count == 0:
                # Insert default branding (SQLite compatible)
                import uuid
                from datetime import datetime, timezone
                
                default_id = str(uuid.uuid4())
                now = datetime.now(timezone.utc).isoformat()
                
                await conn.execute(text("""
                    INSERT INTO branding (
                        id,
                        site_name,
                        site_icon,
                        instance_name,
                        site_tagline,
                        primary_color,
                        accent_color,
                        is_active,
                        created_at,
                        updated_at
                    ) VALUES (
                        :id,
                        :site_name,
                        :site_icon,
                        :instance_name,
                        :site_tagline,
                        :primary_color,
                        :accent_color,
                        :is_active,
                        :created_at,
                        :updated_at
                    )
                """), {
                    "id": default_id,
                    "site_name": "AI Content Platform",
                    "site_icon": "🤖",
                    "instance_name": "My AI Platform",
                    "site_tagline": "AI-Powered Content Generation",
                    "primary_color": "#667eea",
                    "accent_color": "#10b981",
                    "is_active": True,
                    "created_at": now,
                    "updated_at": now,
                })
                print("✅ Default branding configuration created")
            else:
                print("ℹ️  Branding configuration already exists")
        
        print("\n✅ Migration completed successfully!")
        print("\n📝 Next steps:")
        print("   1. Update branding via /admin/settings or API")
        print("   2. Remove branding variables from .env file")
        print("   3. Restart application to use database branding")
        
    except Exception as e:
        print(f"\n❌ Migration failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        await database_manager.disconnect()


if __name__ == "__main__":
    asyncio.run(migrate())
