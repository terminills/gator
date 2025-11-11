#!/usr/bin/env python3
"""
Database Migration: Add System Settings Table

Moves all configuration from .env to database for dynamic management.
"""

import asyncio
import sys
from sqlalchemy import text

from backend.database.connection import database_manager
from backend.models.settings import SystemSettingModel, DEFAULT_SETTINGS


async def migrate():
    """Run the migration to add system settings table."""
    print("🔄 Starting system settings table migration...")
    print("📝 This moves configuration from .env to database")
    
    try:
        # Connect to database
        await database_manager.connect()
        print("✅ Connected to database")
        
        # Create tables
        from backend.database.connection import Base
        async with database_manager.engine.begin() as conn:
            # Create system_settings table
            await conn.run_sync(Base.metadata.create_all)
            print("✅ System settings table created")
            
            # Check if settings exist
            result = await conn.execute(text("SELECT COUNT(*) FROM system_settings"))
            count = result.scalar()
            
            if count == 0:
                # Insert default settings
                print(f"📥 Inserting {len(DEFAULT_SETTINGS)} default settings...")
                
                for key, setting_data in DEFAULT_SETTINGS.items():
                    import uuid
                    from datetime import datetime, timezone
                    import json
                    
                    setting_id = str(uuid.uuid4())
                    now = datetime.now(timezone.utc).isoformat()
                    value_json = json.dumps(setting_data["value"])
                    
                    await conn.execute(text("""
                        INSERT INTO system_settings (
                            id, key, category, value, description, 
                            is_sensitive, is_active, created_at, updated_at
                        ) VALUES (
                            :id, :key, :category, :value, :description,
                            :is_sensitive, :is_active, :created_at, :updated_at
                        )
                    """), {
                        "id": setting_id,
                        "key": key,
                        "category": setting_data["category"],
                        "value": value_json,
                        "description": setting_data.get("description"),
                        "is_sensitive": setting_data.get("is_sensitive", False),
                        "is_active": True,
                        "created_at": now,
                        "updated_at": now,
                    })
                
                print(f"✅ Inserted {len(DEFAULT_SETTINGS)} default settings")
            else:
                print(f"ℹ️  Found {count} existing settings")
        
        print("\n✅ Migration completed successfully!")
        print("\n📝 Next steps:")
        print("   1. Configure settings via /admin/settings or API")
        print("   2. Remove old variables from .env file")
        print("   3. Settings update without restarting app")
        print("\n💡 Tip: API keys and secrets are encrypted in database")
        
    except Exception as e:
        print(f"\n❌ Migration failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        await database_manager.disconnect()


if __name__ == "__main__":
    asyncio.run(migrate())
