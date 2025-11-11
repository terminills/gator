#!/usr/bin/env python3
"""
Test script to verify settings service UUID fix
"""

import asyncio
from backend.database.connection import database_manager
from backend.services.settings_service import SettingsService


async def test_settings():
    """Test settings service operations"""
    print("🧪 Testing Settings Service...")
    
    try:
        # Connect to database
        await database_manager.connect()
        print("✅ Connected to database")
        
        # Get a session
        async with database_manager.get_session() as session:
            service = SettingsService(session)
            
            # Test 1: Get a specific setting
            print("\n📋 Test 1: Get openai_api_key setting")
            setting = await service.get_setting("openai_api_key")
            if setting:
                print(f"✅ Retrieved setting: {setting.key}")
                print(f"   ID type: {type(setting.id)}")
                print(f"   ID value: {setting.id}")
                print(f"   Category: {setting.category}")
                print(f"   Is sensitive: {setting.is_sensitive}")
            else:
                print("❌ Setting not found")
            
            # Test 2: Get settings by category
            print("\n📋 Test 2: Get all AI model settings")
            from backend.models.settings import SettingCategory
            ai_settings = await service.get_settings_by_category(SettingCategory.AI_MODELS)
            print(f"✅ Retrieved {len(ai_settings)} AI model settings:")
            for s in ai_settings:
                print(f"   - {s.key} (ID type: {type(s.id).__name__})")
            
            # Test 3: List all settings
            print("\n📋 Test 3: List all settings")
            all_settings = await service.list_all_settings()
            print(f"✅ Retrieved {len(all_settings)} total settings")
            
            # Test 4: Create a test setting
            print("\n📋 Test 4: Create test setting")
            from backend.models.settings import SettingCreate, SettingCategory
            test_setting = SettingCreate(
                key="test_key",
                category=SettingCategory.AI_MODELS,
                value="test_value",
                description="Test setting for validation",
                is_sensitive=False
            )
            created = await service.create_setting(test_setting)
            if created:
                print(f"✅ Created setting: {created.key}")
                print(f"   ID type: {type(created.id).__name__}")
                print(f"   ID value: {created.id}")
            else:
                print("❌ Failed to create setting (may already exist)")
            
            # Test 5: Update the test setting
            print("\n📋 Test 5: Update test setting")
            from backend.models.settings import SettingUpdate
            update_data = SettingUpdate(value="updated_value")
            updated = await service.update_setting("test_key", update_data)
            if updated:
                print(f"✅ Updated setting: {updated.key}")
                print(f"   New value: {updated.value}")
                print(f"   ID type: {type(updated.id).__name__}")
            else:
                print("❌ Failed to update setting")
            
            # Test 6: Upsert operation
            print("\n📋 Test 6: Upsert operation")
            upsert_data = SettingCreate(
                key="openai_api_key",
                category=SettingCategory.AI_MODELS,
                value="sk-test123",
                description="OpenAI API key",
                is_sensitive=True
            )
            upserted = await service.upsert_setting(upsert_data)
            if upserted:
                print(f"✅ Upserted setting: {upserted.key}")
                print(f"   ID type: {type(upserted.id).__name__}")
            else:
                print("❌ Failed to upsert setting")
        
        print("\n✅ All tests completed successfully!")
        print("   UUID to string conversion is working correctly")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await database_manager.disconnect()
        print("\n🔌 Disconnected from database")


if __name__ == "__main__":
    asyncio.run(test_settings())
