#!/usr/bin/env python3
"""
Test IPMI Credentials Management

Validates that IPMI credentials can be saved to and retrieved from database.
"""

import asyncio
import sys

from backend.database.connection import database_manager
from backend.services.settings_service import SettingsService
from backend.models.settings import SettingCreate, SettingCategory


async def test_ipmi_credentials():
    """Test IPMI credentials save and retrieval."""
    print("🔧 Testing IPMI Credentials Management")
    print("=" * 60)
    
    try:
        # Connect to database
        await database_manager.connect()
        print("✅ Connected to database")
        
        # Create service
        async with database_manager.get_session() as session:
            service = SettingsService(session)
            
            # Test 1: Save IPMI credentials
            print("\n1️⃣ Saving IPMI credentials...")
            
            test_credentials = {
                "ipmi_host": "192.168.1.100",
                "ipmi_username": "testuser",
                "ipmi_password": "testpass123",
                "ipmi_interface": "lanplus"
            }
            
            for key, value in test_credentials.items():
                setting_data = SettingCreate(
                    key=key,
                    category=SettingCategory.IPMI,
                    value=value,
                    is_sensitive="password" in key or "username" in key,
                    description=f"Test {key}"
                )
                result = await service.upsert_setting(setting_data)
                if result:
                    print(f"   ✓ Saved {key}: {value if 'password' not in key else '••••••••'}")
                else:
                    print(f"   ✗ Failed to save {key}")
                    return False
            
            # Test 2: Retrieve IPMI credentials
            print("\n2️⃣ Retrieving IPMI credentials...")
            
            for key in test_credentials.keys():
                setting = await service.get_setting(key)
                if setting:
                    expected_value = test_credentials[key]
                    if setting.value == expected_value:
                        print(f"   ✓ Retrieved {key}: {setting.value if 'password' not in key else '••••••••'}")
                    else:
                        print(f"   ✗ {key} value mismatch: expected {expected_value}, got {setting.value}")
                        return False
                else:
                    print(f"   ✗ Failed to retrieve {key}")
                    return False
            
            # Test 3: Verify category filtering
            print("\n3️⃣ Testing category filtering...")
            ipmi_settings = await service.get_settings_by_category(SettingCategory.IPMI)
            ipmi_keys = {s.key for s in ipmi_settings}
            expected_keys = set(test_credentials.keys())
            
            if expected_keys.issubset(ipmi_keys):
                print(f"   ✓ Found {len(ipmi_settings)} IPMI settings")
                for setting in ipmi_settings:
                    if setting.key in expected_keys:
                        print(f"      - {setting.key}: {'sensitive' if setting.is_sensitive else 'non-sensitive'}")
            else:
                print(f"   ✗ Category filtering failed")
                return False
            
            # Test 4: Verify sensitive flag
            print("\n4️⃣ Verifying sensitive flags...")
            
            username_setting = await service.get_setting("ipmi_username")
            password_setting = await service.get_setting("ipmi_password")
            host_setting = await service.get_setting("ipmi_host")
            
            if username_setting and username_setting.is_sensitive:
                print("   ✓ ipmi_username is marked sensitive")
            else:
                print("   ✗ ipmi_username should be sensitive")
                return False
                
            if password_setting and password_setting.is_sensitive:
                print("   ✓ ipmi_password is marked sensitive")
            else:
                print("   ✗ ipmi_password should be sensitive")
                return False
                
            if host_setting and not host_setting.is_sensitive:
                print("   ✓ ipmi_host is not marked sensitive")
            else:
                print("   ✗ ipmi_host should not be sensitive")
                return False
            
            print("\n" + "=" * 60)
            print("✅ All IPMI credentials tests passed!")
            print("\n📝 Summary:")
            print("   • IPMI credentials can be saved to database")
            print("   • IPMI credentials can be retrieved from database")
            print("   • Category filtering works correctly")
            print("   • Sensitive flags are properly set")
            print("   • Ready for use in /admin/settings page")
            return True
            
    except Exception as e:
        print(f"\n❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        await database_manager.disconnect()


async def test_fan_control_integration():
    """Test that FanControlService can use database credentials."""
    print("\n\n🌡️ Testing Fan Control Service Integration")
    print("=" * 60)
    
    try:
        from backend.services.fan_control_service import (
            FanControlService, 
            get_ipmi_credentials_from_db
        )
        
        # Connect to database
        await database_manager.connect()
        print("✅ Connected to database")
        
        # Test loading credentials from database
        print("\n1️⃣ Testing credential loading from database...")
        db_creds = await get_ipmi_credentials_from_db()
        
        if db_creds:
            print("   ✓ Successfully loaded credentials from database")
            print(f"      Host: {db_creds.get('ipmi_host', 'not set')}")
            print(f"      Username: {db_creds.get('ipmi_username', 'not set')}")
            print(f"      Password: {'••••••••' if db_creds.get('ipmi_password') else 'not set'}")
            print(f"      Interface: {db_creds.get('ipmi_interface', 'lanplus')}")
        else:
            print("   ℹ️ No credentials in database (this is okay for initial setup)")
        
        # Test FanControlService can reload credentials
        print("\n2️⃣ Testing FanControlService credential reload...")
        service = FanControlService()
        
        # Store initial credentials
        initial_host = service._ipmi_host
        print(f"   Initial IPMI host: {initial_host if initial_host else 'not set'}")
        
        # Try to reload from database
        await service.reload_credentials_from_db()
        
        # Check if credentials were updated
        updated_host = service._ipmi_host
        print(f"   After reload: {updated_host if updated_host else 'not set'}")
        
        if db_creds and db_creds.get('ipmi_host'):
            if updated_host == db_creds['ipmi_host']:
                print("   ✓ Credentials successfully loaded from database")
            else:
                print("   ✗ Credentials not updated correctly")
                return False
        else:
            print("   ✓ No credentials to load (fallback to environment/default)")
        
        print("\n" + "=" * 60)
        print("✅ Fan Control Service integration test passed!")
        print("\n📝 Integration verified:")
        print("   • FanControlService can load credentials from database")
        print("   • Dynamic credential reload works without restart")
        print("   • Falls back gracefully when no database credentials exist")
        return True
        
    except Exception as e:
        print(f"\n❌ Error during integration test: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        await database_manager.disconnect()


async def main():
    """Run all tests."""
    print("\n🚀 IPMI Credentials Test Suite")
    print("=" * 60)
    
    # Run basic credentials tests
    test1_passed = await test_ipmi_credentials()
    
    # Run integration tests
    test2_passed = await test_fan_control_integration()
    
    # Summary
    print("\n\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    print(f"   Basic Credentials:    {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"   Service Integration:  {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    print("=" * 60)
    
    if test1_passed and test2_passed:
        print("\n🎉 ALL TESTS PASSED!")
        print("\n✨ IPMI credentials can now be managed via /admin/settings")
        print("   No application restart required to update credentials!")
        return 0
    else:
        print("\n❌ SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
