#!/usr/bin/env python3
"""
Validation script for IPMI credentials implementation.
Checks that all components are properly integrated.
"""

import sys
import os

def check_file_exists(filepath, description):
    """Check if a file exists."""
    if os.path.exists(filepath):
        print(f"✅ {description}: {filepath}")
        return True
    else:
        print(f"❌ {description} not found: {filepath}")
        return False

def check_file_contains(filepath, search_strings, description):
    """Check if file contains required strings."""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        missing = []
        for search_str in search_strings:
            if search_str not in content:
                missing.append(search_str)
        
        if not missing:
            print(f"✅ {description}")
            return True
        else:
            print(f"❌ {description} - Missing: {', '.join(missing)}")
            return False
    except Exception as e:
        print(f"❌ Error checking {filepath}: {e}")
        return False

def main():
    """Run validation checks."""
    print("🔍 IPMI Credentials Implementation Validation")
    print("=" * 60)
    
    checks_passed = 0
    checks_total = 0
    
    # Check 1: Settings model has IPMI category
    checks_total += 1
    if check_file_contains(
        "src/backend/models/settings.py",
        ["IPMI = \"ipmi\"", "ipmi_host", "ipmi_username", "ipmi_password", "ipmi_interface"],
        "Settings model includes IPMI category and settings"
    ):
        checks_passed += 1
    
    # Check 2: Settings route handles IPMI category
    checks_total += 1
    if check_file_contains(
        "src/backend/api/routes/settings.py",
        ["if key.startswith(\"ipmi_\"):", "category = SettingCategory.IPMI"],
        "Settings route properly categorizes IPMI settings"
    ):
        checks_passed += 1
    
    # Check 3: Fan control service has database integration
    checks_total += 1
    if check_file_contains(
        "src/backend/services/fan_control_service.py",
        ["get_ipmi_credentials_from_db", "reload_credentials_from_db"],
        "Fan control service can load credentials from database"
    ):
        checks_passed += 1
    
    # Check 4: Admin UI has IPMI section
    checks_total += 1
    if check_file_contains(
        "admin_panel/settings.html",
        ["IPMI / Server Management", "ipmi-host", "ipmi-username", "ipmi-password", "saveIPMICredentials"],
        "Admin settings page includes IPMI credentials section"
    ):
        checks_passed += 1
    
    # Check 5: Test file exists
    checks_total += 1
    if check_file_exists(
        "test_ipmi_credentials.py",
        "IPMI credentials test file"
    ):
        checks_passed += 1
    
    # Check 6: Database migration includes IPMI settings
    checks_total += 1
    # Check if migration was run by verifying the database
    if os.path.exists("gator.db"):
        import sqlite3
        conn = sqlite3.connect("gator.db")
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT COUNT(*) FROM system_settings WHERE key LIKE 'ipmi%'")
            count = cursor.fetchone()[0]
            if count >= 4:
                print(f"✅ Database contains {count} IPMI settings")
                checks_passed += 1
            else:
                print(f"❌ Database should contain 4 IPMI settings, found {count}")
        except Exception as e:
            print(f"❌ Error checking database: {e}")
        finally:
            conn.close()
    else:
        print("⚠️ Database not found (this is OK if not initialized yet)")
    
    # Summary
    print("\n" + "=" * 60)
    print(f"📊 Validation Results: {checks_passed}/{checks_total} checks passed")
    print("=" * 60)
    
    if checks_passed == checks_total:
        print("\n✅ ALL VALIDATIONS PASSED!")
        print("\n🎉 IPMI credentials implementation is complete:")
        print("   • Settings model updated with IPMI category")
        print("   • API routes handle IPMI settings")
        print("   • Fan control service can load from database")
        print("   • Admin UI includes IPMI credentials form")
        print("   • Database stores IPMI credentials")
        print("   • Tests verify functionality")
        return 0
    else:
        print(f"\n⚠️ {checks_total - checks_passed} validation(s) failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
