#!/usr/bin/env python3
"""
Validation script for enhancement implementation.

Verifies that all enhancements are working correctly.
"""

import sys
import subprocess
from pathlib import Path


def run_command(cmd, description):
    """Run a command and report results."""
    print(f"\n{'='*60}")
    print(f"Testing: {description}")
    print(f"Command: {cmd}")
    print(f"{'='*60}")
    
    result = subprocess.run(
        cmd,
        shell=True,
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent
    )
    
    print(result.stdout)
    if result.stderr:
        print(f"STDERR: {result.stderr}")
    
    return result.returncode == 0


def main():
    """Run all validation checks."""
    print("🔍 Enhancement Implementation Validation")
    print("=" * 60)
    
    checks = [
        # Check Black formatting
        (
            "python -m black --check src/",
            "Black formatting - Source files"
        ),
        (
            "python -m black --check tests/",
            "Black formatting - Test files"
        ),
        
        # Run plugin tests
        (
            "python -m pytest tests/unit/test_plugin_system.py -v",
            "Plugin system tests (including JSON schema validation)"
        ),
        
        # Verify demo works
        (
            "python demo.py",
            "Demo script execution"
        ),
    ]
    
    results = []
    for cmd, description in checks:
        success = run_command(cmd, description)
        results.append((description, success))
    
    # Print summary
    print("\n" + "="*60)
    print("📊 VALIDATION SUMMARY")
    print("="*60)
    
    for description, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {description}")
    
    all_passed = all(success for _, success in results)
    
    print("\n" + "="*60)
    if all_passed:
        print("🎉 ALL VALIDATIONS PASSED!")
        print("="*60)
        return 0
    else:
        print("❌ SOME VALIDATIONS FAILED")
        print("="*60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
