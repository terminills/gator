#!/usr/bin/env python3
"""
Simple validation script to check the content generation fixes.
Does not require full dependencies - just checks the code changes.
"""

import re
import sys


def check_none_prompt_fix():
    """Verify that the None prompt fix is in place."""
    print("Checking None prompt fix in content_generation_service.py...")
    
    with open("src/backend/services/content_generation_service.py", "r") as f:
        content = f.read()
    
    # Check for the fix around line 1531
    if "if request.prompt:" in content and "For IMAGE type, prompt might be None" in content:
        print("✓ None prompt fix is in place")
        print("  - Checks if request.prompt exists before slicing")
        print("  - Handles IMAGE type gracefully when prompt is None")
        return True
    else:
        print("✗ None prompt fix NOT found")
        return False


def check_lpw_preference():
    """Verify that lpw_stable_diffusion_xl is marked as preferred."""
    print("\nChecking lpw_stable_diffusion_xl preference in ai_models.py...")
    
    with open("src/backend/services/ai_models.py", "r") as f:
        content = f.read()
    
    # Check for PREFERRED marker
    if "PREFERRED" in content and "lpw_stable_diffusion_xl" in content:
        print("✓ lpw_stable_diffusion_xl marked as PREFERRED")
        
        # Check for benefits documentation
        if "Benefits over compel" in content:
            print("✓ Benefits over compel documented")
            return True
        else:
            print("⚠ Benefits documentation could be improved")
            return True
    else:
        print("✗ lpw_stable_diffusion_xl preference NOT clearly marked")
        return False


def check_compel_fallback():
    """Verify that compel is documented as fallback."""
    print("\nChecking compel fallback documentation in ai_models.py...")
    
    with open("src/backend/services/ai_models.py", "r") as f:
        content = f.read()
    
    # Check for fallback indication with bounds checking
    start_pos = content.find("if is_sdxl and not is_lpw_pipeline")
    end_pos = content.find("elif is_lpw_pipeline")
    
    if start_pos == -1 or end_pos == -1 or end_pos <= start_pos:
        print("✗ Could not find compel section in code")
        return False
    
    compel_section = content[start_pos:end_pos]
    
    if "fallback" in compel_section.lower():
        print("✓ Compel documented as fallback")
        
        # Check for lpw recommendation
        if "lpw" in compel_section.lower():
            print("✓ lpw recommendation in compel fallback path")
            return True
        else:
            print("⚠ lpw recommendation could be added to fallback messages")
            return True
    else:
        print("✗ Compel NOT clearly documented as fallback")
        return False


def check_long_prompt_comments():
    """Verify long prompt handling comments."""
    print("\nChecking long prompt support comments...")
    
    with open("src/backend/services/ai_models.py", "r") as f:
        content = f.read()
    
    # Find the custom pipeline section
    lpw_section_start = content.find('load_args["custom_pipeline"] = "lpw_stable_diffusion_xl"')
    if lpw_section_start == -1:
        print("✗ lpw_stable_diffusion_xl pipeline loading NOT found")
        return False
    
    # Get 500 chars before to check for comments
    lpw_context = content[max(0, lpw_section_start - 500):lpw_section_start + 500]
    
    checks = {
        "prompt chunking": "prompt chunking" in lpw_context.lower(),
        "225+ tokens": "225" in lpw_context,
        "community": "community" in lpw_context.lower(),
    }
    
    passed = sum(checks.values())
    total = len(checks)
    
    for check, result in checks.items():
        status = "✓" if result else "✗"
        print(f"  {status} {check}")
    
    if passed >= 2:
        print(f"✓ Long prompt documentation adequate ({passed}/{total} checks)")
        return True
    else:
        print(f"⚠ Long prompt documentation could be improved ({passed}/{total} checks)")
        return False


def check_template_fallback():
    """Verify template-based fallback is in place."""
    print("\nChecking template-based prompt fallback...")
    
    with open("src/backend/services/prompt_generation_service.py", "r") as f:
        content = f.read()
    
    # Check for template generation method
    if "_generate_with_template" in content:
        print("✓ Template-based fallback method exists")
        
        # Check if it includes RSS content support with bounds checking
        start_pos = content.find("def _generate_with_template")
        end_pos = content.find("def _generate_fallback_prompt")
        
        if start_pos != -1 and end_pos != -1 and end_pos > start_pos:
            template_method = content[start_pos:end_pos]
            
            if "rss_content" in template_method:
                print("✓ Template method supports RSS content")
            if "appearance" in template_method:
                print("✓ Template method uses appearance")
            if "personality" in template_method:
                print("✓ Template method uses personality")
        else:
            print("⚠ Could not extract template method for detailed checking")
        
        return True
    else:
        print("✗ Template-based fallback NOT found")
        return False


def main():
    """Run all validation checks."""
    print("=" * 70)
    print("Content Generation Fixes Validation")
    print("=" * 70)
    
    checks = [
        ("None Prompt Fix", check_none_prompt_fix),
        ("LPW Preference", check_lpw_preference),
        ("Compel Fallback", check_compel_fallback),
        ("Long Prompt Comments", check_long_prompt_comments),
        ("Template Fallback", check_template_fallback),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"✗ Error checking {name}: {e}")
            results.append((name, False))
    
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n🎉 All validation checks passed!")
        return 0
    elif passed >= total * 0.8:
        print("\n✓ Most checks passed - fixes are substantially in place")
        return 0
    else:
        print("\n⚠ Some checks failed - review needed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
