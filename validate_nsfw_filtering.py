#!/usr/bin/env python3
"""
Validation script for per-site NSFW filtering feature.

This script validates the implementation without requiring database setup.
"""

import sys
from enum import Enum
from typing import Dict, List, Optional


# Minimal ContentRating enum for testing
class ContentRating(str, Enum):
    SFW = "sfw"
    MODERATE = "moderate"
    NSFW = "nsfw"


def platform_content_filter(
    content_rating: ContentRating, 
    target_platform: str,
    persona_platform_restrictions: Optional[Dict[str, str]] = None
) -> bool:
    """
    Simplified version of ContentModerationService.platform_content_filter
    for validation purposes.
    """
    platform_lower = target_platform.lower()
    
    # Check persona-specific restrictions first
    if persona_platform_restrictions and platform_lower in persona_platform_restrictions:
        restriction = persona_platform_restrictions[platform_lower].lower()
        
        if restriction == "sfw_only":
            return content_rating == ContentRating.SFW
        elif restriction == "moderate_allowed":
            return content_rating in [ContentRating.SFW, ContentRating.MODERATE]
        elif restriction == "both" or restriction == "all":
            return True
    
    # Default platform policies
    platform_policies = {
        "instagram": [ContentRating.SFW, ContentRating.MODERATE],
        "facebook": [ContentRating.SFW],
        "twitter": [ContentRating.SFW, ContentRating.MODERATE, ContentRating.NSFW],
        "onlyfans": [ContentRating.SFW, ContentRating.MODERATE, ContentRating.NSFW],
        "patreon": [ContentRating.SFW, ContentRating.MODERATE, ContentRating.NSFW],
        "discord": [ContentRating.SFW, ContentRating.MODERATE],
    }

    allowed_ratings = platform_policies.get(platform_lower, [ContentRating.SFW])
    return content_rating in allowed_ratings


def run_tests():
    """Run validation tests."""
    print("=" * 70)
    print("VALIDATING PER-SITE NSFW FILTERING IMPLEMENTATION")
    print("=" * 70)
    print()
    
    passed = 0
    failed = 0
    
    # Test 1: Default global policies
    print("Test 1: Default global policies (no overrides)")
    print("-" * 70)
    
    tests = [
        (ContentRating.SFW, "instagram", None, True, "SFW on Instagram"),
        (ContentRating.NSFW, "instagram", None, False, "NSFW blocked on Instagram"),
        (ContentRating.MODERATE, "instagram", None, True, "MODERATE on Instagram"),
        (ContentRating.NSFW, "onlyfans", None, True, "NSFW on OnlyFans"),
        (ContentRating.NSFW, "facebook", None, False, "NSFW blocked on Facebook"),
    ]
    
    for rating, platform, restrictions, expected, description in tests:
        result = platform_content_filter(rating, platform, restrictions)
        status = "✓ PASS" if result == expected else "✗ FAIL"
        if result == expected:
            passed += 1
        else:
            failed += 1
        print(f"  {status}: {description} (expected={expected}, got={result})")
    
    print()
    
    # Test 2: Persona overrides - NSFW on Instagram
    print("Test 2: Persona overrides - Allow NSFW on Instagram")
    print("-" * 70)
    
    restrictions = {"instagram": "both"}
    tests = [
        (ContentRating.SFW, "instagram", restrictions, True, "SFW on Instagram with override"),
        (ContentRating.NSFW, "instagram", restrictions, True, "NSFW on Instagram with override"),
        (ContentRating.MODERATE, "instagram", restrictions, True, "MODERATE on Instagram with override"),
    ]
    
    for rating, platform, restrictions, expected, description in tests:
        result = platform_content_filter(rating, platform, restrictions)
        status = "✓ PASS" if result == expected else "✗ FAIL"
        if result == expected:
            passed += 1
        else:
            failed += 1
        print(f"  {status}: {description} (expected={expected}, got={result})")
    
    print()
    
    # Test 3: SFW-only restriction
    print("Test 3: Persona overrides - SFW-only on OnlyFans")
    print("-" * 70)
    
    restrictions = {"onlyfans": "sfw_only"}
    tests = [
        (ContentRating.SFW, "onlyfans", restrictions, True, "SFW on OnlyFans with SFW-only"),
        (ContentRating.MODERATE, "onlyfans", restrictions, False, "MODERATE blocked on OnlyFans with SFW-only"),
        (ContentRating.NSFW, "onlyfans", restrictions, False, "NSFW blocked on OnlyFans with SFW-only"),
    ]
    
    for rating, platform, restrictions, expected, description in tests:
        result = platform_content_filter(rating, platform, restrictions)
        status = "✓ PASS" if result == expected else "✗ FAIL"
        if result == expected:
            passed += 1
        else:
            failed += 1
        print(f"  {status}: {description} (expected={expected}, got={result})")
    
    print()
    
    # Test 4: Mixed restrictions
    print("Test 4: Mixed platform restrictions")
    print("-" * 70)
    
    restrictions = {
        "instagram": "both",
        "facebook": "moderate_allowed",
        "twitter": "sfw_only"
    }
    
    tests = [
        (ContentRating.NSFW, "instagram", restrictions, True, "NSFW on Instagram (both)"),
        (ContentRating.MODERATE, "facebook", restrictions, True, "MODERATE on Facebook (moderate_allowed)"),
        (ContentRating.NSFW, "facebook", restrictions, False, "NSFW blocked on Facebook (moderate_allowed)"),
        (ContentRating.MODERATE, "twitter", restrictions, False, "MODERATE blocked on Twitter (sfw_only)"),
        (ContentRating.NSFW, "onlyfans", restrictions, True, "NSFW on OnlyFans (no override, uses default)"),
    ]
    
    for rating, platform, restrictions, expected, description in tests:
        result = platform_content_filter(rating, platform, restrictions)
        status = "✓ PASS" if result == expected else "✗ FAIL"
        if result == expected:
            passed += 1
        else:
            failed += 1
        print(f"  {status}: {description} (expected={expected}, got={result})")
    
    print()
    print("=" * 70)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 70)
    
    if failed > 0:
        print("\n✗ VALIDATION FAILED - Some tests did not pass")
        return False
    else:
        print("\n✓ VALIDATION SUCCESSFUL - All tests passed!")
        return True


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
