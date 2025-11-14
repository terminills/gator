#!/usr/bin/env python3
"""
Quick test to verify the content generation fix for None prompt handling
"""
import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from backend.services.content_generation_service import ContentModerationService
from backend.models.content import ContentRating


def test_analyze_content_rating_with_none():
    """Test that analyze_content_rating handles None prompt correctly"""
    
    # Test with None prompt
    result = ContentModerationService.analyze_content_rating(None, "sfw")
    assert result == ContentRating.SFW, f"Expected SFW rating for None prompt, got {result}"
    print("✓ None prompt returns SFW rating")
    
    # Test with empty string
    result = ContentModerationService.analyze_content_rating("", "sfw")
    assert result == ContentRating.SFW, f"Expected SFW rating for empty prompt, got {result}"
    print("✓ Empty prompt returns SFW rating")
    
    # Test with valid prompt (SFW)
    result = ContentModerationService.analyze_content_rating("a beautiful landscape", "sfw")
    assert result == ContentRating.SFW, f"Expected SFW rating for safe prompt, got {result}"
    print("✓ SFW prompt returns SFW rating")
    
    # Test with valid prompt containing NSFW keywords
    result = ContentModerationService.analyze_content_rating("sexy bikini model", "nsfw")
    assert result == ContentRating.NSFW, f"Expected NSFW rating for NSFW prompt, got {result}"
    print("✓ NSFW prompt returns NSFW rating")
    
    # Test with valid prompt containing moderate keywords
    result = ContentModerationService.analyze_content_rating("romantic dinner scene", "sfw")
    assert result == ContentRating.MODERATE, f"Expected MODERATE rating for romantic prompt, got {result}"
    print("✓ Moderate prompt returns MODERATE rating")
    
    print("\n✅ All tests passed! The fix correctly handles None prompts.")


if __name__ == "__main__":
    test_analyze_content_rating_with_none()
