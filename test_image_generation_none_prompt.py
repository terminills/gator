#!/usr/bin/env python3
"""
Test to verify image generation with None prompt works correctly.
This tests the specific scenario from the error log where prompt is None.
"""
import asyncio
import sys
from pathlib import Path
from uuid import uuid4

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from backend.models.content import ContentType, ContentRating


async def test_analyze_content_rating_in_generation_flow():
    """
    Simulate the content generation flow where prompt is None for images.
    This is the exact scenario that caused the original error.
    """
    from backend.services.content_generation_service import ContentModerationService
    
    print("Testing content generation flow with None prompt...")
    print("=" * 60)
    
    # Simulate the scenario from generate_content method
    # Line 275-280 in content_generation_service.py shows that for IMAGE type,
    # prompt is not generated and remains None
    request_prompt = None  # This is what happens for IMAGE content type
    persona_default_rating = "sfw"
    
    print(f"Request prompt: {request_prompt}")
    print(f"Persona default rating: {persona_default_rating}")
    print()
    
    # This is the call that was failing at line 289-290
    print("Calling analyze_content_rating with None prompt...")
    try:
        analyzed_rating = ContentModerationService.analyze_content_rating(
            request_prompt, persona_default_rating
        )
        print(f"✅ SUCCESS! Analyzed rating: {analyzed_rating.value}")
        print(f"   The fix correctly handles None prompts.")
        return True
    except AttributeError as e:
        print(f"❌ FAILED! Error: {e}")
        print(f"   The fix did not work correctly.")
        return False


async def test_full_generation_scenario():
    """
    Test a more complete scenario similar to the error log.
    """
    from backend.services.content_generation_service import ContentModerationService
    from backend.models.content import GenerationRequest
    
    print("\nTesting full generation scenario...")
    print("=" * 60)
    
    # Create a generation request similar to the error log
    request = GenerationRequest(
        persona_id=uuid4(),
        content_type=ContentType.IMAGE,
        content_rating=None,  # Will use persona default
        prompt=None,  # Not provided for image generation
        quality="standard",
    )
    
    print(f"Content type: {request.content_type.value}")
    print(f"Request prompt: {request.prompt}")
    print(f"Request rating: {request.content_rating}")
    print()
    
    # Test the moderation service directly
    persona_rating = "sfw"
    print(f"Analyzing content rating with persona default: {persona_rating}")
    
    try:
        analyzed_rating = ContentModerationService.analyze_content_rating(
            request.prompt, persona_rating
        )
        print(f"✅ SUCCESS! Analyzed rating: {analyzed_rating.value}")
        print(f"   The content generation flow will work correctly.")
        return True
    except AttributeError as e:
        print(f"❌ FAILED! Error: {e}")
        return False


async def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("CONTENT GENERATION FIX VALIDATION")
    print("=" * 60)
    print()
    print("Original error from logs:")
    print("  'NoneType' object has no attribute 'lower'")
    print("  at analyze_content_rating when prompt is None")
    print()
    
    test1_passed = await test_analyze_content_rating_in_generation_flow()
    test2_passed = await test_full_generation_scenario()
    
    print("\n" + "=" * 60)
    if test1_passed and test2_passed:
        print("✅ ALL TESTS PASSED!")
        print("   The fix successfully handles None prompts in image generation.")
        print("   Content generation should now work without errors.")
    else:
        print("❌ SOME TESTS FAILED!")
        print("   The fix needs additional work.")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
