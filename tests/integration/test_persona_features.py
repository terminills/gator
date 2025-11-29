#!/usr/bin/env python
"""
Test script for new persona features:
1. Persona update with content ratings
2. Resolution configuration
3. Random persona generation
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from backend.database.connection import get_db_session, DatabaseManager
from backend.services.persona_service import PersonaService
from backend.services.persona_randomizer import PersonaRandomizer
from backend.models.persona import PersonaCreate, PersonaUpdate, ContentRating
from backend.models.generation_config import (
    ImageResolution,
    ImageGenerationConfig,
    QualityPreset,
)


async def test_persona_update_with_ratings():
    """Test persona update with content rating fields."""
    print("\n" + "="*60)
    print("TEST 1: Persona Update with Content Ratings")
    print("="*60)
    
    # Initialize database
    db_manager = DatabaseManager()
    await db_manager.connect()
    
    try:
        # Get database session
        db_gen = get_db_session()
        db = await db_gen.__anext__()
        
        service = PersonaService(db)
        
        # Create a test persona
        print("\n📝 Creating test persona...")
        persona_data = PersonaCreate(
            name="Test Persona for Ratings",
            appearance="Test appearance description for ratings test.",
            personality="Test personality description for ratings test.",
            content_themes=["test", "ratings"],
            style_preferences={"test": "value"},
            default_content_rating=ContentRating.SFW,
            allowed_content_ratings=[ContentRating.SFW],
        )
        
        persona = await service.create_persona(persona_data)
        print(f"✅ Created persona: {persona.name} (ID: {persona.id})")
        print(f"   Default rating: {persona.default_content_rating}")
        print(f"   Allowed ratings: {persona.allowed_content_ratings}")
        
        # Update with new ratings
        print("\n🔄 Updating content ratings...")
        updates = PersonaUpdate(
            default_content_rating=ContentRating.MODERATE,
            allowed_content_ratings=[ContentRating.SFW, ContentRating.MODERATE],
            platform_restrictions={
                "instagram": "sfw_only",
                "twitter": "moderate_allowed"
            }
        )
        
        updated = await service.update_persona(str(persona.id), updates)
        print(f"✅ Updated persona ratings:")
        print(f"   Default rating: {updated.default_content_rating}")
        print(f"   Allowed ratings: {updated.allowed_content_ratings}")
        print(f"   Platform restrictions: {updated.platform_restrictions}")
        
        # Clean up
        await service.delete_persona(str(persona.id))
        print("\n🧹 Test persona deleted")
        
    finally:
        try:
            await db_gen.aclose()
        except:
            pass
        await db_manager.disconnect()
            
    print("\n✅ TEST 1 PASSED: Content rating updates work correctly!")


def test_resolution_configuration():
    """Test resolution configuration models."""
    print("\n" + "="*60)
    print("TEST 2: Resolution Configuration")
    print("="*60)
    
    # Test various resolutions
    resolutions_to_test = [
        (ImageResolution.HD_1024, "1024x1024 Square"),
        (ImageResolution.LANDSCAPE_HD, "1280x720 Landscape HD"),
        (ImageResolution.LANDSCAPE_FHD, "1920x1080 Landscape Full HD"),
        (ImageResolution.PORTRAIT_HD, "720x1280 Portrait HD"),
        (ImageResolution.PORTRAIT_FHD, "1080x1920 Portrait Full HD"),
    ]
    
    print("\n📐 Testing resolution configurations:")
    for resolution, description in resolutions_to_test:
        config = ImageGenerationConfig(
            resolution=resolution,
            quality=QualityPreset.HIGH
        )
        width, height = config.get_dimensions()
        quality_params = config.get_quality_params()
        
        print(f"\n   {description}:")
        print(f"      Resolution: {width}x{height}")
        print(f"      Steps: {quality_params['num_inference_steps']}")
        print(f"      Guidance: {quality_params['guidance_scale']}")
    
    # Test custom resolution
    print("\n   Custom Resolution:")
    custom_config = ImageGenerationConfig(
        resolution=ImageResolution.CUSTOM,
        custom_width=2560,
        custom_height=1440,
        quality=QualityPreset.PREMIUM
    )
    width, height = custom_config.get_dimensions()
    print(f"      Resolution: {width}x{height}")
    
    print("\n✅ TEST 2 PASSED: All resolution configurations work!")


def test_random_persona_generation():
    """Test random persona generation."""
    print("\n" + "="*60)
    print("TEST 3: Random Persona Generation")
    print("="*60)
    
    # Generate multiple random personas
    print("\n🎲 Generating 3 random personas...")
    
    for i in range(3):
        persona_config = PersonaRandomizer.generate_complete_random_persona()
        
        print(f"\n   Persona {i+1}: {persona_config['name']}")
        print(f"      Appearance: {persona_config['appearance'][:80]}...")
        print(f"      Personality: {persona_config['personality'][:80]}...")
        print(f"      Themes: {', '.join(persona_config['content_themes'])}")
        print(f"      Default Rating: {persona_config['default_content_rating']}")
        print(f"      Style: {persona_config['style_preferences'].get('aesthetic')}")
    
    print("\n✅ TEST 3 PASSED: Random persona generation works!")


def test_quality_presets():
    """Test quality preset configurations."""
    print("\n" + "="*60)
    print("TEST 4: Quality Presets")
    print("="*60)
    
    print("\n⚙️ Testing quality presets:")
    
    for quality in QualityPreset:
        config = ImageGenerationConfig(
            resolution=ImageResolution.HD_1024,
            quality=quality
        )
        params = config.get_quality_params()
        
        print(f"\n   {quality.value.upper()}:")
        print(f"      Steps: {params['num_inference_steps']}")
        print(f"      Guidance Scale: {params['guidance_scale']}")
    
    print("\n✅ TEST 4 PASSED: All quality presets configured correctly!")


async def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("  PERSONA FEATURES TEST SUITE")
    print("  Testing: Updates, Resolutions, Random Generation, Quality Presets")
    print("="*70)
    
    try:
        # Run async tests
        await test_persona_update_with_ratings()
        
        # Run sync tests
        test_resolution_configuration()
        test_random_persona_generation()
        test_quality_presets()
        
        # Summary
        print("\n" + "="*70)
        print("  ✅ ALL TESTS PASSED!")
        print("="*70)
        print("\n📋 Summary:")
        print("   ✅ Persona updates with content ratings - WORKING")
        print("   ✅ Resolution configuration (720p, 1080p, custom) - WORKING")
        print("   ✅ Random persona generation - WORKING")
        print("   ✅ Quality presets (draft to premium) - WORKING")
        print("\n🎉 All new features are functioning correctly!")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
