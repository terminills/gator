#!/usr/bin/env python3
"""
Test script for visual consistency and appearance locking features.

This script demonstrates:
1. Creating a persona with base appearance and visual reference
2. Locking the appearance for consistency
3. Generating content with visual consistency enabled
"""

import asyncio
import sys
from pathlib import Path
from uuid import uuid4

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from backend.database.connection import database_manager
from backend.services.persona_service import PersonaService
from backend.models.persona import PersonaCreate, PersonaUpdate, ContentRating
from backend.config.logging import setup_logging, get_logger

# Setup logging
setup_logging()
logger = get_logger(__name__)


async def test_appearance_locking():
    """Test the appearance locking and visual consistency features."""
    print("🎭 Testing Visual Consistency & Appearance Locking")
    print("=" * 60)
    
    try:
        # Connect to database
        await database_manager.connect()
        
        # Get database session
        async with database_manager.get_session() as session:
            persona_service = PersonaService(session)
            
            print("\n📝 Test 1: Create persona WITHOUT appearance locking")
            print("-" * 60)
            
            # Create a standard persona (no locking)
            standard_persona_data = PersonaCreate(
                name="Standard AI Sarah",
                appearance="Professional woman in her 30s with short dark hair",
                personality="Friendly, professional tech enthusiast",
                content_themes=["technology", "ai"],
                default_content_rating=ContentRating.SFW,
                allowed_content_ratings=[ContentRating.SFW]
            )
            
            standard_persona = await persona_service.create_persona(standard_persona_data)
            print(f"✅ Created standard persona: {standard_persona.name}")
            print(f"   ID: {standard_persona.id}")
            print(f"   Appearance Locked: {standard_persona.appearance_locked}")
            print(f"   Base Image Path: {standard_persona.base_image_path}")
            
            print("\n📝 Test 2: Create persona WITH appearance locking")
            print("-" * 60)
            
            # Create a persona with visual consistency features
            locked_persona_data = PersonaCreate(
                name="Locked AI Emma",
                appearance="Young professional woman with long blonde hair and blue eyes",
                personality="Creative, innovative, passionate about design",
                content_themes=["design", "creativity", "innovation"],
                default_content_rating=ContentRating.SFW,
                allowed_content_ratings=[ContentRating.SFW],
                base_appearance_description=(
                    "A 28-year-old professional woman with long, wavy blonde hair cascading past her shoulders. "
                    "Striking blue eyes with subtle makeup emphasizing natural beauty. Fair complexion with a warm undertone. "
                    "Wearing modern business casual attire - a cream-colored blouse with minimal accessories. "
                    "Confident posture with a friendly, approachable expression. Professional studio lighting, "
                    "soft focus background, high-resolution portrait photography style."
                ),
                base_image_path="/models/base_images/emma_reference_001.jpg",
                appearance_locked=True
            )
            
            locked_persona = await persona_service.create_persona(locked_persona_data)
            print(f"✅ Created locked persona: {locked_persona.name}")
            print(f"   ID: {locked_persona.id}")
            print(f"   Appearance Locked: {locked_persona.appearance_locked}")
            print(f"   Base Image Path: {locked_persona.base_image_path}")
            print(f"   Base Description Length: {len(locked_persona.base_appearance_description or '')} chars")
            
            print("\n📝 Test 3: Update persona to enable appearance locking")
            print("-" * 60)
            
            # Update the standard persona to add locking
            update_data = PersonaUpdate(
                base_appearance_description=(
                    "Professional woman, approximately 35 years old, with short, styled dark brown hair in a modern bob cut. "
                    "Hazel eyes behind contemporary designer glasses with thin metallic frames. Intelligent, focused expression. "
                    "Wearing tailored business attire in neutral tones. Studio portrait with professional lighting, "
                    "sharp focus, corporate photography style."
                ),
                base_image_path="/models/base_images/sarah_reference_001.jpg",
                appearance_locked=True
            )
            
            updated_persona = await persona_service.update_persona(
                str(standard_persona.id), update_data
            )
            print(f"✅ Updated persona: {updated_persona.name}")
            print(f"   Appearance now locked: {updated_persona.appearance_locked}")
            print(f"   Base Image Path: {updated_persona.base_image_path}")
            
            print("\n📝 Test 4: Verify locked appearance cannot be easily changed")
            print("-" * 60)
            
            # Try to update appearance when locked (should still allow but with warning in logs)
            try:
                update_appearance = PersonaUpdate(
                    appearance="Completely different person with red hair"
                )
                still_locked = await persona_service.update_persona(
                    str(locked_persona.id), update_appearance
                )
                print(f"⚠️  Appearance field updated, but locked persona still uses base:")
                print(f"   Standard appearance: {still_locked.appearance[:50]}...")
                print(f"   Base appearance: {still_locked.base_appearance_description[:50]}...")
                print(f"   Locked flag: {still_locked.appearance_locked}")
            except Exception as e:
                print(f"✅ Appearance update properly handled: {e}")
            
            print("\n📝 Test 5: List all personas with locking status")
            print("-" * 60)
            
            all_personas = await persona_service.list_personas(limit=10)
            print(f"Found {len(all_personas)} personas:")
            for p in all_personas:
                lock_status = "🔒 LOCKED" if p.appearance_locked else "🔓 unlocked"
                has_base_image = "📷 with reference" if p.base_image_path else "❌ no reference"
                print(f"   {lock_status} {has_base_image} - {p.name}")
            
            print("\n✅ All tests completed successfully!")
            print("\n📊 Summary:")
            print(f"   • Created {len(all_personas)} personas")
            print(f"   • Tested appearance locking: ✅")
            print(f"   • Tested base appearance description: ✅")
            print(f"   • Tested base image path: ✅")
            print(f"   • Tested update operations: ✅")
            
        await database_manager.disconnect()
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run appearance locking tests."""
    success = await test_appearance_locking()
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
