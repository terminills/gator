#!/usr/bin/env python3
"""
Test script to demonstrate persona edit feature
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from backend.database.connection import database_manager
from backend.services.persona_service import PersonaService
from backend.models.persona import PersonaUpdate
from backend.config.logging import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)


async def test_edit_feature():
    """Test the persona edit feature with comprehensive updates"""
    print("\n🎭 Testing Persona Edit Feature")
    print("=" * 60)

    try:
        # Initialize database connection
        await database_manager.connect()

        async with database_manager.get_session() as session:
            service = PersonaService(session)

            # Get all personas
            print("\n1️⃣  Fetching existing personas...")
            personas = await service.list_personas()

            if not personas:
                print("❌ No personas found. Please create one first.")
                return False

            persona = personas[0]
            print(f"✅ Found persona: {persona.name} (ID: {persona.id})")
            print(f"   Current themes: {persona.content_themes}")
            print(f"   Current rating: {persona.default_content_rating}")
            print(f"   Platform restrictions: {persona.platform_restrictions}")

            # Prepare comprehensive update
            print("\n2️⃣  Preparing comprehensive update...")
            updates = PersonaUpdate(
                name="Tech Sarah - Updated",
                content_themes=["technology", "AI", "machine learning", "startups"],
                style_preferences={"tone": "professional", "format": "educational"},
                default_content_rating="moderate",
                allowed_content_ratings=["sfw", "moderate", "nsfw"],
                platform_restrictions={
                    "instagram": "sfw_only",
                    "twitter": "moderate_allowed",
                    "reddit": "both",
                },
                base_appearance_description="Professional tech influencer with modern aesthetic",
                base_image_path="/models/base_images/tech_sarah.jpg",
                base_image_status="draft",
                appearance_locked=False,
            )

            # Apply updates
            print("\n3️⃣  Applying updates...")
            updated_persona = await service.update_persona(str(persona.id), updates)

            if updated_persona:
                print(f"✅ Successfully updated persona!")
                print(f"   New name: {updated_persona.name}")
                print(f"   New themes: {updated_persona.content_themes}")
                print(f"   New rating: {updated_persona.default_content_rating}")
                print(f"   Allowed ratings: {updated_persona.allowed_content_ratings}")
                print(
                    f"   Platform restrictions: {updated_persona.platform_restrictions}"
                )
                print(f"   Base image: {updated_persona.base_image_path}")
                print(f"   Image status: {updated_persona.base_image_status}")
                print(f"   Appearance locked: {updated_persona.appearance_locked}")

                # Verify update persistence
                print("\n4️⃣  Verifying update persistence...")
                fetched = await service.get_persona(str(persona.id))
                if fetched and fetched.name == updated_persona.name:
                    print("✅ Update persisted correctly!")
                else:
                    print("❌ Update did not persist correctly")
                    return False
            else:
                print("❌ Failed to update persona")
                return False

            print("\n🎉 All tests passed!")
            print("=" * 60)
            return True

    except Exception as e:
        logger.error(f"Test failed: {str(e)}", exc_info=True)
        print(f"\n❌ Test failed: {str(e)}")
        return False
    finally:
        await database_manager.disconnect()


if __name__ == "__main__":
    success = asyncio.run(test_edit_feature())
    sys.exit(0 if success else 1)
