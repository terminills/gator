#!/usr/bin/env python3
"""
Integration test for seed image workflow.

Tests the complete workflow: create persona, upload image, approve baseline.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from backend.database.connection import database_manager
from backend.services.persona_service import PersonaService
from backend.models.persona import PersonaCreate, PersonaUpdate, BaseImageStatus
from backend.config.logging import setup_logging, get_logger

# Setup logging
setup_logging()
logger = get_logger(__name__)


async def test_seed_image_workflow():
    """Test the complete seed image workflow."""
    print("🧪 Testing Seed Image Workflow")
    print("=" * 60)

    try:
        # Connect to database
        await database_manager.connect()

        # Get a database session
        async with database_manager.get_session() as session:
            service = PersonaService(session)

            # 1. Create a persona
            print("\n1️⃣  Creating test persona...")
            persona_data = PersonaCreate(
                name="Test Seed Image Persona",
                appearance="Professional woman in business attire",
                personality="Confident and tech-savvy",
                content_themes=["technology", "AI"],
                style_preferences={"style": "professional"},
                base_appearance_description="Detailed: Professional woman in her 30s, dark hair, modern business casual",
            )

            persona = await service.create_persona(persona_data)
            print(f"✅ Created persona: {persona.id}")
            print(f"   Name: {persona.name}")
            print(f"   Base Image Status: {persona.base_image_status}")
            print(f"   Appearance Locked: {persona.appearance_locked}")

            assert (
                persona.base_image_status == "pending_upload"
            ), "Should start as pending_upload"
            assert persona.appearance_locked is False, "Should not be locked initially"

            # 2. Simulate adding a base image path (like after upload or generation)
            print("\n2️⃣  Simulating base image upload...")
            updates = PersonaUpdate(
                base_image_path="/opt/gator/data/models/base_images/test_image.png",
                base_image_status=BaseImageStatus.DRAFT,
            )
            persona = await service.update_persona(persona.id, updates)
            print(f"✅ Updated persona with base image")
            print(f"   Base Image Path: {persona.base_image_path}")
            print(f"   Base Image Status: {persona.base_image_status}")

            assert persona.base_image_status == "draft", "Should be in draft status"
            assert persona.base_image_path is not None, "Should have image path"

            # 3. Test rejecting the image
            print("\n3️⃣  Testing rejection workflow...")
            updates = PersonaUpdate(base_image_status=BaseImageStatus.REJECTED)
            persona = await service.update_persona(persona.id, updates)
            print(f"✅ Rejected image")
            print(f"   Base Image Status: {persona.base_image_status}")

            assert persona.base_image_status == "rejected", "Should be rejected"
            assert persona.appearance_locked is False, "Should still not be locked"

            # 4. Update back to draft
            print("\n4️⃣  Updating back to draft...")
            updates = PersonaUpdate(base_image_status=BaseImageStatus.DRAFT)
            persona = await service.update_persona(persona.id, updates)
            print(f"✅ Status updated to draft")

            # 5. Approve the baseline
            print("\n5️⃣  Approving baseline image...")
            persona = await service.approve_baseline(persona.id)
            print(f"✅ Baseline approved!")
            print(f"   Base Image Status: {persona.base_image_status}")
            print(f"   Appearance Locked: {persona.appearance_locked}")

            assert persona.base_image_status == "approved", "Should be approved"
            assert persona.appearance_locked is True, "Should be locked after approval"

            # 6. Try to approve baseline without image (should fail)
            print("\n6️⃣  Testing approval without image (should fail)...")
            persona_without_image = PersonaCreate(
                name="Test No Image Persona",
                appearance="Test appearance",
                personality="Test personality",
                content_themes=["test"],
                style_preferences={},
            )
            persona2 = await service.create_persona(persona_without_image)

            try:
                await service.approve_baseline(persona2.id)
                print("❌ ERROR: Should have raised ValueError")
                return False
            except ValueError as e:
                print(f"✅ Correctly raised error: {e}")

            # Cleanup - delete test personas
            print("\n7️⃣  Cleaning up test data...")
            await service.delete_persona(persona.id)
            await service.delete_persona(persona2.id)
            print("✅ Test personas deleted")

        await database_manager.disconnect()

        print("\n" + "=" * 60)
        print("✅ All workflow tests passed!")
        print("=" * 60)
        return True

    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        print(f"\n❌ Test failed: {str(e)}")
        try:
            await database_manager.disconnect()
        except:
            pass
        return False


async def main():
    """Main test entry point."""
    success = await test_seed_image_workflow()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())
