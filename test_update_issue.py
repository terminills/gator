#!/usr/bin/env python3
"""
Test script to reproduce the persona update issue.
"""

import asyncio
from backend.database.connection import database_manager
from backend.services.persona_service import PersonaService
from backend.models.persona import PersonaCreate, PersonaUpdate, BaseImageStatus


async def test_update_issue():
    """Test persona update to see if changes persist."""
    
    # Connect to database
    await database_manager.connect()
    
    try:
        async with database_manager.get_session() as session:
            service = PersonaService(session)
            
            # Create a test persona
            print("Creating test persona...")
            persona_data = PersonaCreate(
                name="Test Persona",
                appearance="Test appearance",
                personality="Test personality",
                content_themes=["test"],
                style_preferences={"tone": "test"},
                base_appearance_description="Test description",
                appearance_locked=False,
                base_image_status=BaseImageStatus.PENDING_UPLOAD
            )
            
            created = await service.create_persona(persona_data)
            print(f"Created persona: {created.id}, name='{created.name}'")
            
            # Update the persona
            print("\nUpdating persona...")
            updates = PersonaUpdate(
                name="Updated Name",
                appearance="Updated appearance",
                personality="Updated personality"
            )
            
            updated = await service.update_persona(str(created.id), updates)
            if updated:
                print(f"Update returned: {updated.id}, name='{updated.name}'")
            else:
                print("Update returned None!")
                
        # Now check if the update persisted by reading in a new session
        print("\nChecking if update persisted (new session)...")
        async with database_manager.get_session() as session:
            service = PersonaService(session)
            persona = await service.get_persona(str(created.id))
            if persona:
                print(f"Persona retrieved: {persona.id}, name='{persona.name}'")
                if persona.name == "Updated Name":
                    print("✅ SUCCESS: Update persisted correctly!")
                else:
                    print(f"❌ FAILURE: Name is '{persona.name}', expected 'Updated Name'")
            else:
                print("❌ FAILURE: Persona not found!")
                
    finally:
        await database_manager.disconnect()


if __name__ == "__main__":
    asyncio.run(test_update_issue())
