#!/usr/bin/env python3
"""
Test concurrent updates to verify WAL mode fixes the issue.
"""

import asyncio
from backend.database.connection import database_manager
from backend.services.persona_service import PersonaService
from backend.models.persona import PersonaCreate, PersonaUpdate, BaseImageStatus


async def worker_update(worker_id: int, persona_id: str, iteration: int):
    """Simulate a worker updating a persona."""
    async with database_manager.get_session() as session:
        service = PersonaService(session)
        
        updates = PersonaUpdate(
            name=f"Worker {worker_id} Update {iteration}"
        )
        
        updated = await service.update_persona(persona_id, updates)
        if updated:
            print(f"[Worker {worker_id}] Updated to: '{updated.name}'")
            return updated.name
        else:
            print(f"[Worker {worker_id}] Update failed!")
            return None


async def worker_read(worker_id: int, persona_id: str):
    """Simulate a worker reading a persona."""
    async with database_manager.get_session() as session:
        service = PersonaService(session)
        
        persona = await service.get_persona(persona_id)
        if persona:
            print(f"[Worker {worker_id}] Read: '{persona.name}'")
            return persona.name
        else:
            print(f"[Worker {worker_id}] Read failed!")
            return None


async def test_concurrent_updates():
    """Test concurrent updates from multiple 'workers'."""
    
    # Connect to database
    await database_manager.connect()
    
    try:
        # Create a test persona
        print("Creating test persona...")
        async with database_manager.get_session() as session:
            service = PersonaService(session)
            
            persona_data = PersonaCreate(
                name="Concurrent Test Persona",
                appearance="Test appearance",
                personality="Test personality",
                content_themes=["test"],
                style_preferences={"tone": "test"},
                base_appearance_description="Test description",
                appearance_locked=False,
                base_image_status=BaseImageStatus.PENDING_UPLOAD
            )
            
            created = await service.create_persona(persona_data)
            persona_id = str(created.id)
            print(f"Created persona: {persona_id}\n")
        
        # Simulate concurrent updates from different workers
        print("Testing concurrent updates...")
        tasks = []
        for i in range(3):
            for worker_id in range(1, 4):
                tasks.append(worker_update(worker_id, persona_id, i + 1))
        
        # Run all updates concurrently
        results = await asyncio.gather(*tasks)
        print(f"\n✅ All updates completed: {len([r for r in results if r])} successful\n")
        
        # Now have multiple workers read the final state
        print("Testing concurrent reads...")
        read_tasks = []
        for worker_id in range(1, 6):
            read_tasks.append(worker_read(worker_id, persona_id))
        
        read_results = await asyncio.gather(*read_tasks)
        
        # Check if all workers see the same data
        unique_names = set(r for r in read_results if r)
        if len(unique_names) == 1:
            print(f"\n✅ SUCCESS: All workers see consistent data: '{unique_names.pop()}'")
        else:
            print(f"\n❌ FAILURE: Workers see inconsistent data: {unique_names}")
                
    finally:
        await database_manager.disconnect()


if __name__ == "__main__":
    asyncio.run(test_concurrent_updates())
