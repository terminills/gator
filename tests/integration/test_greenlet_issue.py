#!/usr/bin/env python3
"""
Test script to reproduce the greenlet_spawn error in content generation.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


async def test_text_generation_fallback():
    """Test text generation fallback that triggers greenlet error."""
    print("\n" + "="*80)
    print("🧪 Testing text generation fallback (greenlet issue)")
    print("="*80)
    
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
    from sqlalchemy.orm import sessionmaker
    from backend.models.persona import PersonaModel
    from backend.services.content_generation_service import (
        ContentGenerationService,
        GenerationRequest,
    )
    from backend.models.content import ContentType, ContentRating
    
    # Create in-memory database
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    
    # Create tables
    from backend.database.connection import Base
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    # Create session
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    
    async with async_session() as session:
        # Create a test persona
        persona = PersonaModel(
            name="Test Persona",
            appearance="A test appearance",
            personality="friendly, casual",
            content_themes=["tech", "lifestyle"],
            style_preferences={"aesthetic": "modern", "voice_style": "casual"},
            default_content_rating="sfw",
            allowed_content_ratings=["sfw"],
        )
        session.add(persona)
        await session.commit()
        await session.refresh(persona)
        
        print(f"✓ Created persona: {persona.name} ({persona.id})")
        
        # Create content generation service
        service = ContentGenerationService(session, content_dir="/tmp/test_content")
        
        # Create a generation request
        request = GenerationRequest(
            persona_id=persona.id,
            content_type=ContentType.TEXT,
            content_rating=ContentRating.SFW,
            prompt="Write a test post",
            quality="standard",
        )
        
        print(f"\n🔄 Attempting text generation (will trigger fallback)...")
        
        try:
            # This should trigger the fallback path and the greenlet error
            content = await service.generate_content(request)
            print(f"\n✅ Content generated successfully!")
            print(f"   Content ID: {content.id}")
            print(f"   File: {content.file_path}")
            return True
            
        except Exception as e:
            print(f"\n❌ Error during generation:")
            print(f"   {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return False


if __name__ == "__main__":
    result = asyncio.run(test_text_generation_fallback())
    sys.exit(0 if result else 1)
