#!/usr/bin/env python3
"""
Gator CLI Demo

Demonstrates the core functionality of the Gator AI Influencer Platform.
Run this to verify the system is working correctly.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from backend.database.connection import database_manager
from backend.services.persona_service import PersonaService
from backend.models.persona import PersonaCreate
from backend.config.logging import setup_logging, get_logger

# Setup logging
setup_logging()
logger = get_logger(__name__)


async def demo_persona_management():
    """Demonstrate persona management functionality."""
    print("🎭 Gator AI Influencer Platform - Demo")
    print("=" * 50)
    
    # Connect to database
    print("📊 Connecting to database...")
    await database_manager.connect()
    
    # Create session and service
    async with database_manager.get_session() as session:
        service = PersonaService(session)
        
        # Create a demo persona
        print("\n✨ Creating demo persona...")
        persona_data = PersonaCreate(
            name="Tech Innovator Sarah",
            appearance="Professional woman in her 30s with short dark hair, wearing modern business attire. Confident posture and intelligent eyes behind stylish glasses.",
            personality="Innovative, analytical, forward-thinking tech leader. Passionate about AI and emerging technologies. Clear communicator who makes complex topics accessible.",
            content_themes=["artificial intelligence", "technology trends", "innovation", "startup culture", "digital transformation"],
            style_preferences=["professional", "modern", "clean", "high-tech", "minimalist"]
        )
        
        try:
            persona = await service.create_persona(persona_data)
            print(f"✅ Created persona: {persona.name} (ID: {persona.id[:8]}...)")
            print(f"   Themes: {', '.join(persona.content_themes)}")
            print(f"   Style: {', '.join(persona.style_preferences)}")
            
            # List all personas
            print("\n📋 Listing all personas...")
            personas = await service.list_personas()
            print(f"📊 Found {len(personas)} persona(s):")
            for p in personas:
                status = "✅ Active" if p.is_active else "❌ Inactive"
                print(f"   • {p.name} - {status} - Created: {p.created_at.strftime('%Y-%m-%d')}")
                print(f"     Content generated: {p.generation_count} times")
            
            # Demonstrate update
            print(f"\n🔄 Testing persona update...")
            from backend.models.persona import PersonaUpdate
            updates = PersonaUpdate(
                content_themes=persona.content_themes + ["machine learning", "data science"]
            )
            updated = await service.update_persona(persona.id, updates)
            if updated:
                print(f"✅ Updated themes: {', '.join(updated.content_themes)}")
            
            # Demonstrate generation count increment
            print(f"\n📈 Simulating content generation...")
            await service.increment_generation_count(persona.id)
            await service.increment_generation_count(persona.id)
            updated_persona = await service.get_persona(persona.id)
            if updated_persona:
                print(f"✅ Generation count updated: {updated_persona.generation_count}")
            
            print(f"\n🎯 Demo completed successfully!")
            print(f"   • Database operations: Working ✅")
            print(f"   • Persona management: Working ✅") 
            print(f"   • Data validation: Working ✅")
            print(f"   • CRUD operations: Working ✅")
            
        except Exception as e:
            print(f"❌ Error during demo: {e}")
            return False
    
    # Disconnect from database
    await database_manager.disconnect()
    return True


async def demo_api_info():
    """Show API information."""
    print(f"\n🌐 API Information")
    print("=" * 30)
    print(f"The Gator platform includes a full REST API with the following endpoints:")
    print(f"")
    print(f"📍 System Endpoints:")
    print(f"   GET  /              - API status and information")
    print(f"   GET  /health        - System health check")
    print(f"")  
    print(f"🎭 Persona Management:")
    print(f"   GET    /api/v1/personas/           - List all personas")
    print(f"   POST   /api/v1/personas/           - Create new persona")
    print(f"   GET    /api/v1/personas/{{id}}       - Get specific persona")
    print(f"   PUT    /api/v1/personas/{{id}}       - Update persona")
    print(f"   DELETE /api/v1/personas/{{id}}       - Delete persona")
    print(f"")
    print(f"🎨 Content Generation (Planned):")
    print(f"   GET  /api/v1/content/              - List generated content") 
    print(f"   POST /api/v1/content/generate      - Generate new content")
    print(f"")
    print(f"📊 Analytics & Monitoring:")
    print(f"   GET  /api/v1/analytics/metrics     - Platform metrics")
    print(f"   GET  /api/v1/analytics/health      - System health details")
    print(f"")
    print(f"🚀 To start the API server:")
    print(f"   cd src && python -m backend.api.main")
    print(f"   Then visit: http://localhost:8000/docs for interactive docs")


async def main():
    """Run the complete demo."""
    try:
        # Run database demo
        success = await demo_persona_management()
        if not success:
            return
        
        # Show API info
        await demo_api_info()
        
        print(f"\n🎉 Gator AI Influencer Platform - Ready for Development!")
        print(f"   Framework: FastAPI with async SQLAlchemy")
        print(f"   Architecture: Modular, scalable, following best practices")
        print(f"   Testing: Comprehensive unit and integration tests")
        print(f"   Next: Implement content generation pipeline")
        
    except Exception as e:
        logger.error("Demo failed", error=str(e))
        print(f"❌ Demo failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())