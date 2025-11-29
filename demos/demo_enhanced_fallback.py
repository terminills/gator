#!/usr/bin/env python3
"""
Demonstration script for Enhanced Fallback Text Generation

This script demonstrates the improvements in the _create_enhanced_fallback_text method,
showing how it uses deeper data integration from PersonaModel attributes.
"""

import asyncio
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from backend.models.persona import PersonaModel, ContentRating
from backend.models.content import GenerationRequest, ContentType
from backend.services.content_generation_service import ContentGenerationService
from unittest.mock import Mock, AsyncMock
import uuid


async def demonstrate_enhanced_fallback():
    """Demonstrate the enhanced fallback text generation capabilities."""
    
    print("=" * 80)
    print("ENHANCED FALLBACK TEXT GENERATION DEMONSTRATION")
    print("=" * 80)
    print()
    
    # Create a mock database session
    mock_db = AsyncMock()
    service = ContentGenerationService(db_session=mock_db)
    
    # Test Case 1: Creative Persona with style_preferences
    print("1. CREATIVE PERSONA WITH STYLE PREFERENCES")
    print("-" * 80)
    
    creative_persona = Mock(spec=PersonaModel)
    creative_persona.name = "Creative Artist"
    creative_persona.personality = "Creative, artistic, imaginative, passionate about design"
    creative_persona.content_themes = ["art", "design", "innovation"]
    creative_persona.style_preferences = {
        "aesthetic": "creative",
        "voice_style": "expressive",
        "tone": "warm"
    }
    creative_persona.appearance = "Artistic individual with vibrant style"
    creative_persona.base_appearance_description = None
    creative_persona.appearance_locked = False
    
    request = GenerationRequest(
        persona_id=uuid.uuid4(),
        content_type=ContentType.TEXT,
        content_rating=ContentRating.SFW,
        prompt="Share creative insights"
    )
    
    text = await service._create_enhanced_fallback_text(creative_persona, request)
    print(f"Generated Text:\n{text}\n")
    
    # Test Case 2: Professional Persona with confident tone
    print("2. PROFESSIONAL PERSONA WITH CONFIDENT TONE")
    print("-" * 80)
    
    professional_persona = Mock(spec=PersonaModel)
    professional_persona.name = "Business Leader"
    professional_persona.personality = "Professional, strategic, confident executive, data-driven"
    professional_persona.content_themes = ["business strategy", "leadership", "innovation"]
    professional_persona.style_preferences = {
        "aesthetic": "professional",
        "voice_style": "formal",
        "tone": "confident"
    }
    professional_persona.appearance = "Professional business attire"
    professional_persona.base_appearance_description = None
    professional_persona.appearance_locked = False
    
    request = GenerationRequest(
        persona_id=uuid.uuid4(),
        content_type=ContentType.TEXT,
        content_rating=ContentRating.SFW,
        prompt="Discuss business trends"
    )
    
    text = await service._create_enhanced_fallback_text(professional_persona, request)
    print(f"Generated Text:\n{text}\n")
    
    # Test Case 3: Tech Persona with analytical traits
    print("3. TECH PERSONA WITH ANALYTICAL TRAITS")
    print("-" * 80)
    
    tech_persona = Mock(spec=PersonaModel)
    tech_persona.name = "Tech Innovator"
    tech_persona.personality = "Tech-savvy engineer, analytical, passionate about AI, data-driven"
    tech_persona.content_themes = ["artificial intelligence", "machine learning", "software"]
    tech_persona.style_preferences = {
        "aesthetic": "tech",
        "voice_style": "technical",
        "tone": "precise"
    }
    tech_persona.appearance = "Modern tech professional"
    tech_persona.base_appearance_description = None
    tech_persona.appearance_locked = False
    
    request = GenerationRequest(
        persona_id=uuid.uuid4(),
        content_type=ContentType.TEXT,
        content_rating=ContentRating.SFW,
        prompt="Analyze technology developments"
    )
    
    text = await service._create_enhanced_fallback_text(tech_persona, request)
    print(f"Generated Text:\n{text}\n")
    
    # Test Case 4: Casual Persona with warm tone
    print("4. CASUAL PERSONA WITH WARM TONE")
    print("-" * 80)
    
    casual_persona = Mock(spec=PersonaModel)
    casual_persona.name = "Friendly Influencer"
    casual_persona.personality = "Friendly, approachable, warm, passionate communicator"
    casual_persona.content_themes = ["lifestyle", "wellness", "community"]
    casual_persona.style_preferences = {
        "aesthetic": "casual",
        "tone": "warm"
    }
    casual_persona.appearance = "Casual, relaxed style"
    casual_persona.base_appearance_description = None
    casual_persona.appearance_locked = False
    
    request = GenerationRequest(
        persona_id=uuid.uuid4(),
        content_type=ContentType.TEXT,
        content_rating=ContentRating.SFW,
        prompt="Share thoughts on wellness"
    )
    
    text = await service._create_enhanced_fallback_text(casual_persona, request)
    print(f"Generated Text:\n{text}\n")
    
    # Test Case 5: Multi-trait scoring (creative + tech)
    print("5. MULTI-TRAIT PERSONA (CREATIVE + ANALYTICAL)")
    print("-" * 80)
    
    hybrid_persona = Mock(spec=PersonaModel)
    hybrid_persona.name = "Creative Technologist"
    hybrid_persona.personality = "Creative thinker, analytical problem solver, tech-savvy innovator"
    hybrid_persona.content_themes = ["design thinking", "UX", "technology"]
    hybrid_persona.style_preferences = {
        "aesthetic": "modern",
        "voice_style": "expressive",
        "tone": "warm"
    }
    hybrid_persona.appearance = "Modern professional with creative flair"
    hybrid_persona.base_appearance_description = None
    hybrid_persona.appearance_locked = False
    
    request = GenerationRequest(
        persona_id=uuid.uuid4(),
        content_type=ContentType.TEXT,
        content_rating=ContentRating.SFW,
        prompt="Discuss design innovation"
    )
    
    text = await service._create_enhanced_fallback_text(hybrid_persona, request)
    print(f"Generated Text:\n{text}\n")
    
    print("=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)
    print()
    print("Key Enhancements Demonstrated:")
    print("✓ Style preferences (aesthetic, voice_style, tone) influence content")
    print("✓ Multi-attribute scoring for style determination")
    print("✓ Voice modifiers (passionate, analytical, warm, confident)")
    print("✓ Dynamic template selection based on multiple persona attributes")
    print("✓ Context-aware customization based on prompt keywords")
    print()


if __name__ == "__main__":
    asyncio.run(demonstrate_enhanced_fallback())
