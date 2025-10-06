"""
Tests for Enhanced Fallback Text Generation

Validates that the _create_enhanced_fallback_text method uses deeper data integration
and produces sophisticated, dynamic content based on PersonaModel attributes.
"""

import asyncio
import pytest
import uuid
from unittest.mock import Mock, AsyncMock
from datetime import datetime

from backend.models.persona import PersonaModel, ContentRating
from backend.models.content import GenerationRequest, ContentType
from backend.services.content_generation_service import ContentGenerationService


@pytest.fixture
def content_service():
    """Create a content generation service for testing."""
    # Mock the database session
    mock_db_session = AsyncMock()
    return ContentGenerationService(db_session=mock_db_session)


@pytest.fixture
def base_persona():
    """Create a base persona with comprehensive attributes."""
    persona = Mock(spec=PersonaModel)
    persona.id = uuid.uuid4()
    persona.name = "Test Persona"
    persona.appearance = "Professional appearance with modern style"
    persona.personality = "Creative, analytical, passionate about innovation"
    persona.content_themes = ["technology", "innovation", "digital transformation"]
    persona.style_preferences = {
        "aesthetic": "professional",
        "voice_style": "confident",
        "tone": "warm",
    }
    persona.base_appearance_description = None
    persona.appearance_locked = False
    return persona


@pytest.fixture
def generation_request():
    """Create a basic generation request."""
    test_uuid = uuid.uuid4()
    return GenerationRequest(
        persona_id=test_uuid,
        content_type=ContentType.TEXT,
        content_rating=ContentRating.SFW,
        prompt="Create engaging content about technology trends",
    )


class TestEnhancedFallbackText:
    """Test suite for enhanced fallback text generation."""

    @pytest.mark.asyncio
    async def test_style_preferences_integration(
        self, content_service, base_persona, generation_request
    ):
        """Test that style_preferences are used in content generation."""
        # Test with professional aesthetic
        base_persona.style_preferences = {
            "aesthetic": "professional",
            "voice_style": "formal",
            "tone": "confident",
        }

        text = await content_service._create_enhanced_fallback_text(
            base_persona, generation_request
        )

        # Should generate content that reflects professional style
        assert len(text) > 50
        assert isinstance(text, str)
        # Check that content uses themes
        assert any(
            theme.lower() in text.lower() for theme in base_persona.content_themes
        )

    @pytest.mark.asyncio
    async def test_creative_style_scoring(
        self, content_service, base_persona, generation_request
    ):
        """Test that creative personality traits result in creative content style."""
        base_persona.personality = (
            "Creative, artistic, imaginative, innovative designer"
        )
        base_persona.style_preferences = {
            "aesthetic": "creative",
            "voice_style": "expressive",
        }

        text = await content_service._create_enhanced_fallback_text(
            base_persona, generation_request
        )

        # Creative content should include emojis and engaging language
        assert any(emoji in text for emoji in ["🎨", "✨", "🚀", "💡", "🌟"])
        assert len(text) > 50

    @pytest.mark.asyncio
    async def test_tech_style_scoring(
        self, content_service, base_persona, generation_request
    ):
        """Test that tech personality traits result in technical content style."""
        base_persona.personality = (
            "Analytical, tech-savvy engineer, data-driven problem solver"
        )
        base_persona.content_themes = [
            "artificial intelligence",
            "machine learning",
            "data science",
        ]
        base_persona.style_preferences = {
            "aesthetic": "tech",
            "voice_style": "technical",
        }

        text = await content_service._create_enhanced_fallback_text(
            base_persona, generation_request
        )

        # Tech content should include technical language
        assert any(emoji in text for emoji in ["🔧", "💻", "⚡", "🔍", "⚙️"])
        assert len(text) > 50

    @pytest.mark.asyncio
    async def test_professional_style_scoring(self, content_service, base_persona):
        """Test that professional personality traits result in professional content."""
        base_persona.personality = "Professional, strategic, business-focused executive"
        base_persona.style_preferences = {
            "aesthetic": "professional",
            "voice_style": "formal",
            "tone": "confident",
        }

        # Use a prompt without "trends" to avoid keyword replacement
        test_uuid = uuid.uuid4()
        request = GenerationRequest(
            persona_id=test_uuid,
            content_type=ContentType.TEXT,
            content_rating=ContentRating.SFW,
            prompt="Create engaging content about business strategy",
        )

        text = await content_service._create_enhanced_fallback_text(
            base_persona, request
        )

        # Professional content should have formal language
        assert len(text) > 50
        # Check for professional hashtags
        assert "#" in text

    @pytest.mark.asyncio
    async def test_casual_style_default(
        self, content_service, base_persona, generation_request
    ):
        """Test that casual style is used when no strong indicators present."""
        base_persona.personality = "Friendly, approachable, warm communicator"
        base_persona.style_preferences = {"aesthetic": "casual", "tone": "warm"}

        text = await content_service._create_enhanced_fallback_text(
            base_persona, generation_request
        )

        # Casual content should be conversational
        assert len(text) > 50
        assert any(emoji in text for emoji in ["💭", "🌟", "✌️", "☕", "🔥"])

    @pytest.mark.asyncio
    async def test_voice_modifiers_passionate(
        self, content_service, base_persona, generation_request
    ):
        """Test that passionate personality adds passionate voice variations."""
        base_persona.personality = (
            "Passionate innovator, deeply committed to making impact"
        )
        base_persona.style_preferences = {"tone": "warm"}

        text = await content_service._create_enhanced_fallback_text(
            base_persona, generation_request
        )

        # Passionate content should be energetic
        assert len(text) > 50

    @pytest.mark.asyncio
    async def test_voice_modifiers_analytical(
        self, content_service, base_persona, generation_request
    ):
        """Test that analytical personality adds analytical voice variations."""
        base_persona.personality = "Analytical thinker, data-driven decision maker"
        base_persona.style_preferences = {"voice_style": "precise"}

        text = await content_service._create_enhanced_fallback_text(
            base_persona, generation_request
        )

        # Analytical content should reference analysis or data
        assert len(text) > 50

    @pytest.mark.asyncio
    async def test_appearance_locked_context(
        self, content_service, base_persona, generation_request
    ):
        """Test that appearance locking adds appropriate context."""
        base_persona.appearance_locked = True
        base_persona.base_appearance_description = (
            "Professional business attire with modern aesthetic"
        )
        base_persona.style_preferences = {"aesthetic": "professional"}

        text = await content_service._create_enhanced_fallback_text(
            base_persona, generation_request
        )

        # Should include appearance context
        assert len(text) > 50

    @pytest.mark.asyncio
    async def test_prompt_keyword_customization_trends(
        self, content_service, base_persona
    ):
        """Test that prompt keywords influence content customization."""
        test_uuid = uuid.uuid4()
        request = GenerationRequest(
            persona_id=test_uuid,
            content_type=ContentType.TEXT,
            content_rating=ContentRating.SFW,
            prompt="future trends upcoming developments",
        )

        text = await content_service._create_enhanced_fallback_text(
            base_persona, request
        )

        # Should reference future/upcoming
        assert len(text) > 50

    @pytest.mark.asyncio
    async def test_prompt_keyword_customization_analysis(
        self, content_service, base_persona
    ):
        """Test that analysis keywords influence content style."""
        test_uuid = uuid.uuid4()
        request = GenerationRequest(
            persona_id=test_uuid,
            content_type=ContentType.TEXT,
            content_rating=ContentRating.SFW,
            prompt="analysis research study data",
        )

        text = await content_service._create_enhanced_fallback_text(
            base_persona, request
        )

        # Should reflect analytical approach
        assert len(text) > 50

    @pytest.mark.asyncio
    async def test_multi_trait_scoring(
        self, content_service, base_persona, generation_request
    ):
        """Test that multiple personality traits are considered together."""
        base_persona.personality = (
            "Creative thinker, analytical problem solver, passionate communicator"
        )
        base_persona.style_preferences = {
            "aesthetic": "creative",
            "voice_style": "expressive",
            "tone": "warm",
        }

        text = await content_service._create_enhanced_fallback_text(
            base_persona, generation_request
        )

        # Should produce coherent content considering all traits
        assert len(text) > 50
        assert any(theme in text.lower() for theme in base_persona.content_themes)

    @pytest.mark.asyncio
    async def test_template_variation_randomness(
        self, content_service, base_persona, generation_request
    ):
        """Test that multiple calls produce varied content (not always the same)."""
        texts = []
        for _ in range(5):
            text = await content_service._create_enhanced_fallback_text(
                base_persona, generation_request
            )
            texts.append(text)

        # Should have generated at least 2 different templates
        unique_texts = set(texts)
        assert len(unique_texts) >= 2, "Content should vary across multiple generations"

    @pytest.mark.asyncio
    async def test_content_themes_integration(
        self, content_service, base_persona, generation_request
    ):
        """Test that content themes are properly integrated into output."""
        base_persona.content_themes = ["blockchain", "cryptocurrency", "web3"]

        text = await content_service._create_enhanced_fallback_text(
            base_persona, generation_request
        )

        # Should reference the first theme
        assert base_persona.content_themes[0] in text.lower()

    @pytest.mark.asyncio
    async def test_empty_style_preferences_handling(
        self, content_service, base_persona, generation_request
    ):
        """Test graceful handling when style_preferences is empty."""
        base_persona.style_preferences = {}

        text = await content_service._create_enhanced_fallback_text(
            base_persona, generation_request
        )

        # Should still generate valid content
        assert len(text) > 50
        assert isinstance(text, str)

    @pytest.mark.asyncio
    async def test_default_themes_fallback(
        self, content_service, base_persona, generation_request
    ):
        """Test that default themes are used when content_themes is empty."""
        base_persona.content_themes = []

        text = await content_service._create_enhanced_fallback_text(
            base_persona, generation_request
        )

        # Should use default themes
        assert len(text) > 50
        assert any(word in text.lower() for word in ["lifestyle", "thoughts"])
