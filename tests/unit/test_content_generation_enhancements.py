"""
Test Content Generation Enhancements

Tests for the enhanced content generation service with SFW/NSFW support
and new content types (video, voice, audio).
"""

import pytest
import uuid
from unittest.mock import Mock, AsyncMock
from datetime import datetime

from backend.services.content_generation_service import (
    ContentGenerationService,
    ContentModerationService,
    GenerationRequest,
)
from backend.models.content import ContentType, ContentRating, ModerationStatus
from backend.models.persona import PersonaModel


class TestContentModerationService:
    """Test content moderation and rating analysis."""

    def test_analyze_content_rating_sfw(self):
        """Test SFW content detection."""
        prompt = "Beautiful landscape photo with natural lighting"
        result = ContentModerationService.analyze_content_rating(prompt, "sfw")
        assert result == ContentRating.SFW

    def test_analyze_content_rating_nsfw(self):
        """Test NSFW content detection."""
        prompt = "Sexy portrait with provocative poses and lingerie"
        result = ContentModerationService.analyze_content_rating(prompt, "nsfw")
        assert result == ContentRating.NSFW

    def test_analyze_content_rating_moderate(self):
        """Test moderate content detection."""
        prompt = "Romantic sunset photo with suggestive atmosphere"
        result = ContentModerationService.analyze_content_rating(prompt, "both")
        assert result == ContentRating.MODERATE

    def test_platform_content_filter_instagram(self):
        """Test Instagram content filtering."""
        # SFW should be allowed
        assert ContentModerationService.platform_content_filter(
            ContentRating.SFW, "instagram"
        )

        # NSFW should be blocked
        assert not ContentModerationService.platform_content_filter(
            ContentRating.NSFW, "instagram"
        )

        # Moderate should be allowed
        assert ContentModerationService.platform_content_filter(
            ContentRating.MODERATE, "instagram"
        )

    def test_platform_content_filter_onlyfans(self):
        """Test OnlyFans content filtering (allows all types)."""
        assert ContentModerationService.platform_content_filter(
            ContentRating.SFW, "onlyfans"
        )
        assert ContentModerationService.platform_content_filter(
            ContentRating.NSFW, "onlyfans"
        )
        assert ContentModerationService.platform_content_filter(
            ContentRating.MODERATE, "onlyfans"
        )

    def test_platform_content_filter_with_persona_restrictions_sfw_only(self):
        """Test per-persona platform restrictions - SFW only override."""
        # Persona restricts Instagram to SFW only (more restrictive than default)
        restrictions = {"instagram": "sfw_only"}

        # SFW should be allowed
        assert ContentModerationService.platform_content_filter(
            ContentRating.SFW, "instagram", restrictions
        )

        # MODERATE should be blocked (normally allowed on Instagram)
        assert not ContentModerationService.platform_content_filter(
            ContentRating.MODERATE, "instagram", restrictions
        )

        # NSFW should be blocked
        assert not ContentModerationService.platform_content_filter(
            ContentRating.NSFW, "instagram", restrictions
        )

    def test_platform_content_filter_with_persona_restrictions_moderate_allowed(self):
        """Test per-persona platform restrictions - moderate allowed."""
        # Persona allows moderate content on Facebook (more permissive than default)
        restrictions = {"facebook": "moderate_allowed"}

        # SFW should be allowed
        assert ContentModerationService.platform_content_filter(
            ContentRating.SFW, "facebook", restrictions
        )

        # MODERATE should be allowed (normally blocked on Facebook)
        assert ContentModerationService.platform_content_filter(
            ContentRating.MODERATE, "facebook", restrictions
        )

        # NSFW should still be blocked
        assert not ContentModerationService.platform_content_filter(
            ContentRating.NSFW, "facebook", restrictions
        )

    def test_platform_content_filter_with_persona_restrictions_all_content(self):
        """Test per-persona platform restrictions - all content types allowed."""
        # Persona allows all content types on Instagram (NSFW override)
        restrictions = {"instagram": "both"}

        # All content types should be allowed
        assert ContentModerationService.platform_content_filter(
            ContentRating.SFW, "instagram", restrictions
        )
        assert ContentModerationService.platform_content_filter(
            ContentRating.MODERATE, "instagram", restrictions
        )
        assert ContentModerationService.platform_content_filter(
            ContentRating.NSFW, "instagram", restrictions
        )

        # Test with "all" as alternative keyword
        restrictions_all = {"instagram": "all"}
        assert ContentModerationService.platform_content_filter(
            ContentRating.NSFW, "instagram", restrictions_all
        )

    def test_platform_content_filter_persona_restrictions_fallback(self):
        """Test that unspecified platforms fall back to global policies."""
        # Persona has restrictions only for Instagram
        restrictions = {"instagram": "sfw_only"}

        # OnlyFans should use global policy (allows NSFW)
        assert ContentModerationService.platform_content_filter(
            ContentRating.NSFW, "onlyfans", restrictions
        )

        # Facebook should use global policy (blocks NSFW)
        assert not ContentModerationService.platform_content_filter(
            ContentRating.NSFW, "facebook", restrictions
        )


class TestEnhancedContentGeneration:
    """Test enhanced content generation with new content types."""

    @pytest.fixture
    def mock_persona(self):
        """Create a mock persona for testing."""
        persona = Mock(spec=PersonaModel)
        persona.id = uuid.uuid4()
        persona.name = "TestPersona"
        persona.appearance = "attractive young woman"
        persona.personality = "confident and outgoing"
        persona.content_themes = ["lifestyle", "fashion"]
        persona.style_preferences = {
            "visual_style": "realistic",
            "lighting": "natural",
            "voice_style": "warm",
            "voice_pitch": "medium",
        }
        persona.default_content_rating = "sfw"
        persona.allowed_content_ratings = ["sfw", "nsfw"]
        persona.platform_restrictions = {}  # Default to no restrictions
        persona.generation_count = 0
        return persona

    @pytest.fixture
    def mock_db_session(self):
        """Create a mock database session."""
        session = AsyncMock()
        return session

    @pytest.fixture
    def content_service(self, mock_db_session, tmp_path):
        """Create content generation service with temp directory."""
        return ContentGenerationService(mock_db_session, str(tmp_path))

    @pytest.mark.asyncio
    async def test_generate_video_content(self, content_service, mock_persona):
        """Test video content generation."""
        request = GenerationRequest(
            persona_id=mock_persona.id,
            content_type=ContentType.VIDEO,
            content_rating=ContentRating.SFW,
            prompt="Create an engaging lifestyle video",
            quality="high",
        )

        # Mock the persona retrieval
        content_service.db.execute = AsyncMock()
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = mock_persona
        content_service.db.execute.return_value = mock_result

        # Mock database operations
        content_service.db.add = Mock()
        content_service.db.commit = AsyncMock()
        content_service.db.refresh = AsyncMock()

        result = await content_service.generate_content(request)

        assert result is not None
        assert hasattr(result, "content_type")

    @pytest.mark.asyncio
    async def test_generate_voice_content(self, content_service, mock_persona):
        """Test voice content generation."""
        request = GenerationRequest(
            persona_id=mock_persona.id,
            content_type=ContentType.VOICE,
            content_rating=ContentRating.SFW,
            prompt="Record a friendly greeting message",
            quality="high",
        )

        # Mock the persona retrieval
        content_service.db.execute = AsyncMock()
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = mock_persona
        content_service.db.execute.return_value = mock_result

        # Mock database operations
        content_service.db.add = Mock()
        content_service.db.commit = AsyncMock()
        content_service.db.refresh = AsyncMock()

        result = await content_service.generate_content(request)

        assert result is not None
        assert hasattr(result, "content_type")

    @pytest.mark.asyncio
    async def test_generate_audio_content(self, content_service, mock_persona):
        """Test audio content generation."""
        request = GenerationRequest(
            persona_id=mock_persona.id,
            content_type=ContentType.AUDIO,
            content_rating=ContentRating.SFW,
            prompt="Create background music for lifestyle content",
            quality="high",
        )

        # Mock the persona retrieval
        content_service.db.execute = AsyncMock()
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = mock_persona
        content_service.db.execute.return_value = mock_result

        # Mock database operations
        content_service.db.add = Mock()
        content_service.db.commit = AsyncMock()
        content_service.db.refresh = AsyncMock()

        result = await content_service.generate_content(request)

        assert result is not None
        assert hasattr(result, "content_type")

    @pytest.mark.asyncio
    async def test_content_rating_validation(self, content_service, mock_persona):
        """Test content rating validation against persona settings."""
        # Test with allowed rating
        assert await content_service._validate_content_rating(
            mock_persona, ContentRating.SFW
        )
        assert await content_service._validate_content_rating(
            mock_persona, ContentRating.NSFW
        )

        # Test with persona that only allows SFW
        mock_persona.allowed_content_ratings = ["sfw"]
        assert await content_service._validate_content_rating(
            mock_persona, ContentRating.SFW
        )
        assert not await content_service._validate_content_rating(
            mock_persona, ContentRating.NSFW
        )

    @pytest.mark.asyncio
    async def test_platform_adaptations(self, content_service, mock_persona):
        """Test platform-specific content adaptations."""
        content_data = {"width": 1920, "height": 1080}
        target_platforms = ["instagram", "facebook", "onlyfans"]

        # Mock persona with no specific platform restrictions
        mock_persona.platform_restrictions = {}

        adaptations = await content_service._create_platform_adaptations(
            mock_persona, content_data, ContentRating.SFW, target_platforms
        )

        assert "instagram" in adaptations
        assert "facebook" in adaptations
        assert "onlyfans" in adaptations

        # Instagram should have crop ratio adaptation
        assert adaptations["instagram"]["crop_ratio"] == "1:1"
        assert adaptations["instagram"]["modified_for_platform"] is True

    @pytest.mark.asyncio
    async def test_nsfw_platform_blocking(self, content_service, mock_persona):
        """Test NSFW content blocking on restrictive platforms."""
        content_data = {}
        target_platforms = ["instagram", "onlyfans"]

        # Mock persona with no specific platform restrictions (uses global policies)
        mock_persona.platform_restrictions = {}

        adaptations = await content_service._create_platform_adaptations(
            mock_persona, content_data, ContentRating.NSFW, target_platforms
        )

        # Instagram should block NSFW content
        assert adaptations["instagram"]["status"] == "blocked"
        assert "not allowed" in adaptations["instagram"]["reason"]

        # OnlyFans should allow NSFW content
        assert adaptations["onlyfans"]["status"] == "approved"

    @pytest.mark.asyncio
    async def test_nsfw_allowed_with_persona_override(
        self, content_service, mock_persona
    ):
        """Test NSFW content allowed on Instagram with persona override."""
        content_data = {}
        target_platforms = ["instagram", "onlyfans"]

        # Persona allows NSFW on Instagram via platform_restrictions
        mock_persona.platform_restrictions = {"instagram": "both"}

        adaptations = await content_service._create_platform_adaptations(
            mock_persona, content_data, ContentRating.NSFW, target_platforms
        )

        # Instagram should now allow NSFW content due to persona override
        assert adaptations["instagram"]["status"] == "approved"

        # OnlyFans should still allow NSFW content
        assert adaptations["onlyfans"]["status"] == "approved"

    @pytest.mark.asyncio
    async def test_mixed_platform_restrictions(self, content_service, mock_persona):
        """Test mixed platform restrictions for different content ratings."""
        content_data = {}

        # Persona configuration:
        # - Instagram: allows all (NSFW override)
        # - Facebook: allows moderate (more permissive)
        # - Twitter: only SFW (more restrictive)
        mock_persona.platform_restrictions = {
            "instagram": "both",
            "facebook": "moderate_allowed",
            "twitter": "sfw_only",
        }

        # Test NSFW content
        target_platforms = ["instagram", "facebook", "twitter", "onlyfans"]
        adaptations = await content_service._create_platform_adaptations(
            mock_persona, content_data, ContentRating.NSFW, target_platforms
        )

        # Instagram: allowed (persona override)
        assert adaptations["instagram"]["status"] == "approved"

        # Facebook: blocked (moderate_allowed doesn't include NSFW)
        assert adaptations["facebook"]["status"] == "blocked"

        # Twitter: blocked (sfw_only)
        assert adaptations["twitter"]["status"] == "blocked"

        # OnlyFans: allowed (global policy)
        assert adaptations["onlyfans"]["status"] == "approved"

        # Test MODERATE content
        adaptations_moderate = await content_service._create_platform_adaptations(
            mock_persona, content_data, ContentRating.MODERATE, target_platforms
        )

        # Instagram: allowed (both)
        assert adaptations_moderate["instagram"]["status"] == "approved"

        # Facebook: allowed (moderate_allowed)
        assert adaptations_moderate["facebook"]["status"] == "approved"

        # Twitter: blocked (sfw_only)
        assert adaptations_moderate["twitter"]["status"] == "blocked"

        # OnlyFans: allowed (global policy)
        assert adaptations_moderate["onlyfans"]["status"] == "approved"


class TestPersonaContentRatingFields:
    """Test new persona content rating fields."""

    def test_content_rating_enum(self):
        """Test content rating enumeration."""
        assert ContentRating.SFW == "sfw"
        assert ContentRating.NSFW == "nsfw"
        assert ContentRating.MODERATE == "moderate"

    def test_generation_request_validation(self):
        """Test generation request validation."""
        request = GenerationRequest(
            persona_id=uuid.uuid4(),
            content_type=ContentType.IMAGE,
            content_rating=ContentRating.SFW,
            quality="high",
        )

        assert request.persona_id is not None
        assert request.content_type == ContentType.IMAGE
        assert request.content_rating == ContentRating.SFW
        assert request.quality == "high"

    def test_invalid_quality_validation(self):
        """Test validation of invalid quality parameter."""
        import pydantic_core

        with pytest.raises(pydantic_core.ValidationError) as exc_info:
            GenerationRequest(
                persona_id=uuid.uuid4(),
                content_type=ContentType.IMAGE,
                quality="invalid_quality",
            )
        assert "Quality must be one of" in str(exc_info.value)


class TestPlatformRestrictionsValidation:
    """Test platform restrictions validation in PersonaCreate and PersonaUpdate models."""

    def test_valid_platform_restrictions(self):
        """Test that valid platform restrictions are accepted."""
        from backend.models.persona import PersonaCreate

        # All valid restriction values
        valid_restrictions = {
            "instagram": "sfw_only",
            "facebook": "moderate_allowed",
            "twitter": "both",
            "onlyfans": "all",
        }

        persona_data = PersonaCreate(
            name="Test Persona",
            appearance="Test appearance description for the persona",
            personality="Test personality traits and characteristics",
            platform_restrictions=valid_restrictions,
        )

        assert persona_data.platform_restrictions == valid_restrictions

    def test_invalid_platform_restriction_value(self):
        """Test that invalid restriction values are rejected."""
        from backend.models.persona import PersonaCreate
        import pydantic_core

        invalid_restrictions = {"instagram": "invalid_value"}

        with pytest.raises(pydantic_core.ValidationError) as exc_info:
            PersonaCreate(
                name="Test Persona",
                appearance="Test appearance description for the persona",
                personality="Test personality traits and characteristics",
                platform_restrictions=invalid_restrictions,
            )

        error_message = str(exc_info.value)
        assert "Invalid restriction" in error_message
        assert "invalid_value" in error_message

    def test_empty_platform_restrictions(self):
        """Test that empty platform restrictions are valid."""
        from backend.models.persona import PersonaCreate

        persona_data = PersonaCreate(
            name="Test Persona",
            appearance="Test appearance description for the persona",
            personality="Test personality traits and characteristics",
            platform_restrictions={},
        )

        assert persona_data.platform_restrictions == {}

    def test_persona_update_valid_restrictions(self):
        """Test that PersonaUpdate accepts valid platform restrictions."""
        from backend.models.persona import PersonaUpdate

        update_data = PersonaUpdate(
            platform_restrictions={"instagram": "both", "facebook": "moderate_allowed"}
        )

        assert update_data.platform_restrictions is not None
        assert "instagram" in update_data.platform_restrictions

    def test_persona_update_invalid_restrictions(self):
        """Test that PersonaUpdate rejects invalid platform restrictions."""
        from backend.models.persona import PersonaUpdate
        import pydantic_core

        with pytest.raises(pydantic_core.ValidationError) as exc_info:
            PersonaUpdate(platform_restrictions={"instagram": "wrong_value"})

        error_message = str(exc_info.value)
        assert "Invalid restriction" in error_message
