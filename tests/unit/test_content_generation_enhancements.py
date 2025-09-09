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
    GenerationRequest
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
        assert ContentModerationService.platform_content_filter(ContentRating.SFW, "instagram")
        
        # NSFW should be blocked
        assert not ContentModerationService.platform_content_filter(ContentRating.NSFW, "instagram")
        
        # Moderate should be allowed
        assert ContentModerationService.platform_content_filter(ContentRating.MODERATE, "instagram")
    
    def test_platform_content_filter_onlyfans(self):
        """Test OnlyFans content filtering (allows all types)."""
        assert ContentModerationService.platform_content_filter(ContentRating.SFW, "onlyfans")
        assert ContentModerationService.platform_content_filter(ContentRating.NSFW, "onlyfans")
        assert ContentModerationService.platform_content_filter(ContentRating.MODERATE, "onlyfans")


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
            "voice_pitch": "medium"
        }
        persona.default_content_rating = "sfw"
        persona.allowed_content_ratings = ["sfw", "nsfw"]
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
            quality="high"
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
        assert hasattr(result, 'content_type')
    
    @pytest.mark.asyncio 
    async def test_generate_voice_content(self, content_service, mock_persona):
        """Test voice content generation."""
        request = GenerationRequest(
            persona_id=mock_persona.id,
            content_type=ContentType.VOICE,
            content_rating=ContentRating.SFW,
            prompt="Record a friendly greeting message",
            quality="high"
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
        assert hasattr(result, 'content_type')
    
    @pytest.mark.asyncio
    async def test_generate_audio_content(self, content_service, mock_persona):
        """Test audio content generation."""
        request = GenerationRequest(
            persona_id=mock_persona.id,
            content_type=ContentType.AUDIO,
            content_rating=ContentRating.SFW,
            prompt="Create background music for lifestyle content",
            quality="high"
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
        assert hasattr(result, 'content_type')
    
    @pytest.mark.asyncio
    async def test_content_rating_validation(self, content_service, mock_persona):
        """Test content rating validation against persona settings."""
        # Test with allowed rating
        assert await content_service._validate_content_rating(mock_persona, ContentRating.SFW)
        assert await content_service._validate_content_rating(mock_persona, ContentRating.NSFW)
        
        # Test with persona that only allows SFW
        mock_persona.allowed_content_ratings = ["sfw"]
        assert await content_service._validate_content_rating(mock_persona, ContentRating.SFW)
        assert not await content_service._validate_content_rating(mock_persona, ContentRating.NSFW)
    
    @pytest.mark.asyncio
    async def test_platform_adaptations(self, content_service):
        """Test platform-specific content adaptations."""
        content_data = {"width": 1920, "height": 1080}
        target_platforms = ["instagram", "facebook", "onlyfans"]
        
        adaptations = await content_service._create_platform_adaptations(
            content_data, ContentRating.SFW, target_platforms
        )
        
        assert "instagram" in adaptations
        assert "facebook" in adaptations
        assert "onlyfans" in adaptations
        
        # Instagram should have crop ratio adaptation
        assert adaptations["instagram"]["crop_ratio"] == "1:1"
        assert adaptations["instagram"]["modified_for_platform"] is True
    
    @pytest.mark.asyncio
    async def test_nsfw_platform_blocking(self, content_service):
        """Test NSFW content blocking on restrictive platforms."""
        content_data = {}
        target_platforms = ["instagram", "onlyfans"]
        
        adaptations = await content_service._create_platform_adaptations(
            content_data, ContentRating.NSFW, target_platforms
        )
        
        # Instagram should block NSFW content
        assert adaptations["instagram"]["status"] == "blocked"
        assert "not allowed" in adaptations["instagram"]["reason"]
        
        # OnlyFans should allow NSFW content
        assert adaptations["onlyfans"]["status"] == "approved"


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
            quality="high"
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
                quality="invalid_quality"
            )
        assert "Quality must be one of" in str(exc_info.value)