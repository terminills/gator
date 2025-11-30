"""
Unit tests for Content Moderation Pipeline Service.

Tests content analysis, review queue, and moderation workflow.
"""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

from backend.services.content_moderation_service import (
    ContentModerationPipeline,
    ContentType,
    ModerationAction,
    ModerationCategory,
    ModerationConfig,
    ModerationResult,
    ModerationSeverity,
    TextAnalysisResult,
    ImageAnalysisResult,
)


class TestModerationEnums:
    """Tests for moderation enumerations."""

    def test_moderation_category_values(self):
        """Test moderation category enum values."""
        assert ModerationCategory.ADULT == "adult"
        assert ModerationCategory.VIOLENCE == "violence"
        assert ModerationCategory.SAFE == "safe"
        assert ModerationCategory.SPAM == "spam"
        assert ModerationCategory.HATE_SPEECH == "hate_speech"

    def test_moderation_action_values(self):
        """Test moderation action enum values."""
        assert ModerationAction.APPROVE == "approve"
        assert ModerationAction.REJECT == "reject"
        assert ModerationAction.FLAG_FOR_REVIEW == "flag_for_review"
        assert ModerationAction.WARN == "warn"

    def test_moderation_severity_values(self):
        """Test moderation severity enum values."""
        assert ModerationSeverity.LOW == "low"
        assert ModerationSeverity.MEDIUM == "medium"
        assert ModerationSeverity.HIGH == "high"
        assert ModerationSeverity.CRITICAL == "critical"

    def test_content_type_values(self):
        """Test content type enum values."""
        assert ContentType.TEXT == "text"
        assert ContentType.IMAGE == "image"
        assert ContentType.VIDEO == "video"
        assert ContentType.AUDIO == "audio"


class TestModerationModels:
    """Tests for moderation data models."""

    def test_moderation_config_defaults(self):
        """Test ModerationConfig default values."""
        config = ModerationConfig()

        assert config.auto_approve_threshold == 0.9
        assert config.auto_reject_threshold == 0.1
        assert config.human_review_threshold == 0.7
        assert config.max_queue_size == 1000

    def test_moderation_result_creation(self):
        """Test ModerationResult model creation."""
        result = ModerationResult(
            content_id="test123",
            content_type=ContentType.TEXT,
            categories=[ModerationCategory.SAFE],
            severity=ModerationSeverity.LOW,
            confidence=0.95,
            recommended_action=ModerationAction.APPROVE,
        )

        assert result.content_id == "test123"
        assert result.content_type == ContentType.TEXT
        assert ModerationCategory.SAFE in result.categories
        assert result.requires_human_review is False

    def test_text_analysis_result(self):
        """Test TextAnalysisResult model."""
        result = TextAnalysisResult(
            toxicity_score=0.2,
            adult_score=0.1,
        )

        assert result.toxicity_score == 0.2
        assert result.adult_score == 0.1
        assert result.hate_speech_score == 0.0
        assert result.categories == []

    def test_image_analysis_result(self):
        """Test ImageAnalysisResult model."""
        result = ImageAnalysisResult(
            nudity_score=0.1,
            safe_score=0.9,
        )

        assert result.nudity_score == 0.1
        assert result.safe_score == 0.9
        assert result.violence_score == 0.0


class TestContentModerationPipeline:
    """Tests for ContentModerationPipeline service."""

    @pytest.fixture
    def mock_db_session(self):
        """Create mock database session."""
        session = AsyncMock()
        session.add = MagicMock()
        session.commit = AsyncMock()
        session.execute = AsyncMock()
        return session

    @pytest.fixture
    def moderation_service(self, mock_db_session):
        """Create moderation service instance."""
        return ContentModerationPipeline(mock_db_session)

    @pytest.mark.asyncio
    async def test_moderate_safe_text(self, moderation_service):
        """Test moderation of safe text content."""
        result = await moderation_service.moderate_content(
            content_id="test1",
            content_type=ContentType.TEXT,
            content_data="Hello, how are you today?",
        )

        assert result.content_type == ContentType.TEXT
        assert ModerationCategory.SAFE in result.categories
        assert result.recommended_action == ModerationAction.APPROVE
        assert result.severity == ModerationSeverity.LOW

    @pytest.mark.asyncio
    async def test_moderate_toxic_text(self, moderation_service):
        """Test moderation of toxic text content."""
        result = await moderation_service.moderate_content(
            content_id="test2",
            content_type=ContentType.TEXT,
            content_data="I hate you, you're stupid and worthless",
        )

        assert ModerationCategory.HARASSMENT in result.categories
        assert result.severity in [ModerationSeverity.MEDIUM, ModerationSeverity.HIGH]

    @pytest.mark.asyncio
    async def test_moderate_adult_text(self, moderation_service):
        """Test moderation of adult text content."""
        result = await moderation_service.moderate_content(
            content_id="test3",
            content_type=ContentType.TEXT,
            content_data="Check out this explicit nsfw content",
        )

        assert ModerationCategory.ADULT in result.categories
        assert result.severity == ModerationSeverity.HIGH

    @pytest.mark.asyncio
    async def test_moderate_spam_text(self, moderation_service):
        """Test moderation of spam text content."""
        result = await moderation_service.moderate_content(
            content_id="test4",
            content_type=ContentType.TEXT,
            content_data="Buy now! Click here for free money!",
        )

        assert ModerationCategory.SPAM in result.categories

    @pytest.mark.asyncio
    async def test_moderate_violence_text(self, moderation_service):
        """Test moderation of violent text content."""
        result = await moderation_service.moderate_content(
            content_id="test5",
            content_type=ContentType.TEXT,
            content_data="The murder scene was full of blood and gore",
        )

        assert ModerationCategory.VIOLENCE in result.categories

    @pytest.mark.asyncio
    async def test_moderate_hate_speech_text(self, moderation_service):
        """Test moderation of hate speech text content."""
        result = await moderation_service.moderate_content(
            content_id="test6",
            content_type=ContentType.TEXT,
            content_data="Those racist bigot supremacist groups are terrible",
        )

        assert ModerationCategory.HATE_SPEECH in result.categories

    @pytest.mark.asyncio
    async def test_moderate_safe_image(self, moderation_service):
        """Test moderation of safe image content."""
        result = await moderation_service.moderate_content(
            content_id="img1",
            content_type=ContentType.IMAGE,
            content_data="/path/to/safe_landscape.jpg",
        )

        assert result.content_type == ContentType.IMAGE
        assert ModerationCategory.SAFE in result.categories
        assert result.recommended_action == ModerationAction.APPROVE

    @pytest.mark.asyncio
    async def test_moderate_flagged_image(self, moderation_service):
        """Test moderation of flagged image content."""
        result = await moderation_service.moderate_content(
            content_id="img2",
            content_type=ContentType.IMAGE,
            content_data="/path/to/nsfw_adult_content.jpg",
        )

        assert ModerationCategory.ADULT in result.categories
        assert result.recommended_action == ModerationAction.FLAG_FOR_REVIEW

    @pytest.mark.asyncio
    async def test_moderate_video(self, moderation_service):
        """Test moderation of video content."""
        result = await moderation_service.moderate_content(
            content_id="vid1",
            content_type=ContentType.VIDEO,
            content_data="/path/to/safe_video.mp4",
        )

        assert result.content_type == ContentType.VIDEO
        assert ModerationCategory.SAFE in result.categories

    @pytest.mark.asyncio
    async def test_moderate_audio(self, moderation_service):
        """Test moderation of audio content."""
        result = await moderation_service.moderate_content(
            content_id="aud1",
            content_type=ContentType.AUDIO,
            content_data="/path/to/audio.mp3",
        )

        assert result.content_type == ContentType.AUDIO
        assert result.recommended_action == ModerationAction.APPROVE

    @pytest.mark.asyncio
    async def test_needs_human_review_high_severity(self, moderation_service):
        """Test that high severity content needs human review."""
        result = await moderation_service.moderate_content(
            content_id="test_review",
            content_type=ContentType.TEXT,
            content_data="I hate everyone, they should all die",
        )

        # High severity should trigger human review
        assert result.requires_human_review is True

    @pytest.mark.asyncio
    async def test_get_review_queue_empty(self, moderation_service, mock_db_session):
        """Test getting empty review queue."""
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = []
        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars
        mock_db_session.execute = AsyncMock(return_value=mock_result)

        items = await moderation_service.get_review_queue()

        assert items == []

    @pytest.mark.asyncio
    async def test_review_content_not_found(self, moderation_service, mock_db_session):
        """Test reviewing non-existent content."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db_session.execute = AsyncMock(return_value=mock_result)

        success = await moderation_service.review_content(
            queue_item_id="nonexistent",
            action=ModerationAction.APPROVE,
            reviewer_id="reviewer1",
        )

        assert success is False

    @pytest.mark.asyncio
    async def test_get_moderation_stats(self, moderation_service, mock_db_session):
        """Test getting moderation statistics."""
        # Mock count queries
        mock_db_session.execute = AsyncMock()
        mock_db_session.execute.side_effect = [
            MagicMock(scalar=MagicMock(return_value=100)),  # total
            MagicMock(scalar=MagicMock(return_value=25)),  # pending
            MagicMock(scalar=MagicMock(return_value=0.75)),  # avg confidence
        ]

        stats = await moderation_service.get_moderation_stats()

        assert "total_items" in stats
        assert "pending_review" in stats
        assert "queue_health" in stats


class TestModerationQueueModel:
    """Tests for moderation queue database model."""

    def test_moderation_queue_model_import(self):
        """Test importing ModerationQueueModel."""
        from backend.services.content_moderation_service import ModerationQueueModel

        assert ModerationQueueModel.__tablename__ == "moderation_queue"

    def test_moderation_queue_model_columns(self):
        """Test ModerationQueueModel has required columns."""
        from backend.services.content_moderation_service import ModerationQueueModel

        columns = [c.name for c in ModerationQueueModel.__table__.columns]
        assert "id" in columns
        assert "content_id" in columns
        assert "content_type" in columns
        assert "severity" in columns
        assert "confidence_score" in columns
        assert "status" in columns


class TestTextAnalysis:
    """Tests for text content analysis."""

    @pytest.fixture
    def mock_db_session(self):
        """Create mock database session."""
        return AsyncMock()

    @pytest.fixture
    def moderation_service(self, mock_db_session):
        """Create moderation service instance."""
        return ContentModerationPipeline(mock_db_session)

    @pytest.mark.asyncio
    async def test_empty_text(self, moderation_service):
        """Test analysis of empty text."""
        result = await moderation_service._analyze_text("test1", "", None)

        assert ModerationCategory.SAFE in result.categories
        assert result.recommended_action == ModerationAction.APPROVE

    @pytest.mark.asyncio
    async def test_mixed_content_text(self, moderation_service):
        """Test analysis of text with multiple flags."""
        result = await moderation_service._analyze_text(
            "test1",
            "I hate you, click here for explicit nude content",
            None,
        )

        # Should detect multiple categories
        assert len(result.categories) > 1
        assert result.severity in [ModerationSeverity.HIGH, ModerationSeverity.CRITICAL]

    @pytest.mark.asyncio
    async def test_confidence_scores(self, moderation_service):
        """Test that confidence scores are reasonable."""
        result = await moderation_service._analyze_text(
            "test1",
            "Hello, this is a normal message",
            None,
        )

        assert 0.0 <= result.confidence <= 1.0
        assert result.confidence >= 0.5  # Should be fairly confident for safe content
