"""
Unit tests for Scheduled Post functionality.

Tests the ScheduledPostService and ScheduledPostModel.
"""

import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from backend.models.scheduled_post import (
    ScheduledPostCreate,
    ScheduledPostModel,
    ScheduledPostResponse,
    ScheduledPostStatus,
    ScheduledPostUpdate,
)
from backend.services.scheduled_post_service import ScheduledPostService


class TestScheduledPostModel:
    """Test the ScheduledPostModel."""

    def test_scheduled_post_status_enum(self):
        """Test ScheduledPostStatus enum values."""
        assert ScheduledPostStatus.PENDING.value == "pending"
        assert ScheduledPostStatus.PROCESSING.value == "processing"
        assert ScheduledPostStatus.PUBLISHED.value == "published"
        assert ScheduledPostStatus.FAILED.value == "failed"
        assert ScheduledPostStatus.CANCELLED.value == "cancelled"
        assert ScheduledPostStatus.PAUSED.value == "paused"

    def test_scheduled_post_create_model(self):
        """Test ScheduledPostCreate Pydantic model."""
        content_id = uuid4()
        persona_id = uuid4()
        future_time = datetime.now(timezone.utc) + timedelta(hours=1)

        data = ScheduledPostCreate(
            content_id=content_id,
            persona_id=persona_id,
            platform="instagram",
            scheduled_at=future_time,
            caption="Test caption",
            hashtags=["test", "post"],
            priority=5,
        )

        assert data.content_id == content_id
        assert data.persona_id == persona_id
        assert data.platform == "instagram"
        assert data.scheduled_at == future_time
        assert data.caption == "Test caption"
        assert data.hashtags == ["test", "post"]
        assert data.priority == 5
        assert data.is_recurring is False
        assert data.max_retries == 3

    def test_scheduled_post_update_model(self):
        """Test ScheduledPostUpdate Pydantic model with partial update."""
        update = ScheduledPostUpdate(
            caption="Updated caption",
            priority=10,
        )

        assert update.caption == "Updated caption"
        assert update.priority == 10
        assert update.scheduled_at is None
        assert update.status is None

    def test_scheduled_post_response_model(self):
        """Test ScheduledPostResponse Pydantic model."""
        post_id = uuid4()
        content_id = uuid4()
        persona_id = uuid4()
        now = datetime.now(timezone.utc)

        response = ScheduledPostResponse(
            id=post_id,
            content_id=content_id,
            persona_id=persona_id,
            platform="twitter",
            scheduled_at=now,
            status="pending",
            retry_count=0,
            max_retries=3,
            priority=0,
            is_recurring=False,
            created_at=now,
            updated_at=now,
        )

        assert response.id == post_id
        assert response.platform == "twitter"
        assert response.status == "pending"
        assert response.retry_count == 0


class TestScheduledPostService:
    """Test the ScheduledPostService."""

    @pytest.fixture
    def mock_db_session(self):
        """Create a mock database session."""
        session = AsyncMock()
        session.execute = AsyncMock()
        session.commit = AsyncMock()
        session.refresh = AsyncMock()
        session.add = MagicMock()
        return session

    @pytest.fixture
    def service(self, mock_db_session):
        """Create a ScheduledPostService with mock session."""
        return ScheduledPostService(mock_db_session)

    @pytest.mark.asyncio
    async def test_create_scheduled_post_validates_future_time(
        self, service, mock_db_session
    ):
        """Test that creating a scheduled post requires future time."""
        past_time = datetime.now(timezone.utc) - timedelta(hours=1)

        data = ScheduledPostCreate(
            content_id=uuid4(),
            persona_id=uuid4(),
            platform="instagram",
            scheduled_at=past_time,
        )

        with pytest.raises(ValueError, match="future"):
            await service.create_scheduled_post(data)

    @pytest.mark.asyncio
    async def test_get_scheduled_post_returns_none_for_missing(
        self, service, mock_db_session
    ):
        """Test get_scheduled_post returns None for non-existent post."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db_session.execute.return_value = mock_result

        result = await service.get_scheduled_post(uuid4())

        assert result is None

    @pytest.mark.asyncio
    async def test_cancel_celery_task_handles_missing_task(self, service):
        """Test _cancel_celery_task handles None task ID gracefully."""
        result = await service._cancel_celery_task(None)
        assert result is True

    @pytest.mark.asyncio
    @patch("backend.services.scheduled_post_service.ScheduledPostService._schedule_celery_task")
    @patch("backend.services.scheduled_post_service.ScheduledPostService._cancel_celery_task")
    async def test_pause_scheduled_post(
        self, mock_cancel, mock_schedule, service, mock_db_session
    ):
        """Test pausing a scheduled post."""
        post_id = uuid4()
        mock_post = MagicMock()
        mock_post.id = post_id
        mock_post.status = ScheduledPostStatus.PENDING.value
        mock_post.celery_task_id = "task-123"

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_post
        mock_db_session.execute.return_value = mock_result
        mock_cancel.return_value = True

        # We need to mock model_validate to return a proper response
        with patch.object(ScheduledPostResponse, 'model_validate') as mock_validate:
            mock_validate.return_value = ScheduledPostResponse(
                id=post_id,
                content_id=uuid4(),
                persona_id=uuid4(),
                platform="instagram",
                scheduled_at=datetime.now(timezone.utc),
                status=ScheduledPostStatus.PAUSED.value,
                retry_count=0,
                max_retries=3,
                priority=0,
                is_recurring=False,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )

            result = await service.pause_scheduled_post(post_id)

            assert result is not None
            mock_cancel.assert_called_once_with("task-123")

    def test_create_model_default_values(self):
        """Test ScheduledPostCreate default values."""
        data = ScheduledPostCreate(
            content_id=uuid4(),
            persona_id=uuid4(),
            platform="facebook",
            scheduled_at=datetime.now(timezone.utc) + timedelta(hours=1),
        )

        assert data.hashtags == []
        assert data.priority == 0
        assert data.is_recurring is False
        assert data.max_retries == 3
        assert data.caption is None
        assert data.platform_specific_data is None


class TestScheduledPostStats:
    """Test ScheduledPostStats model."""

    def test_stats_model(self):
        """Test ScheduledPostStats creation."""
        from backend.models.scheduled_post import ScheduledPostStats

        stats = ScheduledPostStats(
            total_scheduled=100,
            pending=50,
            processing=5,
            published=30,
            failed=10,
            cancelled=3,
            paused=2,
            upcoming_24h=20,
            by_platform={"instagram": 60, "twitter": 40},
            by_persona={"persona-1": 30, "persona-2": 70},
        )

        assert stats.total_scheduled == 100
        assert stats.pending == 50
        assert stats.published == 30
        assert stats.upcoming_24h == 20
        assert stats.by_platform["instagram"] == 60


class TestScheduledPostListResponse:
    """Test ScheduledPostListResponse model."""

    def test_list_response_model(self):
        """Test ScheduledPostListResponse creation."""
        from backend.models.scheduled_post import ScheduledPostListResponse

        response = ScheduledPostListResponse(
            posts=[],
            total=0,
            page=1,
            page_size=20,
            total_pages=0,
        )

        assert response.posts == []
        assert response.total == 0
        assert response.page == 1
        assert response.page_size == 20
        assert response.total_pages == 0
