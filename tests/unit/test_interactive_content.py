"""
Tests for Interactive Content Service

Tests the interactive content management functionality.
"""

import pytest
from datetime import datetime, timedelta
from uuid import uuid4

from backend.services.interactive_content_service import InteractiveContentService
from backend.models.interactive_content import (
    InteractiveContentType,
    InteractiveContentStatus,
)


@pytest.mark.asyncio
async def test_create_poll(db_session, test_persona):
    """Test creating a poll."""
    service = InteractiveContentService(db_session)

    content = await service.create_content(
        persona_id=str(test_persona.id),
        content_type=InteractiveContentType.POLL,
        title="Favorite workout?",
        question="What's your favorite type of workout?",
        options=[
            {"text": "Running"},
            {"text": "Yoga"},
            {"text": "Weight training"},
            {"text": "Swimming"},
        ],
    )

    assert content is not None
    assert content.content_type == InteractiveContentType.POLL.value
    assert content.title == "Favorite workout?"
    assert len(content.options) == 4
    assert content.options[0]["votes"] == 0
    assert content.status == InteractiveContentStatus.DRAFT.value


@pytest.mark.asyncio
async def test_create_story_with_expiration(db_session, test_persona):
    """Test creating a story with automatic 24-hour expiration."""
    service = InteractiveContentService(db_session)

    content = await service.create_content(
        persona_id=str(test_persona.id),
        content_type=InteractiveContentType.STORY,
        title="My morning routine",
        media_url="https://example.com/story.jpg",
    )

    assert content is not None
    assert content.content_type == InteractiveContentType.STORY.value
    assert content.expires_at is not None
    # Should expire in approximately 24 hours
    time_diff = (content.expires_at - datetime.utcnow()).total_seconds()
    assert 23.5 * 3600 < time_diff < 24.5 * 3600


@pytest.mark.asyncio
async def test_publish_content(db_session, test_persona):
    """Test publishing interactive content."""
    service = InteractiveContentService(db_session)

    content = await service.create_content(
        persona_id=str(test_persona.id),
        content_type=InteractiveContentType.POLL,
        title="Test poll",
        question="Test question?",
        options=[{"text": "Yes"}, {"text": "No"}],
    )

    published = await service.publish_content(str(content.id))

    assert published is not None
    assert published.status == InteractiveContentStatus.ACTIVE.value
    assert published.published_at is not None


@pytest.mark.asyncio
async def test_submit_poll_response(db_session, test_persona):
    """Test submitting a response to a poll."""
    service = InteractiveContentService(db_session)

    # Create and publish poll
    content = await service.create_content(
        persona_id=str(test_persona.id),
        content_type=InteractiveContentType.POLL,
        title="Test poll",
        question="Yes or no?",
        options=[{"text": "Yes"}, {"text": "No"}],
    )
    await service.publish_content(str(content.id))

    # Submit response
    response = await service.submit_response(
        content_id=str(content.id),
        response_data={"option_id": 1},
    )

    assert response is not None
    assert response.response_data["option_id"] == 1

    # Check that vote count updated
    updated_content = await service.get_content(str(content.id))
    assert updated_content.response_count == 1
    assert updated_content.options[0]["votes"] == 1
    assert updated_content.options[0]["percentage"] == 100.0


@pytest.mark.asyncio
async def test_list_content_by_persona(db_session, test_persona):
    """Test listing content filtered by persona."""
    service = InteractiveContentService(db_session)

    # Create multiple content items
    await service.create_content(
        persona_id=str(test_persona.id),
        content_type=InteractiveContentType.POLL,
        title="Poll 1",
    )
    await service.create_content(
        persona_id=str(test_persona.id),
        content_type=InteractiveContentType.STORY,
        title="Story 1",
    )

    contents = await service.list_content(persona_id=str(test_persona.id))

    assert len(contents) == 2


@pytest.mark.asyncio
async def test_list_content_by_type(db_session, test_persona):
    """Test listing content filtered by type."""
    service = InteractiveContentService(db_session)

    # Create different types
    await service.create_content(
        persona_id=str(test_persona.id),
        content_type=InteractiveContentType.POLL,
        title="Poll 1",
    )
    await service.create_content(
        persona_id=str(test_persona.id),
        content_type=InteractiveContentType.STORY,
        title="Story 1",
    )

    polls = await service.list_content(
        persona_id=str(test_persona.id), content_type=InteractiveContentType.POLL
    )

    assert len(polls) == 1
    assert polls[0].content_type == InteractiveContentType.POLL.value


@pytest.mark.asyncio
async def test_get_content_stats(db_session, test_persona):
    """Test getting content statistics."""
    service = InteractiveContentService(db_session)

    # Create poll
    content = await service.create_content(
        persona_id=str(test_persona.id),
        content_type=InteractiveContentType.POLL,
        title="Test poll",
        question="Choose one",
        options=[{"text": "A"}, {"text": "B"}, {"text": "C"}],
    )

    # Add some responses
    await service.submit_response(str(content.id), {"option_id": 1})
    await service.submit_response(str(content.id), {"option_id": 1})
    await service.submit_response(str(content.id), {"option_id": 2})

    # Increment view count
    await service.increment_view_count(str(content.id))
    await service.increment_view_count(str(content.id))
    await service.increment_view_count(str(content.id))
    await service.increment_view_count(str(content.id))
    await service.increment_view_count(str(content.id))

    stats = await service.get_content_stats(str(content.id))

    assert stats["total_views"] == 5
    assert stats["total_responses"] == 3
    assert stats["response_rate"] == 60.0  # 3/5 * 100
    assert len(stats["top_options"]) == 3
    assert stats["top_options"][0]["votes"] == 2  # Option 1 has most votes


@pytest.mark.asyncio
async def test_update_content(db_session, test_persona):
    """Test updating interactive content."""
    service = InteractiveContentService(db_session)

    content = await service.create_content(
        persona_id=str(test_persona.id),
        content_type=InteractiveContentType.POLL,
        title="Original title",
    )

    updated = await service.update_content(
        str(content.id), title="Updated title", description="New description"
    )

    assert updated is not None
    assert updated.title == "Updated title"
    assert updated.description == "New description"


@pytest.mark.asyncio
async def test_delete_content(db_session, test_persona):
    """Test deleting interactive content."""
    service = InteractiveContentService(db_session)

    content = await service.create_content(
        persona_id=str(test_persona.id),
        content_type=InteractiveContentType.POLL,
        title="To be deleted",
    )

    deleted = await service.delete_content(str(content.id))
    assert deleted is True

    # Verify it's gone
    retrieved = await service.get_content(str(content.id))
    assert retrieved is None


@pytest.mark.asyncio
async def test_expire_old_content(db_session, test_persona):
    """Test expiring old content."""
    service = InteractiveContentService(db_session)

    # Create story that expired 1 hour ago
    past_time = datetime.utcnow() - timedelta(hours=1)
    content = await service.create_content(
        persona_id=str(test_persona.id),
        content_type=InteractiveContentType.STORY,
        title="Expired story",
        expires_at=past_time,
    )
    await service.publish_content(str(content.id))

    # Run expiration
    expired_count = await service.expire_old_content()

    assert expired_count == 1

    # Verify status changed
    expired_content = await service.get_content(str(content.id))
    assert expired_content.status == InteractiveContentStatus.EXPIRED.value


@pytest.mark.asyncio
async def test_exclude_expired_from_listing(db_session, test_persona):
    """Test that expired content is excluded from listings by default."""
    service = InteractiveContentService(db_session)

    # Create expired and active content
    past_time = datetime.utcnow() - timedelta(hours=1)
    expired = await service.create_content(
        persona_id=str(test_persona.id),
        content_type=InteractiveContentType.STORY,
        title="Expired",
        expires_at=past_time,
    )

    active = await service.create_content(
        persona_id=str(test_persona.id),
        content_type=InteractiveContentType.STORY,
        title="Active",
        expires_at=datetime.utcnow() + timedelta(hours=1),
    )

    # List without expired
    contents = await service.list_content(
        persona_id=str(test_persona.id), include_expired=False
    )

    assert len(contents) == 1
    assert contents[0].id == active.id

    # List with expired
    contents_with_expired = await service.list_content(
        persona_id=str(test_persona.id), include_expired=True
    )

    assert len(contents_with_expired) == 2
