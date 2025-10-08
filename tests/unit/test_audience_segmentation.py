"""
Tests for Audience Segmentation Service

Tests the audience segmentation and personalized content functionality.
"""

import pytest
from uuid import uuid4

from backend.services.audience_segment_service import AudienceSegmentService
from backend.models.audience_segment import SegmentStatus, PersonalizationStrategy


@pytest.mark.asyncio
async def test_create_segment(db_session, test_persona):
    """Test creating an audience segment."""
    service = AudienceSegmentService(db_session)
    
    segment = await service.create_segment(
        persona_id=str(test_persona.id),
        segment_name="Tech Enthusiasts",
        criteria={
            "age_range": [25, 45],
            "interests": ["technology", "gadgets", "AI"],
            "engagement_level": "high"
        },
        description="Users interested in technology",
        strategy=PersonalizationStrategy.HYBRID,
    )
    
    assert segment is not None
    assert segment.segment_name == "Tech Enthusiasts"
    assert segment.criteria["interests"] == ["technology", "gadgets", "AI"]
    assert segment.status == SegmentStatus.ACTIVE.value


@pytest.mark.asyncio
async def test_get_segment(db_session, test_persona):
    """Test getting a segment by ID."""
    service = AudienceSegmentService(db_session)
    
    created = await service.create_segment(
        persona_id=str(test_persona.id),
        segment_name="Test Segment",
        criteria={"test": True},
    )
    
    retrieved = await service.get_segment(str(created.id))
    
    assert retrieved is not None
    assert retrieved.id == created.id
    assert retrieved.segment_name == "Test Segment"


@pytest.mark.asyncio
async def test_list_segments(db_session, test_persona):
    """Test listing segments."""
    service = AudienceSegmentService(db_session)
    
    # Create multiple segments
    await service.create_segment(
        persona_id=str(test_persona.id),
        segment_name="Segment 1",
        criteria={"type": "a"},
    )
    await service.create_segment(
        persona_id=str(test_persona.id),
        segment_name="Segment 2",
        criteria={"type": "b"},
    )
    
    segments = await service.list_segments(persona_id=str(test_persona.id))
    
    assert len(segments) == 2


@pytest.mark.asyncio
async def test_list_segments_by_status(db_session, test_persona):
    """Test filtering segments by status."""
    service = AudienceSegmentService(db_session)
    
    active = await service.create_segment(
        persona_id=str(test_persona.id),
        segment_name="Active Segment",
        criteria={"active": True},
    )
    
    inactive = await service.create_segment(
        persona_id=str(test_persona.id),
        segment_name="Inactive Segment",
        criteria={"active": False},
    )
    
    # Update one to inactive
    await service.update_segment(
        str(inactive.id),
        status=SegmentStatus.INACTIVE
    )
    
    active_segments = await service.list_segments(
        persona_id=str(test_persona.id),
        status=SegmentStatus.ACTIVE
    )
    
    assert len(active_segments) == 1
    assert active_segments[0].id == active.id


@pytest.mark.asyncio
async def test_update_segment(db_session, test_persona):
    """Test updating a segment."""
    service = AudienceSegmentService(db_session)
    
    segment = await service.create_segment(
        persona_id=str(test_persona.id),
        segment_name="Original Name",
        criteria={"original": True},
    )
    
    updated = await service.update_segment(
        str(segment.id),
        segment_name="Updated Name",
        criteria={"updated": True},
        description="New description"
    )
    
    assert updated is not None
    assert updated.segment_name == "Updated Name"
    assert updated.criteria["updated"] is True
    assert updated.description == "New description"


@pytest.mark.asyncio
async def test_delete_segment(db_session, test_persona):
    """Test deleting a segment."""
    service = AudienceSegmentService(db_session)
    
    segment = await service.create_segment(
        persona_id=str(test_persona.id),
        segment_name="To Delete",
        criteria={"delete": True},
    )
    
    deleted = await service.delete_segment(str(segment.id))
    assert deleted is True
    
    # Verify it's gone
    retrieved = await service.get_segment(str(segment.id))
    assert retrieved is None


@pytest.mark.asyncio
async def test_add_member_to_segment(db_session, test_persona, test_user):
    """Test adding a member to a segment."""
    service = AudienceSegmentService(db_session)
    
    segment = await service.create_segment(
        persona_id=str(test_persona.id),
        segment_name="Test Segment",
        criteria={"test": True},
    )
    
    member = await service.add_member_to_segment(
        segment_id=str(segment.id),
        user_id=str(test_user.id),
        confidence_score=0.85,
    )
    
    assert member is not None
    assert member.segment_id == segment.id
    assert member.user_id == test_user.id
    assert member.confidence_score == 0.85
    
    # Check segment member count updated
    updated_segment = await service.get_segment(str(segment.id))
    assert updated_segment.member_count == 1


@pytest.mark.asyncio
async def test_remove_member_from_segment(db_session, test_persona, test_user):
    """Test removing a member from a segment."""
    service = AudienceSegmentService(db_session)
    
    segment = await service.create_segment(
        persona_id=str(test_persona.id),
        segment_name="Test Segment",
        criteria={"test": True},
    )
    
    # Add member
    await service.add_member_to_segment(
        segment_id=str(segment.id),
        user_id=str(test_user.id),
    )
    
    # Remove member
    removed = await service.remove_member_from_segment(
        segment_id=str(segment.id),
        user_id=str(test_user.id),
    )
    
    assert removed is True
    
    # Check member count
    updated_segment = await service.get_segment(str(segment.id))
    assert updated_segment.member_count == 0


@pytest.mark.asyncio
async def test_get_segment_members(db_session, test_persona, test_user):
    """Test getting members of a segment."""
    service = AudienceSegmentService(db_session)
    
    segment = await service.create_segment(
        persona_id=str(test_persona.id),
        segment_name="Test Segment",
        criteria={"test": True},
    )
    
    # Add member
    await service.add_member_to_segment(
        segment_id=str(segment.id),
        user_id=str(test_user.id),
    )
    
    members = await service.get_segment_members(str(segment.id))
    
    assert len(members) == 1
    assert members[0].user_id == test_user.id


@pytest.mark.asyncio
async def test_create_personalized_content(db_session, test_persona, test_content):
    """Test creating personalized content mapping."""
    service = AudienceSegmentService(db_session)
    
    segment = await service.create_segment(
        persona_id=str(test_persona.id),
        segment_name="Test Segment",
        criteria={"test": True},
    )
    
    personalized = await service.create_personalized_content(
        content_id=str(test_content.id),
        segment_id=str(segment.id),
        variant_id="variant_a",
        is_control=False,
    )
    
    assert personalized is not None
    assert personalized.content_id == test_content.id
    assert personalized.segment_id == segment.id
    assert personalized.variant_id == "variant_a"
    assert personalized.is_control is False


@pytest.mark.asyncio
async def test_get_personalized_content_by_segment(db_session, test_persona, test_content):
    """Test getting personalized content for a segment."""
    service = AudienceSegmentService(db_session)
    
    segment = await service.create_segment(
        persona_id=str(test_persona.id),
        segment_name="Test Segment",
        criteria={"test": True},
    )
    
    # Create personalized content
    await service.create_personalized_content(
        content_id=str(test_content.id),
        segment_id=str(segment.id),
    )
    
    personalized_list = await service.get_personalized_content(
        segment_id=str(segment.id)
    )
    
    assert len(personalized_list) == 1
    assert personalized_list[0].segment_id == segment.id


@pytest.mark.asyncio
async def test_update_content_performance(db_session, test_persona, test_content):
    """Test updating performance metrics for personalized content."""
    service = AudienceSegmentService(db_session)
    
    segment = await service.create_segment(
        persona_id=str(test_persona.id),
        segment_name="Test Segment",
        criteria={"test": True},
    )
    
    personalized = await service.create_personalized_content(
        content_id=str(test_content.id),
        segment_id=str(segment.id),
    )
    
    updated = await service.update_content_performance(
        str(personalized.id),
        {
            "views": 1000,
            "engagement": 150,
            "conversions": 25,
        }
    )
    
    assert updated is not None
    assert updated.view_count == 1000
    assert updated.engagement_count == 150
    assert updated.conversion_count == 25
    assert updated.engagement_rate == 15.0  # 150/1000 * 100


@pytest.mark.asyncio
async def test_get_segment_analytics(db_session, test_persona, test_content):
    """Test getting analytics for a segment."""
    service = AudienceSegmentService(db_session)
    
    segment = await service.create_segment(
        persona_id=str(test_persona.id),
        segment_name="Test Segment",
        criteria={"test": True},
    )
    
    # Create personalized content with performance
    personalized = await service.create_personalized_content(
        content_id=str(test_content.id),
        segment_id=str(segment.id),
    )
    
    await service.update_content_performance(
        str(personalized.id),
        {
            "views": 500,
            "engagement": 100,
            "conversions": 10,
        }
    )
    
    analytics = await service.get_segment_analytics(str(segment.id))
    
    assert analytics["segment_id"] == str(segment.id)
    assert analytics["segment_name"] == "Test Segment"
    assert analytics["performance_summary"]["total_views"] == 500
    assert analytics["performance_summary"]["total_engagement"] == 100
    assert analytics["performance_summary"]["total_conversions"] == 10
    assert len(analytics["recommendations"]) > 0


@pytest.mark.asyncio
async def test_analyze_segment(db_session, test_persona):
    """Test analyzing a segment."""
    service = AudienceSegmentService(db_session)
    
    segment = await service.create_segment(
        persona_id=str(test_persona.id),
        segment_name="Test Segment",
        criteria={"test": True},
    )
    
    # Run analysis
    await service.analyze_segment(str(segment.id))
    
    # Check that analysis timestamp was updated
    analyzed_segment = await service.get_segment(str(segment.id))
    assert analyzed_segment.last_analyzed_at is not None
    assert analyzed_segment.performance_metrics is not None
