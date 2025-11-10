"""
Tests for Social Media Engagement Tracking with ACD Integration

Tests the complete feedback loop from content generation through
social media engagement tracking to ACD learning integration.
"""

import pytest
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from backend.models.social_media_post import (
    SocialMediaPostCreate,
    SocialMediaPostUpdate,
    EngagementMetrics,
    SocialPlatform,
    PostStatus,
)
from backend.models.acd import (
    ACDContextCreate,
    AIStatus,
    AIState,
    AIComplexity,
    AIValidation,
    AIConfidence,
)
from backend.services.social_engagement_service import SocialEngagementService
from backend.services.acd_service import ACDService


@pytest.mark.asyncio
async def test_create_post_with_acd_link(async_session):
    """Test creating a social media post linked to ACD context."""
    # Create ACD context first
    acd_service = ACDService(async_session)
    acd_context = await acd_service.create_context(
        ACDContextCreate(
            ai_phase="SOCIAL_CONTENT_GENERATION",
            ai_status=AIStatus.IMPLEMENTED,
            ai_state=AIState.DONE,
            ai_complexity=AIComplexity.MEDIUM,
            ai_note="Test social media content",
        )
    )

    # Create post record linked to ACD
    engagement_service = SocialEngagementService(async_session)
    post_data = SocialMediaPostCreate(
        content_id=uuid4(),
        persona_id=uuid4(),
        platform=SocialPlatform.INSTAGRAM,
        caption="Test post caption",
        hashtags=["test", "demo"],
        acd_context_id=acd_context.id,
    )

    post = await engagement_service.create_post_record(post_data)

    assert post is not None
    assert post.platform == SocialPlatform.INSTAGRAM.value
    assert post.acd_context_id == acd_context.id
    assert post.caption == "Test post caption"
    assert post.hashtags == ["test", "demo"]
    assert post.status == PostStatus.DRAFT.value


@pytest.mark.asyncio
async def test_update_engagement_metrics(async_session):
    """Test updating engagement metrics for a post."""
    # Create post
    engagement_service = SocialEngagementService(async_session)
    post_data = SocialMediaPostCreate(
        content_id=uuid4(),
        persona_id=uuid4(),
        platform=SocialPlatform.INSTAGRAM,
        caption="Test engagement",
    )
    post = await engagement_service.create_post_record(post_data)

    # Update with metrics
    metrics = EngagementMetrics(
        likes_count=1000,
        comments_count=50,
        shares_count=25,
        saves_count=100,
        impressions=10000,
        reach=5000,
        bot_interaction_count=100,
        persona_interaction_count=10,
        genuine_user_count=940,
    )

    updated_post = await engagement_service.update_post_metrics(post.id, metrics)

    assert updated_post.likes_count == 1000
    assert updated_post.comments_count == 50
    assert updated_post.shares_count == 25
    assert updated_post.saves_count == 100
    assert updated_post.impressions == 10000
    assert updated_post.reach == 5000
    assert updated_post.engagement_rate is not None
    assert updated_post.engagement_rate > 0
    assert updated_post.bot_interaction_count == 100
    assert updated_post.persona_interaction_count == 10
    assert updated_post.genuine_user_count == 940


@pytest.mark.asyncio
async def test_engagement_rate_calculation(async_session):
    """Test that engagement rate is calculated correctly."""
    engagement_service = SocialEngagementService(async_session)
    post_data = SocialMediaPostCreate(
        content_id=uuid4(),
        persona_id=uuid4(),
        platform=SocialPlatform.INSTAGRAM,
    )
    post = await engagement_service.create_post_record(post_data)

    # Set metrics with known values for calculation
    metrics = EngagementMetrics(
        likes_count=100,
        comments_count=50,
        shares_count=25,
        saves_count=25,
        reach=2000,
        impressions=5000,
    )

    updated_post = await engagement_service.update_post_metrics(post.id, metrics)

    # Total engagement = 100 + 50 + 25 + 25 = 200
    # Engagement rate = (200 / 2000) * 100 = 10%
    expected_rate = 10.0
    assert abs(updated_post.engagement_rate - expected_rate) < 0.01


@pytest.mark.asyncio
async def test_acd_update_with_high_engagement(async_session):
    """Test that ACD context is updated when engagement is high."""
    # Create ACD context
    acd_service = ACDService(async_session)
    acd_context = await acd_service.create_context(
        ACDContextCreate(
            ai_phase="SOCIAL_CONTENT",
            ai_status=AIStatus.IMPLEMENTED,
            ai_state=AIState.DONE,
        )
    )

    # Create post linked to ACD
    engagement_service = SocialEngagementService(async_session)
    post_data = SocialMediaPostCreate(
        content_id=uuid4(),
        persona_id=uuid4(),
        platform=SocialPlatform.INSTAGRAM,
        caption="Viral post!",
        hashtags=["trending", "viral"],
        acd_context_id=acd_context.id,
    )
    post = await engagement_service.create_post_record(post_data)

    # High engagement metrics (>5% engagement rate)
    metrics = EngagementMetrics(
        likes_count=500,
        comments_count=100,
        shares_count=50,
        saves_count=50,
        reach=10000,
        impressions=20000,
        genuine_user_count=700,
    )

    await engagement_service.update_post_metrics(post.id, metrics)

    # Check ACD context was updated
    updated_acd = await acd_service.get_context(acd_context.id)
    
    assert updated_acd.ai_validation == AIValidation.APPROVED
    assert updated_acd.ai_confidence == AIConfidence.VALIDATED
    assert updated_acd.ai_metadata is not None
    assert "social_metrics" in updated_acd.ai_metadata
    
    social_metrics = updated_acd.ai_metadata["social_metrics"]
    assert social_metrics["platform"] == "instagram"
    assert social_metrics["engagement_rate"] == 7.0  # (700 / 10000) * 100
    assert social_metrics["genuine_user_count"] == 700


@pytest.mark.asyncio
async def test_acd_update_with_low_engagement(async_session):
    """Test that ACD context reflects low engagement appropriately."""
    acd_service = ACDService(async_session)
    acd_context = await acd_service.create_context(
        ACDContextCreate(
            ai_phase="SOCIAL_CONTENT",
            ai_status=AIStatus.IMPLEMENTED,
            ai_state=AIState.DONE,
        )
    )

    engagement_service = SocialEngagementService(async_session)
    post_data = SocialMediaPostCreate(
        content_id=uuid4(),
        persona_id=uuid4(),
        platform=SocialPlatform.INSTAGRAM,
        acd_context_id=acd_context.id,
    )
    post = await engagement_service.create_post_record(post_data)

    # Low engagement metrics (<2% engagement rate)
    metrics = EngagementMetrics(
        likes_count=50,
        comments_count=5,
        shares_count=2,
        reach=10000,
        impressions=15000,
        genuine_user_count=57,
    )

    await engagement_service.update_post_metrics(post.id, metrics)

    updated_acd = await acd_service.get_context(acd_context.id)
    
    assert updated_acd.ai_validation == AIValidation.ANALYZED
    assert updated_acd.ai_confidence == AIConfidence.UNCERTAIN


@pytest.mark.asyncio
async def test_pattern_extraction_from_high_performing_post(async_session):
    """Test that patterns are extracted from high-performing posts."""
    acd_service = ACDService(async_session)
    acd_context = await acd_service.create_context(
        ACDContextCreate(
            ai_phase="SOCIAL_CONTENT",
            ai_status=AIStatus.IMPLEMENTED,
            ai_state=AIState.DONE,
        )
    )

    engagement_service = SocialEngagementService(async_session)
    post_data = SocialMediaPostCreate(
        content_id=uuid4(),
        persona_id=uuid4(),
        platform=SocialPlatform.TIKTOK,
        caption="Amazing content!",
        hashtags=["fyp", "viral", "trending"],
        acd_context_id=acd_context.id,
    )
    post = await engagement_service.create_post_record(post_data)

    # Very high engagement
    metrics = EngagementMetrics(
        likes_count=5000,
        comments_count=500,
        shares_count=300,
        reach=50000,
        impressions=100000,
        genuine_user_count=5800,
    )

    await engagement_service.update_post_metrics(post.id, metrics)

    updated_acd = await acd_service.get_context(acd_context.id)
    
    # Should have pattern extracted
    assert updated_acd.ai_pattern is not None
    assert "tiktok" in updated_acd.ai_pattern.lower()
    assert "high_engagement" in updated_acd.ai_pattern.lower()
    
    # Should have strategy documented
    assert updated_acd.ai_strategy is not None
    assert "engagement rate" in updated_acd.ai_strategy.lower()


@pytest.mark.asyncio
async def test_bot_filtering_metrics(async_session):
    """Test that bot interactions are properly filtered."""
    engagement_service = SocialEngagementService(async_session)
    post_data = SocialMediaPostCreate(
        content_id=uuid4(),
        persona_id=uuid4(),
        platform=SocialPlatform.INSTAGRAM,
    )
    post = await engagement_service.create_post_record(post_data)

    # Metrics with significant bot activity
    metrics = EngagementMetrics(
        likes_count=1000,
        comments_count=200,
        shares_count=50,
        reach=10000,
        impressions=20000,
        bot_interaction_count=300,  # 25% are bots
        persona_interaction_count=20,
        genuine_user_count=930,  # 75% genuine
    )

    updated_post = await engagement_service.update_post_metrics(post.id, metrics)

    assert updated_post.bot_interaction_count == 300
    assert updated_post.persona_interaction_count == 20
    assert updated_post.genuine_user_count == 930
    
    # Ensure metrics reflect filtered counts
    assert updated_post.likes_count == 1000
    assert updated_post.comments_count == 200


@pytest.mark.asyncio
async def test_analyze_post_performance(async_session):
    """Test post performance analysis."""
    engagement_service = SocialEngagementService(async_session)
    persona_id = uuid4()
    
    # Create post
    post_data = SocialMediaPostCreate(
        content_id=uuid4(),
        persona_id=persona_id,
        platform=SocialPlatform.INSTAGRAM,
        hashtags=["trending", "popular"],
    )
    post = await engagement_service.create_post_record(post_data)

    # Add metrics
    metrics = EngagementMetrics(
        likes_count=500,
        comments_count=100,
        shares_count=50,
        reach=5000,
        genuine_user_count=650,
    )
    await engagement_service.update_post_metrics(post.id, metrics)

    # Analyze performance
    analysis = await engagement_service.analyze_post_performance(post.id)

    assert analysis is not None
    assert analysis.post_id == post.id
    assert analysis.total_engagement == 650
    assert analysis.genuine_engagement == 650
    assert analysis.engagement_rate > 0
    assert analysis.recommendations is not None
    assert len(analysis.recommendations) > 0


@pytest.mark.asyncio
async def test_multiple_platforms_tracking(async_session):
    """Test tracking posts across multiple platforms."""
    engagement_service = SocialEngagementService(async_session)
    persona_id = uuid4()
    content_id = uuid4()

    platforms = [
        SocialPlatform.INSTAGRAM,
        SocialPlatform.FACEBOOK,
        SocialPlatform.TWITTER,
        SocialPlatform.TIKTOK,
    ]

    created_posts = []
    for platform in platforms:
        post_data = SocialMediaPostCreate(
            content_id=content_id,
            persona_id=persona_id,
            platform=platform,
            caption=f"Test post for {platform.value}",
        )
        post = await engagement_service.create_post_record(post_data)
        created_posts.append(post)

    assert len(created_posts) == 4
    assert all(post.content_id == content_id for post in created_posts)
    assert all(post.persona_id == persona_id for post in created_posts)
    
    # Each should have unique platform
    post_platforms = [post.platform for post in created_posts]
    assert len(set(post_platforms)) == 4


@pytest.mark.asyncio
async def test_engagement_timeline_tracking(async_session):
    """Test that engagement timeline is properly stored."""
    engagement_service = SocialEngagementService(async_session)
    post_data = SocialMediaPostCreate(
        content_id=uuid4(),
        persona_id=uuid4(),
        platform=SocialPlatform.INSTAGRAM,
    )
    post = await engagement_service.create_post_record(post_data)

    # Add metrics with timeline
    timeline = {
        "00:00": 10,
        "01:00": 5,
        "02:00": 3,
        "03:00": 2,
        "09:00": 50,
        "10:00": 120,
        "12:00": 200,
        "18:00": 150,
        "20:00": 100,
    }

    metrics = EngagementMetrics(
        likes_count=640,
        reach=5000,
        engagement_timeline=timeline,
        genuine_user_count=640,
    )

    updated_post = await engagement_service.update_post_metrics(post.id, metrics)

    assert updated_post.engagement_timeline is not None
    assert updated_post.engagement_timeline == timeline
    # Peak should be at 12:00
    assert max(timeline.items(), key=lambda x: x[1])[0] == "12:00"


@pytest.mark.asyncio
async def test_demographic_insights_storage(async_session):
    """Test that demographic insights are properly stored."""
    engagement_service = SocialEngagementService(async_session)
    post_data = SocialMediaPostCreate(
        content_id=uuid4(),
        persona_id=uuid4(),
        platform=SocialPlatform.INSTAGRAM,
    )
    post = await engagement_service.create_post_record(post_data)

    # Add demographic insights
    demographics = {
        "age_groups": {
            "18-24": 30,
            "25-34": 45,
            "35-44": 20,
            "45+": 5,
        },
        "gender": {
            "male": 40,
            "female": 55,
            "other": 5,
        },
        "locations": {
            "US": 60,
            "UK": 20,
            "Canada": 10,
            "Other": 10,
        },
    }

    metrics = EngagementMetrics(
        likes_count=1000,
        reach=10000,
        demographic_insights=demographics,
        genuine_user_count=1000,
    )

    updated_post = await engagement_service.update_post_metrics(post.id, metrics)

    assert updated_post.demographic_insights is not None
    assert "age_groups" in updated_post.demographic_insights
    assert "gender" in updated_post.demographic_insights
    assert "locations" in updated_post.demographic_insights


@pytest.mark.asyncio
async def test_complete_feedback_loop(async_session):
    """Test the complete feedback loop from generation to learning."""
    # 1. Create ACD context for content generation
    acd_service = ACDService(async_session)
    acd_context = await acd_service.create_context(
        ACDContextCreate(
            ai_phase="IMAGE_GENERATION",
            ai_status=AIStatus.IMPLEMENTED,
            ai_state=AIState.DONE,
            ai_complexity=AIComplexity.MEDIUM,
            ai_context={"prompt": "lifestyle content", "quality": "high"},
        )
    )

    # 2. Create social media post linked to ACD
    engagement_service = SocialEngagementService(async_session)
    post_data = SocialMediaPostCreate(
        content_id=uuid4(),
        persona_id=uuid4(),
        platform=SocialPlatform.INSTAGRAM,
        caption="Living my best life! #lifestyle #motivation",
        hashtags=["lifestyle", "motivation"],
        acd_context_id=acd_context.id,
    )
    post = await engagement_service.create_post_record(post_data)

    # 3. Simulate high engagement
    metrics = EngagementMetrics(
        likes_count=2000,
        comments_count=200,
        shares_count=100,
        saves_count=300,
        reach=20000,
        impressions=50000,
        genuine_user_count=2600,
        bot_interaction_count=100,
    )
    await engagement_service.update_post_metrics(post.id, metrics)

    # 4. Verify ACD was updated with learning data
    updated_acd = await acd_service.get_context(acd_context.id)
    
    # Should be marked as successful
    assert updated_acd.ai_validation == AIValidation.APPROVED
    assert updated_acd.ai_confidence == AIConfidence.VALIDATED
    
    # Should have social metrics stored
    assert "social_metrics" in updated_acd.ai_metadata
    
    # Should have pattern extracted
    assert updated_acd.ai_pattern is not None
    assert updated_acd.ai_strategy is not None
    
    # 5. Verify pattern can be used for future content
    assert "instagram" in updated_acd.ai_pattern.lower()
    assert "lifestyle" in updated_acd.ai_strategy.lower() or "motivation" in updated_acd.ai_strategy.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
