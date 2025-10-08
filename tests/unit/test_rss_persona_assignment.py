"""
Test RSS Feed and Persona Assignment functionality.

Tests the RSS feed management, topic extraction, and persona-feed assignment.
"""

import pytest
from uuid import uuid4
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

from backend.services.rss_ingestion_service import RSSIngestionService
from backend.models.feed import (
    RSSFeedCreate,
    PersonaFeedAssignment,
    RSSFeedModel,
    PersonaFeedModel,
    FeedItemModel,
)
from backend.models.persona import PersonaModel


class TestRSSPersonaAssignment:
    """Test suite for RSS feed and persona assignment functionality."""

    @pytest.fixture
    def sample_feed(self, db_session):
        """Create a sample RSS feed."""
        # Use unique URL for each test to avoid conflicts
        feed = RSSFeedModel(
            id=uuid4(),
            name="TechCrunch",
            url=f"https://techcrunch.com/feed/{uuid4()}",
            description="Tech news",
            categories=["technology", "startups"],
            fetch_frequency_hours=6,
            is_active=True,
        )
        return feed

    @pytest.fixture
    def sample_persona(self, db_session):
        """Create a sample persona."""
        persona = PersonaModel(
            id=uuid4(),
            name="Tech Influencer",
            appearance="Professional tech enthusiast",
            personality="Knowledgeable and engaging",
            content_themes=["technology", "innovation", "startups"],
            style_preferences={"tone": "professional"},
        )
        return persona

    @pytest.fixture
    def sample_feed_items(self, sample_feed):
        """Create sample feed items."""
        items = []
        for i in range(3):
            item = FeedItemModel(
                id=uuid4(),
                feed_id=sample_feed.id,
                title=f"Tech Article {i+1}",
                link=f"https://example.com/article-{i+1}",
                description=f"Article about technology {i+1}",
                published_date=datetime.now(timezone.utc),
                categories=["technology", "ai"],
                topics=["artificial intelligence", "machine learning"],
                keywords=["ai", "ml", "tech"],
                sentiment_score=0.5,
                relevance_score=0.8,
                processed=True,
            )
            items.append(item)
        return items

    async def test_assign_feed_to_persona(
        self, db_session, sample_feed, sample_persona
    ):
        """Test assigning an RSS feed to a persona."""
        # Add feed and persona to database
        db_session.add(sample_feed)
        db_session.add(sample_persona)
        await db_session.commit()

        service = RSSIngestionService(db_session)

        # Assign feed to persona
        assignment = PersonaFeedAssignment(
            feed_id=sample_feed.id, topics=["technology", "ai"], priority=80
        )

        result = await service.assign_feed_to_persona(sample_persona.id, assignment)

        # Verify assignment
        assert result.persona_id == sample_persona.id
        assert result.feed_id == sample_feed.id
        assert result.topics == ["technology", "ai"]
        assert result.priority == 80
        assert result.is_active is True
        assert result.feed_name == "TechCrunch"
        assert result.feed_url == sample_feed.url  # URL is unique per test

    async def test_list_persona_feeds(self, db_session, sample_feed, sample_persona):
        """Test listing feeds assigned to a persona."""
        # Setup
        db_session.add(sample_feed)
        db_session.add(sample_persona)
        await db_session.commit()

        # Create assignment
        assignment = PersonaFeedModel(
            persona_id=sample_persona.id,
            feed_id=sample_feed.id,
            topics=["technology"],
            priority=70,
            is_active=True,
        )
        db_session.add(assignment)
        await db_session.commit()

        service = RSSIngestionService(db_session)

        # List feeds
        feeds = await service.list_persona_feeds(sample_persona.id)

        # Verify
        assert len(feeds) == 1
        assert feeds[0].feed_id == sample_feed.id
        assert feeds[0].topics == ["technology"]
        assert feeds[0].priority == 70
        assert feeds[0].feed_name == "TechCrunch"

    async def test_unassign_feed_from_persona(
        self, db_session, sample_feed, sample_persona
    ):
        """Test unassigning a feed from a persona."""
        # Setup
        db_session.add(sample_feed)
        db_session.add(sample_persona)
        await db_session.commit()

        # Create assignment
        assignment = PersonaFeedModel(
            persona_id=sample_persona.id,
            feed_id=sample_feed.id,
            topics=["technology"],
            priority=70,
            is_active=True,
        )
        db_session.add(assignment)
        await db_session.commit()

        service = RSSIngestionService(db_session)

        # Unassign
        success = await service.unassign_feed_from_persona(
            sample_persona.id, sample_feed.id
        )

        # Verify
        assert success is True

        # Verify assignment is now inactive
        feeds = await service.list_persona_feeds(sample_persona.id)
        assert len(feeds) == 0

    async def test_list_feeds_by_topic(self, db_session):
        """Test listing feeds filtered by topic."""
        # Create multiple feeds with different topics (unique URLs and unique test topic)
        test_id = str(uuid4())
        unique_topic = (
            f"quantumphysics_{test_id[:8]}"  # Unique topic unlikely to collide
        )

        feed1 = RSSFeedModel(
            id=uuid4(),
            name="TechCrunch",
            url=f"https://techcrunch.com/feed/{test_id}/1",
            categories=[unique_topic, "startups"],
            is_active=True,
        )
        feed2 = RSSFeedModel(
            id=uuid4(),
            name="Business Insider",
            url=f"https://businessinsider.com/feed/{test_id}/2",
            categories=["business", "finance"],
            is_active=True,
        )
        feed3 = RSSFeedModel(
            id=uuid4(),
            name="Wired",
            url=f"https://wired.com/feed/{test_id}/3",
            categories=[unique_topic, "science"],
            is_active=True,
        )

        db_session.add_all([feed1, feed2, feed3])
        await db_session.commit()

        service = RSSIngestionService(db_session)

        # List feeds by unique topic
        topic_feeds = await service.list_feeds_by_topic(unique_topic)

        # Verify
        assert len(topic_feeds) == 2
        feed_names = [f.name for f in topic_feeds]
        assert "TechCrunch" in feed_names
        assert "Wired" in feed_names
        assert "Business Insider" not in feed_names

    async def test_get_content_suggestions_with_assigned_feeds(
        self, db_session, sample_feed, sample_persona, sample_feed_items
    ):
        """Test getting content suggestions based on assigned feeds."""
        # Setup
        db_session.add(sample_feed)
        db_session.add(sample_persona)
        await db_session.commit()

        # Add feed items
        for item in sample_feed_items:
            db_session.add(item)
        await db_session.commit()

        # Create assignment
        assignment = PersonaFeedModel(
            persona_id=sample_persona.id,
            feed_id=sample_feed.id,
            topics=["technology", "ai"],
            priority=80,
            is_active=True,
        )
        db_session.add(assignment)
        await db_session.commit()

        service = RSSIngestionService(db_session)

        # Get suggestions
        suggestions = await service.get_content_suggestions(sample_persona.id, limit=5)

        # Verify
        assert len(suggestions) > 0
        assert all(s.feed_id == sample_feed.id for s in suggestions)

    async def test_get_content_suggestions_no_assigned_feeds(
        self, db_session, sample_persona
    ):
        """Test content suggestions when no feeds are assigned."""
        db_session.add(sample_persona)
        await db_session.commit()

        service = RSSIngestionService(db_session)

        # Get suggestions
        suggestions = await service.get_content_suggestions(sample_persona.id, limit=5)

        # Verify - should return empty list when no feeds assigned
        assert len(suggestions) == 0

    async def test_update_existing_assignment(
        self, db_session, sample_feed, sample_persona
    ):
        """Test updating an existing feed assignment."""
        # Setup
        db_session.add(sample_feed)
        db_session.add(sample_persona)
        await db_session.commit()

        service = RSSIngestionService(db_session)

        # Create initial assignment
        assignment1 = PersonaFeedAssignment(
            feed_id=sample_feed.id, topics=["technology"], priority=50
        )
        result1 = await service.assign_feed_to_persona(sample_persona.id, assignment1)

        # Update assignment
        assignment2 = PersonaFeedAssignment(
            feed_id=sample_feed.id, topics=["technology", "ai", "startups"], priority=90
        )
        result2 = await service.assign_feed_to_persona(sample_persona.id, assignment2)

        # Verify update
        assert result2.topics == ["technology", "ai", "startups"]
        assert result2.priority == 90

        # Verify only one assignment exists
        feeds = await service.list_persona_feeds(sample_persona.id)
        assert len(feeds) == 1

    async def test_assign_nonexistent_feed(self, db_session, sample_persona):
        """Test assigning a feed that doesn't exist."""
        db_session.add(sample_persona)
        await db_session.commit()

        service = RSSIngestionService(db_session)

        # Try to assign non-existent feed
        assignment = PersonaFeedAssignment(
            feed_id=uuid4(), topics=["technology"], priority=50  # Random UUID
        )

        # Should raise ValueError
        with pytest.raises(ValueError, match="Feed not found"):
            await service.assign_feed_to_persona(sample_persona.id, assignment)
