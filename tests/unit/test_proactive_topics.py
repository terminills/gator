"""
Tests for the Proactive Topics Service

Tests the proactive topic generation and opinion features for personas.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from backend.services.proactive_topics_service import ProactiveTopicsService


class TestProactiveTopicsService:
    """Tests for ProactiveTopicsService."""
    
    @pytest.fixture
    def mock_db_session(self):
        """Create a mock database session."""
        session = MagicMock()
        session.execute = AsyncMock()
        session.commit = AsyncMock()
        session.refresh = AsyncMock()
        return session
    
    @pytest.fixture
    def service(self, mock_db_session):
        """Create a ProactiveTopicsService instance with mock database."""
        return ProactiveTopicsService(mock_db_session)
    
    @pytest.fixture
    def mock_persona(self):
        """Create a mock persona for testing."""
        persona = MagicMock()
        persona.id = uuid4()
        persona.name = "Test Persona"
        persona.personality = "Friendly and tech-savvy"
        persona.content_themes = ["technology", "AI", "startups"]
        persona.warmth_level = "warm"
        persona.signature_phrases = ["That's interesting!", "Let me think..."]
        persona.forbidden_phrases = []
        return persona
    
    @pytest.mark.asyncio
    async def test_generate_topics_from_themes(self, service, mock_persona):
        """Test generating topics from persona themes when no RSS feeds assigned."""
        # Call the internal method directly
        topics = await service._generate_topics_from_themes(mock_persona, limit=3)
        
        # Verify topics were generated
        assert len(topics) <= 3
        assert all(isinstance(t, dict) for t in topics)
        
        # Verify each topic has expected structure
        for topic in topics:
            assert "title" in topic
            assert "summary" in topic
            assert "source" in topic
            assert topic["source"] == "interests"
            assert "relevance_score" in topic
            assert "categories" in topic
    
    @pytest.mark.asyncio
    async def test_generate_topics_from_themes_no_themes(self, service, mock_persona):
        """Test fallback when persona has no content themes."""
        mock_persona.content_themes = []
        mock_persona.personality = "A friendly person"
        
        topics = await service._generate_topics_from_themes(mock_persona, limit=3)
        
        # Should still generate topics using fallback themes
        assert len(topics) > 0
    
    def test_extract_themes_from_personality(self, service):
        """Test extracting themes from personality description."""
        personality = "Sarah is passionate about technology and loves machine learning."
        themes = service._extract_themes_from_personality(personality)
        
        assert len(themes) > 0
        assert any("technology" in t.lower() for t in themes) or any("machine learning" in t.lower() for t in themes)
    
    def test_analyze_opinion_sentiment_positive(self, service):
        """Test positive sentiment analysis."""
        opinion = "I love this topic! It's amazing and really exciting to see."
        sentiment = service._analyze_opinion_sentiment(opinion)
        assert sentiment == "positive"
    
    def test_analyze_opinion_sentiment_negative(self, service):
        """Test negative sentiment analysis."""
        opinion = "This is terrible. I hate this trend, it's awful."
        sentiment = service._analyze_opinion_sentiment(opinion)
        assert sentiment == "negative"
    
    def test_analyze_opinion_sentiment_neutral(self, service):
        """Test neutral sentiment analysis."""
        opinion = "I think there are some interesting points to consider here."
        sentiment = service._analyze_opinion_sentiment(opinion)
        assert sentiment == "neutral"
    
    def test_generate_hashtags(self, service, mock_persona):
        """Test hashtag generation from topic and persona."""
        topic = {
            "title": "AI Trends",
            "categories": ["technology", "artificial-intelligence"],
            "keywords": ["ai", "machine-learning"],
        }
        
        hashtags = service._generate_hashtags(topic, mock_persona)
        
        # Verify hashtags were generated
        assert len(hashtags) > 0
        assert len(hashtags) <= 5
        
        # Verify hashtag format
        for tag in hashtags:
            assert tag.startswith("#")
    
    @pytest.mark.asyncio
    async def test_get_proactive_topics_with_mock_persona(self, service, mock_db_session, mock_persona):
        """Test get_proactive_topics with mocked database."""
        # Mock the persona query
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_persona
        mock_db_session.execute.return_value = mock_result
        
        # Mock the RSS feeds query to return empty (triggers fallback)
        mock_empty_result = MagicMock()
        mock_empty_result.all.return_value = []
        
        # First call returns persona, second returns empty feeds
        mock_db_session.execute.side_effect = [mock_result, mock_empty_result]
        
        topics = await service.get_proactive_topics(
            persona_id=mock_persona.id,
            limit=3,
            hours_window=48,
        )
        
        # Should return topics from themes since no RSS feeds
        assert len(topics) <= 3


class TestProactiveTopicsEdgeCases:
    """Edge case tests for proactive topics."""
    
    @pytest.fixture
    def mock_db_session(self):
        """Create a mock database session."""
        session = MagicMock()
        session.execute = AsyncMock()
        session.commit = AsyncMock()
        return session
    
    @pytest.fixture
    def service(self, mock_db_session):
        """Create a ProactiveTopicsService instance."""
        return ProactiveTopicsService(mock_db_session)
    
    @pytest.mark.asyncio
    async def test_get_proactive_topics_persona_not_found(self, service, mock_db_session):
        """Test handling of non-existent persona."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db_session.execute.return_value = mock_result
        
        topics = await service.get_proactive_topics(
            persona_id=uuid4(),
            limit=5,
            hours_window=48,
        )
        
        # Should return empty list for non-existent persona
        assert topics == []
    
    @pytest.mark.asyncio
    async def test_generate_opinion_persona_not_found(self, service, mock_db_session):
        """Test handling opinion generation for non-existent persona."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db_session.execute.return_value = mock_result
        
        result = await service.generate_opinion(
            persona_id=uuid4(),
            topic="Test Topic",
        )
        
        # Should return error response
        assert result["success"] is False
        assert "error" in result
