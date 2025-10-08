"""
Tests for Gator Agent Service

Tests the functionality of the Gator help agent including persona responses,
knowledge base lookups, and conversation handling.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, patch

from backend.services.gator_agent_service import GatorAgentService, gator_agent


class TestGatorAgentService:
    """Test cases for Gator Agent Service."""

    @pytest.fixture
    def agent(self):
        """Create a fresh Gator agent instance for testing."""
        return GatorAgentService()

    @pytest.mark.asyncio
    async def test_gator_greeting_response(self, agent):
        """Test that Gator responds appropriately to greetings."""
        response = await agent.process_message("Hello Gator")

        assert isinstance(response, str)
        assert len(response) > 0
        # Should be one of Gator's greeting responses
        expected_responses = [
            "Yeah, what do you need? I'm a peacock, you gotta let me fly!",
            "Speak up! What's the problem? I'm like a tiny peacock with a big beak.",
            "I'm listening. Make it quick - I'm a lion and I want to be free like a lion.",
            "What brings you to Gator? Better be important.",
        ]
        assert any(expected in response for expected in expected_responses)

    @pytest.mark.asyncio
    async def test_gator_persona_help(self, agent):
        """Test Gator's persona-related help responses."""
        response = await agent.process_message("How do I create a persona?")

        assert "persona" in response.lower()
        assert "create new persona" in response.lower()
        assert "don't half-ass it" in response.lower()

    @pytest.mark.asyncio
    async def test_gator_dns_help(self, agent):
        """Test Gator's DNS-related help responses."""
        response = await agent.process_message("Help me with DNS setup")

        assert "dns" in response.lower()
        assert (
            "dns management tab" in response.lower() or "auto-setup" in response.lower()
        )

    @pytest.mark.asyncio
    async def test_gator_error_handling(self, agent):
        """Test Gator's error handling responses."""
        response = await agent.process_message("I'm getting an error")

        assert any(
            phrase in response
            for phrase in ["Hold up", "not right", "broken", "gonna fix"]
        )

    @pytest.mark.asyncio
    async def test_gator_goodbye_response(self, agent):
        """Test Gator's goodbye responses."""
        response = await agent.process_message("Thanks Gator, goodbye")

        expected_responses = [
            "Alright, we're done here",
            "You're good to go",
            "That's all from Gator",
            "Peace out",
            "I'm a lion, and I want to be free like a lion",
        ]
        assert any(expected in response for expected in expected_responses)

    @pytest.mark.asyncio
    async def test_conversation_history_tracking(self, agent):
        """Test that conversation history is properly tracked."""
        await agent.process_message("Hello")
        await agent.process_message("How do I create personas?")

        history = agent.get_conversation_history()

        assert len(history) == 2
        assert history[0]["user_message"] == "hello"
        assert history[1]["user_message"] == "how do i create personas?"
        assert "gator_response" in history[0]
        assert "gator_response" in history[1]

    def test_clear_conversation_history(self, agent):
        """Test clearing conversation history."""
        agent.conversation_history = [{"test": "data"}]
        agent.clear_conversation_history()

        assert len(agent.get_conversation_history()) == 0

    def test_quick_help_topics(self, agent):
        """Test that quick help topics are properly formatted."""
        topics = agent.get_quick_help_topics()

        assert isinstance(topics, list)
        assert len(topics) > 0

        for topic in topics:
            assert "topic" in topic
            assert "message" in topic
            assert isinstance(topic["topic"], str)
            assert isinstance(topic["message"], str)

    @pytest.mark.asyncio
    async def test_gator_attitude_consistency(self, agent):
        """Test that Gator maintains his tough attitude in responses."""
        messages = ["Hello", "I need help", "How do I do this?", "Thanks"]

        attitude_phrases = [
            "listen",
            "don't",
            "ain't",
            "better",
            "gator",
            "pay attention",
            "make it",
            "gonna",
            "alright",
            "peacock",
            "lion",
            "pimp",
        ]

        attitude_count = 0
        for message in messages:
            response = await agent.process_message(message)
            if any(phrase in response.lower() for phrase in attitude_phrases):
                attitude_count += 1

        # Gator should maintain his attitude in most responses
        assert attitude_count >= len(messages) * 0.5

    @pytest.mark.asyncio
    async def test_context_handling(self, agent):
        """Test that Gator can handle context information."""
        context = {"current_tab": "dns", "timestamp": "2024-01-01T00:00:00Z"}
        response = await agent.process_message("I need help", context)

        # Should get a response regardless of context
        assert isinstance(response, str)
        assert len(response) > 0

    @pytest.mark.asyncio
    async def test_long_message_handling(self, agent):
        """Test handling of longer messages."""
        long_message = (
            "Hey Gator, " * 20 + "can you help me understand how to set up personas?"
        )
        response = await agent.process_message(long_message)

        assert isinstance(response, str)
        assert len(response) > 0
        # Should still provide persona help
        assert "persona" in response.lower()

    @pytest.mark.asyncio
    async def test_empty_message_handling(self, agent):
        """Test handling of empty or whitespace messages."""
        response = await agent.process_message("   ")

        # Should get some kind of default response
        assert isinstance(response, str)
        assert len(response) > 0

    @pytest.mark.asyncio
    async def test_global_agent_instance(self):
        """Test that the global gator_agent instance works."""
        response = await gator_agent.process_message("Hello Gator")

        assert isinstance(response, str)
        assert len(response) > 0


class TestGatorKnowledgeBase:
    """Test Gator's knowledge base responses."""

    @pytest.fixture
    def agent(self):
        return GatorAgentService()

    @pytest.mark.asyncio
    async def test_troubleshooting_responses(self, agent):
        """Test troubleshooting knowledge base responses."""
        slow_response = await agent.process_message("The system is running slow")

        assert "slow" in slow_response.lower()
        assert any(
            word in slow_response.lower()
            for word in ["resources", "hardware", "cpu", "ram"]
        )

    @pytest.mark.asyncio
    async def test_content_generation_help(self, agent):
        """Test content generation help responses."""
        response = await agent.process_message("How do I generate content?")

        assert "content" in response.lower()
        assert "generate" in response.lower()

    @pytest.mark.asyncio
    async def test_system_status_help(self, agent):
        """Test system status help responses."""
        response = await agent.process_message("How do I check system status?")

        assert any(
            word in response.lower() for word in ["status", "dashboard", "system"]
        )

    @pytest.mark.asyncio
    async def test_unknown_topic_fallback(self, agent):
        """Test fallback response for unknown topics."""
        response = await agent.process_message("What about quantum computing?")

        # Should get a fallback response asking for more specifics
        assert any(
            phrase in response.lower()
            for phrase in ["not sure", "more specific", "what do you need"]
        )


class TestGatorPersonality:
    """Test Gator's personality traits."""

    @pytest.fixture
    def agent(self):
        return GatorAgentService()

    @pytest.mark.asyncio
    async def test_no_nonsense_attitude(self, agent):
        """Test that Gator maintains his no-nonsense attitude."""
        response = await agent.process_message("Can you maybe possibly help me?")

        # Should be direct and to the point
        attitude_indicators = [
            "listen",
            "don't",
            "ain't",
            "gonna",
            "pay attention",
            "make it",
            "better",
            "specific",
            "peacock",
            "lion",
        ]

        assert any(indicator in response.lower() for indicator in attitude_indicators)

    @pytest.mark.asyncio
    async def test_helpful_but_tough(self, agent):
        """Test that Gator is helpful despite being tough."""
        response = await agent.process_message("I need help with personas")

        # Should be tough but still provide help
        assert "persona" in response.lower()
        assert any(
            tough_word in response.lower()
            for tough_word in ["listen", "don't", "ain't", "better", "peacock", "lion"]
        )

    def test_gator_phrases_variety(self, agent):
        """Test that Gator has variety in his phrases."""
        phrases = agent.gator_phrases

        assert len(phrases) >= 5
        assert all(isinstance(phrase, str) for phrase in phrases)
        assert len(set(phrases)) == len(phrases)  # No duplicates

    def test_authentic_gator_quotes_included(self, agent):
        """Test that authentic movie quotes are included."""
        authentic_quotes = [
            "I'm a peacock, you gotta let me fly!",
            "I'm like a tiny peacock with a big beak",
            "I'm a lion, and I want to be free like a lion",
        ]

        all_phrases = agent.gator_phrases + agent.gator_confidence

        for quote in authentic_quotes:
            assert quote in all_phrases, f"Missing authentic quote: {quote}"

    @pytest.mark.asyncio
    async def test_confidence_quotes_in_responses(self, agent):
        """Test that confidence quotes appear in responses."""
        # Test multiple times to increase chance of getting confidence quotes
        responses = []
        for _ in range(10):
            response = await agent.process_message("I don't understand")
            responses.append(response.lower())

        confidence_indicators = ["peacock", "lion", "pimp"]
        quote_found = any(
            any(indicator in response for indicator in confidence_indicators)
            for response in responses
        )

        assert quote_found, "Confidence quotes should appear in some responses"
