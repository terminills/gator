"""
Integration tests for Gator Agent API endpoints.
"""

import pytest


class TestGatorAgentAPI:
    """Test Gator Agent API endpoints."""

    def test_gator_agent_status(self, test_client):
        """Test getting Gator agent status at the main endpoint."""
        response = test_client.get("/api/v1/gator-agent/status")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure
        assert "status" in data
        assert "agent" in data
        assert "attitude" in data
        assert "conversation_count" in data
        assert "last_interaction" in data
        assert "available_topics" in data
        
        # Verify values
        assert data["status"] == "operational"
        assert "Gator" in data["agent"]
        assert isinstance(data["conversation_count"], int)
        assert isinstance(data["available_topics"], list)
        
    def test_gator_agent_status_alias(self, test_client):
        """Test the backward compatibility alias for Gator agent status."""
        response = test_client.get("/gator-agent/status")
        
        assert response.status_code == 200
        data = response.json()
        
        # Should have the same structure as /api/v1/gator-agent/status
        assert "status" in data
        assert "agent" in data
        assert "attitude" in data
        assert "conversation_count" in data
        assert "last_interaction" in data
        assert "available_topics" in data
        
        # Verify values
        assert data["status"] == "operational"
        assert "Gator" in data["agent"]
        
    @pytest.mark.skip(reason="Gator agent chat functionality has pre-existing issues")
    def test_gator_agent_chat(self, test_client):
        """Test chatting with Gator agent."""
        request = {
            "message": "Hello Gator",
            "context": None,
            "verbose": False
        }
        
        response = test_client.post("/api/v1/gator-agent/chat", json=request)
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure
        assert "response" in data
        assert "timestamp" in data
        
        # Verify response content
        assert isinstance(data["response"], str)
        assert len(data["response"]) > 0
        
    def test_gator_agent_quick_help(self, test_client):
        """Test getting quick help topics from Gator."""
        response = test_client.get("/api/v1/gator-agent/quick-help")
        
        assert response.status_code == 200
        data = response.json()
        
        # Should return a list of topics
        assert isinstance(data, list)
        
        # Each topic should have topic and message fields
        if data:
            topic = data[0]
            assert "topic" in topic
            assert "message" in topic
