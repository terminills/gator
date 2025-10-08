"""
Test Analytics API endpoints.

Tests the analytics API endpoints for metrics and health checks
with real database queries.
"""

import pytest
from fastapi.testclient import TestClient


class TestAnalyticsAPI:
    """Test analytics API endpoints."""

    def test_metrics_endpoint_returns_data(self, test_client):
        """Test that /metrics endpoint returns real metrics."""
        response = test_client.get("/api/v1/analytics/metrics")

        assert response.status_code == 200
        data = response.json()

        # Verify all expected fields are present
        assert "personas_created" in data
        assert "content_generated" in data
        assert "api_requests_today" in data
        assert "system_uptime" in data
        assert "status" in data

        # Verify data types
        assert isinstance(data["personas_created"], int)
        assert isinstance(data["content_generated"], int)
        assert isinstance(data["api_requests_today"], int)
        assert isinstance(data["system_uptime"], str)
        assert isinstance(data["status"], str)

        # Status should be operational or degraded
        assert data["status"] in ["operational", "degraded"]

    def test_metrics_with_personas(self, test_client, sample_persona_data):
        """Test that metrics reflect actual database content."""
        # Create a persona
        create_response = test_client.post(
            "/api/v1/personas/", json=sample_persona_data
        )
        assert create_response.status_code == 201

        # Get metrics
        metrics_response = test_client.get("/api/v1/analytics/metrics")
        assert metrics_response.status_code == 200

        data = metrics_response.json()

        # Should have at least 1 persona now
        assert data["personas_created"] >= 1
        assert data["status"] == "operational"

    def test_health_endpoint_returns_status(self, test_client):
        """Test that /health endpoint returns system health."""
        response = test_client.get("/api/v1/analytics/health")

        assert response.status_code == 200
        data = response.json()

        # Verify all expected fields are present
        assert "api" in data
        assert "database" in data
        assert "ai_models" in data
        assert "content_generation" in data
        assert "timestamp" in data

        # API should always be healthy if we can reach it
        assert data["api"] == "healthy"

        # Database should be healthy or unhealthy
        assert data["database"] in ["healthy", "unhealthy", "unknown"]

        # AI models status
        assert data["ai_models"] in ["not_loaded", "configured"]

        # Content generation status
        assert data["content_generation"] in ["not_configured", "configured"]

    def test_health_database_connectivity(self, test_client):
        """Test that health check actually tests database connection."""
        response = test_client.get("/api/v1/analytics/health")

        assert response.status_code == 200
        data = response.json()

        # With a working test database, should report healthy
        assert data["database"] == "healthy"

    def test_uptime_format(self, test_client):
        """Test that uptime is in correct format."""
        response = test_client.get("/api/v1/analytics/metrics")

        assert response.status_code == 200
        data = response.json()

        uptime = data["system_uptime"]

        # Should be in format like "0h 0m" or "unknown"
        assert "h" in uptime or uptime == "unknown"
        if "h" in uptime:
            assert "m" in uptime


@pytest.fixture
def sample_persona_data():
    """Sample persona data for testing."""
    return {
        "name": "Analytics Test Persona",
        "appearance": "Professional appearance for testing",
        "personality": "Analytical, focused, detail-oriented",
        "content_themes": ["analytics", "testing", "data"],
        "style_preferences": {
            "aesthetic": "professional",
            "tone": "analytical",
            "voice_style": "technical",
        },
    }
