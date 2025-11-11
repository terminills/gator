"""
Integration tests for RSS API endpoints.

Tests the new /api/v1/feeds/rss endpoints that match frontend expectations.
"""

import pytest
from uuid import uuid4
from unittest.mock import AsyncMock, MagicMock
from fastapi.testclient import TestClient

from backend.api.main import app
from backend.services.rss_ingestion_service import RSSIngestionService
from backend.models.feed import RSSFeedResponse


@pytest.fixture
def mock_rss_service():
    """Create a mock RSS ingestion service."""
    service = MagicMock(spec=RSSIngestionService)

    # Mock list_feeds
    service.list_feeds = AsyncMock(
        return_value=[
            RSSFeedResponse(
                id=uuid4(),
                name="Test Feed",
                url="https://example.com/rss",
                description="Test RSS feed",
                categories=["technology"],
                fetch_frequency_hours=6,
                last_fetched=None,
                is_active=True,
                created_at="2024-01-01T00:00:00Z",
                updated_at="2024-01-01T00:00:00Z",
            )
        ]
    )

    # Mock add_feed
    service.add_feed = AsyncMock(
        return_value=RSSFeedResponse(
            id=uuid4(),
            name="New Feed",
            url="https://example.com/new-rss",
            description="New RSS feed",
            categories=["news"],
            fetch_frequency_hours=6,
            last_fetched=None,
            is_active=True,
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z",
        )
    )

    # Mock fetch_feed
    service.fetch_feed = AsyncMock(
        return_value={
            "feed_id": str(uuid4()),
            "new_items": 5,
            "last_fetched": "2024-01-01T00:00:00Z",
            "status": "success",
        }
    )

    # Mock delete_feed
    service.delete_feed = AsyncMock(return_value=True)

    return service


class TestRSSAPIEndpoints:
    """Test suite for RSS API endpoints."""

    def test_list_rss_feeds_endpoint_exists(self):
        """Test that GET /api/v1/feeds/rss endpoint exists."""
        client = TestClient(app)
        response = client.get("/api/v1/feeds/rss")
        # Should not return 404
        assert response.status_code != 404

    def test_add_rss_feed_endpoint_exists(self):
        """Test that POST /api/v1/feeds/rss endpoint exists."""
        client = TestClient(app)
        response = client.post(
            "/api/v1/feeds/rss",
            json={
                "name": "Test Feed",
                "url": "https://example.com/rss",
                "categories": ["test"],
            },
        )
        # Should not return 404
        assert response.status_code != 404

    def test_refresh_rss_feed_endpoint_exists(self):
        """Test that POST /api/v1/feeds/rss/{id}/refresh endpoint exists."""
        client = TestClient(app)
        test_id = str(uuid4())
        response = client.post(f"/api/v1/feeds/rss/{test_id}/refresh")
        # Should not return 404 (may return 400/500 for invalid ID, but not 404)
        assert response.status_code != 404

    def test_delete_rss_feed_endpoint_exists(self):
        """Test that DELETE /api/v1/feeds/rss/{id} endpoint exists."""
        client = TestClient(app)
        test_id = str(uuid4())
        response = client.delete(f"/api/v1/feeds/rss/{test_id}")
        # Should not return 404 (may return 400/500 for invalid ID, but not 404)
        assert response.status_code != 404

    def test_list_rss_feeds_response_structure(self):
        """Test that GET /api/v1/feeds/rss returns correct structure."""
        client = TestClient(app)
        response = client.get("/api/v1/feeds/rss")

        # Should return 200 or error, but response should be JSON
        assert response.headers.get("content-type") == "application/json"

        data = response.json()

        # Check that response has the expected structure for frontend
        if response.status_code == 200:
            assert "feeds" in data
            assert "total" in data
            assert isinstance(data["feeds"], list)
            assert isinstance(data["total"], int)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
