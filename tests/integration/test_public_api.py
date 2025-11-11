"""
Test public API endpoints integration.

Tests the public-facing API endpoints for viewing personas and content.
"""

import pytest
from fastapi.testclient import TestClient


class TestPublicAPI:
    """Test public API endpoints."""

    def test_public_categories(self, test_client):
        """Test public categories endpoint."""
        response = test_client.get("/api/v1/public/categories")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        # Should return empty list or default categories when no personas exist

    def test_public_personas(self, test_client):
        """Test public personas listing endpoint."""
        response = test_client.get("/api/v1/public/personas")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_public_personas_with_limit(self, test_client):
        """Test public personas listing with limit parameter."""
        response = test_client.get("/api/v1/public/personas?limit=5")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) <= 5

    def test_public_personas_with_category(self, test_client):
        """Test public personas listing filtered by category."""
        response = test_client.get("/api/v1/public/personas?category=technology")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_public_feed(self, test_client):
        """Test public feed endpoint."""
        response = test_client.get("/api/v1/public/feed")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_public_feed_with_limit(self, test_client):
        """Test public feed with limit parameter."""
        response = test_client.get("/api/v1/public/feed?limit=10")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) <= 10

    def test_old_public_path_returns_404(self, test_client):
        """Test that old /public path (without /api/v1) returns 404."""
        response = test_client.get("/public/categories")

        assert response.status_code == 404

    def test_public_persona_detail_not_found(self, test_client):
        """Test getting a non-existent public persona."""
        response = test_client.get("/api/v1/public/personas/00000000-0000-0000-0000-000000000000")

        assert response.status_code == 404

    def test_public_persona_gallery_empty(self, test_client):
        """Test getting gallery for a persona (returns empty list for invalid ID)."""
        response = test_client.get("/api/v1/public/personas/00000000-0000-0000-0000-000000000000/gallery")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 0
