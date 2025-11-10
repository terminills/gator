"""
Test host header configuration.

Tests that the TrustedHostMiddleware is properly configured to allow
access from different network hosts.
"""

import pytest
from fastapi.testclient import TestClient
from backend.api.main import create_app
from backend.config.settings import Settings


class TestHostHeaderConfiguration:
    """Test host header middleware configuration."""

    def test_localhost_access_allowed(self, test_client):
        """Test that localhost access is allowed."""
        response = test_client.get(
            "/health",
            headers={"Host": "localhost:8000"}
        )
        assert response.status_code == 200

    def test_ip_address_access_allowed(self, test_client):
        """Test that IP address access is allowed."""
        response = test_client.get(
            "/health",
            headers={"Host": "192.168.1.100:8000"}
        )
        assert response.status_code == 200

    def test_custom_hostname_access_allowed(self, test_client):
        """Test that custom hostname access is allowed."""
        response = test_client.get(
            "/health",
            headers={"Host": "gator.local:8000"}
        )
        assert response.status_code == 200

    def test_domain_name_access_allowed(self, test_client):
        """Test that domain name access is allowed."""
        response = test_client.get(
            "/health",
            headers={"Host": "example.com"}
        )
        assert response.status_code == 200

    def test_wildcard_hosts_default_setting(self):
        """Test that default settings allow wildcard hosts."""
        settings = Settings()
        assert "*" in settings.allowed_hosts
        assert "*" in settings.allowed_origins

    def test_cors_headers_present(self, test_client):
        """Test that CORS headers are properly set."""
        response = test_client.options(
            "/health",
            headers={
                "Origin": "http://192.168.1.50:3000",
                "Access-Control-Request-Method": "GET",
            }
        )
        # Should have CORS headers with wildcard origin
        assert "access-control-allow-origin" in response.headers
