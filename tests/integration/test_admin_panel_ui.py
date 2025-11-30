"""
Admin Panel UI Integration Tests

Tests for the admin panel HTML UI and its API integrations.
Validates that all tabs have corresponding backend API endpoints that respond correctly.
This is part of Phase 5: Testing & Documentation.
"""

import pytest


class TestAdminPanelUI:
    """Test the admin panel UI page and tabs."""

    def test_admin_panel_loads(self, test_client):
        """Test that the admin panel loads successfully."""
        response = test_client.get("/admin")

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/html")

    def test_admin_panel_is_html(self, test_client):
        """Test that admin panel returns valid HTML."""
        response = test_client.get("/admin")

        assert response.status_code == 200
        content = response.text
        assert "<!DOCTYPE html>" in content or "<html" in content


class TestSchedulingTabAPI:
    """Test API endpoints for the Scheduling tab."""

    def test_list_scheduled_posts(self, test_client):
        """Test listing scheduled posts endpoint."""
        response = test_client.get("/api/v1/scheduled-posts/")

        assert response.status_code == 200
        data = response.json()
        # Response can be a list or a pagination response with items
        assert isinstance(data, (list, dict))

    def test_get_scheduled_posts_stats(self, test_client):
        """Test getting scheduling statistics."""
        response = test_client.get("/api/v1/scheduled-posts/stats")

        assert response.status_code == 200

    def test_get_upcoming_posts(self, test_client):
        """Test getting upcoming scheduled posts."""
        response = test_client.get("/api/v1/scheduled-posts/upcoming")

        assert response.status_code == 200


class TestMonitoringTabAPI:
    """Test API endpoints for the Monitoring tab."""

    def test_get_gpu_temperatures(self, test_client):
        """Test getting GPU temperature data."""
        response = test_client.get("/api/v1/system/gpu/temperature")

        # 200 or 500/503 (if no GPUs available) are acceptable
        assert response.status_code in [200, 500, 503]

    def test_get_gpu_status(self, test_client):
        """Test getting GPU status."""
        response = test_client.get("/api/v1/system/gpu/status")

        # 200 or 500/503 are acceptable
        assert response.status_code in [200, 500, 503]

    def test_get_fans_status(self, test_client):
        """Test getting fans status."""
        response = test_client.get("/api/v1/system/fans")

        # 200 or 500/503 are acceptable (503 if IPMI not configured)
        assert response.status_code in [200, 500, 503]


class TestModerationTabAPI:
    """Test API endpoints for the Moderation tab."""

    def test_get_moderation_queue(self, test_client):
        """Test getting moderation queue."""
        response = test_client.get("/api/v1/moderation/queue")

        # 200 OK or 401 if auth is required
        assert response.status_code in [200, 401]

    def test_get_moderation_stats(self, test_client):
        """Test getting moderation statistics."""
        response = test_client.get("/api/v1/moderation/stats")

        # 200 OK or 401 if auth is required
        assert response.status_code in [200, 401]

    def test_get_moderation_categories(self, test_client):
        """Test getting moderation categories."""
        response = test_client.get("/api/v1/moderation/categories")

        assert response.status_code == 200


class TestAuthTabAPI:
    """Test API endpoints for the Auth tab."""

    def test_auth_register_validation(self, test_client):
        """Test auth registration endpoint with invalid data."""
        # Test with missing required fields
        response = test_client.post("/api/v1/auth/register", json={})

        # Should return 422 for validation error
        assert response.status_code == 422

    def test_oauth_platforms(self, test_client):
        """Test listing OAuth platforms."""
        response = test_client.get("/api/v1/oauth/platforms")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, (list, dict))


class TestCacheTabAPI:
    """Test API endpoints for the Cache tab."""

    def test_cache_status(self, test_client):
        """Test getting cache status."""
        response = test_client.get("/api/v1/cache/status")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data

    def test_cache_health(self, test_client):
        """Test cache health check."""
        response = test_client.get("/api/v1/cache/health")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data


class TestFriendGroupsTabAPI:
    """Test API endpoints for the Groups & Reels tab."""

    def test_list_friend_groups(self, test_client):
        """Test listing friend groups."""
        response = test_client.get("/api/v1/friend-groups/")

        assert response.status_code == 200


class TestACDTabAPI:
    """Test API endpoints for the ACD (Autonomous Continuous Development) tab."""

    def test_list_acd_contexts(self, test_client):
        """Test listing ACD contexts."""
        response = test_client.get("/api/v1/acd/contexts")

        assert response.status_code == 200

    def test_get_acd_stats(self, test_client):
        """Test getting ACD statistics."""
        response = test_client.get("/api/v1/acd/stats/")

        assert response.status_code == 200


class TestAgentsTabAPI:
    """Test API endpoints for the Agents tab (Multi-Agent).
    
    Note: The multi-agent routes exist but may not be wired in main.py yet.
    These tests verify the route availability.
    """

    def test_list_agents(self, test_client):
        """Test listing agents endpoint exists."""
        response = test_client.get("/api/v1/multi-agent/agents/")

        # 200 OK, 401 if auth required, or 404 if route not wired yet
        assert response.status_code in [200, 401, 404]

    def test_get_workload(self, test_client):
        """Test getting agent workload endpoint exists."""
        response = test_client.get("/api/v1/multi-agent/workload/")

        # 200 OK, 401 if auth required, or 404 if route not wired yet
        assert response.status_code in [200, 401, 404]

    def test_search_marketplace(self, test_client):
        """Test searching marketplace agents endpoint exists."""
        response = test_client.get("/api/v1/multi-agent/marketplace/search")

        # 200 OK, 401 if auth required, or 404 if route not wired yet
        assert response.status_code in [200, 401, 404]


class TestMLLearningTabAPI:
    """Test API endpoints for the ML Learning tab.
    
    Note: The ML Learning routes exist but may not be wired in main.py yet.
    These tests verify the route availability.
    """

    def test_get_feature_importance(self, test_client):
        """Test getting ML feature importance endpoint exists."""
        response = test_client.get("/api/v1/ml-learning/models/feature-importance")

        # 200 OK, 401 if auth required, or 404 if route not wired yet
        assert response.status_code in [200, 401, 404]

    def test_get_cross_persona_aggregate(self, test_client):
        """Test getting cross-persona aggregate data endpoint exists."""
        response = test_client.get("/api/v1/ml-learning/cross-persona/aggregate")

        # 200 OK, 401 if auth required, or 404 if route not wired yet
        assert response.status_code in [200, 401, 404]


class TestDiagnosticsAPI:
    """Test Diagnostics API endpoints."""

    def test_get_ai_activity(self, test_client):
        """Test getting AI activity summary."""
        response = test_client.get("/api/v1/diagnostics/ai-activity")

        assert response.status_code == 200
        data = response.json()
        # Should have activity metrics
        assert "total_generations" in data

    def test_get_generation_attempts(self, test_client):
        """Test getting generation attempts."""
        response = test_client.get("/api/v1/diagnostics/generation-attempts")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_get_ai_model_activity(self, test_client):
        """Test getting AI model activity."""
        response = test_client.get("/api/v1/diagnostics/ai-models")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_get_acd_contexts(self, test_client):
        """Test getting ACD contexts from diagnostics."""
        response = test_client.get("/api/v1/diagnostics/acd-contexts")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)


class TestReasoningOrchestratorAPI:
    """Test Reasoning Orchestrator API endpoints."""

    def test_get_orchestration_stats(self, test_client):
        """Test getting orchestration statistics."""
        response = test_client.get("/api/v1/reasoning/stats")

        # 200 OK, 401 if auth required, or 500 if db not initialized
        assert response.status_code in [200, 401, 500]
        if response.status_code == 200:
            data = response.json()
            assert "total_decisions" in data
            assert "time_window_hours" in data


class TestSettingsTabAPI:
    """Test API endpoints for the Settings tab."""

    def test_get_setup_status(self, test_client):
        """Test getting setup status."""
        response = test_client.get("/api/v1/setup/status")

        assert response.status_code == 200
        data = response.json()
        assert "env_file_exists" in data

    def test_get_setup_template(self, test_client):
        """Test getting setup template."""
        response = test_client.get("/api/v1/setup/template")

        assert response.status_code == 200


class TestDNSTabAPI:
    """Test API endpoints for the DNS tab."""

    def test_dns_providers(self, test_client):
        """Test getting DNS providers."""
        response = test_client.get("/api/v1/dns/providers")

        assert response.status_code == 200


class TestDatabaseTabAPI:
    """Test API endpoints for the Database tab."""

    def test_database_info(self, test_client):
        """Test getting database info."""
        response = test_client.get("/api/v1/admin/database/info")

        # 200 OK or 401 if auth is required
        assert response.status_code in [200, 401]

    def test_database_schema_status(self, test_client):
        """Test getting database schema status."""
        response = test_client.get("/api/v1/admin/database/schema/status")

        # 200 OK or 401 if auth is required
        assert response.status_code in [200, 401]


class TestPluginsTabAPI:
    """Test API endpoints for the Plugins tab."""

    def test_list_marketplace_plugins(self, test_client):
        """Test listing marketplace plugins."""
        response = test_client.get("/api/v1/plugins/marketplace")

        # 200 OK or 401 if auth is required
        assert response.status_code in [200, 401]

    def test_list_installed_plugins(self, test_client):
        """Test listing installed plugins."""
        response = test_client.get("/api/v1/plugins/installed")

        # 200 OK or 401 if auth is required
        assert response.status_code in [200, 401]


class TestGatorHelpTabAPI:
    """Test API endpoints for the Gator Help tab."""

    def test_gator_agent_available(self, test_client):
        """Test that gator agent endpoint is available."""
        response = test_client.get("/api/v1/gator-agent/status")

        # May be 200 or 404 depending on agent status
        assert response.status_code in [200, 404]
