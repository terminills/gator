"""
Test API endpoints integration.

Tests the full API stack including routing, validation, and database operations.
"""

import pytest
from fastapi.testclient import TestClient


class TestPersonaAPI:
    """Test persona API endpoints."""
    
    def test_root_endpoint(self, test_client):
        """Test the root API endpoint."""
        response = test_client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Gator AI Influencer Platform"
        assert data["version"] == "0.1.0"
        assert data["status"] == "operational"
    
    def test_health_check(self, test_client):
        """Test the health check endpoint."""
        response = test_client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "database" in data
        assert "timestamp" in data
    
    def test_create_persona_success(self, test_client):
        """Test successful persona creation via API."""
        persona_data = {
            "name": "Test API Persona",
            "appearance": "Young professional with dark hair and glasses",
            "personality": "Analytical, detail-oriented, friendly, approachable",
            "content_themes": ["technology", "education", "productivity"],
            "style_preferences": {"style": "minimalist", "tone": "professional", "aesthetic": "modern"}
        }
        
        response = test_client.post("/api/v1/personas/", json=persona_data)
        
        assert response.status_code == 201
        data = response.json()
        
        assert data["name"] == persona_data["name"]
        assert data["appearance"] == persona_data["appearance"] 
        assert data["personality"] == persona_data["personality"]
        assert data["content_themes"] == persona_data["content_themes"]
        assert data["style_preferences"] == persona_data["style_preferences"]
        assert data["id"] is not None
        assert data["created_at"] is not None
        assert data["is_active"] is True
        assert data["generation_count"] == 0
    
    def test_create_persona_validation_error(self, test_client):
        """Test persona creation with validation errors."""
        # Missing required fields
        persona_data = {
            "name": "Test",
            # Missing appearance and personality
        }
        
        response = test_client.post("/api/v1/personas/", json=persona_data)
        
        assert response.status_code == 422  # Validation error
        assert "detail" in response.json()
    
    def test_list_personas_empty(self, test_client):
        """Test listing personas when none exist."""
        response = test_client.get("/api/v1/personas/")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 0
    
    def test_get_persona_not_found(self, test_client):
        """Test getting a non-existent persona."""
        response = test_client.get("/api/v1/personas/nonexistent-id")
        
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]
    
    def test_persona_crud_flow(self, test_client):
        """Test complete CRUD flow for personas."""
        # 1. Create persona
        persona_data = {
            "name": "CRUD Test Persona",
            "appearance": "Test appearance for CRUD operations",
            "personality": "Test personality for CRUD operations", 
            "content_themes": ["test", "crud"],
            "style_preferences": {"style": "test-style"}
        }
        
        create_response = test_client.post("/api/v1/personas/", json=persona_data)
        assert create_response.status_code == 201
        created = create_response.json()
        persona_id = created["id"]
        
        # 2. Get persona
        get_response = test_client.get(f"/api/v1/personas/{persona_id}")
        assert get_response.status_code == 200
        retrieved = get_response.json()
        assert retrieved["id"] == persona_id
        assert retrieved["name"] == persona_data["name"]
        
        # 3. List personas (should include our created one)
        list_response = test_client.get("/api/v1/personas/")
        assert list_response.status_code == 200
        personas_list = list_response.json()
        assert len(personas_list) >= 1
        assert any(p["id"] == persona_id for p in personas_list)
        
        # 4. Update persona
        update_data = {"name": "Updated CRUD Persona"}
        update_response = test_client.put(f"/api/v1/personas/{persona_id}", json=update_data)
        assert update_response.status_code == 200
        updated = update_response.json()
        assert updated["name"] == "Updated CRUD Persona"
        assert updated["updated_at"] is not None
        
        # 5. Delete persona
        delete_response = test_client.delete(f"/api/v1/personas/{persona_id}")
        assert delete_response.status_code == 204
        
        # 6. Verify persona is soft deleted (inactive)
        final_get = test_client.get(f"/api/v1/personas/{persona_id}")
        assert final_get.status_code == 200
        final_persona = final_get.json()
        assert final_persona["is_active"] is False


class TestContentAPI:
    """Test content API endpoints."""
    
    def test_list_content_placeholder(self, test_client):
        """Test content listing placeholder endpoint."""
        response = test_client.get("/api/v1/content/")
        
        assert response.status_code == 200
        data = response.json()
        assert "placeholder" in data["status"]
    
    def test_generate_content_placeholder(self, test_client):
        """Test content generation placeholder endpoint."""
        response = test_client.post("/api/v1/content/generate")
        
        assert response.status_code == 202
        data = response.json()
        assert "placeholder" in data["status"]


class TestAnalyticsAPI:
    """Test analytics API endpoints."""
    
    def test_get_metrics(self, test_client):
        """Test metrics endpoint."""
        response = test_client.get("/api/v1/analytics/metrics")
        
        assert response.status_code == 200
        data = response.json()
        assert "personas_created" in data
        assert "content_generated" in data
        assert "api_requests_today" in data
        assert data["status"] == "operational"
    
    def test_get_system_health(self, test_client):
        """Test system health endpoint."""
        response = test_client.get("/api/v1/analytics/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["api"] == "healthy"
        assert data["database"] == "healthy"
        assert "timestamp" in data