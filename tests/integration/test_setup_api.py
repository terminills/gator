"""
Test Setup API endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import tempfile
import shutil


class TestSetupAPI:
    """Test setup API endpoints."""
    
    def test_get_setup_status(self, test_client):
        """Test getting setup status."""
        response = test_client.get("/api/v1/setup/status")
        
        assert response.status_code == 200
        data = response.json()
        assert "env_file_exists" in data
        assert "env_file_path" in data
        assert "configured_sections" in data
        assert "current_config" in data
    
    def test_get_configuration_template(self, test_client):
        """Test getting configuration template."""
        response = test_client.get("/api/v1/setup/template")
        
        assert response.status_code == 200
        data = response.json()
        assert "sections" in data
        
        sections = data["sections"]
        # Verify all expected sections are present
        assert "database" in sections
        assert "ai_models" in sections
        assert "security" in sections
        assert "social_media" in sections
        assert "dns" in sections
        assert "application" in sections
        
        # Verify section structure
        db_section = sections["database"]
        assert "title" in db_section
        assert "fields" in db_section
        assert "DATABASE_URL" in db_section["fields"]
    
    def test_update_configuration_validation(self, test_client):
        """Test configuration update with validation."""
        # Test with invalid SMTP port
        config = {
            "smtp_port": 99999  # Invalid port
        }
        
        response = test_client.post("/api/v1/setup/config", json=config)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == False
        assert "validation" in data
        assert len(data["validation"]["errors"]) > 0
    
    def test_update_configuration_success(self, test_client):
        """Test successful configuration update."""
        config = {
            "environment": "test",
            "debug": True,
            "log_level": "INFO"
        }
        
        response = test_client.post("/api/v1/setup/config", json=config)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        assert data["restart_required"] == True
        assert "message" in data
        assert "validation" in data
    
    def test_update_configuration_empty(self, test_client):
        """Test configuration update with no values."""
        config = {}
        
        response = test_client.post("/api/v1/setup/config", json=config)
        
        assert response.status_code == 400
        data = response.json()
        assert "detail" in data
        assert "No configuration values" in data["detail"]
