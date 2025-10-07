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
    
    def test_ai_models_status(self, test_client):
        """Test getting AI models status."""
        response = test_client.get("/api/v1/setup/ai-models/status")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure
        assert "system" in data
        assert "models_directory" in data
        assert "installed_models" in data
        assert "available_models" in data
        assert "setup_script_available" in data
        
        # Verify system info
        system = data["system"]
        assert "python_version" in system
        assert "platform" in system
        assert "gpu_available" in system
        
        # The setup script should be available in the repository
        assert data["setup_script_available"] is True, \
            "setup_ai_models.py should be found in repository root"
    
    def test_ai_models_analyze(self, test_client):
        """Test analyzing system for AI models."""
        response = test_client.post("/api/v1/setup/ai-models/analyze")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure
        assert "success" in data
        assert "output" in data
        
        # If successful, output should contain system analysis
        if data["success"]:
            assert "System Information" in data["output"] or \
                   "Analyzing system" in data["output"], \
                   "Output should contain system analysis information"
        else:
            # If it fails, there should be an error message
            assert "error" in data
    
    def test_setup_script_path_calculation(self):
        """Test that the setup script path is calculated correctly.
        
        This test verifies the fix for the issue where setup_ai_models.py
        was not found because of incorrect path calculation in the API.
        """
        from pathlib import Path
        
        # Get project root from test file location
        # Test file is at: tests/integration/test_setup_api.py
        # Project root is 2 parents up
        project_root = Path(__file__).parent.parent.parent
        
        # Verify setup script exists at project root
        setup_script = project_root / "setup_ai_models.py"
        assert setup_script.exists(), \
            f"setup_ai_models.py should exist at {setup_script}"
        
        # Now verify the routes/setup.py uses correct path calculation
        routes_setup = project_root / "src" / "backend" / "api" / "routes" / "setup.py"
        assert routes_setup.exists(), f"routes/setup.py should exist at {routes_setup}"
        
        # From routes/setup.py, parents[4] should lead to project root
        # routes/ -> api/ -> backend/ -> src/ -> project_root
        calculated_root = routes_setup.parents[4]
        assert calculated_root == project_root, \
            f"Path calculation should lead to {project_root}, got {calculated_root}"
