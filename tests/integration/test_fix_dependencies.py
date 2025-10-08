"""
Test for the fix_dependencies endpoint behavior.
This test verifies that the endpoint correctly excludes torch/torchvision.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import subprocess


class TestFixDependenciesEndpoint:
    """Test the fix_dependencies endpoint."""

    def test_fix_dependencies_excludes_torch_torchvision(self, test_client):
        """Test that fix_dependencies excludes torch and torchvision."""
        
        # We'll mock subprocess.run to avoid actually installing packages
        # and verify that torch/torchvision are not in the install commands
        
        install_commands = []
        
        def mock_subprocess_run(cmd, **kwargs):
            """Mock subprocess.run to capture commands."""
            install_commands.append(cmd)
            
            # Create a mock result
            result = MagicMock()
            result.returncode = 0
            result.stdout = f"Successfully installed {cmd[-1]}"
            result.stderr = ""
            return result
        
        with patch('subprocess.run', side_effect=mock_subprocess_run):
            # Call the endpoint
            response = test_client.post("/api/v1/setup/ai-models/fix-dependencies")
            
            # Verify response structure
            assert response.status_code == 200
            data = response.json()
            
            assert "success" in data
            assert "message" in data
            assert "packages_installed" in data
            assert "packages_failed" in data
            assert "failed_packages" in data
            
            # Verify torch/torchvision are NOT in any install command
            for cmd in install_commands:
                package = cmd[-1]  # Last argument is the package name
                assert not package.startswith("torch=="), \
                    f"torch should not be installed, but found in command: {cmd}"
                assert not package.startswith("torchvision=="), \
                    f"torchvision should not be installed, but found in command: {cmd}"
            
            # Verify key ML packages ARE in the commands
            installed_packages = [cmd[-1] for cmd in install_commands]
            
            # Check for essential ML packages
            ml_packages = ["diffusers>=", "transformers>=", "accelerate>=", "huggingface_hub>="]
            for ml_pkg in ml_packages:
                found = any(ml_pkg in pkg for pkg in installed_packages)
                assert found, f"{ml_pkg} should be in install commands"
            
            print(f"✓ Verified {len(install_commands)} packages were processed")
            print("✓ torch and torchvision correctly excluded")
            print("✓ Essential ML packages included")
    
    def test_fix_dependencies_response_structure(self, test_client):
        """Test that fix_dependencies returns correct response structure."""
        
        def mock_subprocess_run(cmd, **kwargs):
            """Mock subprocess.run."""
            result = MagicMock()
            result.returncode = 0
            result.stdout = "Success"
            result.stderr = ""
            return result
        
        with patch('subprocess.run', side_effect=mock_subprocess_run):
            response = test_client.post("/api/v1/setup/ai-models/fix-dependencies")
            
            assert response.status_code == 200
            data = response.json()
            
            # Verify all expected fields
            assert "success" in data
            assert "message" in data
            assert "stdout" in data
            assert "stderr" in data
            assert "packages_installed" in data
            assert "packages_failed" in data
            assert "failed_packages" in data
            
            # Verify types
            assert isinstance(data["success"], bool)
            assert isinstance(data["message"], str)
            assert isinstance(data["stdout"], str)
            assert isinstance(data["stderr"], str)
            assert isinstance(data["packages_installed"], int)
            assert isinstance(data["packages_failed"], int)
            assert isinstance(data["failed_packages"], list)
    
    def test_fix_dependencies_handles_failures(self, test_client):
        """Test that fix_dependencies handles package installation failures."""
        
        def mock_subprocess_run(cmd, **kwargs):
            """Mock subprocess.run with some failures."""
            package = cmd[-1]
            result = MagicMock()
            
            # Fail on beautifulsoup4 to test error handling
            if "beautifulsoup4" in package:
                result.returncode = 1
                result.stdout = ""
                result.stderr = "Error installing beautifulsoup4"
            else:
                result.returncode = 0
                result.stdout = f"Successfully installed {package}"
                result.stderr = ""
            
            return result
        
        with patch('subprocess.run', side_effect=mock_subprocess_run):
            response = test_client.post("/api/v1/setup/ai-models/fix-dependencies")
            
            assert response.status_code == 200
            data = response.json()
            
            # Should report partial success
            assert data["success"] == False  # At least one package failed
            assert data["packages_failed"] > 0
            assert len(data["failed_packages"]) > 0
            assert "beautifulsoup4" in data["message"] or any(
                "beautifulsoup4" in pkg for pkg in data["failed_packages"]
            )
