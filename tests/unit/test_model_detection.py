"""
Unit tests for model detection utilities.
"""

import os
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import tempfile
import shutil

from backend.utils.model_detection import (
    find_comfyui_installation,
    check_inference_engine_available,
)


class TestComfyUIDetection:
    """Test ComfyUI detection functionality."""

    def test_find_comfyui_with_env_var(self, tmp_path):
        """Test that ComfyUI is found via COMFYUI_DIR environment variable."""
        # Create a mock ComfyUI installation
        comfyui_dir = tmp_path / "custom_comfyui"
        comfyui_dir.mkdir()
        (comfyui_dir / "main.py").touch()
        
        # Set environment variable
        with patch.dict(os.environ, {"COMFYUI_DIR": str(comfyui_dir)}):
            result = find_comfyui_installation()
            assert result == comfyui_dir

    def test_find_comfyui_without_main_py(self, tmp_path):
        """Test that directory without main.py is not considered valid."""
        # Create directory without main.py
        comfyui_dir = tmp_path / "invalid_comfyui"
        comfyui_dir.mkdir()
        
        with patch.dict(os.environ, {"COMFYUI_DIR": str(comfyui_dir)}):
            result = find_comfyui_installation()
            assert result is None

    def test_find_comfyui_in_base_dir(self, tmp_path):
        """Test that ComfyUI is found next to base directory."""
        # Create mock structure: base_dir/../ComfyUI/
        base_dir = tmp_path / "models"
        base_dir.mkdir()
        comfyui_dir = tmp_path / "ComfyUI"
        comfyui_dir.mkdir()
        (comfyui_dir / "main.py").touch()
        
        result = find_comfyui_installation(base_dir=base_dir.parent)
        assert result == comfyui_dir

    def test_find_comfyui_in_current_dir(self, tmp_path, monkeypatch):
        """Test that ComfyUI is found in current directory."""
        comfyui_dir = tmp_path / "ComfyUI"
        comfyui_dir.mkdir()
        (comfyui_dir / "main.py").touch()
        
        # Change to temp directory
        monkeypatch.chdir(tmp_path)
        
        result = find_comfyui_installation()
        # Compare resolved paths since function returns absolute path
        assert result.resolve() == comfyui_dir.resolve()

    def test_find_comfyui_returns_none_when_not_found(self):
        """Test that None is returned when ComfyUI is not found."""
        # Clear environment variable if set
        with patch.dict(os.environ, {}, clear=True):
            # Use a non-existent base directory
            result = find_comfyui_installation(base_dir=Path("/nonexistent"))
            assert result is None

    def test_find_comfyui_checks_multiple_locations(self, tmp_path):
        """Test that multiple locations are checked in order."""
        # Create ComfyUI in the home-like location (last in list)
        comfyui_dir = tmp_path / "home" / "ComfyUI"
        comfyui_dir.mkdir(parents=True)
        (comfyui_dir / "main.py").touch()
        
        # Mock Path.home() to return our tmp_path/home
        with patch("pathlib.Path.home", return_value=tmp_path / "home"):
            result = find_comfyui_installation()
            assert result == comfyui_dir


class TestInferenceEngineDetection:
    """Test inference engine detection functionality."""

    def test_check_vllm_available(self):
        """Test vLLM detection when module is available."""
        # Simply test that the function works - result depends on whether vllm is installed
        result = check_inference_engine_available("vllm")
        assert isinstance(result, bool)

    def test_check_vllm_not_available(self):
        """Test vLLM detection when module is not available."""
        with patch("builtins.__import__", side_effect=ImportError):
            result = check_inference_engine_available("vllm")
            assert result is False

    def test_check_comfyui_available(self, tmp_path):
        """Test ComfyUI detection when installation exists."""
        # Create mock ComfyUI
        base_dir = tmp_path / "models"
        base_dir.mkdir()
        comfyui_dir = tmp_path / "ComfyUI"
        comfyui_dir.mkdir()
        (comfyui_dir / "main.py").touch()
        
        result = check_inference_engine_available("comfyui", base_dir=base_dir.parent)
        assert result is True

    def test_check_comfyui_not_available(self, tmp_path):
        """Test ComfyUI detection when installation doesn't exist."""
        base_dir = tmp_path / "models"
        base_dir.mkdir()
        
        result = check_inference_engine_available("comfyui", base_dir=base_dir.parent)
        assert result is False

    def test_check_diffusers_available(self):
        """Test diffusers detection."""
        # diffusers should be installed as a dependency
        result = check_inference_engine_available("diffusers")
        assert isinstance(result, bool)

    def test_check_transformers_available(self):
        """Test transformers detection."""
        # transformers should be installed as a dependency
        result = check_inference_engine_available("transformers")
        assert isinstance(result, bool)

    def test_check_unknown_engine(self):
        """Test detection of unknown engine returns False."""
        result = check_inference_engine_available("unknown_engine")
        assert result is False

    def test_check_engine_with_base_dir(self, tmp_path):
        """Test that base_dir is properly passed for comfyui detection."""
        base_dir = tmp_path / "models"
        base_dir.mkdir()
        comfyui_dir = base_dir.parent / "ComfyUI"
        comfyui_dir.mkdir()
        (comfyui_dir / "main.py").touch()
        
        result = check_inference_engine_available("comfyui", base_dir=base_dir.parent)
        assert result is True


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_comfyui_dir_is_file_not_directory(self, tmp_path):
        """Test behavior when ComfyUI path exists but is a file."""
        comfyui_file = tmp_path / "ComfyUI"
        comfyui_file.touch()  # Create as file, not directory
        
        with patch.dict(os.environ, {"COMFYUI_DIR": str(comfyui_file)}):
            result = find_comfyui_installation()
            assert result is None

    def test_main_py_is_directory_not_file(self, tmp_path):
        """Test behavior when main.py exists but is a directory."""
        comfyui_dir = tmp_path / "ComfyUI"
        comfyui_dir.mkdir()
        (comfyui_dir / "main.py").mkdir()  # Create as directory
        
        result = find_comfyui_installation(base_dir=tmp_path)
        assert result is None

    def test_symlink_to_comfyui(self, tmp_path):
        """Test that symlinks to ComfyUI are followed."""
        # Create actual ComfyUI installation
        actual_dir = tmp_path / "actual_comfyui"
        actual_dir.mkdir()
        (actual_dir / "main.py").touch()
        
        # Create symlink
        link_dir = tmp_path / "ComfyUI"
        link_dir.symlink_to(actual_dir)
        
        result = find_comfyui_installation(base_dir=tmp_path)
        # Should find either the symlink or the actual directory
        assert result is not None
        assert (result / "main.py").exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
