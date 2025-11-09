"""
Unit tests for ROCm detection utilities.
"""

import pytest
from unittest.mock import patch, mock_open, MagicMock
from pathlib import Path

# Import the module to test
from backend.utils.rocm_utils import (
    ROCmVersionInfo,
    detect_rocm_version,
    parse_rocm_version,
    get_pytorch_index_url,
    get_pytorch_install_command,
    get_recommended_pytorch_version,
    check_pytorch_installation,
)


class TestROCmVersionInfo:
    """Test ROCmVersionInfo class."""
    
    def test_version_info_creation(self):
        """Test creating version info object."""
        version = ROCmVersionInfo("6.5.0", 6, 5, 0)
        assert version.version == "6.5.0"
        assert version.major == 6
        assert version.minor == 5
        assert version.patch == 0
        assert version.is_installed is True
    
    def test_is_6_5_or_later(self):
        """Test detection of ROCm 6.5+."""
        # ROCm 6.5 should be True
        version = ROCmVersionInfo("6.5.0", 6, 5, 0)
        assert version.is_6_5_or_later is True
        
        # ROCm 6.6 should be True
        version = ROCmVersionInfo("6.6.0", 6, 6, 0)
        assert version.is_6_5_or_later is True
        
        # ROCm 7.0 should be True
        version = ROCmVersionInfo("7.0.0", 7, 0, 0)
        assert version.is_6_5_or_later is True
        
        # ROCm 6.4 should be False
        version = ROCmVersionInfo("6.4.0", 6, 4, 0)
        assert version.is_6_5_or_later is False
        
        # ROCm 5.7 should be False
        version = ROCmVersionInfo("5.7.1", 5, 7, 1)
        assert version.is_6_5_or_later is False
    
    def test_short_version(self):
        """Test short version property."""
        version = ROCmVersionInfo("6.5.0", 6, 5, 0)
        assert version.short_version == "6.5"
        
        version = ROCmVersionInfo("5.7.1", 5, 7, 1)
        assert version.short_version == "5.7"


class TestParseROCmVersion:
    """Test ROCm version parsing."""
    
    def test_parse_standard_version(self):
        """Test parsing standard version strings."""
        version = parse_rocm_version("6.5.0")
        assert version.major == 6
        assert version.minor == 5
        assert version.patch == 0
        assert version.version == "6.5.0"
    
    def test_parse_version_with_suffix(self):
        """Test parsing version with build suffix."""
        version = parse_rocm_version("6.5.0-98")
        assert version.major == 6
        assert version.minor == 5
        assert version.patch == 0
        assert version.version == "6.5.0"
    
    def test_parse_two_part_version(self):
        """Test parsing two-part version."""
        version = parse_rocm_version("6.5")
        assert version.major == 6
        assert version.minor == 5
        assert version.patch == 0
    
    def test_parse_invalid_version(self):
        """Test parsing invalid version returns None."""
        assert parse_rocm_version("invalid") is None
        assert parse_rocm_version("") is None


class TestGetPyTorchIndexURL:
    """Test PyTorch index URL generation."""
    
    def test_rocm_6_5_stable(self):
        """Test URL for ROCm 6.5 stable."""
        version = ROCmVersionInfo("6.5.0", 6, 5, 0)
        url = get_pytorch_index_url(version, use_nightly=False)
        assert url == "https://download.pytorch.org/whl/rocm6.5/"
    
    def test_rocm_6_5_nightly(self):
        """Test URL for ROCm 6.5 nightly."""
        version = ROCmVersionInfo("6.5.0", 6, 5, 0)
        url = get_pytorch_index_url(version, use_nightly=True)
        assert url == "https://download.pytorch.org/whl/nightly/rocm6.5/"
    
    def test_rocm_6_6_stable(self):
        """Test URL for ROCm 6.6 stable."""
        version = ROCmVersionInfo("6.6.0", 6, 6, 0)
        url = get_pytorch_index_url(version, use_nightly=False)
        assert url == "https://download.pytorch.org/whl/rocm6.6/"
    
    def test_rocm_6_4(self):
        """Test URL for ROCm 6.4."""
        version = ROCmVersionInfo("6.4.0", 6, 4, 0)
        url = get_pytorch_index_url(version, use_nightly=False)
        assert url == "https://download.pytorch.org/whl/rocm6.4/"
    
    def test_rocm_5_7(self):
        """Test URL for ROCm 5.7 (legacy)."""
        version = ROCmVersionInfo("5.7.1", 5, 7, 1)
        url = get_pytorch_index_url(version, use_nightly=False)
        assert url == "https://download.pytorch.org/whl/rocm5.7/"
    
    def test_no_rocm(self):
        """Test URL when no ROCm detected."""
        url = get_pytorch_index_url(None, use_nightly=False)
        assert url == "https://download.pytorch.org/whl/cpu"


class TestGetPyTorchInstallCommand:
    """Test PyTorch installation command generation."""
    
    def test_rocm_6_5_stable(self):
        """Test install command for ROCm 6.5 stable."""
        version = ROCmVersionInfo("6.5.0", 6, 5, 0)
        command, metadata = get_pytorch_install_command(
            version, use_nightly=False, include_torchvision=True
        )
        
        assert "torch" in command
        assert "torchvision" in command
        assert "rocm6.5" in command
        assert "--pre" not in command
        assert metadata["rocm_version"] == "6.5.0"
        assert metadata["nightly"] is False
    
    def test_rocm_6_5_nightly(self):
        """Test install command for ROCm 6.5 nightly."""
        version = ROCmVersionInfo("6.5.0", 6, 5, 0)
        command, metadata = get_pytorch_install_command(
            version, use_nightly=True, include_torchvision=True, include_torchaudio=True
        )
        
        assert "torch" in command
        assert "torchvision" in command
        assert "torchaudio" in command
        assert "--pre" in command
        assert "nightly/rocm6.5" in command
        assert metadata["nightly"] is True
    
    def test_rocm_5_7(self):
        """Test install command for ROCm 5.7."""
        version = ROCmVersionInfo("5.7.1", 5, 7, 1)
        command, metadata = get_pytorch_install_command(
            version, use_nightly=False, include_torchvision=True
        )
        
        assert "torch" in command
        assert "torchvision" in command
        assert "rocm5.7" in command
        assert metadata["rocm_version"] == "5.7.1"


class TestGetRecommendedPyTorchVersion:
    """Test recommended PyTorch version detection."""
    
    def test_rocm_6_5_recommendations(self):
        """Test recommendations for ROCm 6.5+."""
        version = ROCmVersionInfo("6.5.0", 6, 5, 0)
        recommended = get_recommended_pytorch_version(version)
        
        assert "torch" in recommended
        assert "nightly_available" in recommended
        assert recommended["nightly_available"] is True
        assert "6.5" in recommended["note"]
    
    def test_rocm_5_7_recommendations(self):
        """Test recommendations for ROCm 5.7."""
        version = ROCmVersionInfo("5.7.1", 5, 7, 1)
        recommended = get_recommended_pytorch_version(version)
        
        assert "torch" in recommended
        assert "2.3.1+rocm5.7" in recommended["torch"]
        assert recommended.get("nightly_available", True) is False
    
    def test_no_rocm_recommendations(self):
        """Test recommendations when no ROCm detected."""
        recommended = get_recommended_pytorch_version(None)
        
        assert "CPU" in recommended["torch"]
        assert "No ROCm detected" in recommended["note"]


class TestDetectROCmVersion:
    """Test ROCm version detection."""
    
    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.read_text")
    def test_detect_from_version_file(self, mock_read_text, mock_exists):
        """Test detecting ROCm version from /opt/rocm/.info/version."""
        mock_exists.return_value = True
        mock_read_text.return_value = "6.5.0-98\n"
        
        version = detect_rocm_version()
        
        assert version is not None
        assert version.major == 6
        assert version.minor == 5
        assert version.patch == 0
    
    @patch("pathlib.Path.exists")
    @patch("subprocess.run")
    def test_detect_from_rocminfo(self, mock_run, mock_exists):
        """Test detecting ROCm version from rocminfo command."""
        mock_exists.return_value = False  # No version file
        
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "ROCm Version: 6.5.0"
        mock_run.return_value = mock_result
        
        version = detect_rocm_version()
        
        assert version is not None
        assert version.major == 6
        assert version.minor == 5
    
    @patch("pathlib.Path.exists")
    @patch("subprocess.run")
    @patch.dict("os.environ", {"ROCM_PATH": "/opt/rocm"})
    def test_detect_from_env_var(self, mock_run, mock_exists):
        """Test detecting ROCm from environment variable."""
        # First path check fails, rocminfo fails, but env var path succeeds
        def exists_side_effect(path):
            if ".info/version" in str(path):
                return True
            return False
        
        mock_exists.side_effect = exists_side_effect
        mock_run.side_effect = FileNotFoundError()
        
        with patch("pathlib.Path.read_text", return_value="6.5.0"):
            version = detect_rocm_version()
            
            assert version is not None
            assert version.major == 6
            assert version.minor == 5
    
    @patch("pathlib.Path.exists")
    @patch("subprocess.run")
    def test_no_rocm_detected(self, mock_run, mock_exists):
        """Test when no ROCm is detected."""
        mock_exists.return_value = False
        mock_run.side_effect = FileNotFoundError()
        
        version = detect_rocm_version()
        
        assert version is None


class TestCheckPyTorchInstallation:
    """Test PyTorch installation checking."""
    
    @patch("backend.utils.rocm_utils.torch")
    def test_pytorch_installed_with_rocm(self, mock_torch):
        """Test checking installed PyTorch with ROCm."""
        mock_torch.__version__ = "2.4.0+rocm6.5"
        mock_torch.version.hip = "6.5.0"
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 2
        
        info = check_pytorch_installation()
        
        assert info["installed"] is True
        assert info["version"] == "2.4.0+rocm6.5"
        assert info["is_rocm_build"] is True
        assert info["rocm_build_version"] == "6.5.0"
        assert info["gpu_available"] is True
        assert info["gpu_count"] == 2
    
    def test_pytorch_not_installed(self):
        """Test checking when PyTorch is not installed."""
        with patch("backend.utils.rocm_utils.torch", side_effect=ImportError()):
            info = check_pytorch_installation()
            
            assert info["installed"] is False
            assert info["version"] is None
            assert info["is_rocm_build"] is False
