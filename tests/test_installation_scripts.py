"""
Test suite for installation scripts
Tests that the vllm and ComfyUI installation scripts exist and are properly configured
"""

import os
import subprocess
from pathlib import Path
import pytest


@pytest.fixture
def project_root():
    """Get the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def scripts_dir(project_root):
    """Get the scripts directory."""
    return project_root / "scripts"


class TestVLLMInstallationScript:
    """Tests for vLLM installation script."""

    def test_script_exists(self, scripts_dir):
        """Test that the vLLM installation script exists."""
        script_path = scripts_dir / "install_vllm_rocm.sh"
        assert script_path.exists(), "vLLM installation script not found"

    def test_script_is_executable(self, scripts_dir):
        """Test that the script has executable permissions."""
        script_path = scripts_dir / "install_vllm_rocm.sh"
        assert os.access(script_path, os.X_OK), "Script is not executable"

    def test_script_has_shebang(self, scripts_dir):
        """Test that the script has a proper shebang."""
        script_path = scripts_dir / "install_vllm_rocm.sh"
        with open(script_path, "r") as f:
            first_line = f.readline().strip()
        assert first_line.startswith("#!"), "Script missing shebang"
        assert "bash" in first_line, "Script should use bash"

    def test_script_syntax(self, scripts_dir):
        """Test that the script has valid bash syntax."""
        script_path = scripts_dir / "install_vllm_rocm.sh"
        result = subprocess.run(
            ["bash", "-n", str(script_path)], capture_output=True, text=True
        )
        assert result.returncode == 0, f"Script has syntax errors: {result.stderr}"

    def test_script_contains_key_functions(self, scripts_dir):
        """Test that the script contains essential functions."""
        script_path = scripts_dir / "install_vllm_rocm.sh"
        with open(script_path, "r") as f:
            content = f.read()

        # Check for key functions
        assert "check_venv()" in content, "Missing venv check function"
        assert "detect_rocm()" in content, "Missing ROCm detection function"
        assert "install_pytorch_rocm()" in content, "Missing PyTorch installation"
        assert "build_vllm()" in content, "Missing vLLM build function"
        assert "VLLM_TARGET_DEVICE=rocm" in content, "Missing ROCm target device"

    def test_script_has_error_handling(self, scripts_dir):
        """Test that the script has proper error handling."""
        script_path = scripts_dir / "install_vllm_rocm.sh"
        with open(script_path, "r") as f:
            content = f.read()

        assert "set -e" in content, "Missing 'set -e' for error handling"
        assert "print_error" in content, "Missing error printing function"

    def test_script_checks_ninja_command_not_package(self, scripts_dir):
        """Test that the script checks for 'ninja' command, not 'ninja-build'."""
        script_path = scripts_dir / "install_vllm_rocm.sh"
        with open(script_path, "r") as f:
            content = f.read()

        # Should check for ninja command (executable)
        assert 'for cmd in gcc g++ make cmake git ninja;' in content, \
            "Script should check for 'ninja' command"
        
        # Should map ninja to ninja-build package for error messages
        assert 'ninja-build' in content, \
            "Script should reference 'ninja-build' package name for apt-get"

    def test_script_preserves_existing_pytorch(self, scripts_dir):
        """Test that the vLLM script preserves existing PyTorch installation."""
        script_path = scripts_dir / "install_vllm_rocm.sh"
        with open(script_path, "r") as f:
            content = f.read()

        # Should check if PyTorch is already installed
        assert 'if python3 -c "import torch' in content, \
            "Script should check if PyTorch is already installed"
        
        # Should skip installation if PyTorch exists
        assert 'already installed' in content, \
            "Script should indicate when PyTorch is already present"
        
        # Should preserve existing setup
        assert 'preserve existing' in content or 'Skipping PyTorch installation' in content, \
            "Script should preserve existing PyTorch installation"


class TestComfyUIInstallationScript:
    """Tests for ComfyUI installation script."""

    def test_script_exists(self, scripts_dir):
        """Test that the ComfyUI installation script exists."""
        script_path = scripts_dir / "install_comfyui_rocm.sh"
        assert script_path.exists(), "ComfyUI installation script not found"

    def test_script_is_executable(self, scripts_dir):
        """Test that the script has executable permissions."""
        script_path = scripts_dir / "install_comfyui_rocm.sh"
        assert os.access(script_path, os.X_OK), "Script is not executable"

    def test_script_has_shebang(self, scripts_dir):
        """Test that the script has a proper shebang."""
        script_path = scripts_dir / "install_comfyui_rocm.sh"
        with open(script_path, "r") as f:
            first_line = f.readline().strip()
        assert first_line.startswith("#!"), "Script missing shebang"
        assert "bash" in first_line, "Script should use bash"

    def test_script_syntax(self, scripts_dir):
        """Test that the script has valid bash syntax."""
        script_path = scripts_dir / "install_comfyui_rocm.sh"
        result = subprocess.run(
            ["bash", "-n", str(script_path)], capture_output=True, text=True
        )
        assert result.returncode == 0, f"Script has syntax errors: {result.stderr}"

    def test_script_contains_key_functions(self, scripts_dir):
        """Test that the script contains essential functions."""
        script_path = scripts_dir / "install_comfyui_rocm.sh"
        with open(script_path, "r") as f:
            content = f.read()

        # Check for key functions
        assert "check_venv()" in content, "Missing venv check function"
        assert "detect_rocm()" in content, "Missing ROCm detection function"
        assert "install_pytorch_rocm()" in content, "Missing PyTorch installation"
        assert "clone_comfyui()" in content, "Missing ComfyUI clone function"
        assert "install_comfyui_deps()" in content, "Missing dependency installation"
        assert "create_launch_script()" in content, "Missing launch script creation"

    def test_script_has_error_handling(self, scripts_dir):
        """Test that the script has proper error handling."""
        script_path = scripts_dir / "install_comfyui_rocm.sh"
        with open(script_path, "r") as f:
            content = f.read()

        assert "set -e" in content, "Missing 'set -e' for error handling"
        assert "print_error" in content, "Missing error printing function"

    def test_script_checks_existing_pytorch(self, scripts_dir):
        """Test that the script checks for existing PyTorch installation."""
        script_path = scripts_dir / "install_comfyui_rocm.sh"
        with open(script_path, "r") as f:
            content = f.read()

        # Should check if PyTorch is already installed
        assert 'if python3 -c "import torch' in content, \
            "Script should check if PyTorch is already installed"
        
        # Should skip installation if PyTorch exists
        assert 'already installed' in content, \
            "Script should indicate when PyTorch is already present"
        
        # Should preserve existing setup
        assert 'preserve existing' in content or 'Skipping PyTorch installation' in content, \
            "Script should preserve existing PyTorch installation"

    def test_script_supports_cpu_fallback(self, scripts_dir):
        """Test that the script supports CPU-only mode."""
        script_path = scripts_dir / "install_comfyui_rocm.sh"
        with open(script_path, "r") as f:
            content = f.read()

        assert (
            "CPU mode" in content or "cpu" in content.lower()
        ), "Missing CPU fallback support"

    def test_script_exports_comfyui_dir_correctly(self, scripts_dir):
        """Test that the script correctly exports COMFYUI_DIR variable."""
        script_path = scripts_dir / "install_comfyui_rocm.sh"
        with open(script_path, "r") as f:
            content = f.read()

        # Check that COMFYUI_DIR is not declared as local before export
        # This ensures it's available to subsequent functions
        lines = content.split('\n')
        in_clone_function = False
        for i, line in enumerate(lines):
            if 'clone_comfyui()' in line:
                in_clone_function = True
            if in_clone_function and 'COMFYUI_DIR=' in line and 'export' not in line:
                # If we find COMFYUI_DIR assignment, make sure it's not local
                assert 'local COMFYUI_DIR=' not in line, \
                    "COMFYUI_DIR should not be declared as local before export"
            if in_clone_function and 'export COMFYUI_DIR' in line:
                break


class TestScriptsDocumentation:
    """Tests for scripts documentation."""

    def test_readme_exists(self, scripts_dir):
        """Test that the README exists."""
        readme_path = scripts_dir / "README.md"
        assert readme_path.exists(), "Scripts README not found"

    def test_readme_has_content(self, scripts_dir):
        """Test that the README has substantial content."""
        readme_path = scripts_dir / "README.md"
        with open(readme_path, "r") as f:
            content = f.read()

        assert len(content) > 1000, "README is too short"
        assert "vllm" in content.lower(), "README missing vLLM documentation"
        assert "comfyui" in content.lower(), "README missing ComfyUI documentation"
        assert "Usage" in content or "usage" in content, "README missing usage section"

    def test_readme_documents_both_scripts(self, scripts_dir):
        """Test that README documents both scripts."""
        readme_path = scripts_dir / "README.md"
        with open(readme_path, "r") as f:
            content = f.read()

        assert "install_vllm_rocm.sh" in content, "README doesn't document vLLM script"
        assert (
            "install_comfyui_rocm.sh" in content
        ), "README doesn't document ComfyUI script"

    def test_readme_has_troubleshooting(self, scripts_dir):
        """Test that README includes troubleshooting section."""
        readme_path = scripts_dir / "README.md"
        with open(readme_path, "r") as f:
            content = f.read()

        assert (
            "Troubleshooting" in content or "troubleshooting" in content
        ), "README missing troubleshooting section"


class TestSetupAIModelsIntegration:
    """Tests for integration with setup_ai_models.py."""

    def test_setup_ai_models_exists(self, project_root):
        """Test that setup_ai_models.py exists."""
        script_path = project_root / "setup_ai_models.py"
        assert script_path.exists(), "setup_ai_models.py not found"

    def test_setup_ai_models_has_vllm_method(self, project_root):
        """Test that setup_ai_models.py has vLLM setup method."""
        script_path = project_root / "setup_ai_models.py"
        with open(script_path, "r") as f:
            content = f.read()

        assert "_setup_vllm_rocm" in content, "Missing vLLM ROCm setup method"
        assert (
            "install_vllm_rocm.sh" in content
        ), "Setup method doesn't reference script"

    def test_setup_ai_models_has_comfyui_method(self, project_root):
        """Test that setup_ai_models.py has ComfyUI setup method."""
        script_path = project_root / "setup_ai_models.py"
        with open(script_path, "r") as f:
            content = f.read()

        assert "_setup_comfyui_rocm" in content, "Missing ComfyUI ROCm setup method"
        assert (
            "install_comfyui_rocm.sh" in content
        ), "Setup method doesn't reference script"

    def test_setup_ai_models_syntax(self, project_root):
        """Test that setup_ai_models.py has valid Python syntax."""
        script_path = project_root / "setup_ai_models.py"
        result = subprocess.run(
            ["python3", "-m", "py_compile", str(script_path)],
            capture_output=True,
            text=True,
        )
        assert (
            result.returncode == 0
        ), f"setup_ai_models.py has syntax errors: {result.stderr}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
