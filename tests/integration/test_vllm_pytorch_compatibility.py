#!/usr/bin/env python3
"""
Test to verify vLLM installation script handles PyTorch 2.10 compatibility correctly.

This test validates that the install_vllm_rocm.sh script:
1. Detects ROCm 7.0+ correctly
2. Uses the appropriate PyTorch index URL
3. Uses --no-build-isolation flag
4. Ensures torchvision/torchaudio compatibility
"""

import subprocess
import re
from pathlib import Path


def test_script_syntax():
    """Verify the script has valid bash syntax."""
    script_path = Path(__file__).parent / "scripts" / "install_vllm_rocm.sh"
    result = subprocess.run(
        ["bash", "-n", str(script_path)],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0, f"Script syntax error: {result.stderr}"
    print("✓ Script syntax is valid")


def test_rocm_7_detection():
    """Verify ROCm 7.0+ detection logic is correct."""
    script_path = Path(__file__).parent / "scripts" / "install_vllm_rocm.sh"
    content = script_path.read_text()
    
    # Check for ROCm 7.0+ detection
    assert 'ROCM_MAJOR >= 7' in content, "Missing ROCm 7.0+ detection"
    assert 'nightly/rocm' in content, "Missing nightly URL for ROCm 7.0+"
    print("✓ ROCm 7.0+ detection logic is present")


def test_pytorch_nightly_url():
    """Verify PyTorch nightly URL format is correct."""
    script_path = Path(__file__).parent / "scripts" / "install_vllm_rocm.sh"
    content = script_path.read_text()
    
    # Check for correct nightly URL format
    pattern = r'https://download\.pytorch\.org/whl/nightly/rocm\$\{ROCM_MAJOR\}\.\$\{ROCM_MINOR\}'
    assert re.search(pattern, content), "PyTorch nightly URL format incorrect"
    print("✓ PyTorch nightly URL format is correct")


def test_no_build_isolation_flag():
    """Verify --no-build-isolation flag is used for vLLM installation."""
    script_path = Path(__file__).parent / "scripts" / "install_vllm_rocm.sh"
    content = script_path.read_text()
    
    # Check for --no-build-isolation flag in pip install
    assert '--no-build-isolation' in content, "Missing --no-build-isolation flag"
    print("✓ --no-build-isolation flag is present")


def test_torchvision_verification():
    """Verify torchvision/torchaudio compatibility check exists."""
    script_path = Path(__file__).parent / "scripts" / "install_vllm_rocm.sh"
    content = script_path.read_text()
    
    # Check for torchvision/torchaudio verification
    assert 'verify_pytorch_packages' in content, "Missing PyTorch packages verification"
    assert 'torchvision' in content, "Missing torchvision check"
    assert 'torchaudio' in content, "Missing torchaudio check"
    print("✓ PyTorch packages verification is present")


def test_get_pytorch_index_url_function():
    """Verify get_pytorch_index_url function exists and is called."""
    script_path = Path(__file__).parent / "scripts" / "install_vllm_rocm.sh"
    content = script_path.read_text()
    
    # Check for function definition
    assert 'get_pytorch_index_url()' in content, "Missing get_pytorch_index_url function"
    
    # Check function is called in main flow
    lines = content.split('\n')
    main_section = False
    found_call = False
    for line in lines:
        if 'main()' in line and '{' in line:
            main_section = True
        if main_section and 'get_pytorch_index_url' in line and 'def' not in line:
            found_call = True
            break
    
    assert found_call, "get_pytorch_index_url is not called in main()"
    print("✓ get_pytorch_index_url function exists and is called")


def test_vllm_build_deps_function():
    """Verify install_vllm_build_deps function exists."""
    script_path = Path(__file__).parent / "scripts" / "install_vllm_rocm.sh"
    content = script_path.read_text()
    
    # Check for function definition
    assert 'install_vllm_build_deps()' in content, "Missing install_vllm_build_deps function"
    
    # Check required dependencies
    assert 'packaging' in content, "Missing packaging dependency"
    assert 'psutil' in content or 'ray' in content, "Missing build dependencies"
    print("✓ install_vllm_build_deps function exists")


def test_pytorch_version_logging():
    """Verify PyTorch version is logged during build."""
    script_path = Path(__file__).parent / "scripts" / "install_vllm_rocm.sh"
    content = script_path.read_text()
    
    # Check for PyTorch version logging
    assert 'PYTORCH_VERSION' in content, "Missing PyTorch version variable"
    assert 'torch.__version__' in content, "Missing PyTorch version check"
    print("✓ PyTorch version logging is present")


def test_documentation_updated():
    """Verify documentation mentions PyTorch 2.10 support."""
    readme_path = Path(__file__).parent / "scripts" / "README.md"
    content = readme_path.read_text()
    
    # Check for PyTorch 2.10 mentions
    assert '2.10' in content or 'PyTorch 2.10' in content.lower(), "Missing PyTorch 2.10 documentation"
    assert 'ROCm 7.0' in content, "Missing ROCm 7.0 documentation"
    assert '--no-build-isolation' in content, "Missing --no-build-isolation documentation"
    print("✓ Documentation mentions PyTorch 2.10 support")


def test_vllm_comfyui_doc_updated():
    """Verify VLLM_COMFYUI_INSTALLATION.md is updated."""
    doc_path = Path(__file__).parent / "VLLM_COMFYUI_INSTALLATION.md"
    if not doc_path.exists():
        print("⚠ VLLM_COMFYUI_INSTALLATION.md not found, skipping")
        return
    
    content = doc_path.read_text()
    
    # Check for PyTorch 2.10 mentions
    assert '2.10' in content or 'PyTorch 2.10' in content.lower(), "Missing PyTorch 2.10 in doc"
    print("✓ VLLM_COMFYUI_INSTALLATION.md mentions PyTorch 2.10")


def test_amd_repo_function():
    """Verify AMD ROCm repository installation function exists."""
    script_path = Path(__file__).parent / "scripts" / "install_vllm_rocm.sh"
    content = script_path.read_text()
    
    # Check for function definition
    assert 'install_pytorch_amd_repo()' in content, "Missing install_pytorch_amd_repo function"
    
    # Check for AMD repository URL
    assert 'repo.radeon.com/rocm/manylinux' in content, "Missing AMD repository URL"
    assert 'torch==2.8.0' in content, "Missing PyTorch 2.8.0 version pin"
    assert 'torchaudio==2.8.0' in content, "Missing torchaudio 2.8.0 version pin"
    assert 'triton' in content, "Missing triton installation"
    
    print("✓ AMD ROCm repository installation function exists")


def test_repair_mode():
    """Verify repair mode functionality exists."""
    script_path = Path(__file__).parent / "scripts" / "install_vllm_rocm.sh"
    content = script_path.read_text()
    
    # Check for repair function
    assert 'repair_pytorch()' in content, "Missing repair_pytorch function"
    
    # Check for --repair flag handling
    assert '--repair' in content, "Missing --repair flag"
    assert 'REPAIR_MODE' in content, "Missing REPAIR_MODE variable"
    
    print("✓ Repair mode functionality exists")


def test_amd_repo_flag():
    """Verify --amd-repo flag exists."""
    script_path = Path(__file__).parent / "scripts" / "install_vllm_rocm.sh"
    content = script_path.read_text()
    
    # Check for --amd-repo flag handling
    assert '--amd-repo' in content, "Missing --amd-repo flag"
    assert 'USE_AMD_REPO' in content, "Missing USE_AMD_REPO variable"
    
    print("✓ --amd-repo flag exists")


def test_help_flag():
    """Verify --help flag provides usage information."""
    script_path = Path(__file__).parent / "scripts" / "install_vllm_rocm.sh"
    content = script_path.read_text()
    
    # Check for help flag and usage text
    assert '--help' in content, "Missing --help flag"
    assert 'Usage:' in content, "Missing usage text"
    assert 'Examples:' in content, "Missing examples in help"
    
    print("✓ --help flag exists with usage information")


def test_fallback_to_amd_repo():
    """Verify automatic fallback to AMD repo on nightly failure."""
    script_path = Path(__file__).parent / "scripts" / "install_vllm_rocm.sh"
    content = script_path.read_text()
    
    # Check for fallback logic
    assert 'Falling back to AMD ROCm repository' in content, "Missing fallback message"
    assert 'install_pytorch_amd_repo' in content, "Missing fallback call"
    
    print("✓ Automatic fallback to AMD repository exists")


def test_documentation_repair_instructions():
    """Verify documentation includes repair instructions."""
    readme_path = Path(__file__).parent / "scripts" / "README.md"
    content = readme_path.read_text()
    
    # Check for repair instructions
    assert '--repair' in content, "Missing --repair in documentation"
    assert 'torch==2.8.0' in content, "Missing PyTorch 2.8.0 in repair instructions"
    assert 'repo.radeon.com' in content, "Missing AMD repository URL in documentation"
    
    print("✓ Documentation includes repair instructions")


def main():
    """Run all tests."""
    print("=" * 70)
    print("Testing vLLM PyTorch 2.10 Compatibility Fix")
    print("=" * 70)
    print()
    
    tests = [
        test_script_syntax,
        test_rocm_7_detection,
        test_pytorch_nightly_url,
        test_no_build_isolation_flag,
        test_torchvision_verification,
        test_get_pytorch_index_url_function,
        test_vllm_build_deps_function,
        test_pytorch_version_logging,
        test_documentation_updated,
        test_vllm_comfyui_doc_updated,
        test_amd_repo_function,
        test_repair_mode,
        test_amd_repo_flag,
        test_help_flag,
        test_fallback_to_amd_repo,
        test_documentation_repair_instructions,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"✗ {test.__name__} failed: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test.__name__} error: {e}")
            failed += 1
    
    print()
    print("=" * 70)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 70)
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    exit(main())
