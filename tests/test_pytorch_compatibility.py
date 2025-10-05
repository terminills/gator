#!/usr/bin/env python3
"""
Test PyTorch 2.2.0 compatibility.

Validates that all PyTorch version references in the codebase are consistent
and compatible with PyTorch 2.2.0 for ROCm 5.7.1.
"""

import subprocess
from pathlib import Path
import re


def test_pyproject_toml_pytorch_version():
    """Test that pyproject.toml specifies PyTorch 2.2.0 with ROCm 5.7.1."""
    pyproject_path = Path('pyproject.toml')
    content = pyproject_path.read_text()
    
    # Check for exact PyTorch version in rocm extras
    assert 'torch==2.2.0+rocm5.7.1' in content, \
        "pyproject.toml must specify torch==2.2.0+rocm5.7.1 in rocm extras"
    assert 'torchvision==0.17.0+rocm5.7.1' in content, \
        "pyproject.toml must specify torchvision==0.17.0+rocm5.7.1 in rocm extras"
    
    # Verify no older versions are specified
    assert 'torch==2.2.0+rocm5.7"' not in content, \
        "pyproject.toml should not use rocm5.7 (missing .1)"
    
    print("✓ pyproject.toml specifies PyTorch 2.2.0+rocm5.7.1")


def test_setup_script_pytorch_version():
    """Test that server-setup.sh installs PyTorch 2.2.0 with ROCm 5.7.1."""
    script_path = Path('server-setup.sh')
    content = script_path.read_text()
    
    # Check for PyTorch installation command with correct version
    assert 'torch==2.2.0+rocm5.7.1' in content, \
        "server-setup.sh must install torch==2.2.0+rocm5.7.1"
    assert 'torchvision==0.17.0+rocm5.7.1' in content, \
        "server-setup.sh must install torchvision==0.17.0+rocm5.7.1"
    
    print("✓ server-setup.sh installs PyTorch 2.2.0+rocm5.7.1")


def test_setup_ai_models_pytorch_version():
    """Test that setup_ai_models.py requires compatible PyTorch version."""
    script_path = Path('setup_ai_models.py')
    content = script_path.read_text()
    
    # Check for PyTorch version requirement
    # Should be torch>=2.2.0 or torch>=2.0.0 (both are compatible)
    torch_version_pattern = r'torch>=2\.[0-9]+\.[0-9]+'
    matches = re.findall(torch_version_pattern, content)
    
    assert len(matches) > 0, \
        "setup_ai_models.py must specify a torch version requirement"
    
    # Extract version and check it's at least 2.2.0
    for match in matches:
        version_str = match.replace('torch>=', '')
        major, minor, patch = map(int, version_str.split('.'))
        assert (major, minor) >= (2, 2), \
            f"setup_ai_models.py torch requirement {match} must be >= 2.2.0"
    
    print(f"✓ setup_ai_models.py requires torch>=2.2.0")


def test_version_consistency():
    """Test that all PyTorch version references are consistent."""
    files_to_check = {
        'pyproject.toml': '2.2.0+rocm5.7.1',
        'server-setup.sh': '2.2.0+rocm5.7.1',
    }
    
    for filename, expected_version in files_to_check.items():
        filepath = Path(filename)
        content = filepath.read_text()
        
        # Find all torch version references
        torch_versions = re.findall(r'torch==([0-9.]+\+rocm[0-9.]+)', content)
        
        if torch_versions:
            for version in torch_versions:
                assert version == expected_version, \
                    f"{filename} has torch=={version}, expected {expected_version}"
    
    print("✓ PyTorch version references are consistent across files")


def test_rocm_version_alignment():
    """Test that PyTorch version aligns with ROCm 5.7.1."""
    # Check that we're using ROCm 5.7.1
    script_path = Path('server-setup.sh')
    content = script_path.read_text()
    
    # Should have ROCM_VERSION="5.7.1"
    assert 'ROCM_VERSION="5.7.1"' in content, \
        "server-setup.sh must set ROCM_VERSION to 5.7.1"
    
    # PyTorch version should match ROCm version
    assert 'torch==2.2.0+rocm5.7.1' in content, \
        "PyTorch version must match ROCm version 5.7.1"
    
    print("✓ PyTorch 2.2.0 aligns with ROCm 5.7.1")


def test_no_conflicting_versions():
    """Test that there are no conflicting or outdated PyTorch versions."""
    files_to_check = ['pyproject.toml', 'server-setup.sh', 'setup_ai_models.py']
    
    outdated_patterns = [
        r'torch==2\.0\.',
        r'torch==2\.1\.',
        r'torch>=2\.0\.0[^,\]]',  # torch>=2.0.0 not followed by higher requirement
    ]
    
    for filename in files_to_check:
        filepath = Path(filename)
        if not filepath.exists():
            continue
            
        content = filepath.read_text()
        
        for pattern in outdated_patterns:
            matches = re.findall(pattern, content)
            # Allow torch>=2.0.0 in setup_ai_models.py as it's just showing compatibility
            if filename == 'setup_ai_models.py' and pattern.startswith(r'torch>=2\.0\.0'):
                continue
                
            assert len(matches) == 0, \
                f"{filename} contains outdated pattern: {pattern}"
    
    print("✓ No conflicting PyTorch versions found")


def run_all_tests():
    """Run all PyTorch compatibility tests."""
    import sys
    
    tests = [
        test_pyproject_toml_pytorch_version,
        test_setup_script_pytorch_version,
        test_setup_ai_models_pytorch_version,
        test_version_consistency,
        test_rocm_version_alignment,
        test_no_conflicting_versions,
    ]
    
    print("Running PyTorch 2.2.0 compatibility tests...\n")
    
    failed = []
    for test in tests:
        try:
            test()
        except AssertionError as e:
            print(f"✗ {test.__name__} failed: {e}")
            failed.append(test.__name__)
        except Exception as e:
            print(f"✗ {test.__name__} error: {e}")
            failed.append(test.__name__)
    
    print(f"\n{'='*60}")
    if failed:
        print(f"❌ {len(failed)} test(s) failed:")
        for name in failed:
            print(f"   - {name}")
        sys.exit(1)
    else:
        print(f"✅ All {len(tests)} tests passed!")
        print("PyTorch 2.2.0 compatibility confirmed for ROCm 5.7.1")
        sys.exit(0)


if __name__ == '__main__':
    import os
    # Change to repo root if needed
    if not Path('server-setup.sh').exists():
        os.chdir('/home/runner/work/gator/gator')
    
    run_all_tests()
