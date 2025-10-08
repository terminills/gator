#!/usr/bin/env python3
"""
Test PyTorch 2.3.1 compatibility.

Validates that all PyTorch version references in the codebase are consistent
and compatible with PyTorch 2.3.1 for ROCm 5.7.
"""

import subprocess
from pathlib import Path
import re


def test_pyproject_toml_pytorch_version():
    """Test that pyproject.toml specifies PyTorch 2.3.1 with ROCm 5.7."""
    pyproject_path = Path('pyproject.toml')
    content = pyproject_path.read_text()
    
    # Check for exact PyTorch version in rocm extras
    assert 'torch==2.3.1+rocm5.7' in content, \
        "pyproject.toml must specify torch==2.3.1+rocm5.7 in rocm extras"
    assert 'torchvision==0.18.1+rocm5.7' in content, \
        "pyproject.toml must specify torchvision==0.18.1+rocm5.7 in rocm extras"
    
    print("✓ pyproject.toml specifies PyTorch 2.3.1+rocm5.7")


def test_setup_script_pytorch_version():
    """Test that server-setup.sh installs PyTorch 2.3.1 with ROCm 5.7."""
    script_path = Path('server-setup.sh')
    content = script_path.read_text()
    
    # Check for PyTorch installation command with correct version
    assert 'torch==2.3.1+rocm5.7' in content, \
        "server-setup.sh must install torch==2.3.1+rocm5.7"
    assert 'torchvision==0.18.1+rocm5.7' in content, \
        "server-setup.sh must install torchvision==0.18.1+rocm5.7"
    
    print("✓ server-setup.sh installs PyTorch 2.3.1+rocm5.7")


def test_setup_ai_models_pytorch_version():
    """Test that setup_ai_models.py requires compatible PyTorch version."""
    script_path = Path('setup_ai_models.py')
    content = script_path.read_text()
    
    # Check for PyTorch version requirement
    # Should be torch>=2.3.0 or higher
    torch_version_pattern = r'torch>=2\.[0-9]+\.[0-9]+'
    matches = re.findall(torch_version_pattern, content)
    
    assert len(matches) > 0, \
        "setup_ai_models.py must specify a torch version requirement"
    
    # Extract version and check it's at least 2.3.0
    for match in matches:
        version_str = match.replace('torch>=', '')
        major, minor, patch = map(int, version_str.split('.'))
        assert (major, minor) >= (2, 3), \
            f"setup_ai_models.py torch requirement {match} must be >= 2.3.0"
    
    print(f"✓ setup_ai_models.py requires torch>=2.3.0")


def test_version_consistency():
    """Test that all PyTorch version references are consistent."""
    files_to_check = {
        'pyproject.toml': '2.3.1+rocm5.7',
        'server-setup.sh': '2.3.1+rocm5.7',
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
    """Test that PyTorch version aligns with ROCm 5.7."""
    # Check that we're using ROCm 5.7.1 as the base ROCm version
    script_path = Path('server-setup.sh')
    content = script_path.read_text()
    
    # Should have ROCM_VERSION="5.7.1"
    assert 'ROCM_VERSION="5.7.1"' in content, \
        "server-setup.sh must set ROCM_VERSION to 5.7.1"
    
    # PyTorch version tag uses rocm5.7 (not rocm5.7.1)
    assert 'torch==2.3.1+rocm5.7' in content, \
        "PyTorch version must use rocm5.7 tag (PyTorch naming convention)"
    
    print("✓ PyTorch 2.3.1+rocm5.7 aligns with ROCm 5.7.1")


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


def test_ml_dependencies_compatibility():
    """Test that ML dependencies are compatible with PyTorch 2.3.1."""
    pyproject_path = Path('pyproject.toml')
    content = pyproject_path.read_text()
    
    # Check for updated versions compatible with PyTorch 2.3.1
    assert 'transformers>=4.41.0' in content, \
        "transformers must be >=4.41.0 for PyTorch 2.3.1 compatibility"
    assert 'diffusers>=0.28.0' in content, \
        "diffusers must be >=0.28.0 for PyTorch 2.3.1 compatibility"
    assert 'accelerate>=0.29.0' in content, \
        "accelerate must be >=0.29.0 for PyTorch 2.3.1 compatibility"
    assert 'huggingface_hub>=0.23.0' in content, \
        "huggingface_hub must be >=0.23.0 for modern API"
    
    # Check setup_ai_models.py as well
    setup_path = Path('setup_ai_models.py')
    setup_content = setup_path.read_text()
    
    assert 'transformers>=4.41.0' in setup_content, \
        "setup_ai_models.py must specify transformers>=4.41.0"
    assert 'diffusers>=0.28.0' in setup_content, \
        "setup_ai_models.py must specify diffusers>=0.28.0"
    assert 'accelerate>=0.29.0' in setup_content, \
        "setup_ai_models.py must specify accelerate>=0.29.0"
    assert 'huggingface_hub>=0.23.0' in setup_content, \
        "setup_ai_models.py must specify huggingface_hub>=0.23.0"
    
    print("✓ ML dependencies are compatible with PyTorch 2.3.1")


def test_numpy_version_constraint():
    """Test that numpy version is constrained to be compatible with PyTorch 2.3.1."""
    pyproject_path = Path('pyproject.toml')
    content = pyproject_path.read_text()
    
    # PyTorch 2.3.1 requires numpy < 2.0
    # Check that numpy has an upper bound constraint
    numpy_pattern = r'numpy>=1\.[0-9]+\.[0-9]+,<2\.0'
    matches = re.findall(numpy_pattern, content)
    
    assert len(matches) > 0, \
        "numpy must be constrained to <2.0 for PyTorch 2.3.1 compatibility (requires numpy<2.0)"
    
    # Verify the lower bound is at least 1.24.0
    assert 'numpy>=1.24.0,<2.0' in content, \
        "numpy should be >=1.24.0,<2.0 for PyTorch 2.3.1 compatibility"
    
    print("✓ numpy version is constrained to <2.0 for PyTorch 2.3.1 compatibility")


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
        test_ml_dependencies_compatibility,
        test_numpy_version_constraint,
    ]
    
    print("Running PyTorch 2.3.1 compatibility tests...\n")
    
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
        print("PyTorch 2.3.1 compatibility confirmed for ROCm 5.7.1")
        sys.exit(0)


if __name__ == '__main__':
    import os
    # Change to repo root if needed
    if not Path('server-setup.sh').exists():
        os.chdir('/home/runner/work/gator/gator')
    
    run_all_tests()
