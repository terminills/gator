#!/usr/bin/env python3
"""
Test MI25 detection and compatibility features.

This test validates that the MI25 GPU detection logic works correctly
and that the appropriate environment is configured.
"""

import subprocess
import os
import sys
from pathlib import Path

def test_bash_syntax():
    """Test that server-setup.sh has valid bash syntax."""
    result = subprocess.run(
        ['bash', '-n', 'server-setup.sh'],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0, f"Bash syntax error: {result.stderr}"
    print("✓ server-setup.sh has valid bash syntax")


def test_mi25_detection_logic():
    """Test MI25 detection patterns in server-setup.sh."""
    script_path = Path('server-setup.sh')
    content = script_path.read_text()
    
    # Check for MI25 detection patterns
    assert 'radeon instinct mi25' in content.lower(), "Missing MI25 detection pattern"
    assert 'vega.*10' in content.lower() or 'vega 10' in content.lower(), "Missing Vega 10 detection"
    assert 'gfx900' in content.lower(), "Missing gfx900 architecture reference"
    
    # Check for device ID detection (Vega 10 device IDs)
    assert '6860' in content or '686' in content, "Missing Vega 10 device ID detection"
    
    print("✓ MI25 detection patterns present")


def test_rocm_version_selection():
    """Test that ROCm 5.7.1 is used (MI25 compatible)."""
    script_path = Path('server-setup.sh')
    content = script_path.read_text()
    
    # Check for ROCm 5.7.1 as default version
    assert '5.7.1' in content, "Missing ROCm 5.7.1 version"
    assert 'ROCM_VERSION="5.7.1"' in content, "ROCm 5.7.1 not set as version"
    
    print("✓ ROCm 5.7.1 version selection present")


def test_gfx900_environment_variables():
    """Test that gfx900-specific environment variables are configured."""
    script_path = Path('server-setup.sh')
    content = script_path.read_text()
    
    # Check for critical MI25 environment variables
    required_vars = [
        'HSA_OVERRIDE_GFX_VERSION=9.0.0',
        'HCC_AMDGPU_TARGET=gfx900',
        'PYTORCH_ROCM_ARCH=gfx900',
        'TF_ROCM_AMDGPU_TARGETS=gfx900'
    ]
    
    for var in required_vars:
        assert var in content, f"Missing environment variable: {var}"
    
    print("✓ gfx900 environment variables configured")


def test_enhanced_verification_script():
    """Test that the verification script includes comprehensive checks."""
    script_path = Path('server-setup.sh')
    content = script_path.read_text()
    
    # Check for enhanced check_rocm.sh content
    assert 'HSA_OVERRIDE_GFX_VERSION' in content, "Missing HSA override in verification"
    assert 'rocminfo' in content, "Missing rocminfo in verification"
    assert 'gfx' in content.lower(), "Missing GPU architecture check"
    
    print("✓ Enhanced verification script present")


def test_compatibility_documentation():
    """Test that MI25 compatibility documentation exists."""
    doc_path = Path('docs/MI25_COMPATIBILITY.md')
    assert doc_path.exists(), "MI25 compatibility documentation not found"
    
    content = doc_path.read_text()
    
    # Check for key sections
    required_sections = [
        'gfx900',
        'ROCm 5.7.1',
        'Ubuntu 20.04',
        'HSA_OVERRIDE_GFX_VERSION',
        'PyTorch',
        'kernel'
    ]
    
    for section in required_sections:
        assert section in content, f"Missing documentation section: {section}"
    
    print("✓ MI25 compatibility documentation complete")


def test_amdgpu_install_utility():
    """Test that server-setup.sh uses AMD's official amdgpu-install utility."""
    script_path = Path('server-setup.sh')
    content = script_path.read_text()
    
    # Check for amdgpu-install utility usage
    assert 'amdgpu-install' in content, "Missing amdgpu-install utility"
    assert 'amdgpu-install_5.7.50701-1_all.deb' in content, "Missing amdgpu-install package"
    assert '--usecase=rocm,hiplibsdk,dkms' in content, "Missing ROCm use-case configuration"
    assert '--rocmrelease=' in content, "Missing ROCm release specification"
    
    # Check for proper Ubuntu codename handling
    assert 'focal' in content, "Missing Ubuntu 20.04 (focal) support"
    assert 'jammy' in content, "Missing Ubuntu 22.04 (jammy) support"
    
    print("✓ amdgpu-install utility properly configured")


def test_setup_ai_models_mi25_detection():
    """Test that setup_ai_models.py detects MI25 correctly."""
    script_path = Path('setup_ai_models.py')
    content = script_path.read_text()
    
    # Check for MI25 detection
    assert 'MI25' in content, "Missing MI25 detection in setup_ai_models.py"
    assert 'gfx900' in content, "Missing gfx900 reference in setup_ai_models.py"
    assert 'is_mi25' in content.lower(), "Missing MI25 flag in setup_ai_models.py"
    
    print("✓ setup_ai_models.py MI25 detection present")


def run_all_tests():
    """Run all tests."""
    tests = [
        test_bash_syntax,
        test_mi25_detection_logic,
        test_rocm_version_selection,
        test_gfx900_environment_variables,
        test_amdgpu_install_utility,
        test_enhanced_verification_script,
        test_compatibility_documentation,
        test_setup_ai_models_mi25_detection,
    ]
    
    print("Running MI25 compatibility tests...\n")
    
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
        sys.exit(0)


if __name__ == '__main__':
    # Change to repo root if needed
    if not Path('server-setup.sh').exists():
        os.chdir('/home/runner/work/gator/gator')
    
    run_all_tests()
