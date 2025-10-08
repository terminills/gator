#!/usr/bin/env python3
"""
Test HuggingFace Hub Compatibility.

Validates that the diffusers and huggingface_hub versions are compatible
and that the cached_download deprecation is properly handled.
"""

import subprocess
from pathlib import Path
import re


def test_diffusers_version_compatibility():
    """Test that diffusers version is >= 0.25.0 for cached_download fix."""
    pyproject_path = Path('pyproject.toml')
    content = pyproject_path.read_text()
    
    # Check that diffusers is >= 0.25.0
    assert 'diffusers>=0.25.0' in content, \
        "diffusers must be >=0.25.0 to avoid cached_download deprecation error"
    
    # Ensure no old versions are referenced
    old_versions = re.findall(r'diffusers>=0\.2[0-4]\.[0-9]+', content)
    assert len(old_versions) == 0, \
        f"Found old diffusers version references: {old_versions}"
    
    print("✓ pyproject.toml specifies diffusers>=0.25.0")


def test_huggingface_hub_explicit_dependency():
    """Test that huggingface_hub is explicitly declared as a dependency."""
    pyproject_path = Path('pyproject.toml')
    content = pyproject_path.read_text()
    
    # Check that huggingface_hub is explicitly listed with version >= 0.20.0
    assert 'huggingface_hub>=0.20.0' in content, \
        "huggingface_hub must be explicitly declared as >=0.20.0 (cached_download removed in 0.20.0)"
    
    print("✓ pyproject.toml explicitly declares huggingface_hub>=0.20.0")


def test_setup_ai_models_dependencies():
    """Test that setup_ai_models.py uses compatible versions."""
    setup_path = Path('setup_ai_models.py')
    content = setup_path.read_text()
    
    # Check for diffusers>=0.25.0
    assert 'diffusers>=0.25.0' in content, \
        "setup_ai_models.py must specify diffusers>=0.25.0"
    
    # Check for huggingface_hub>=0.20.0
    assert 'huggingface_hub>=0.20.0' in content, \
        "setup_ai_models.py must specify huggingface_hub>=0.20.0"
    
    # Ensure no old versions
    old_versions = re.findall(r'diffusers>=0\.2[0-4]\.[0-9]+', content)
    assert len(old_versions) == 0, \
        f"Found old diffusers version references in setup_ai_models.py: {old_versions}"
    
    print("✓ setup_ai_models.py specifies compatible versions")


def test_no_cached_download_usage():
    """Test that cached_download is not used anywhere in the codebase."""
    # Search for cached_download usage in Python files
    result = subprocess.run(
        ['grep', '-r', 'from huggingface_hub import.*cached_download', 'src/'],
        capture_output=True,
        text=True
    )
    
    # grep returns 1 if no matches found, which is what we want
    assert result.returncode == 1, \
        f"Found cached_download imports in codebase:\n{result.stdout}"
    
    print("✓ No cached_download imports found in codebase")


def test_diffusers_api_compatibility():
    """Test that the codebase is ready for modern diffusers API."""
    # Check that StableDiffusionPipeline.from_pretrained is used correctly
    setup_path = Path('setup_ai_models.py')
    content = setup_path.read_text()
    
    # Should use from_pretrained (modern API)
    assert 'from_pretrained' in content, \
        "setup_ai_models.py should use from_pretrained for model loading"
    
    print("✓ setup_ai_models.py uses modern diffusers API")


def test_version_constraint_format():
    """Test that version constraints follow proper format."""
    pyproject_path = Path('pyproject.toml')
    content = pyproject_path.read_text()
    
    # Check that constraints are >= not ==
    diffusers_constraints = re.findall(r'"diffusers([><=]+)[0-9.]+', content)
    for constraint in diffusers_constraints:
        assert constraint == '>=', \
            f"diffusers constraint should use '>=' not '{constraint}' for flexibility"
    
    huggingface_constraints = re.findall(r'"huggingface_hub([><=]+)[0-9.]+', content)
    for constraint in huggingface_constraints:
        assert constraint == '>=', \
            f"huggingface_hub constraint should use '>=' not '{constraint}' for flexibility"
    
    print("✓ Version constraints use proper >= format")


def test_documentation_references_fix():
    """Test that the fix is properly documented."""
    doc_path = Path('MODEL_INSTALL_FIX_VERIFICATION.md')
    
    if doc_path.exists():
        content = doc_path.read_text()
        
        # Check that documentation mentions the fix
        assert 'cached_download' in content, \
            "Documentation should reference the cached_download issue"
        assert 'diffusers>=0.25.0' in content, \
            "Documentation should reference the diffusers>=0.25.0 fix"
        assert 'huggingface_hub>=0.20.0' in content, \
            "Documentation should reference the huggingface_hub>=0.20.0 fix"
        
        print("✓ Fix is documented in MODEL_INSTALL_FIX_VERIFICATION.md")
    else:
        print("⚠ MODEL_INSTALL_FIX_VERIFICATION.md not found (optional)")


def run_all_tests():
    """Run all HuggingFace compatibility tests."""
    import sys
    
    tests = [
        test_diffusers_version_compatibility,
        test_huggingface_hub_explicit_dependency,
        test_setup_ai_models_dependencies,
        test_no_cached_download_usage,
        test_diffusers_api_compatibility,
        test_version_constraint_format,
        test_documentation_references_fix,
    ]
    
    print("Running HuggingFace Hub compatibility tests...\n")
    
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
        print("HuggingFace Hub compatibility confirmed - cached_download issue resolved")
        sys.exit(0)


if __name__ == '__main__':
    import os
    # Change to repo root if needed
    if not Path('setup_ai_models.py').exists():
        os.chdir('/home/runner/work/gator/gator')
    
    run_all_tests()
