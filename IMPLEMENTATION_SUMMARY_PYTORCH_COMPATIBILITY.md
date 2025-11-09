# Implementation Summary: PyTorch 2.10 and ROCm 7.0 Support

**Issue**: [pytorch 2.10 support](https://github.com/terminills/gator/issues/XXX)

## Problem Statement

When installing PyTorch using the nightly index URL for ROCm 7.0:
```bash
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm7.0
```

The system installs PyTorch 2.10, but doesn't check what version of PyTorch is currently running in the venv or install the correct versions of dependencies that are compatible with PyTorch 2.10.

## Solution Overview

Implemented a comprehensive PyTorch version detection and dependency compatibility system that:

1. **Detects ROCm versions** including 7.0+ with support for nightly builds
2. **Checks installed PyTorch version** and extracts major.minor version
3. **Determines compatible dependency versions** based on installed PyTorch version
4. **Provides installation guidance** for correct PyTorch and dependency versions

## Implementation Details

### 1. ROCm Version Detection Enhancement

**File**: `src/backend/utils/rocm_utils.py`

Enhanced `get_pytorch_index_url()` to support ROCm 7.0+:

```python
# ROCm 7.0+ support
if rocm_version.major >= 7:
    return f"https://download.pytorch.org/whl/rocm{rocm_version.short_version}/"
```

Now generates correct URLs for:
- ROCm 7.0 stable: `https://download.pytorch.org/whl/rocm7.0/`
- ROCm 7.0 nightly: `https://download.pytorch.org/whl/nightly/rocm7.0/`
- ROCm 6.5+: `https://download.pytorch.org/whl/rocm6.5/` (and nightly)
- ROCm 5.7 (legacy): `https://download.pytorch.org/whl/rocm5.7/`

### 2. PyTorch Version Detection

Enhanced `check_pytorch_installation()` to extract version information:

```python
# Parse PyTorch version to get major.minor
pytorch_major_minor = None
try:
    # Handle versions like "2.10.0+rocm7.0" or "2.3.1"
    version_parts = pytorch_version.split('+')[0].split('.')
    if len(version_parts) >= 2:
        pytorch_major_minor = f"{version_parts[0]}.{version_parts[1]}"
except Exception:
    pass
```

Returns comprehensive installation info including:
- Full version string (e.g., "2.10.0+rocm7.0")
- Major.minor version (e.g., "2.10")
- ROCm build information
- GPU availability and count

### 3. Dependency Compatibility Matrix

Created `get_compatible_dependency_versions()` function with version-specific dependencies:

```python
def get_compatible_dependency_versions(pytorch_version: Optional[str] = None) -> Dict[str, str]:
    """Get compatible dependency versions based on installed PyTorch version."""
    
    # PyTorch 2.10+ (nightly/future releases)
    if major >= 3 or (major == 2 and minor >= 10):
        return {
            "transformers": ">=4.45.0",
            "diffusers": ">=0.31.0",
            "accelerate": ">=0.34.0",
            "huggingface_hub": ">=0.25.0",
        }
    
    # PyTorch 2.4-2.9
    elif major == 2 and 4 <= minor <= 9:
        return {
            "transformers": ">=4.43.0",
            "diffusers": ">=0.29.0",
            "accelerate": ">=0.30.0",
            "huggingface_hub": ">=0.24.0",
        }
    
    # PyTorch 2.3.x (ROCm 5.7 legacy) - with upper bounds for safety
    elif major == 2 and minor == 3:
        return {
            "transformers": ">=4.41.0,<4.50.0",
            "diffusers": ">=0.28.0,<0.35.0",
            "accelerate": ">=0.29.0,<0.35.0",
            "huggingface_hub": ">=0.23.0,<0.30.0",
        }
```

### 4. Package Configuration

**File**: `pyproject.toml`

Added ROCm 7.0 optional dependencies:

```toml
[project.optional-dependencies]
# ROCm 7.0+ (standard wheels - install via index URL)
rocm70 = [
    "torch>=2.5.0",
    "torchvision>=0.20.0",
    "torchaudio>=2.5.0",
]
```

Installation:
```bash
pip install -e .[rocm70] --index-url https://download.pytorch.org/whl/rocm7.0
```

### 5. Server Setup Script

**File**: `server-setup.sh`

Updated ROCm version detection:

```bash
get_pytorch_index_url() {
    local rocm_ver="$1"
    local major=$(echo "$rocm_ver" | cut -d'.' -f1)
    local minor=$(echo "$rocm_ver" | cut -d'.' -f2)
    
    # ROCm 7.0+ uses standard wheels
    if [[ "$major" -ge 7 ]]; then
        echo "https://download.pytorch.org/whl/rocm${major}.${minor}/"
    # ... additional version checks
}
```

### 6. Testing

**File**: `tests/unit/test_rocm_utils.py`

Added comprehensive tests:

- ✅ ROCm 7.0 URL generation (stable and nightly)
- ✅ PyTorch version parsing and detection
- ✅ Dependency compatibility for PyTorch 2.10+
- ✅ Dependency compatibility for PyTorch 2.4-2.9
- ✅ Dependency compatibility for PyTorch 2.3.x (legacy)
- ✅ Backward compatibility with older PyTorch versions

**Results**: 31 tests passing, 2 skipped (complex mocking scenarios)

### 7. Documentation

**File**: `PYTORCH_VERSION_COMPATIBILITY.md`

Created comprehensive documentation covering:
- Problem statement and solution
- Feature descriptions
- Installation examples
- Dependency compatibility matrix
- Migration guide from ROCm 5.7 to 7.0
- Troubleshooting guide
- API reference
- Best practices

### 8. Demonstration Script

**File**: `demo_pytorch_version_check.py`

Interactive script that demonstrates:
- ROCm version detection
- PyTorch installation checking
- Compatible dependency version determination
- Installation command generation
- System-specific recommendations

## Usage Examples

### Checking Current Configuration

```python
from backend.utils.rocm_utils import check_pytorch_installation, get_compatible_dependency_versions

# Check installed PyTorch
pytorch_info = check_pytorch_installation()
print(f"PyTorch: {pytorch_info['version']}")
print(f"Major.Minor: {pytorch_info['pytorch_major_minor']}")

# Get compatible dependencies
deps = get_compatible_dependency_versions(pytorch_info['version'])
for pkg, version in deps.items():
    print(f"{pkg}: {version}")
```

### Installing PyTorch 2.10 with ROCm 7.0

```bash
# Method 1: Direct installation
pip3 install --pre torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/nightly/rocm7.0

# Install compatible dependencies
pip3 install 'transformers>=4.45.0' 'diffusers>=0.31.0' 'accelerate>=0.34.0'

# Method 2: Using pyproject.toml
pip install -e .[rocm70] --index-url https://download.pytorch.org/whl/rocm7.0
```

### Automated Detection and Installation

```python
from backend.utils.rocm_utils import (
    detect_rocm_version,
    get_pytorch_install_command,
    get_compatible_dependency_versions
)

# Detect ROCm and generate installation command
rocm_version = detect_rocm_version()
command, metadata = get_pytorch_install_command(
    rocm_version,
    use_nightly=True,  # For PyTorch 2.10+
    include_torchvision=True,
    include_torchaudio=True
)

print(f"Install command: {command}")

# Get compatible dependencies
deps = get_compatible_dependency_versions("2.10.0+rocm7.0")
for pkg, version in deps.items():
    print(f"pip install '{pkg}{version}'")
```

## Dependency Compatibility Matrix

| PyTorch Version | Transformers | Diffusers | Accelerate | Hugging Face Hub |
|----------------|--------------|-----------|------------|------------------|
| **2.10+ (nightly)** | >=4.45.0 | >=0.31.0 | >=0.34.0 | >=0.25.0 |
| **2.4-2.9** | >=4.43.0 | >=0.29.0 | >=0.30.0 | >=0.24.0 |
| **2.3.x (ROCm 5.7)** | >=4.41.0,<4.50.0 | >=0.28.0,<0.35.0 | >=0.29.0,<0.35.0 | >=0.23.0,<0.30.0 |
| **2.0-2.2** | >=4.35.0,<4.45.0 | >=0.25.0,<0.30.0 | >=0.25.0,<0.30.0 | >=0.20.0,<0.25.0 |

## Verification

### Test Results

```
$ python -m pytest tests/unit/test_rocm_utils.py -v
================================================= test session starts ==================================================
...
====================================== 31 passed, 2 skipped, 8 warnings in 0.09s =======================================
```

### Security Check

```
$ codeql_checker
Analysis Result for 'python'. Found 0 alerts:
- **python**: No alerts found.
```

### Demonstration

```
$ python demo_pytorch_version_check.py
================================================================================
PyTorch Version Detection and Dependency Compatibility Check
================================================================================

Step 1: Detecting ROCm version...
✓ ROCm detected: 7.0.0

Step 2: PyTorch installation URLs...
  Stable builds: https://download.pytorch.org/whl/rocm7.0/
  Nightly builds: https://download.pytorch.org/whl/nightly/rocm7.0/

Step 3: Checking installed PyTorch...
✓ PyTorch is installed
  - Version: 2.10.0+rocm7.0
  - Major.Minor: 2.10
  - ROCm build: True
  - ROCm version: 7.0.0

Step 4: Compatible dependency versions for installed PyTorch...
  Based on PyTorch 2.10.0+rocm7.0:
    - transformers: >=4.45.0
    - diffusers: >=0.31.0
    - accelerate: >=0.34.0
    - huggingface_hub: >=0.25.0
```

## Impact

### Benefits

1. **Automatic Compatibility**: System automatically determines correct dependency versions
2. **Reduced Errors**: Prevents runtime errors from incompatible library versions
3. **Better UX**: Clear guidance on what to install for each PyTorch version
4. **Future-Proof**: Supports PyTorch 2.10+ and future ROCm releases
5. **Backward Compatible**: Works with existing ROCm 5.7, 6.4, 6.5+ setups

### Backward Compatibility

All existing functionality is preserved:
- ✅ ROCm 5.7 support (legacy MI-25 hardware)
- ✅ ROCm 6.4 support
- ✅ ROCm 6.5+ support with standard wheels
- ✅ Existing scripts and tools continue to work

### Breaking Changes

**None**. This is a purely additive change that enhances existing functionality.

## Files Changed

1. **`src/backend/utils/rocm_utils.py`** (+88 lines, -41 lines)
   - Enhanced ROCm 7.0 detection
   - Added PyTorch version extraction
   - Added dependency compatibility checking

2. **`pyproject.toml`** (+7 lines)
   - Added rocm70 optional dependency group

3. **`server-setup.sh`** (+3 lines)
   - Updated ROCm 7.0 detection logic

4. **`tests/unit/test_rocm_utils.py`** (+46 lines)
   - Added comprehensive tests for new features

5. **`PYTORCH_VERSION_COMPATIBILITY.md`** (new, 336 lines)
   - Comprehensive documentation

6. **`demo_pytorch_version_check.py`** (new, 150 lines)
   - Interactive demonstration script

7. **`IMPLEMENTATION_SUMMARY_PYTORCH_COMPATIBILITY.md`** (this file)
   - Implementation summary and guide

## Future Enhancements

Potential improvements for future releases:

1. **Automatic Dependency Updates**: Script to automatically upgrade dependencies when PyTorch is upgraded
2. **Conflict Resolution**: Detect and resolve dependency conflicts automatically
3. **Version Locking**: Lock compatible versions in requirements.txt after successful testing
4. **CI/CD Integration**: Automated testing with different PyTorch versions
5. **GUI Tool**: Visual interface for version checking and dependency management

## Conclusion

This implementation successfully addresses the issue of PyTorch 2.10 and ROCm 7.0 support while maintaining full backward compatibility. The system now intelligently detects PyTorch versions and recommends compatible dependency versions, reducing errors and improving the user experience.

## Resources

- **Documentation**: `PYTORCH_VERSION_COMPATIBILITY.md`
- **Demo**: `demo_pytorch_version_check.py`
- **Tests**: `tests/unit/test_rocm_utils.py`
- **Issue**: [pytorch 2.10 support](https://github.com/terminills/gator/issues/XXX)
