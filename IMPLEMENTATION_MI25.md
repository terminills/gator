# MI-25 Compatibility Enhancement - Implementation Summary

## Issue
**MI-25 Compatibility** - Use learnings from https://github.com/terminills/mi-25 to fix compatibility with the MI-25 and Ubuntu 20.04

## Solution Overview

This implementation adds comprehensive MI25 (AMD Radeon Instinct MI25, gfx900 architecture) compatibility to the Gator AI Influencer Platform for Ubuntu 20.04 systems.

## Key Changes

### 1. Enhanced MI25 GPU Detection (`server-setup.sh`)

**Problem**: Previous detection only used lspci string matching which could be unreliable.

**Solution**: Added multiple detection methods:
- Text-based detection: "Radeon Instinct MI25" and "Vega 10" strings
- Device ID detection: Check for Vega 10 PCI device IDs (0x6860-0x686f)
- Set `IS_MI25` flag for configuration throughout the script

```bash
# Check via device ID (Vega 10 = 0x6860-0x686f)
if lspci -n | grep -E "1002:(6860|6861|...)" > /dev/null; then
    IS_MI25=true
fi
```

### 2. Fixed ROCm 4.5.2 Repository URL

**Problem**: ROCm 4.5.2 repository URL structure differs from newer versions on Ubuntu 20.04.

**Solution**: Added specific repository URL handling:
```bash
if [[ "$UBUNTU_VERSION" == "20.04" ]]; then
    if [[ "$ROCM_VERSION" == "4.5.2" ]]; then
        REPO_URL="deb [arch=amd64] https://repo.radeon.com/rocm/apt/4.5.2 ubuntu main"
    fi
fi
```

### 3. Kernel Version Validation

**Problem**: ROCm 4.5.2 requires kernel 5.4+ but Ubuntu 20.04 may have older kernels.

**Solution**: Added kernel version check with clear upgrade guidance:
```bash
KERNEL_VERSION=$(uname -r | cut -d. -f1-2)
if [[ $KERNEL_MAJOR -lt 5 ]] || [[ $KERNEL_MAJOR -eq 5 && $KERNEL_MINOR -lt 4 ]]; then
    warn "Consider upgrading kernel: sudo apt install linux-generic-hwe-20.04"
fi
```

### 4. gfx900-Specific Environment Variables

**Problem**: Many ML frameworks don't recognize or support gfx900 without specific environment configuration.

**Solution**: Added critical environment variables for MI25:
```bash
export HSA_OVERRIDE_GFX_VERSION=9.0.0      # Critical for gfx900 compatibility
export HCC_AMDGPU_TARGET=gfx900             # Compiler target
export PYTORCH_ROCM_ARCH=gfx900             # PyTorch optimization
export TF_ROCM_AMDGPU_TARGETS=gfx900        # TensorFlow support
```

`HSA_OVERRIDE_GFX_VERSION=9.0.0` is particularly important - it tells the ROCm runtime to treat gfx900 as supported even when newer libraries might not explicitly list it.

### 5. ROCm 4.5.2 Package Installation

**Problem**: Package names differ between ROCm versions and some packages may not be available.

**Solution**: Added version-specific package installation with fallback:
```bash
if [[ "$ROCM_VERSION" == "4.5.2" ]]; then
    apt install -y hip-runtime-amd hip-dev rocrand rocblas rocsparse || {
        warn "Some math libraries may not be available for ROCm 4.5.2"
    }
fi
```

### 6. Enhanced Verification Script

**Problem**: Original check_rocm.sh had limited information for troubleshooting.

**Solution**: Enhanced with comprehensive checks:
- ROCm version from multiple sources
- Kernel version
- GPU detection via lspci, rocm-smi, and rocminfo
- Architecture detection (gfx900)
- Environment variable validation
- User group membership
- Library presence checks

### 7. Improved setup_ai_models.py Detection

**Problem**: Limited MI25 detection in AI model setup.

**Solution**: Added multiple detection methods and MI25-specific system info:
```python
# Detect via rocm-smi, lspci, and directory checks
is_mi25 = False
rocm_arch = "unknown"

# Multiple detection methods...
if is_mi25:
    sys_info["gpu_architecture"] = "gfx900"
    sys_info["is_mi25"] = True
    sys_info["compatibility_notes"] = [
        "MI25 (gfx900) detected - ROCm 4.5.2 recommended",
        "HSA_OVERRIDE_GFX_VERSION=9.0.0 should be set",
        ...
    ]
```

### 8. Comprehensive Documentation

Created `docs/MI25_COMPATIBILITY.md` with:
- System requirements
- Installation instructions (automated and manual)
- Post-installation configuration
- Kernel requirements and upgrade process
- Verification procedures
- ML framework compatibility (PyTorch, TensorFlow, etc.)
- Inference engine recommendations
- Common issues and solutions
- Performance optimization tips
- Production deployment guidelines

### 9. Test Suite

Created `tests/test_mi25_compatibility.py` to validate:
- Bash syntax correctness
- MI25 detection patterns
- ROCm 4.5.2 version selection
- gfx900 environment variables
- Kernel version checks
- Enhanced verification script
- Documentation completeness
- setup_ai_models.py detection

All 8 tests passing ✅

### 10. README Updates

Added GPU Support section explaining:
- AMD GPU support with MI25 special mention
- Link to MI25 compatibility guide
- NVIDIA GPU support
- CPU-only mode
- Installation commands for each

## Technical Details

### Why ROCm 4.5.2?
- Last version with full gfx900 (MI25/Vega 10) support
- Newer versions (5.x+) have dropped or limited gfx900 support
- Best tested and most stable for production MI25 workloads

### Why HSA_OVERRIDE_GFX_VERSION?
- Many newer ROCm libraries check GPU architecture
- They may refuse to run on gfx900 even when technically compatible
- Override tells runtime to treat gfx900 as supported
- Required for PyTorch 1.10+, some versions of TensorFlow, etc.

### Kernel 5.4+ Requirement
- ROCm 4.5.2 kernel modules require features from kernel 5.4+
- Ubuntu 20.04 ships with 5.4.0 by default (good)
- Older systems may have kernel 5.3 or earlier (need upgrade)
- HWE (Hardware Enablement) kernel recommended: `linux-generic-hwe-20.04`

### Device ID Detection
- Vega 10 architecture uses PCI device IDs 0x6860-0x686f
- More reliable than string matching
- Detects MI25 even with custom firmware or modified names
- Works even if GPU doesn't report "MI25" string

## Testing Performed

1. **Bash Syntax Validation**: `bash -n server-setup.sh` - PASSED ✅
2. **Python Syntax Validation**: `python3 -m py_compile setup_ai_models.py` - PASSED ✅
3. **MI25 Compatibility Tests**: All 8 tests - PASSED ✅
4. **Documentation Completeness**: All sections present - PASSED ✅

## Files Modified

1. `server-setup.sh` - Core MI25 detection and installation logic
2. `setup_ai_models.py` - Enhanced GPU detection and MI25 handling
3. `README.md` - Added GPU support section and installation examples
4. `docs/MI25_COMPATIBILITY.md` - NEW comprehensive guide
5. `tests/test_mi25_compatibility.py` - NEW test suite

## Backward Compatibility

All changes are backward compatible:
- Non-MI25 systems continue using ROCm 5.7.1 (or latest)
- Environment variables are MI25-specific, don't affect other GPUs
- Package installation has fallbacks for missing packages
- Documentation is additive, doesn't replace existing docs

## Usage

### Automatic Installation
```bash
# Detects MI25 automatically
curl -sSL https://raw.githubusercontent.com/terminills/gator/main/server-setup.sh | sudo bash -s -- --rocm

# Or with full options
sudo bash server-setup.sh --rocm --domain example.com --email admin@example.com
```

### Verification
```bash
# After reboot
/opt/gator/check_rocm.sh
rocm-smi
```

### Expected Output for MI25
```
ROCm Version: 4.5.2
GPU Architecture: gfx900 (MI25/Vega10)
HSA_OVERRIDE_GFX_VERSION: 9.0.0
```

## Known Limitations

1. ROCm 4.5.2 is end-of-life but necessary for MI25
2. Some cutting-edge ML frameworks don't support gfx900
3. PyTorch 2.0+ has limited gfx900 support (recommend 1.10-1.12)
4. TensorFlow 2.8+ has dropped gfx900 (recommend 2.7 or earlier)

## Future Improvements

1. Add automatic PyTorch ROCm 4.5.2 installation
2. Pre-built model configurations for MI25 limitations
3. Automated performance tuning scripts
4. Multi-node MI25 cluster setup
5. Container images with MI25 optimization

## References

- Issue: MI-25 Compatibility
- ROCm Documentation: https://rocmdocs.amd.com/
- MI25 Specifications: AMD Radeon Instinct MI25 (Vega 10, gfx900)
- Ubuntu 20.04 LTS Support
