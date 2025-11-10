# vLLM and ComfyUI Installation Scripts

## Quick Reference

This document provides a quick reference for the new installation scripts added to support vLLM and ComfyUI on AMD ROCm systems.

### Problem Statement

vLLM on AMD does not have standard wheels available and requires building from source. ComfyUI also needs proper configuration for ROCm support. These scripts automate the entire process.

### Solution Overview

Two automated bash scripts that handle:
1. **vLLM** - Build from source for AMD ROCm (no wheels available)
2. **ComfyUI** - Install and configure with ROCm support

## Quick Start

### Install vLLM

```bash
# Activate your virtual environment
source venv/bin/activate

# Run installation script
bash scripts/install_vllm_rocm.sh

# Verify installation
python3 -c "import vllm; print(vllm.__version__)"
```

**Build time**: 10-30 minutes (one-time build)

### Install ComfyUI

```bash
# Activate your virtual environment
source venv/bin/activate

# Run installation script
bash scripts/install_comfyui_rocm.sh

# Download models (optional)
cd ComfyUI && python3 download_models.py

# Launch ComfyUI
./launch_rocm.sh
```

**Installation time**: 5-10 minutes
**Access UI**: http://localhost:8188

## What's Included

### Scripts

1. **scripts/install_vllm_rocm.sh** - Automated vLLM build and installation
   - Detects ROCm version
   - Installs PyTorch with ROCm support
   - Compiles vLLM from source
   - Configures for optimal performance

2. **scripts/install_comfyui_rocm.sh** - ComfyUI installation with ROCm
   - Clones ComfyUI repository
   - Installs all dependencies
   - Adds ComfyUI Manager
   - Creates helper scripts

### Documentation

- **scripts/README.md** - Detailed script documentation
- **docs/INSTALLATION_SCRIPTS_GUIDE.md** - Complete installation guide
- **README.md** - Updated with installation instructions

### Tests

- **tests/test_installation_scripts.py** - Validation test suite
  - 10 comprehensive tests
  - Validates syntax, permissions, content
  - Verifies integration

## Key Features

✅ **Automatic ROCm detection** - Detects version and installs compatible packages
✅ **Virtual environment support** - Works with active venv
✅ **Build dependency checking** - Verifies all requirements
✅ **Progress indicators** - Clear, colored output
✅ **Error handling** - Comprehensive validation
✅ **Custom directories** - Flexible installation paths
✅ **CPU fallback** - ComfyUI works without GPU
✅ **Extension support** - ComfyUI Manager included

## System Requirements

### Minimum
- Ubuntu 20.04+ or Debian 11+
- Python 3.9+
- 16GB RAM
- 50GB disk space

### Recommended
- ROCm 5.7+ or 6.x+
- AMD GPU with ROCm support
- 32GB+ RAM
- 100GB+ disk space

## File Manifest

```
scripts/
├── install_vllm_rocm.sh        # vLLM installation script (6.6KB)
├── install_comfyui_rocm.sh     # ComfyUI installation script (10.4KB)
└── README.md                    # Script documentation (7.2KB)

docs/
└── INSTALLATION_SCRIPTS_GUIDE.md  # Comprehensive guide (9.5KB)

tests/
└── test_installation_scripts.py   # Validation tests (9.2KB)

Updated files:
├── setup_ai_models.py           # Integration updates
└── README.md                    # Main documentation updates
```

## Validation Status

All validation checks passed:
- ✅ Script syntax validated
- ✅ Python syntax validated
- ✅ All tests passing (10/10)
- ✅ Security scan clean (CodeQL)
- ✅ Documentation complete
- ✅ Integration verified

## Usage Examples

### vLLM Usage

```python
from vllm import LLM, SamplingParams

# Initialize model
llm = LLM(model="meta-llama/Llama-2-7b-chat-hf")

# Configure sampling
params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=512
)

# Generate
outputs = llm.generate(["Tell me about AI:"], params)
print(outputs[0].outputs[0].text)
```

### ComfyUI Usage

1. Launch: `cd ComfyUI && ./launch_rocm.sh`
2. Open browser: http://localhost:8188
3. Load workflow or create new
4. Queue prompt to generate images

## Troubleshooting

### Common Issues

**ROCm not detected**:
```bash
# Install ROCm first
# See: https://rocm.docs.amd.com/
```

**Build fails (vLLM)**:
```bash
# Reduce parallel jobs
export MAX_JOBS=1
bash scripts/install_vllm_rocm.sh
```

**GPU not detected (ComfyUI)**:
```bash
# Set GFX version for your GPU
export HSA_OVERRIDE_GFX_VERSION=10.3.0  # Example for gfx1030
cd ComfyUI && ./launch_rocm.sh
```

## Reference Links

- **vLLM Documentation**: https://docs.vllm.ai/
- **ComfyUI Repository**: https://github.com/comfyanonymous/ComfyUI
- **ROCm Documentation**: https://rocm.docs.amd.com/
- **Issue Reference**: https://docs.vllm.ai/en/stable/getting_started/installation/gpu.html#unsupported-os-build

## Support

For issues or questions:
1. Check documentation: `docs/INSTALLATION_SCRIPTS_GUIDE.md`
2. Review troubleshooting: `scripts/README.md`
3. Open issue: https://github.com/terminills/gator/issues

## Credits

Implementation by GitHub Copilot
Issue: vllm on AMD requires custom build scripts
Date: November 10, 2025

---

**Note**: These scripts automate the entire installation process. No manual intervention required after running the scripts, assuming system requirements are met.
