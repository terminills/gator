# vLLM PyTorch 2.10 Build Fix and Repair Guide

## Problem Overview

When building vLLM from source on ROCm 7.0+ systems with PyTorch 2.10 nightly builds, you may encounter version conflicts:

```
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. 
This behaviour is the source of the following dependency conflicts.
torchvision 0.25.0.dev20251108+rocm7.0 requires torch==2.10.0.dev20251107, 
but you have torch 2.9.0 which is incompatible.
```

## Root Cause

The issue occurs because:
1. vLLM's build process creates an isolated pip environment
2. This isolated environment installs PyTorch dependencies independently
3. The isolated environment may install PyTorch 2.9.0 from vLLM's build requirements
4. The system has torchvision 0.25.0 nightly which requires PyTorch 2.10.0
5. Result: Version mismatch and build failure

## Solutions

### Solution 1: Standard Installation with --no-build-isolation (Recommended)

The `install_vllm_rocm.sh` script now uses `--no-build-isolation` by default to prevent this issue:

```bash
# Standard installation - uses existing PyTorch
bash scripts/install_vllm_rocm.sh
```

**How it works:**
- Uses `pip install -e . --no-build-isolation` when building vLLM
- Reuses your existing PyTorch installation instead of creating isolated environment
- Prevents version conflicts automatically

### Solution 2: Use Stable PyTorch 2.8.0 from AMD Repository

For a more stable installation, use PyTorch 2.8.0 from AMD's official ROCm repository:

```bash
# Install using AMD repository (PyTorch 2.8.0)
bash scripts/install_vllm_rocm.sh --amd-repo
```

**Advantages:**
- Stable PyTorch 2.8.0 release instead of nightly
- Officially supported by AMD
- Better compatibility with vLLM build process
- Includes optimized triton kernels

**Source:** AMD ROCm manylinux repository at `https://repo.radeon.com/rocm/manylinux/rocm-rel-7.0.2/`

### Solution 3: Repair Mode (For Failed Installations)

If vLLM installation has already failed due to PyTorch conflicts, use repair mode:

```bash
# Repair existing PyTorch installation
bash scripts/install_vllm_rocm.sh --repair
```

**What it does:**
1. Detects ROCm version (must be 7.0+)
2. Uninstalls existing torch, torchvision, torchaudio, triton
3. Reinstalls PyTorch 2.8.0 from AMD repository
4. Installs triton for optimized kernels
5. Verifies GPU support

After repair, run the standard installation again:
```bash
bash scripts/install_vllm_rocm.sh
```

### Solution 4: Manual Repair Command

If you prefer to fix manually without using the script:

```bash
# Uninstall existing PyTorch
pip uninstall -y torch torchvision torchaudio triton

# Install PyTorch 2.8.0 from AMD repository
pip install --pre torch==2.8.0 torchvision torchaudio==2.8.0 \
  -f https://repo.radeon.com/rocm/manylinux/rocm-rel-7.0.2/

# Install triton
pip install triton

# Verify installation
python3 -c "import torch; print(f'PyTorch {torch.__version__}')"
python3 -c "import torch; print(f'GPU available: {torch.cuda.is_available()}')"

# Now retry vLLM installation
bash scripts/install_vllm_rocm.sh
```

## Comparison: Nightly vs Stable

| Feature | PyTorch Nightly (2.10+) | AMD Repo (2.8.0) |
|---------|-------------------------|-------------------|
| **Version** | 2.10.0.dev (development) | 2.8.0 (stable) |
| **Stability** | May have bugs | Production-ready |
| **Features** | Latest features | Proven features |
| **vLLM Compatibility** | Good with --no-build-isolation | Excellent |
| **ROCm Support** | Full ROCm 7.0+ | Full ROCm 7.0+ |
| **Build Conflicts** | Possible without --no-build-isolation | Rare |
| **Recommended For** | Testing new features | Production use |

## Installation Methods Summary

### Quick Reference

```bash
# 1. Standard installation (PyTorch nightly, recommended)
bash scripts/install_vllm_rocm.sh

# 2. Stable installation (PyTorch 2.8.0 from AMD)
bash scripts/install_vllm_rocm.sh --amd-repo

# 3. Repair after failed build
bash scripts/install_vllm_rocm.sh --repair

# 4. Show all options
bash scripts/install_vllm_rocm.sh --help

# 5. Custom directory
bash scripts/install_vllm_rocm.sh /path/to/vllm
```

## Verification

After installation, verify everything works:

```bash
# Check PyTorch version
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"

# Check GPU availability
python3 -c "import torch; print(f'GPU available: {torch.cuda.is_available()}')"
python3 -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"

# Check vLLM installation
python3 -c "import vllm; print(f'vLLM: {vllm.__version__}')"

# Quick inference test
python3 << 'EOF'
from vllm import LLM, SamplingParams
llm = LLM(model='facebook/opt-125m')
outputs = llm.generate('Hello, my name is', SamplingParams(max_tokens=50))
print(outputs[0].outputs[0].text)
EOF
```

## Troubleshooting

### Issue: "GPU not available" after repair

**Solution:**
```bash
# Check ROCm is loaded
rocm-smi

# Verify HIP is working
hipinfo

# Check environment variables
echo $ROCM_HOME
echo $HIP_PATH

# Test PyTorch GPU detection
python3 -c "import torch; print(torch.cuda.is_available())"
```

### Issue: "triton not found" error

**Solution:**
```bash
# Install triton separately
pip install triton

# Verify installation
python3 -c "import triton; print(triton.__version__)"
```

### Issue: Build still fails after repair

**Solution:**
```bash
# Clean vLLM build artifacts
cd vllm-rocm
rm -rf build/ dist/ *.egg-info

# Ensure all build dependencies are installed
pip install cmake ninja packaging wheel setuptools-scm

# Retry build with verbose output
pip install -e . --no-build-isolation --verbose 2>&1 | tee build.log

# Check build.log for specific errors
```

### Issue: ImportError for torch extensions

**Solution:**
```bash
# Rebuild PyTorch extensions
pip install --force-reinstall --no-cache-dir torch torchvision torchaudio \
  -f https://repo.radeon.com/rocm/manylinux/rocm-rel-7.0.2/
```

## Best Practices

1. **Always use virtual environments:**
   ```bash
   python3 -m venv vllm-env
   source vllm-env/bin/activate
   ```

2. **For production systems, prefer stable builds:**
   ```bash
   bash scripts/install_vllm_rocm.sh --amd-repo
   ```

3. **For development, use nightly with --no-build-isolation:**
   ```bash
   bash scripts/install_vllm_rocm.sh
   ```

4. **Document your exact versions:**
   ```bash
   pip freeze > requirements.txt
   ```

5. **Keep ROCm updated:**
   ```bash
   sudo apt update && sudo apt upgrade rocm-dkms
   ```

## Technical Details

### --no-build-isolation Flag

The `--no-build-isolation` flag tells pip to use the current Python environment for building instead of creating a temporary isolated environment. This:

- Uses your existing PyTorch installation
- Prevents pip from installing incompatible PyTorch versions
- Requires all build dependencies to be pre-installed
- Is faster since it skips dependency resolution in isolated environment

### AMD ROCm Repository

The AMD repository at `https://repo.radeon.com/rocm/manylinux/rocm-rel-7.0.2/` provides:

- Pre-built PyTorch wheels for ROCm 7.0.2
- PyTorch 2.8.0 stable release
- Optimized torchvision and torchaudio
- ROCm-specific optimizations
- Better ABI compatibility with vLLM

### Automatic Fallback

The script includes automatic fallback logic for ROCm 7.0+:

1. First tries PyTorch nightly from `download.pytorch.org`
2. If that fails, automatically falls back to AMD repository
3. Installs PyTorch 2.8.0 with triton
4. Continues with vLLM build

## Support

- **GitHub Issues**: [terminills/gator/issues](https://github.com/terminills/gator/issues)
- **vLLM Documentation**: [docs.vllm.ai](https://docs.vllm.ai/)
- **ROCm Documentation**: [rocm.docs.amd.com](https://rocm.docs.amd.com/)
- **AMD Support**: [community.amd.com](https://community.amd.com/)

## References

- [vLLM GitHub](https://github.com/vllm-project/vllm)
- [AMD ROCm Installation Guide](https://rocm.docs.amd.com/en/latest/deploy/linux/quick_start.html)
- [PyTorch ROCm Documentation](https://pytorch.org/get-started/locally/)
- [pip build isolation](https://pip.pypa.io/en/stable/cli/pip_install/#cmdoption-no-build-isolation)

## License

This guide is part of the Gator project and is licensed under the MIT License.
