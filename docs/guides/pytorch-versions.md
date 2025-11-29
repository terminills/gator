# PyTorch Version Compatibility and Dependency Management

This document describes how Gator handles PyTorch version detection and ensures compatible dependency versions are installed.

## Problem Statement

When installing PyTorch with ROCm support, different ROCm versions provide different PyTorch versions:

- **ROCm 7.0+ nightly**: Provides PyTorch 2.10+ (nightly builds)
- **ROCm 6.5+**: Provides PyTorch 2.4+ (stable builds with standard wheels)
- **ROCm 5.7**: Provides PyTorch 2.3.1 (legacy builds for older hardware like MI-25)

Each PyTorch version requires specific versions of ML libraries (transformers, diffusers, accelerate) for compatibility. Installing mismatched versions can cause runtime errors or unexpected behavior.

## Solution

Gator now includes intelligent PyTorch version detection and dependency compatibility checking in the `backend.utils.rocm_utils` module.

## Features

### 1. ROCm 7.0 Detection

The system automatically detects ROCm 7.0+ installations and provides appropriate PyTorch index URLs:

```python
from backend.utils.rocm_utils import detect_rocm_version, get_pytorch_index_url

# Detect installed ROCm version
rocm_version = detect_rocm_version()  # Returns ROCmVersionInfo("7.0.0", 7, 0, 0)

# Get appropriate PyTorch index URL
stable_url = get_pytorch_index_url(rocm_version, use_nightly=False)
# Returns: "https://download.pytorch.org/whl/rocm7.0/"

nightly_url = get_pytorch_index_url(rocm_version, use_nightly=True)
# Returns: "https://download.pytorch.org/whl/nightly/rocm7.0/"
```

### 2. PyTorch Version Detection

Check the currently installed PyTorch version and extract compatibility information:

```python
from backend.utils.rocm_utils import check_pytorch_installation

pytorch_info = check_pytorch_installation()
# Returns:
# {
#     "installed": True,
#     "version": "2.10.0+rocm7.0",
#     "pytorch_major_minor": "2.10",
#     "is_rocm_build": True,
#     "rocm_build_version": "7.0.0",
#     "gpu_available": True,
#     "gpu_count": 4,
#     "gpu_architecture": {...}
# }
```

### 3. Compatible Dependency Versions

Automatically determine compatible ML library versions based on installed PyTorch:

```python
from backend.utils.rocm_utils import get_compatible_dependency_versions

# For PyTorch 2.10+
deps = get_compatible_dependency_versions("2.10.0+rocm7.0")
# Returns:
# {
#     "transformers": ">=4.45.0",
#     "diffusers": ">=0.31.0",
#     "accelerate": ">=0.34.0",
#     "huggingface_hub": ">=0.25.0"
# }

# For PyTorch 2.3.1 (ROCm 5.7 legacy)
deps = get_compatible_dependency_versions("2.3.1+rocm5.7")
# Returns:
# {
#     "transformers": ">=4.41.0,<4.50.0",  # Upper bounds for safety
#     "diffusers": ">=0.28.0,<0.35.0",
#     "accelerate": ">=0.29.0,<0.35.0",
#     "huggingface_hub": ">=0.23.0,<0.30.0"
# }
```

## Installation Examples

### Installing PyTorch 2.10 with ROCm 7.0

```bash
# Install PyTorch nightly with ROCm 7.0
pip3 install --pre torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/nightly/rocm7.0

# Install compatible dependencies
pip3 install 'transformers>=4.45.0' 'diffusers>=0.31.0' 'accelerate>=0.34.0'
```

### Using pyproject.toml

Add to your installation:

```bash
# For ROCm 7.0+
pip install -e .[rocm70] --index-url https://download.pytorch.org/whl/rocm7.0

# For ROCm 6.5+
pip install -e .[rocm65] --index-url https://download.pytorch.org/whl/rocm6.5

# For ROCm 5.7 (legacy)
pip install -e .[rocm57] --index-url https://download.pytorch.org/whl/rocm5.7
```

### Automated Installation

Use the `get_pytorch_install_command()` function:

```python
from backend.utils.rocm_utils import get_pytorch_install_command, detect_rocm_version

rocm_version = detect_rocm_version()
command, metadata = get_pytorch_install_command(
    rocm_version,
    use_nightly=True,  # For PyTorch 2.10+
    include_torchvision=True,
    include_torchaudio=True
)

print(command)
# Output: pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm7.0/
```

## Dependency Compatibility Matrix

| PyTorch Version | Transformers | Diffusers | Accelerate | Hugging Face Hub |
|----------------|--------------|-----------|------------|------------------|
| 2.10+ (nightly) | >=4.45.0 | >=0.31.0 | >=0.34.0 | >=0.25.0 |
| 2.4-2.9 | >=4.43.0 | >=0.29.0 | >=0.30.0 | >=0.24.0 |
| 2.3.x (ROCm 5.7) | >=4.41.0,<4.50.0 | >=0.28.0,<0.35.0 | >=0.29.0,<0.35.0 | >=0.23.0,<0.30.0 |
| 2.0-2.2 | >=4.35.0,<4.45.0 | >=0.25.0,<0.30.0 | >=0.25.0,<0.30.0 | >=0.20.0,<0.25.0 |

## Testing

Run the demonstration script to verify your system configuration:

```bash
python demo_pytorch_version_check.py
```

Run unit tests:

```bash
python -m pytest tests/unit/test_rocm_utils.py -v
```

## Server Setup Integration

The `server-setup.sh` script automatically handles ROCm 7.0 detection:

```bash
# Automatic ROCm 7.0+ detection and PyTorch installation
sudo ./server-setup.sh --rocm --domain myserver.com --email admin@myserver.com
```

The script will:
1. Detect installed ROCm version (5.7, 6.5, 7.0, etc.)
2. Determine appropriate PyTorch index URL
3. Install compatible PyTorch and ML libraries
4. Configure environment variables for optimal performance

## Migration Guide

### Upgrading from ROCm 5.7 to ROCm 7.0

1. **Backup your environment**:
   ```bash
   pip freeze > requirements_backup.txt
   ```

2. **Upgrade ROCm** (follow AMD ROCm documentation)

3. **Verify ROCm installation**:
   ```bash
   rocm-smi --version
   cat /opt/rocm/.info/version
   ```

4. **Reinstall PyTorch**:
   ```bash
   pip uninstall torch torchvision torchaudio
   pip3 install --pre torch torchvision torchaudio \
     --index-url https://download.pytorch.org/whl/nightly/rocm7.0
   ```

5. **Update ML libraries**:
   ```bash
   pip install --upgrade 'transformers>=4.45.0' 'diffusers>=0.31.0' 'accelerate>=0.34.0'
   ```

6. **Verify installation**:
   ```bash
   python demo_pytorch_version_check.py
   ```

### Checking Current Configuration

```python
from backend.utils.rocm_utils import check_pytorch_installation

info = check_pytorch_installation()
print(f"PyTorch: {info['version']}")
print(f"ROCm build: {info['is_rocm_build']}")
print(f"ROCm version: {info['rocm_build_version']}")
```

## Troubleshooting

### Issue: PyTorch 2.10 installed but ML libraries incompatible

**Solution**: Check and upgrade ML libraries:
```bash
python demo_pytorch_version_check.py  # See recommended versions
pip install --upgrade transformers diffusers accelerate
```

### Issue: ROCm not detected

**Solution**: Verify ROCm installation:
```bash
ls -la /opt/rocm/.info/version
rocminfo
```

### Issue: GPU not available in PyTorch

**Solution**: Check ROCm environment variables:
```bash
echo $ROCM_PATH
echo $HIP_VISIBLE_DEVICES
python -c "import torch; print(torch.cuda.is_available())"
```

## API Reference

### `detect_rocm_version() -> Optional[ROCmVersionInfo]`

Detects installed ROCm version from multiple sources.

**Returns**: ROCmVersionInfo object or None if not detected

### `get_pytorch_index_url(rocm_version, use_nightly=False) -> str`

Gets appropriate PyTorch index URL for the given ROCm version.

**Parameters**:
- `rocm_version`: ROCmVersionInfo object or None
- `use_nightly`: Whether to use nightly builds (ROCm 6.5+ only)

**Returns**: PyTorch wheel index URL

### `check_pytorch_installation() -> Dict[str, any]`

Checks installed PyTorch and extracts version information.

**Returns**: Dictionary with installation details

### `get_compatible_dependency_versions(pytorch_version=None) -> Dict[str, str]`

Gets compatible ML library versions for the given PyTorch version.

**Parameters**:
- `pytorch_version`: PyTorch version string or None (auto-detect)

**Returns**: Dictionary mapping package names to version specifiers

### `get_pytorch_install_command(rocm_version, use_nightly, ...) -> Tuple[str, Dict]`

Generates pip install command for PyTorch with ROCm support.

**Returns**: Tuple of (pip command string, metadata dict)

## Best Practices

1. **Always check compatibility** before upgrading PyTorch or ML libraries
2. **Use version specifiers** when installing packages to ensure compatibility
3. **Test your code** after upgrades with your specific workload
4. **Keep ROCm updated** to benefit from latest optimizations
5. **Document your environment** with `pip freeze` after successful configuration

## Support

- ROCm issues: https://github.com/RadeonOpenCompute/ROCm/issues
- PyTorch issues: https://github.com/pytorch/pytorch/issues
- Gator issues: https://github.com/terminills/gator/issues

## References

- [PyTorch ROCm Installation Guide](https://pytorch.org/get-started/locally/)
- [ROCm Documentation](https://rocm.docs.amd.com/)
- [Hugging Face Model Hub](https://huggingface.co/models)
