# AMD MI25 (Vega 10 / gfx900) Compatibility Guide

This guide provides specific instructions for running Gator AI Influencer Platform on systems with AMD Radeon Instinct MI25 GPUs with Ubuntu 20.04.

## Overview

The AMD MI25 is based on the Vega 10 architecture (gfx900) and requires specific ROCm versions and configurations for optimal compatibility, especially on Ubuntu 20.04.

## System Requirements

### Minimum Requirements
- **OS**: Ubuntu 20.04 LTS (Focal Fossa)
- **Kernel**: 5.4 or higher (5.4.0-generic recommended)
- **RAM**: 64GB+ recommended (32GB minimum)
- **GPU**: AMD Radeon Instinct MI25 (one or multiple)
- **Storage**: 500GB+ for models and data

### Recommended Configuration
- **OS**: Ubuntu 20.04.6 LTS
- **Kernel**: 5.4.0-generic (HWE kernel for best compatibility)
- **RAM**: 128GB+ for production workloads
- **GPUs**: 4-8x MI25 for optimal performance
- **Storage**: 1TB+ NVMe SSD

## ROCm Installation

### Supported ROCm Version

**ROCm 5.7.1** is confirmed working with MI25 on Ubuntu 20.04. The MI25 (gfx900 architecture) works well with this version when properly configured with the `HSA_OVERRIDE_GFX_VERSION=9.0.0` environment variable.

### Automatic Installation

The Gator setup script automatically detects MI25 GPUs and installs the correct ROCm version:

```bash
# Full installation with ROCm support
curl -sSL https://raw.githubusercontent.com/terminills/gator/main/server-setup.sh | sudo bash -s -- --rocm --domain your-domain.com --email your@email.com

# Or with explicit AMD GPU support
sudo bash server-setup.sh --rocm
```

### Manual Installation

If you need to install ROCm 5.7.1 manually, use AMD's official installer utility:

```bash
# Download AMD GPU installer package
wget https://repo.radeon.com/amdgpu-install/5.7.1/ubuntu/focal/amdgpu-install_5.7.50701-1_all.deb

# Install the package
sudo dpkg -i ./amdgpu-install_5.7.50701-1_all.deb
sudo apt install -f

# Install ROCm with required components
sudo amdgpu-install --usecase=rocm,hiplibsdk,dkms --rocmrelease=5.7.1

# Add users to required groups
sudo usermod -aG render,video $USER
sudo usermod -aG render,video root
```

**Note**: For Ubuntu 22.04, replace `focal` with `jammy` in the installer URL.

### Post-Installation Configuration

After installing ROCm, configure environment variables:

```bash
# Add to /etc/environment or ~/.bashrc
export PATH=/opt/rocm/bin:/opt/rocm/opencl/bin:$PATH
export LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64:$LD_LIBRARY_PATH
export ROCM_PATH=/opt/rocm
export HIP_PATH=/opt/rocm/hip

# Critical for MI25/gfx900 compatibility
export HSA_OVERRIDE_GFX_VERSION=9.0.0
export HCC_AMDGPU_TARGET=gfx900
export PYTORCH_ROCM_ARCH=gfx900
export TF_ROCM_AMDGPU_TARGETS=gfx900

# For multiple GPUs
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export GPU_DEVICE_ORDINAL=0,1,2,3,4,5,6,7
```

**Important**: The `HSA_OVERRIDE_GFX_VERSION=9.0.0` variable is critical for MI25. It enables ROCm 5.7.1 libraries to work with gfx900 architecture even though some libraries may not explicitly list gfx900 as supported.

## Kernel Requirements

### Checking Kernel Version

```bash
uname -r
```

Ubuntu 20.04 typically ships with kernel 5.4.0 or higher, which is compatible with ROCm 5.7.1.

### Upgrading Kernel (if needed)

If you're on an older kernel:

```bash
sudo apt update
sudo apt install -y linux-generic-hwe-20.04
sudo reboot
```

## Verification

### Check ROCm Installation

Use the provided verification script:

```bash
/opt/gator/check_rocm.sh
```

Expected output:
```
=== ROCm Installation Check ===

ROCm Version:
5.7.1

Kernel Version:
5.4.0-xxx-generic

GPU Devices (lspci):
[GPU information]

GPU Devices (rocm-smi):
GPU[0] : MI25
GPU[1] : MI25
...

HIP Platform:
AMD

GPU Architecture Detection:
  Name:                    gfx900
  ...

Environment Variables:
  ROCM_PATH: /opt/rocm
  HIP_PATH: /opt/rocm/hip
  HSA_OVERRIDE_GFX_VERSION: 9.0.0
  ...
```

### Manual Verification Commands

```bash
# Check GPUs
rocm-smi
rocm-smi --showid --showproductname

# Check HIP installation
hipconfig --platform
hipconfig --version

# Check GPU architecture
rocminfo | grep gfx

# Test GPU access
/opt/rocm/bin/rocm-smi --showmeminfo --showuse
```

## ML Framework Compatibility

### PyTorch

**Recommended Version**: PyTorch with ROCm 5.7.1 support

```bash
# Install PyTorch for ROCm 5.7
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7
```

**Testing PyTorch**:
```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device count: {torch.cuda.device_count()}")
print(f"Device name: {torch.cuda.get_device_name(0)}")
```

### TensorFlow

**Recommended Version**: TensorFlow-ROCm compatible builds

```bash
# Install TensorFlow for ROCm
pip3 install tensorflow-rocm
```

### Other Frameworks

- **ONNX Runtime**: Use ROCm 5.7.1 compatible builds
- **JAX**: May require HSA override for gfx900 compatibility
- **Transformers**: Works with compatible PyTorch version
- **Diffusers**: Works with compatible PyTorch version

## Inference Engines

### Recommended Engines for MI25

1. **Text Generation**:
   - vLLM (ROCm build for 4.5.2) - Best performance
   - llama.cpp with HIP support - Good compatibility
   - Transformers library - Wide model support

2. **Image Generation**:
   - ComfyUI (ROCm build) - Recommended
   - Automatic1111 (ROCm fork) - Good compatibility  
   - Diffusers library - Wide model support

3. **Voice/Audio**:
   - XTTS-v2 (with ROCm) - High quality
   - Piper TTS - Lightweight, CPU-based
   - Coqui TTS - Good quality

### Installation Notes

Some inference engines may need special configuration:

```bash
# vLLM with ROCm 5.7
export PYTORCH_ROCM_ARCH=gfx900
pip install vllm

# For ComfyUI
git clone https://github.com/comfyanonymous/ComfyUI
cd ComfyUI
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm5.7
pip install -r requirements.txt
```

## Common Issues and Solutions

### Issue 1: "Unsupported GPU architecture"

**Solution**: Set the HSA override:
```bash
export HSA_OVERRIDE_GFX_VERSION=9.0.0
```

### Issue 2: ROCm not detecting GPUs

**Solutions**:
1. Check user is in render and video groups: `groups | grep render`
2. Verify kernel module: `lsmod | grep amdgpu`
3. Check permissions: `ls -la /dev/kfd`
4. Reboot after installation

### Issue 3: Out of memory errors

**Solutions**:
1. Reduce batch size in model configurations
2. Use model quantization (4-bit or 8-bit)
3. Enable gradient checkpointing for training
4. Distribute across multiple MI25 GPUs

### Issue 4: Package installation failures

**Solutions**:
1. Use the official AMD GPU installer utility (recommended):
   ```bash
   wget https://repo.radeon.com/amdgpu-install/5.7.1/ubuntu/focal/amdgpu-install_5.7.50701-1_all.deb
   sudo dpkg -i ./amdgpu-install_5.7.50701-1_all.deb
   sudo apt install -f
   sudo amdgpu-install --usecase=rocm,hiplibsdk,dkms --rocmrelease=5.7.1
   ```
2. Check repository URL is correct for ROCm 5.7.1
3. Clear apt cache: `sudo apt clean && sudo apt update`
4. Check network connectivity to repo.radeon.com

### Issue 5: PyTorch not recognizing GPU

**Solutions**:
1. Verify ROCm installation: `/opt/gator/check_rocm.sh`
2. Check environment variables are set
3. Reinstall PyTorch with correct ROCm version
4. Verify HSA_OVERRIDE_GFX_VERSION is set

## Performance Optimization

### Multi-GPU Configuration

For systems with multiple MI25 GPUs:

```bash
# Distribute work across all GPUs
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# In Python
import torch
torch.cuda.device_count()  # Should show number of GPUs
```

### Memory Management

```bash
# Monitor GPU memory
watch -n 1 rocm-smi --showmeminfo --showuse

# Set memory allocation strategy
export HSA_ENABLE_SDMA=0  # Disable if having memory issues
```

### Performance Tuning

```bash
# Enable performance mode
echo performance | sudo tee /sys/class/drm/card*/device/power_dpm_force_performance_level

# Monitor GPU clocks
rocm-smi --showclocks
```

## Production Deployment

### Recommended Setup

1. **Hardware**: 4-8x MI25 GPUs per node
2. **RAM**: 128GB+ per node
3. **Storage**: NVMe SSD for models and temp files
4. **Network**: 10Gbps+ for multi-node deployments

### Load Balancing

Distribute inference workloads:
- Use separate GPUs for different model types
- Implement queue system for inference requests
- Monitor GPU utilization with `rocm-smi`

### Monitoring

```bash
# Real-time GPU monitoring
watch -n 1 rocm-smi

# Log GPU metrics
rocm-smi --showuse --csv > gpu_metrics.log

# Monitor system
htop
```

## Known Limitations

1. **gfx900 Support**: Some cutting-edge frameworks may have limited gfx900 support
2. **HSA Override Required**: HSA_OVERRIDE_GFX_VERSION=9.0.0 is essential for many libraries
3. **Model Size**: MI25 has 16GB memory per GPU - large models need quantization or multi-GPU
4. **Framework Compatibility**: Always check framework documentation for gfx900 support

## Additional Resources

- [AMD ROCm Documentation](https://rocmdocs.amd.com/)
- [ROCm GitHub](https://github.com/RadeonOpenCompute/ROCm)
- [PyTorch ROCm](https://pytorch.org/get-started/locally/)
- [Gator Platform Documentation](../README.md)

## Support

For MI25-specific issues:
1. Check this compatibility guide first
2. Verify all environment variables are set
3. Run `/opt/gator/check_rocm.sh` and share output
4. Check Gator logs: `journalctl -u gator -f`
5. Report issues with full system information

## Changelog

- **2024-01**: MI25 compatibility improvements for ROCm 5.7.1
  - Confirmed ROCm 5.7.1 works with MI25 on Ubuntu 20.04
  - Added gfx900 device ID detection
  - Added HSA_OVERRIDE_GFX_VERSION configuration for compatibility
  - Enhanced verification script for MI25 validation
  - Added comprehensive environment variable setup
