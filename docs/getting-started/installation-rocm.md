# ROCm 6.5+ Installation Guide for Gator AI Platform

## Overview

This guide covers installing Gator with ROCm 6.5+ support for modern AMD GPUs, including automatic detection and configuration for single and multi-GPU setups.

## Supported Hardware

### GPU Support Matrix

| GPU Model | Architecture | ROCm Version | Multi-GPU | Status |
|-----------|--------------|--------------|-----------|--------|
| **Radeon Pro V620** | RDNA2 (gfx1030) | 6.5+ | 2-8 GPUs | ✅ Recommended |
| **RX 7900 XTX/XT** | RDNA3 (gfx1100) | 6.5+ | 2-4 GPUs | ✅ Supported |
| **RX 6900/6800 XT** | RDNA2 (gfx1030) | 6.5+ | 2-4 GPUs | ✅ Supported |
| **MI210/MI250** | CDNA2 (gfx90a) | 6.5+ | 2-8 GPUs | ✅ Supported |
| **MI25** | Vega (gfx900) | 5.7 | 1-5 GPUs | ✅ Legacy Support |

## Installation Methods

### Method 1: Automated Installation (Recommended)

The automated installer detects your ROCm version and configures PyTorch accordingly.

```bash
# Clone repository
git clone https://github.com/terminills/gator.git
cd gator

# Run automated setup with ROCm detection
sudo ./server-setup.sh --rocm --domain your-domain.com --email admin@your-domain.com
```

**What the installer does:**
1. Detects installed ROCm version from `/opt/rocm/.info/version`
2. Determines appropriate PyTorch index URL based on ROCm version
3. Installs PyTorch with correct ROCm support (5.7, 6.4, or 6.5+)
4. Configures multi-GPU environment variables automatically
5. Sets up optimized GPU settings for detected architecture

### Method 2: Manual Installation

#### Step 1: Install ROCm 6.5+

For Ubuntu 22.04/24.04:

```bash
# Add AMD repository
wget https://repo.radeon.com/amdgpu-install/latest/ubuntu/jammy/amdgpu-install_latest_all.deb
sudo dpkg -i amdgpu-install_latest_all.deb

# Install ROCm (choose version based on your GPU)
# For V620, RX 7900, and other modern GPUs
sudo amdgpu-install -y --usecase=rocm --rocmrelease=6.5.0

# Verify installation
rocm-smi
rocminfo
```

#### Step 2: Install PyTorch with ROCm 6.5+

```bash
# Standard wheels for ROCm 6.5
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.5

# OR for nightly builds (latest features)
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm6.5

# Verify PyTorch GPU detection
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'ROCm: {torch.version.hip if hasattr(torch.version, \"hip\") else \"N/A\"}'); print(f'GPU Available: {torch.cuda.is_available()}'); print(f'GPU Count: {torch.cuda.device_count()}')"
```

#### Step 3: Install Gator Platform

```bash
# Install dependencies
cd gator
pip install -e .

# Or for ROCm 6.5+ specific installation
pip install -e .[rocm65] --index-url https://download.pytorch.org/whl/rocm6.5

# Initialize database
python setup_db.py

# Verify installation
python demo.py
```

#### Step 4: Configure Environment

Copy and edit the environment file:

```bash
cp .env.template .env
```

Edit `.env` with your GPU configuration:

```bash
# GPU Configuration for 2x V620
HIP_VISIBLE_DEVICES=0,1
ROCR_VISIBLE_DEVICES=0,1
HSA_OVERRIDE_GFX_VERSION=10.3.0
PYTORCH_ROCM_ARCH=gfx1030
GPU_MAX_ALLOC_PERCENT=100
GPU_MAX_HEAP_SIZE=100
HSA_ENABLE_SDMA=0
NCCL_IB_DISABLE=1
HIP_FORCE_DEV_KERNARG=1

# Add GPU 2 when available
# HIP_VISIBLE_DEVICES=0,1,2
# ROCR_VISIBLE_DEVICES=0,1,2
```

#### Step 5: Start the Platform

```bash
cd src
python -m backend.api.main
```

Access the dashboard at `http://localhost:8000`

## Verification Steps

### 1. Check ROCm Installation

```bash
# ROCm version
cat /opt/rocm/.info/version

# GPU detection
rocm-smi

# Expected output for 2x V620:
# GPU[0]     : GPU ID: 0x73bf
# GPU[0]     : Temperature: 35.0°C
# GPU[0]     : Memory Total: 32768 MB
# 
# GPU[1]     : GPU ID: 0x73bf
# GPU[1]     : Temperature: 35.0°C
# GPU[1]     : Memory Total: 32768 MB
```

### 2. Verify PyTorch Installation

```bash
python3 << EOF
import torch
print(f"PyTorch Version: {torch.__version__}")
print(f"ROCm Build: {hasattr(torch.version, 'hip')}")
if hasattr(torch.version, 'hip'):
    print(f"ROCm Version: {torch.version.hip}")
print(f"GPU Available: {torch.cuda.is_available()}")
print(f"GPU Count: {torch.cuda.device_count()}")

for i in range(torch.cuda.device_count()):
    print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
    props = torch.cuda.get_device_properties(i)
    print(f"  Memory: {props.total_memory / 1024**3:.2f} GB")
EOF
```

Expected output:
```
PyTorch Version: 2.4.0+rocm6.5
ROCm Build: True
ROCm Version: 6.5.0
GPU Available: True
GPU Count: 2

GPU 0: AMD Radeon PRO V620 MxGPU
  Memory: 32.00 GB

GPU 1: AMD Radeon PRO V620 MxGPU
  Memory: 32.00 GB
```

### 3. Test Gator Multi-GPU Detection

```bash
cd /home/runner/work/gator/gator
PYTHONPATH=/home/runner/work/gator/gator/src python src/backend/utils/rocm_utils.py
```

Expected output should show:
- ✓ ROCm detected: 6.5.0
- ✓ Multi-GPU setup detected!
- GPU architecture details
- Recommended environment variables
- Multi-GPU configuration strategies

### 4. Run Platform Demo

```bash
python demo.py
```

Should complete without errors and display persona operations.

## Multi-GPU Configuration

### 2 GPU Setup (Current)

```bash
# .env configuration
HIP_VISIBLE_DEVICES=0,1
ROCR_VISIBLE_DEVICES=0,1
```

**Recommended Usage:**
- GPU 0: Primary inference (50% workload)
- GPU 1: Secondary inference (50% workload)
- Load balancing: Round-robin or least-loaded

### 3 GPU Setup (Future)

```bash
# .env configuration
HIP_VISIBLE_DEVICES=0,1,2
ROCR_VISIBLE_DEVICES=0,1,2
```

**Recommended Usage:**
- GPU 0: Text generation (LLMs)
- GPU 1: Image generation (SDXL, FLUX)
- GPU 2: Video/Audio processing

### 4+ GPU Setup (Enterprise)

```bash
# .env configuration
HIP_VISIBLE_DEVICES=0,1,2,3
ROCR_VISIBLE_DEVICES=0,1,2,3
```

**Recommended Usage:**
- GPU 0-1: Distributed LLM inference
- GPU 2: Image generation
- GPU 3: Video + backup capacity

## Performance Optimization

### V620-Specific Optimizations

```bash
# Add to .env for optimal V620 performance
HSA_OVERRIDE_GFX_VERSION=10.3.0
PYTORCH_ROCM_ARCH=gfx1030
GPU_MAX_ALLOC_PERCENT=100
GPU_MAX_HEAP_SIZE=100
HSA_ENABLE_SDMA=0
HIP_FORCE_DEV_KERNARG=1

# Multi-GPU communication
NCCL_DEBUG=INFO
NCCL_IB_DISABLE=1
```

### Monitoring GPU Usage

```bash
# Real-time monitoring (updates every 1 second)
watch -n 1 rocm-smi

# Detailed stats
rocm-smi --showuse --showmeminfo --showtemp

# Per-GPU stats
rocm-smi -d 0  # GPU 0
rocm-smi -d 1  # GPU 1
rocm-smi -d 2  # GPU 2 (when added)
```

## Troubleshooting

### Issue: ROCm not detected

```bash
# Check if ROCm is installed
ls -la /opt/rocm

# Verify kernel module
lsmod | grep amdgpu

# Check GPU visibility
rocm-smi --showid

# Reinstall ROCm if needed
sudo amdgpu-install -y --usecase=rocm --rocmrelease=6.5.0
```

### Issue: PyTorch not detecting GPUs

```bash
# Verify ROCm environment
echo $ROCM_PATH
echo $HIP_PATH
echo $HIP_VISIBLE_DEVICES

# Reinstall PyTorch
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.5

# Test detection
python3 -c "import torch; print(torch.cuda.is_available())"
```

### Issue: Only detecting 1 GPU instead of 2+

```bash
# Check GPU visibility settings
echo $HIP_VISIBLE_DEVICES
echo $ROCR_VISIBLE_DEVICES

# List all GPUs
rocm-smi --showid

# Set environment manually
export HIP_VISIBLE_DEVICES=0,1
export ROCR_VISIBLE_DEVICES=0,1

# Test in Python
python3 -c "import torch; print(f'GPU Count: {torch.cuda.device_count()}')"
```

### Issue: Out of memory errors

```bash
# Clear GPU cache
python3 << EOF
import torch
for i in range(torch.cuda.device_count()):
    with torch.cuda.device(i):
        torch.cuda.empty_cache()
print("Cache cleared for all GPUs")
EOF

# Check current memory usage
rocm-smi --showmeminfo

# Reduce batch size in .env
MAX_CONTENT_GENERATION_CONCURRENT=2  # Default is 4
```

## Upgrading from ROCm 5.7 to 6.5+

If you have an existing Gator installation with ROCm 5.7:

```bash
# 1. Backup your data
sudo systemctl stop gator
cp -r /opt/gator/data /opt/gator/data.backup

# 2. Uninstall old ROCm
sudo amdgpu-uninstall

# 3. Install ROCm 6.5
sudo amdgpu-install -y --usecase=rocm --rocmrelease=6.5.0

# 4. Reinstall PyTorch
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.5

# 5. Update Gator
cd /opt/gator/app
git pull
./update.sh

# 6. Update environment variables
# Edit /opt/gator/app/.env with new GPU settings

# 7. Restart service
sudo systemctl start gator
```

## Additional Resources

- [ROCm Official Documentation](https://rocm.docs.amd.com/)
- [PyTorch ROCm Installation](https://pytorch.org/get-started/locally/)
- [Multi-GPU Setup Guide](./MULTI_GPU_SETUP.md)
- [V620 Product Page](https://www.amd.com/en/products/server-accelerators/amd-radeon-pro-v620)
- [Gator GitHub Repository](https://github.com/terminills/gator)

## Support

For ROCm 6.5+ specific issues:
1. Check logs: `journalctl -u gator -f`
2. Run diagnostics: `python -m backend.utils.rocm_utils`
3. Review GPU status: `rocm-smi --showuse`
4. Open issue on GitHub with diagnostic output

## Version Compatibility

| Gator Version | ROCm Version | PyTorch Version | Notes |
|---------------|--------------|-----------------|-------|
| 0.1.0+ | 6.5+ | 2.4.0+ | Recommended for V620 and modern GPUs |
| 0.1.0+ | 6.4 | 2.3.1+ | Supported |
| 0.1.0+ | 5.7 | 2.3.1 | Legacy support for MI25 |
| < 0.1.0 | 5.7 | 2.3.1 | Legacy only |
