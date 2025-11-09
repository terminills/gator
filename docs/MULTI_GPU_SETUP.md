# Multi-GPU Setup Guide for Gator AI Platform

## Overview

This guide covers setting up Gator with multiple AMD GPUs, specifically optimized for modern RDNA2/RDNA3 cards like the **Radeon Pro V620** and future CDNA architectures.

## Supported Configurations

### Radeon Pro V620 (RDNA2 - gfx1030)
- **Architecture**: RDNA2 (gfx1030)
- **Memory**: 32GB GDDR6 per card
- **Optimal ROCm**: 6.5+ (standard wheels available)
- **Recommended for**: Production multi-tenant AI workloads
- **Multi-GPU Support**: Excellent (2-8 GPUs)

### Scaling Recommendations

| GPU Count | Configuration | Use Case |
|-----------|--------------|----------|
| 1 GPU | Single instance | Development, small-scale |
| 2 GPUs | Dual parallel | Production, load balancing |
| 3 GPUs | Task-specialized | Multi-model serving (LLM + Image + Video) |
| 4+ GPUs | Distributed | Enterprise, multi-tenant, high-throughput |

## Installation for Multi-GPU Systems

### 1. Install ROCm 6.5+

For Ubuntu 22.04/24.04:

```bash
# Add ROCm repository
wget https://repo.radeon.com/amdgpu-install/latest/ubuntu/jammy/amdgpu-install_latest_all.deb
sudo dpkg -i amdgpu-install_latest_all.deb
sudo amdgpu-install -y --usecase=rocm

# Verify installation
rocm-smi
```

Expected output for V620 pair:
```
GPU[0]     : GPU ID: 0x73bf
GPU[0]     : GPU use (%): 0
GPU[0]     : Temperature: 35.0°C
GPU[0]     : Memory Total: 32768 MB

GPU[1]     : GPU ID: 0x73bf
GPU[1]     : GPU use (%): 0
GPU[1]     : Temperature: 35.0°C
GPU[1]     : Memory Total: 32768 MB
```

### 2. Install PyTorch with ROCm 6.5+

```bash
# For ROCm 6.5 stable wheels
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.5

# For ROCm 6.5 nightly builds (latest features)
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm6.5

# For ROCm 6.6+ (when available)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.6
```

### 3. Verify Multi-GPU Setup

```bash
# Check ROCm detection
python -m backend.utils.rocm_utils

# Expected output:
# ROCm Detection & Multi-GPU Configuration Utility
# ======================================================================
# ✓ ROCm detected: 6.5.0
#   Version: 6.5.0
#   ROCm 6.5+: True
#
# Current PyTorch Installation:
#   ✓ Installed: 2.4.0+rocm6.5
#   ROCm build: True
#   ROCm version: 6.5.0
#   GPU available: True
#   GPU count: 2
#
#   GPU Devices:
#     [0] AMD Radeon PRO V620 MxGPU
#         Architecture: gfx1030
#         Memory: 32.00 GB
#     [1] AMD Radeon PRO V620 MxGPU
#         Architecture: gfx1030
#         Memory: 32.00 GB
#
#   Total GPU Memory: 64.00 GB
#   Architectures: gfx1030
#   ✓ Multi-GPU setup detected!
```

### 4. Configure Environment Variables

Create or update `.env` file with multi-GPU settings:

```bash
# ROCm Configuration
ROCM_PATH=/opt/rocm
HIP_PATH=/opt/rocm/hip

# Multi-GPU Settings (for 2 GPUs, adjust for 3+)
HIP_VISIBLE_DEVICES=0,1
ROCR_VISIBLE_DEVICES=0,1

# V620/RDNA2 Specific Optimizations
HSA_OVERRIDE_GFX_VERSION=10.3.0
PYTORCH_ROCM_ARCH=gfx1030
GPU_MAX_ALLOC_PERCENT=100
GPU_MAX_HEAP_SIZE=100

# Multi-GPU Communication
NCCL_DEBUG=INFO
NCCL_IB_DISABLE=1
HIP_FORCE_DEV_KERNARG=1
HSA_ENABLE_SDMA=0
```

For 3 GPUs:
```bash
HIP_VISIBLE_DEVICES=0,1,2
ROCR_VISIBLE_DEVICES=0,1,2
```

## Multi-GPU Strategies

### Strategy 1: Data Parallelism (Recommended for Image Generation)

Replicate the model across GPUs and split batches:

```python
# Automatic with PyTorch DataParallel
import torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")
pipe = torch.nn.DataParallel(pipe)
pipe.to("cuda")

# Generates batch across both GPUs automatically
images = pipe(["prompt1", "prompt2", "prompt3", "prompt4"])
```

### Strategy 2: Pipeline Parallelism (For Large Models)

Split model layers across GPUs:

```python
# For large language models
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-70B-Instruct",
    device_map="auto"  # Automatically splits across GPUs
)
```

### Strategy 3: Task Parallelism (Best for Multi-Persona)

Assign different tasks to different GPUs:

```python
# GPU 0: Text generation
text_model = load_text_model(device="cuda:0")

# GPU 1: Image generation  
image_model = load_image_model(device="cuda:1")

# GPU 2: Video processing (when you add 3rd GPU)
video_model = load_video_model(device="cuda:2")
```

## Performance Optimization

### 3-GPU V620 Configuration (Your Future Setup)

**Recommended Assignment:**

```yaml
GPU 0 (32GB):
  - Primary LLM inference (Llama 3.1 70B with quantization)
  - Text generation for all personas
  - Response time: <100ms

GPU 1 (32GB):
  - Image generation (SDXL/FLUX)
  - Avatar rendering
  - Response time: 2-5s per image

GPU 2 (32GB):
  - Video generation/processing
  - Audio processing (TTS/voice cloning)
  - Background tasks (training, fine-tuning)
```

**Total Capacity:**
- 96GB VRAM
- Can handle 10+ simultaneous personas
- Can process 100+ requests/minute
- Suitable for production deployment

### Load Balancing

Use round-robin or weighted load balancing:

```python
# Gator auto-detects and balances across GPUs
from backend.services.gpu_manager import GPUManager

gpu_manager = GPUManager()
# Automatically assigns tasks to least-loaded GPU
device = gpu_manager.get_next_available_device()
```

## Monitoring Multi-GPU Setup

### Real-time Monitoring

```bash
# Watch all GPUs
watch -n 1 rocm-smi

# Detailed stats
rocm-smi --showuse --showmeminfo --showtemp

# Per-GPU utilization
rocm-smi -d 0  # GPU 0
rocm-smi -d 1  # GPU 1
rocm-smi -d 2  # GPU 2 (when added)
```

### Programmatic Monitoring

```python
import torch

for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    memory_allocated = torch.cuda.memory_allocated(i) / 1024**3
    memory_reserved = torch.cuda.memory_reserved(i) / 1024**3
    
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    print(f"  Memory Allocated: {memory_allocated:.2f} GB")
    print(f"  Memory Reserved: {memory_reserved:.2f} GB")
    print(f"  Total Memory: {props.total_memory / 1024**3:.2f} GB")
```

## Troubleshooting

### Issue: GPU not detected

```bash
# Check GPU visibility
echo $HIP_VISIBLE_DEVICES
echo $ROCR_VISIBLE_DEVICES

# List all GPUs
rocm-smi --showid

# Reset GPU
sudo systemctl restart gpu-manager
```

### Issue: Out of memory on multi-GPU

```python
# Clear cache on all GPUs
import torch
for i in range(torch.cuda.device_count()):
    with torch.cuda.device(i):
        torch.cuda.empty_cache()
```

### Issue: Uneven GPU utilization

1. Check if models are properly distributed
2. Verify environment variables are set
3. Use explicit device assignment
4. Monitor with `rocm-smi` to identify bottlenecks

## Future-Proofing for 4+ GPUs

When scaling beyond 3 GPUs:

1. **Network Topology**: Ensure GPUs are on same PCIe root complex for best peer-to-peer
2. **Cooling**: Monitor temperatures, consider additional cooling
3. **Power**: V620 is 225W TDP per card, plan power accordingly
4. **Software**: Consider vLLM or Ray for distributed inference
5. **Monitoring**: Implement Prometheus + Grafana for production monitoring

## ROCm 6.5+ Benefits for V620

- **Standard Wheels**: No custom builds needed
- **Nightly Builds**: Access to latest features
- **Better RDNA2 Support**: Optimized kernels for gfx1030
- **Faster Updates**: Regular PyTorch releases
- **Community Support**: Broader ecosystem compatibility

## Recommended Workflows

### Development (2 GPUs)
```
GPU 0: Active development and testing
GPU 1: Production serving
```

### Production (3 GPUs)
```
GPU 0: LLM inference (primary)
GPU 1: Image/Video generation
GPU 2: Backup/overflow + background tasks
```

### Enterprise (4+ GPUs)
```
GPU 0-1: LLM inference (distributed)
GPU 2: Image generation
GPU 3: Video + Audio processing
GPU 4+: Scaling and redundancy
```

## Additional Resources

- [ROCm Documentation](https://rocm.docs.amd.com/)
- [PyTorch ROCm Support](https://pytorch.org/get-started/locally/)
- [Radeon Pro V620 Specs](https://www.amd.com/en/products/server-accelerators/amd-radeon-pro-v620)
- [Multi-GPU Training Guide](https://pytorch.org/tutorials/beginner/dist_overview.html)

## Support

For issues specific to multi-GPU V620 setups, check:
1. Gator platform logs: `/opt/gator/logs/`
2. ROCm logs: `/var/log/rocm/`
3. PyTorch GPU detection: Run `python -m backend.utils.rocm_utils`
