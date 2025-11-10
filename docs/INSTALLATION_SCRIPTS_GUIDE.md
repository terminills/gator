# Installation Scripts Guide

## Overview

Gator provides automated installation scripts for AI frameworks that require special handling on AMD ROCm systems. This guide covers the installation of vLLM and ComfyUI.

## Quick Start

### Prerequisites

1. **System Requirements**:
   - Ubuntu 20.04+ or Debian 11+
   - Python 3.9+
   - ROCm 5.7+ or 6.x+ (for GPU support)
   - 16GB+ RAM
   - 50GB+ free disk space

2. **Virtual Environment** (recommended):
   ```bash
   # Create and activate virtual environment
   python3 -m venv gator-venv
   source gator-venv/bin/activate
   ```

3. **ROCm Installation** (if using AMD GPU):
   ```bash
   # Check if ROCm is installed
   rocminfo
   
   # If not installed, follow AMD's guide:
   # https://rocm.docs.amd.com/en/latest/deploy/linux/quick_start.html
   ```

## Installing vLLM for AMD ROCm

vLLM is a high-performance inference engine for large language models. On AMD systems, it must be built from source.

### Installation

```bash
# Ensure you're in your virtual environment
source gator-venv/bin/activate

# Run the installation script
cd /path/to/gator
bash scripts/install_vllm_rocm.sh
```

The script will:
1. ✅ Verify virtual environment
2. ✅ Detect ROCm version
3. ✅ Check build dependencies
4. ✅ Install PyTorch with ROCm support
5. ✅ Clone vLLM repository
6. ✅ Build vLLM from source (10-30 minutes)
7. ✅ Install into active environment

### Custom Installation Directory

```bash
# Install vLLM to a specific directory
bash scripts/install_vllm_rocm.sh /path/to/custom/vllm-dir
```

### Verification

```python
# Test vLLM installation
python3 -c "import vllm; print(f'vLLM version: {vllm.__version__}')"

# Quick inference test
python3 << 'EOF'
from vllm import LLM, SamplingParams

# Initialize model (downloads if not cached)
llm = LLM(model='facebook/opt-125m')

# Generate text
prompts = ["Hello, my name is", "The capital of France is"]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=50)
outputs = llm.generate(prompts, sampling_params)

# Print outputs
for output in outputs:
    print(f"Prompt: {output.prompt}")
    print(f"Generated: {output.outputs[0].text}")
    print("-" * 50)
EOF
```

### Usage Example

```python
from vllm import LLM, SamplingParams

# Initialize LLM
llm = LLM(
    model="meta-llama/Llama-2-7b-chat-hf",  # Use your preferred model
    tensor_parallel_size=1,  # Set to number of GPUs for multi-GPU
    dtype="auto"
)

# Configure sampling
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=512
)

# Generate
prompts = ["Write a short poem about AI:"]
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(output.outputs[0].text)
```

## Installing ComfyUI with ROCm Support

ComfyUI is a powerful node-based UI for Stable Diffusion and other generative AI models.

### Installation

```bash
# Ensure you're in your virtual environment
source gator-venv/bin/activate

# Run the installation script
cd /path/to/gator
bash scripts/install_comfyui_rocm.sh
```

The script will:
1. ✅ Verify virtual environment
2. ✅ Detect ROCm version (falls back to CPU if not available)
3. ✅ Install PyTorch with ROCm support
4. ✅ Clone ComfyUI repository
5. ✅ Install all dependencies
6. ✅ Install ComfyUI Manager (extension manager)
7. ✅ Create launch script and helpers

### Custom Installation Directory

```bash
# Install ComfyUI to a specific directory
bash scripts/install_comfyui_rocm.sh /path/to/custom/ComfyUI
```

### Downloading Models

ComfyUI requires models to generate images. Use the included downloader:

```bash
cd ComfyUI
python3 download_models.py
```

This downloads:
- Stable Diffusion 1.5 base model (~4GB)
- VAE for improved quality (~350MB)

You can also manually download models to:
- `ComfyUI/models/checkpoints/` - Main models
- `ComfyUI/models/vae/` - VAE models
- `ComfyUI/models/loras/` - LoRA models
- `ComfyUI/models/controlnet/` - ControlNet models

### Launching ComfyUI

```bash
cd ComfyUI

# Method 1: Use the provided launch script
./launch_rocm.sh

# Method 2: Direct launch
python3 main.py

# Method 3: With options
python3 main.py --listen --port 8188

# Method 4: CPU mode (if no GPU)
python3 main.py --cpu
```

### Accessing the Web Interface

Once ComfyUI is running, open your browser to:
- **Local access**: http://localhost:8188
- **Network access**: http://your-ip:8188 (if using `--listen`)

### AMD GPU Configuration

For optimal performance on AMD GPUs, set the appropriate GFX version:

```bash
# For Vega 10 / MI25 (gfx900)
export HSA_OVERRIDE_GFX_VERSION=9.0.0

# For Vega 20 / MI50/60 (gfx906)
export HSA_OVERRIDE_GFX_VERSION=9.0.6

# For RDNA2 / RX 6000 / V620 (gfx1030)
export HSA_OVERRIDE_GFX_VERSION=10.3.0

# For RDNA3 / RX 7000 (gfx1100)
export HSA_OVERRIDE_GFX_VERSION=11.0.0

# Then launch ComfyUI
cd ComfyUI && ./launch_rocm.sh
```

## Integration with Gator Platform

Both vLLM and ComfyUI can be accessed through Gator's AI model setup interface:

### Web Interface

1. Start Gator server:
   ```bash
   cd src && python -m backend.api.main
   ```

2. Navigate to setup page:
   ```
   http://localhost:8000/ai_models_setup.html
   ```

3. Follow the guided installation process

### Python API

```python
from setup_ai_models import ModelSetupManager

# Initialize manager
manager = ModelSetupManager()

# Setup inference engines
await manager.setup_inference_engines()

# This will provide instructions for running the installation scripts
```

## Troubleshooting

### Common Issues

#### vLLM Build Errors

**Problem**: Out of memory during build
```bash
# Solution: Reduce parallel compilation
export MAX_JOBS=1
bash scripts/install_vllm_rocm.sh
```

**Problem**: ROCm not detected
```bash
# Solution: Set ROCM_HOME manually
export ROCM_HOME=/opt/rocm
export HIP_PATH=$ROCM_HOME
bash scripts/install_vllm_rocm.sh
```

**Problem**: Unsupported GPU architecture
```bash
# Solution: Set supported architectures
export PYTORCH_ROCM_ARCH="gfx900;gfx906;gfx908;gfx90a;gfx1030"
bash scripts/install_vllm_rocm.sh
```

#### ComfyUI Issues

**Problem**: GPU not detected
```python
# Check PyTorch GPU support
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device count: {torch.cuda.device_count()}")
```

**Problem**: Out of VRAM
```bash
# Solution 1: Use lowvram mode
python3 main.py --lowvram

# Solution 2: Use normalvram mode
python3 main.py --normalvram

# Solution 3: CPU mode
python3 main.py --cpu
```

**Problem**: Models not loading
```bash
# Verify model paths
ls -la ComfyUI/models/checkpoints/
ls -la ComfyUI/models/vae/

# Re-download if needed
cd ComfyUI && python3 download_models.py
```

### Performance Optimization

#### vLLM

1. **Multi-GPU setup**:
   ```python
   llm = LLM(
       model="your-model",
       tensor_parallel_size=2,  # Number of GPUs
   )
   ```

2. **Quantization** (lower memory):
   ```python
   llm = LLM(
       model="your-model",
       quantization="awq",  # or "gptq", "squeezellm"
   )
   ```

3. **Batch size tuning**:
   ```python
   llm = LLM(
       model="your-model",
       max_num_batched_tokens=8192,  # Adjust based on VRAM
       max_num_seqs=256,
   )
   ```

#### ComfyUI

1. **Enable xformers** (faster attention):
   - Automatically enabled if installed
   - Check in ComfyUI settings

2. **Use VAE tiling**:
   - For large images (4K+)
   - Reduces VRAM usage
   - Enable in workflow settings

3. **Use TAESD for previews**:
   - Much faster previews
   - Lower quality but faster iteration
   - Download to `models/vae_approx/`

## Advanced Configuration

### Environment Variables Reference

```bash
# ROCm Configuration
export ROCM_HOME=/opt/rocm
export HIP_PATH=$ROCM_HOME
export HSA_OVERRIDE_GFX_VERSION=<version>

# PyTorch Configuration
export PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:512

# vLLM Configuration
export VLLM_TARGET_DEVICE=rocm
export PYTORCH_ROCM_ARCH="gfx900;gfx906;gfx908;gfx90a;gfx1030;gfx1100"

# Build Configuration
export MAX_JOBS=4  # Number of parallel build jobs
export CMAKE_BUILD_TYPE=Release
```

### Custom PyTorch Build

If you need a specific PyTorch version:

```bash
# For ROCm 6.2
pip3 install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/rocm6.2

# For ROCm 5.7
pip3 install torch==2.3.1+rocm5.7 torchvision==0.18.1+rocm5.7 \
  --index-url https://download.pytorch.org/whl/rocm5.7
```

## Additional Resources

### Documentation

- **vLLM**: https://docs.vllm.ai/
- **vLLM GitHub**: https://github.com/vllm-project/vllm
- **ComfyUI GitHub**: https://github.com/comfyanonymous/ComfyUI
- **ComfyUI Wiki**: https://github.com/comfyanonymous/ComfyUI/wiki
- **ROCm Documentation**: https://rocm.docs.amd.com/

### Community

- **Gator Issues**: https://github.com/terminills/gator/issues
- **vLLM Discord**: Join vLLM community
- **ComfyUI Discord**: Active community support
- **ROCm Forums**: AMD developer forums

### Support

For issues specific to:
- **Installation scripts**: Open issue on Gator repository
- **vLLM functionality**: Check vLLM documentation/issues
- **ComfyUI functionality**: Check ComfyUI wiki/issues
- **ROCm issues**: Check AMD ROCm documentation

## See Also

- [Scripts README](../scripts/README.md) - Detailed script documentation
- [ROCm 6.5 Upgrade Summary](../ROCM_6.5_UPGRADE_SUMMARY.md) - ROCm upgrade guide
- [PyTorch Version Compatibility](../PYTORCH_VERSION_COMPATIBILITY.md) - PyTorch compatibility guide
- [Multi-GPU Enhancement](../MULTI_GPU_ENHANCEMENT.md) - Multi-GPU setup guide
