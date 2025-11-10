# Gator AI Installation Scripts

This directory contains installation scripts for various AI frameworks and tools used by the Gator platform.

## Available Scripts

### 1. install_vllm_rocm.sh

Builds and installs vLLM (Very Large Language Model inference engine) for AMD ROCm systems.

**Purpose**: vLLM is a high-performance inference engine for large language models. AMD systems require building from source as there are no pre-built wheels available.

**Requirements**:
- ROCm 5.7+ or 6.x+
- Python 3.9+
- Build tools (gcc, g++, cmake, ninja)
- 16GB+ RAM for building
- AMD GPU with ROCm support

**Usage**:
```bash
# Activate your virtual environment first
source venv/bin/activate  # or conda activate your_env

# Run installation script
bash scripts/install_vllm_rocm.sh [optional-install-dir]

# Default install directory is ./vllm-rocm
# Example with custom directory:
bash scripts/install_vllm_rocm.sh /path/to/vllm
```

**What it does**:
1. Checks for virtual environment activation
2. Detects ROCm version
3. Verifies build dependencies
4. Installs/upgrades PyTorch with ROCm support
5. Clones vLLM repository
6. Builds vLLM from source with ROCm optimizations
7. Installs vLLM into current Python environment

**Build time**: 10-30 minutes depending on system specs

**Verification**:
```python
# Test vLLM installation
python3 -c "import vllm; print(vllm.__version__)"

# Quick inference test
python3 << EOF
from vllm import LLM, SamplingParams
llm = LLM(model='facebook/opt-125m')
outputs = llm.generate('Hello, my name is', SamplingParams(max_tokens=50))
print(outputs[0].outputs[0].text)
EOF
```

**Reference**: https://docs.vllm.ai/en/stable/getting_started/installation/gpu.html#unsupported-os-build

---

### 2. install_comfyui_rocm.sh

Installs ComfyUI node-based UI for Stable Diffusion with AMD ROCm support.

**Purpose**: ComfyUI is a powerful and modular GUI for Stable Diffusion and other generative AI models. Works with both ROCm and CUDA.

**Requirements**:
- Python 3.9+
- ROCm 5.7+ or 6.x+ (optional, CPU mode available)
- 8GB+ RAM
- AMD/NVIDIA GPU recommended (optional)
- ~10GB disk space for ComfyUI and models

**Usage**:
```bash
# Activate your virtual environment first
source venv/bin/activate  # or conda activate your_env

# Run installation script
bash scripts/install_comfyui_rocm.sh [optional-install-dir]

# Default install directory is ./ComfyUI
# Example with custom directory:
bash scripts/install_comfyui_rocm.sh /path/to/ComfyUI
```

**What it does**:
1. Checks for virtual environment activation
2. Detects ROCm version (or falls back to CPU mode)
3. Installs/upgrades PyTorch with ROCm support
4. Clones ComfyUI repository
5. Installs ComfyUI dependencies
6. Installs ComfyUI Manager (extension manager)
7. Creates launch script and model downloader

**Installation time**: 5-10 minutes (excluding model downloads)

**Post-Installation**:

1. **Download models** (optional but recommended):
   ```bash
   cd ComfyUI
   python3 download_models.py
   ```
   This downloads Stable Diffusion 1.5 base model (~4GB) and VAE (~350MB).

2. **Launch ComfyUI**:
   ```bash
   cd ComfyUI
   ./launch_rocm.sh
   # or directly:
   python3 main.py
   ```

3. **Access web interface**:
   - Open browser to: http://localhost:8188
   - Default port is 8188
   - Use `--listen` flag to allow external access
   - Use `--port` to change port

**AMD GPU Configuration**:

For AMD GPUs, you may need to set the GFX version override:

```bash
# For gfx900 (Vega 10, MI25):
export HSA_OVERRIDE_GFX_VERSION=9.0.0

# For gfx906 (Vega 20, MI50/60):
export HSA_OVERRIDE_GFX_VERSION=9.0.6

# For gfx1030 (RDNA2, RX 6000 series, V620):
export HSA_OVERRIDE_GFX_VERSION=10.3.0

# For gfx1100 (RDNA3, RX 7000 series):
export HSA_OVERRIDE_GFX_VERSION=11.0.0

# Then launch:
cd ComfyUI && ./launch_rocm.sh
```

**Verification**:
```bash
# Check if ComfyUI is running
curl http://localhost:8188

# Check GPU detection in logs
cd ComfyUI && python3 main.py | grep -i "device\|gpu\|cuda"
```

---

## General Tips

### Virtual Environment Management

Always use a virtual environment to avoid conflicts:

```bash
# Create virtual environment
python3 -m venv gator-venv

# Activate (Linux/Mac)
source gator-venv/bin/activate

# Activate (Windows)
gator-venv\Scripts\activate

# Verify activation
which python3  # Should show path in venv
```

### ROCm Environment Variables

Common ROCm environment variables you may need:

```bash
# Set ROCm installation path (if non-standard)
export ROCM_HOME=/opt/rocm
export HIP_PATH=$ROCM_HOME

# GPU architecture override (see AMD GPU Configuration above)
export HSA_OVERRIDE_GFX_VERSION=<version>

# Memory management
export PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:512

# Enable debug output
export ROCM_DEBUG_ENABLE=1
```

### Troubleshooting

**Build fails with "out of memory"**:
- Increase system swap space
- Close other applications
- Use `--jobs 1` to reduce parallel compilation

**PyTorch not detecting GPU**:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"Device name: {torch.cuda.get_device_name(0)}")
```

**vLLM build errors**:
- Ensure ROCm is properly installed: `rocminfo`
- Check PyTorch version: `python3 -c "import torch; print(torch.__version__)"`
- Verify ROCM_HOME: `echo $ROCM_HOME`

**ComfyUI not starting**:
- Check dependencies: `cd ComfyUI && pip install -r requirements.txt`
- Try CPU mode: `python3 main.py --cpu`
- Check logs for detailed error messages

### Performance Optimization

**vLLM**:
- Use quantized models (Q4, Q8) for lower memory usage
- Enable tensor parallelism for multi-GPU: `--tensor-parallel-size N`
- Tune `--max-num-batched-tokens` and `--max-num-seqs`

**ComfyUI**:
- Use lower precision: `--lowvram` or `--normalvram` flags
- Enable attention optimizations in settings
- Use VAE tiling for large images
- Consider using TAESD for previews (faster, lower quality)

---

## Integration with Gator Platform

These scripts are integrated with the main Gator setup:

```python
# From Python code
from setup_ai_models import ModelSetupManager

manager = ModelSetupManager()
await manager.setup_inference_engines()
```

Or use the web-based setup wizard:
1. Navigate to http://localhost:8000/ai_models_setup.html
2. Follow the guided installation process
3. Scripts will be automatically invoked as needed

---

## Additional Resources

- **vLLM Documentation**: https://docs.vllm.ai/
- **vLLM GitHub**: https://github.com/vllm-project/vllm
- **ComfyUI GitHub**: https://github.com/comfyanonymous/ComfyUI
- **ComfyUI Wiki**: https://github.com/comfyanonymous/ComfyUI/wiki
- **ROCm Documentation**: https://rocm.docs.amd.com/
- **Gator Documentation**: https://github.com/terminills/gator

---

## Support

For issues related to:
- **Installation scripts**: Open an issue on the Gator repository
- **vLLM**: Check vLLM documentation or GitHub issues
- **ComfyUI**: Check ComfyUI GitHub issues and wiki
- **ROCm**: Check AMD ROCm documentation and forums

## License

These scripts are part of the Gator project and are licensed under the MIT License.
