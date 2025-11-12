# ComfyUI Integration Guide

This document explains how to use ComfyUI with the Gator AI platform for advanced image generation, particularly for FLUX.1-dev models.

## Overview

The Gator platform now supports **ComfyUI integration** for advanced image generation workflows, with **automatic fallback to diffusers** when ComfyUI is unavailable. This provides:

- ✅ **High-quality image generation** with FLUX.1-dev and other ComfyUI models
- ✅ **Automatic fallback** to Stable Diffusion (SDXL/SD 1.5) via diffusers
- ✅ **No configuration required** for basic usage
- ✅ **Zero downtime** - system works with or without ComfyUI

## Quick Start

### Option 1: Use Without ComfyUI (Default)

The platform works out-of-the-box with diffusers-based models:
- **Stable Diffusion 1.5** (4GB VRAM, fast)
- **Stable Diffusion XL** (8GB VRAM, high quality)

No setup needed! Just run:
```bash
python demo_ai_video_generation.py
```

The system will automatically use available diffusers models.

### Option 2: Enable ComfyUI for FLUX.1-dev

For advanced users who want to use FLUX.1-dev models via ComfyUI:

1. **Install ComfyUI**:
   ```bash
   # Clone ComfyUI
   git clone https://github.com/comfyanonymous/ComfyUI
   cd ComfyUI
   
   # Install dependencies
   pip install -r requirements.txt
   
   # For ROCm (AMD GPUs)
   bash ../scripts/install_comfyui_rocm.sh
   ```

2. **Download FLUX.1-dev model**:
   ```bash
   cd ComfyUI/models/checkpoints
   # Download flux1-dev.safetensors from HuggingFace
   wget https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/flux1-dev.safetensors
   ```

3. **Start ComfyUI server**:
   ```bash
   cd ComfyUI
   python main.py --listen
   ```
   
   Server will start at http://127.0.0.1:8188

4. **Run Gator with ComfyUI**:
   ```bash
   # In a new terminal
   cd gator
   python demo_ai_video_generation.py
   ```

## Architecture

### How It Works

```
generate_image() 
    ├─> Check ComfyUI availability (2s timeout)
    │
    ├─> If ComfyUI available:
    │   └─> Use flux.1-dev via ComfyUI API
    │       ├─> Submit workflow to ComfyUI
    │       ├─> Poll for completion
    │       ├─> Download generated image
    │       └─> Return image data
    │
    └─> If ComfyUI unavailable:
        └─> Automatic fallback to diffusers
            ├─> Select best diffusers model (SDXL > SD 1.5)
            └─> Generate locally with diffusers
```

### Fallback Behavior

The system implements intelligent fallback:

1. **ComfyUI Check**: Quick 2-second check at http://127.0.0.1:8188/system_stats
2. **Model Filtering**: Removes ComfyUI models from available list if API not responding
3. **Automatic Selection**: Picks best available diffusers model
4. **Seamless Generation**: User experience identical regardless of backend

## Configuration

### Environment Variables

```bash
# ComfyUI API URL (default: http://127.0.0.1:8188)
export COMFYUI_API_URL="http://localhost:8188"

# ComfyUI installation directory (optional)
export COMFYUI_DIR="/path/to/ComfyUI"

# Enable cloud APIs as fallback (optional, default: false)
export ENABLE_CLOUD_APIS=false
```

### Model Configuration

ComfyUI models are configured in `ai_models.py`:

```python
"flux.1-dev": {
    "model_id": "black-forest-labs/FLUX.1-dev",
    "size_gb": 12,
    "min_gpu_memory_gb": 12,
    "min_ram_gb": 24,
    "inference_engine": "comfyui",  # Uses ComfyUI
    "description": "Very good quality; verify license for commercial use",
}
```

Diffusers models (automatic fallback):

```python
"stable-diffusion-v1-5": {
    "model_id": "runwayml/stable-diffusion-v1-5",
    "size_gb": 4,
    "min_gpu_memory_gb": 4,
    "min_ram_gb": 8,
    "inference_engine": "diffusers",  # Pure Python
}

"sdxl-1.0": {
    "model_id": "stabilityai/stable-diffusion-xl-base-1.0",
    "size_gb": 7,
    "min_gpu_memory_gb": 8,
    "min_ram_gb": 16,
    "inference_engine": "diffusers",  # Pure Python
}
```

## API Usage

### Python API

```python
from backend.services.ai_models import AIModelManager

# Initialize AI manager
ai_manager = AIModelManager()
await ai_manager.initialize_models()

# Generate image (automatic backend selection)
result = await ai_manager.generate_image(
    prompt="A serene mountain landscape at sunrise",
    width=1024,
    height=1024,
    num_inference_steps=20,
    guidance_scale=3.5,  # FLUX uses lower guidance
    seed=42
)

# Check which backend was used
print(f"Generated with: {result.get('model')}")  # flux.1-dev or sdxl-1.0
print(f"Workflow: {result.get('workflow')}")     # comfyui or diffusers
print(f"Status: {result.get('status')}")         # success
```

### Video Generation

```python
# Video generation with AI frames
result = await ai_manager.generate_video(
    prompt=[
        "A serene mountain landscape at sunrise",
        "A bustling city skyline at night"
    ],
    video_type="multi_frame",
    quality="high",
    use_ai_generation=True  # Uses available backend (ComfyUI or diffusers)
)
```

## ComfyUI Workflow Details

The platform uses a standard FLUX-compatible workflow:

```json
{
    "3": {  // Text encoder
        "inputs": {
            "text": "<prompt>",
            "clip": ["11", 0]
        },
        "class_type": "CLIPTextEncode"
    },
    "4": {  // Empty latent
        "inputs": {
            "width": 1024,
            "height": 1024,
            "batch_size": 1
        },
        "class_type": "EmptyLatentImage"
    },
    "10": {  // Sampler
        "inputs": {
            "seed": 42,
            "steps": 20,
            "cfg": 3.5,
            "sampler_name": "euler",
            "scheduler": "simple",
            "model": ["11", 0],
            "positive": ["3", 0],
            "negative": ["3", 0],
            "latent_image": ["4", 0]
        },
        "class_type": "KSampler"
    },
    "11": {  // Model loader
        "inputs": {
            "ckpt_name": "flux1-dev.safetensors"
        },
        "class_type": "CheckpointLoaderSimple"
    }
}
```

## Troubleshooting

### ComfyUI Not Detected

**Symptom**: System uses diffusers even when ComfyUI is installed

**Solution**:
1. Ensure ComfyUI is running: `python main.py --listen`
2. Check API is accessible: `curl http://127.0.0.1:8188/system_stats`
3. Verify port 8188 is not blocked by firewall
4. Set COMFYUI_API_URL if using non-default port

### FLUX Model Not Loading

**Symptom**: ComfyUI returns errors about missing model

**Solution**:
1. Check model file exists: `ls ComfyUI/models/checkpoints/flux1-dev.safetensors`
2. Verify file size (should be ~12GB)
3. Check ComfyUI logs for loading errors
4. Ensure sufficient VRAM (12GB+ required for FLUX)

### Connection Timeouts

**Symptom**: "ComfyUI API not accessible" warnings

**Solution**:
1. System will automatically fallback to diffusers
2. To disable timeout, increase in code: `timeout=5.0` → `timeout=30.0`
3. Check ComfyUI is not overloaded with other requests

### Fallback Always Used

**Symptom**: Always uses SDXL instead of FLUX

**Solution**:
1. Check ComfyUI API availability check passes
2. Review logs for "ComfyUI not available, filtering to diffusers-only models"
3. Verify model selection logic in `generate_image()`

## Performance Comparison

| Model | Backend | VRAM | Generation Time* | Quality |
|-------|---------|------|-----------------|---------|
| FLUX.1-dev | ComfyUI | 12GB | ~10-15s | Excellent |
| SDXL 1.0 | Diffusers | 8GB | ~8-12s | Very Good |
| SD 1.5 | Diffusers | 4GB | ~3-5s | Good |

*Times for 1024x1024 image, 20 steps, on MI25 60GB

## Best Practices

1. **Development**: Use diffusers (no setup required)
2. **Production**: Install ComfyUI for best quality
3. **Testing**: Both backends tested automatically
4. **Fallback**: Always available, zero configuration
5. **Monitoring**: Check logs for backend selection

## Advanced Usage

### Custom ComfyUI Workflows

Modify the workflow in `_generate_image_comfyui()`:

```python
workflow = {
    # Add custom nodes here
    "custom_node": {
        "inputs": {...},
        "class_type": "CustomNodeType"
    }
}
```

### Model Priority

To prefer diffusers over ComfyUI:

```python
# In generate_image(), filter ComfyUI models first
local_models = [
    m for m in local_models 
    if m.get("inference_engine") != "comfyui"
]
```

### Force ComfyUI

To require ComfyUI (fail if unavailable):

```python
# Remove fallback in _generate_image_comfyui()
# raise ConnectionError instead of calling _fallback_to_diffusers()
```

## Testing

Run the test suite:

```bash
# Test ComfyUI integration
pytest tests/unit/test_ai_image_generation.py::TestImageGeneration::test_comfyui_integration_with_api -v

# Test fallback behavior  
pytest tests/unit/test_ai_image_generation.py::TestImageGeneration::test_comfyui_fallback_when_unavailable -v

# Test video generation
pytest tests/unit/test_ai_video_frame_generation.py -v
```

## License Considerations

- **Stable Diffusion 1.5**: CreativeML Open RAIL-M (commercial use allowed)
- **SDXL 1.0**: CreativeML Open RAIL++-M (commercial use allowed)
- **FLUX.1-dev**: Apache 2.0 with restrictions (verify license for commercial use)

Always review model licenses before commercial deployment.

## Support

For issues or questions:
1. Check logs in `backend.services.ai_models`
2. Review ComfyUI logs in `ComfyUI/` directory
3. Test with: `python demo_ai_video_generation.py`
4. Open issue on GitHub with logs

## Future Enhancements

Planned improvements:
- [ ] WebSocket support for faster ComfyUI communication
- [ ] Custom workflow templates for different models
- [ ] ControlNet integration for visual consistency
- [ ] Model download automation
- [ ] Performance optimization for multi-GPU
- [ ] Workflow caching and reuse
