# Local Image Generation Guide

This guide explains how to use the local image generation feature in Gator AI Platform.

## Overview

Gator now supports local image generation using Stable Diffusion models via the `diffusers` library. This provides privacy-focused, cost-effective image generation without relying on external APIs.

## Features

- 🎨 **Local Image Generation** - Generate images entirely on your hardware
- 🔒 **Privacy First** - No data sent to external services
- 💰 **Cost Effective** - No per-image API costs
- ⚡ **Model Caching** - Models loaded once and cached for fast subsequent generations
- 🎯 **Full Control** - Customize generation parameters (size, steps, guidance, seed)
- 🚀 **Optimized** - Automatic memory optimizations for GPU efficiency

## Requirements

### Hardware Requirements

**Minimum Requirements (stable-diffusion-v1-5):**
- GPU: 4 GB VRAM (CUDA or ROCm supported)
- RAM: 8 GB
- Storage: ~4 GB for model

**Recommended Requirements (sdxl-1.0):**
- GPU: 8 GB VRAM
- RAM: 16 GB  
- Storage: ~7 GB for model

**CPU Mode (Slower):**
- RAM: 8 GB minimum
- Storage: ~4 GB for model
- Note: Generation will be significantly slower

### Software Requirements

- Python 3.9+
- PyTorch 2.0+ (automatically installed)
- diffusers library (automatically installed)
- transformers, accelerate (automatically installed)

## Installation

The required dependencies are installed automatically when you install Gator:

```bash
pip install -e .
```

## Usage

### Basic Image Generation

```python
import asyncio
from backend.services.ai_models import AIModelManager

async def generate_image():
    # Initialize the AI Model Manager
    manager = AIModelManager()
    await manager.initialize_models()
    
    # Generate an image
    result = await manager.generate_image(
        "A serene mountain landscape at sunset, digital art"
    )
    
    # Save the image
    with open("output.png", "wb") as f:
        f.write(result["image_data"])
    
    print(f"Generated {len(result['image_data'])} bytes")
    print(f"Model used: {result['model']}")

asyncio.run(generate_image())
```

### Advanced Parameters

```python
result = await manager.generate_image(
    prompt="A cyberpunk city street at night, neon lights, rain",
    
    # Image dimensions
    width=768,
    height=512,
    
    # Quality settings
    num_inference_steps=30,  # More steps = better quality (default: 25)
    guidance_scale=8.0,      # Higher = closer to prompt (default: 7.5)
    
    # Consistency
    seed=42,                 # Set seed for reproducible results
    
    # Negative prompt to avoid unwanted elements
    negative_prompt="blurry, low quality, distorted, ugly"
)
```

### Parameter Guide

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | str | Required | Text description of desired image |
| `width` | int | 512 | Image width in pixels |
| `height` | int | 512 | Image height in pixels |
| `num_inference_steps` | int | 25 | Number of denoising steps (higher = better quality but slower) |
| `guidance_scale` | float | 7.5 | How closely to follow the prompt (1-20, typically 7-9) |
| `seed` | int | None | Random seed for reproducibility |
| `negative_prompt` | str | "ugly, blurry..." | What to avoid in the image |

## Model Selection

Gator automatically selects the best available model based on your hardware:

### Available Models

**stable-diffusion-v1-5** (Recommended for most users)
- Model ID: `runwayml/stable-diffusion-v1-5`
- Size: ~4 GB
- Requirements: 4 GB VRAM, 8 GB RAM
- Speed: Fast
- Quality: Good
- Best for: General use, quick iterations

**sdxl-1.0** (Higher quality)
- Model ID: `stabilityai/stable-diffusion-xl-base-1.0`
- Size: ~7 GB
- Requirements: 8 GB VRAM, 16 GB RAM
- Speed: Slower
- Quality: Excellent
- Best for: High-quality final images

## First Run

On first run, the model will be downloaded from HuggingFace Hub:

```
🎨 Testing Local Image Generation
============================================================

1. Initializing AI Model Manager...
   - GPU Type: cuda
   - GPU Memory: 8.0 GB
   
2. Initializing models...
   Loading model from HuggingFace Hub: runwayml/stable-diffusion-v1-5
   Downloading... (this may take several minutes)
   Model saved to: ./models/image/stable-diffusion-v1-5
   
3. Generating image...
   ✅ Image generated successfully!
```

Subsequent runs will use the cached model and be much faster.

## Troubleshooting

### "No image generation models available"

This means your system doesn't meet the minimum hardware requirements. Options:

1. **Use API-based generation**: Set `OPENAI_API_KEY` to use DALL-E 3
2. **Upgrade hardware**: Add more GPU memory
3. **Use CPU mode**: Will work but be very slow

### "CUDA out of memory"

Your GPU doesn't have enough memory. Try:

1. Reduce image size: `width=512, height=512`
2. Lower inference steps: `num_inference_steps=20`
3. Use CPU mode (automatic fallback)
4. Close other GPU-intensive applications

### Slow generation on CPU

CPU generation is 10-50x slower than GPU. Consider:

1. Using smaller images (512x512 or less)
2. Fewer inference steps (15-20)
3. Using API-based generation for production
4. Upgrading to a system with GPU

### Model download fails

If downloading fails due to network issues:

1. Check internet connection
2. Try again (downloads resume automatically)
3. Manually download from HuggingFace Hub
4. Use a HuggingFace token for private models

## Performance Tips

### Speed Optimization
- Use `num_inference_steps=20-25` for good quality/speed balance
- Generate smaller images (512x512) for faster results
- Batch multiple generations when possible
- Keep the model loaded (pipeline caching handles this)

### Quality Optimization
- Use `num_inference_steps=30-50` for best quality
- Set `guidance_scale=7.5-9.0` for balanced results
- Use descriptive, specific prompts
- Add negative prompts to avoid unwanted elements
- Use higher resolution (768x768 or 1024x1024) with SDXL

### Memory Optimization
- Model automatically uses attention slicing
- xformers enabled if available (faster and less memory)
- Pipeline cached to avoid reloading
- GPU memory cleared after generation

## API Integration

Generate images via the REST API:

```bash
curl -X POST http://localhost:8000/api/v1/content/generate-image \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A serene mountain landscape at sunset",
    "width": 512,
    "height": 512,
    "num_inference_steps": 25
  }'
```

## Testing

Run the test script to verify your setup:

```bash
python test_local_image_generation.py
```

This will:
1. Check hardware capabilities
2. Display available models
3. Generate a test image
4. Save output to `test_generated_image.png`

## Examples

### Portrait Generation
```python
result = await manager.generate_image(
    prompt="Professional headshot photo of a woman in business attire, studio lighting, sharp focus",
    width=512,
    height=768,
    guidance_scale=8.0,
    negative_prompt="blurry, cartoon, anime, low quality"
)
```

### Landscape Generation
```python
result = await manager.generate_image(
    prompt="Majestic mountain range at golden hour, dramatic clouds, photorealistic",
    width=768,
    height=512,
    num_inference_steps=30,
    guidance_scale=7.5
)
```

### Product Photography
```python
result = await manager.generate_image(
    prompt="Modern smartphone on white background, product photography, studio lighting",
    width=512,
    height=512,
    guidance_scale=9.0,
    negative_prompt="blurry, dirty, damaged, low quality"
)
```

## Comparison: Local vs API

| Feature | Local (Diffusers) | API (DALL-E 3) |
|---------|------------------|----------------|
| Privacy | ✅ Complete | ❌ Sent to OpenAI |
| Cost | ✅ Free (after setup) | ❌ ~$0.04-0.12 per image |
| Speed | ⚡ Fast (with GPU) | ⚡ Fast |
| Quality | 🎨 Good-Excellent | 🎨 Excellent |
| Setup | 🔧 Requires GPU | ✅ API key only |
| Offline | ✅ Works offline | ❌ Requires internet |
| Customization | ✅ Full control | ⚠️ Limited options |

## Next Steps

- Try generating images with different prompts
- Experiment with generation parameters
- Integrate into your workflow
- Consider upgrading hardware for better performance
- Explore fine-tuned models on HuggingFace Hub

## Support

For issues or questions:
1. Check troubleshooting section above
2. Review test output: `python test_local_image_generation.py`
3. Open an issue on GitHub
4. Check system requirements match your hardware

---

**Note**: First generation will download the model (~4-7 GB). This is a one-time operation. Subsequent generations will be much faster using the cached model.
