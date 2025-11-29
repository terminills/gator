# AI-Powered Video Generation

This document describes the enhanced video generation pipeline that uses actual AI image generation for video frames instead of placeholder frames.

## Overview

The Gator platform now supports two modes of video generation:

1. **Placeholder Mode** (Fast): Generates gradient frames with text overlays - useful for testing and previews
2. **AI Generation Mode** (Quality): Uses AI image generation models to create actual images from prompts for each frame

## Features

### Core Capabilities

- ✅ **AI-Powered Frame Generation**: Each video frame can be generated using Stable Diffusion or other image models
- ✅ **Automatic Fallback**: If AI generation fails, automatically falls back to placeholder frames
- ✅ **Quality Presets**: Supports DRAFT (480p), STANDARD (720p), HIGH (1080p), and PREMIUM (4K) resolutions
- ✅ **Multiple Transitions**: FADE, CROSSFADE, WIPE, SLIDE, ZOOM, DISSOLVE
- ✅ **Flexible Control**: Enable/disable AI generation per video
- ✅ **Multi-Scene Support**: Generate videos with multiple scenes, each from different prompts

### Technical Features

- Image format conversion (PIL RGB → OpenCV BGR)
- Resolution matching for different quality presets
- Configurable inference steps and guidance scale
- GPU-aware generation (uses available GPU memory)
- Error handling with graceful degradation

## Usage

### Basic Example

```python
from backend.services.ai_models import AIModelManager

# Initialize AI manager
ai_manager = AIModelManager()
await ai_manager.initialize_models()

# Generate video with AI frames
result = await ai_manager.generate_video(
    prompt=[
        "A serene mountain landscape at sunrise",
        "A bustling city skyline at night",
    ],
    video_type="multi_frame",
    quality="high",
    transition="crossfade",
    duration_per_frame=3.0,
    use_ai_generation=True,  # Enable AI generation
    num_inference_steps=25,  # Quality vs speed tradeoff
    guidance_scale=7.5,      # How closely to follow prompt
)

print(f"Video saved to: {result['file_path']}")
```

### Direct VideoProcessingService Usage

```python
from backend.services.video_processing_service import (
    VideoProcessingService,
    VideoQuality,
    TransitionType,
)
from backend.services.ai_models import AIModelManager

# Initialize services
ai_manager = AIModelManager()
await ai_manager.initialize_models()

video_service = VideoProcessingService()

# Generate video
result = await video_service.generate_frame_by_frame_video(
    prompts=[
        "Scene 1 description",
        "Scene 2 description",
        "Scene 3 description",
    ],
    duration_per_frame=2.0,
    quality=VideoQuality.HIGH,
    transition=TransitionType.CROSSFADE,
    use_ai_generation=True,
    ai_model_manager=ai_manager,
    num_inference_steps=20,
)
```

### Placeholder Mode (Fast Preview)

```python
# Generate video with placeholder frames (fast)
result = await ai_manager.generate_video(
    prompt=["Scene 1", "Scene 2"],
    video_type="multi_frame",
    quality="standard",
    use_ai_generation=False,  # Use placeholders
)
```

## Configuration

### Quality Presets

| Quality | Resolution | FPS | Bitrate | Use Case |
|---------|-----------|-----|---------|----------|
| DRAFT | 854x480 | 24 | 1500k | Fast testing |
| STANDARD | 1280x720 | 30 | 3000k | General use |
| HIGH | 1920x1080 | 30 | 6000k | High quality |
| PREMIUM | 3840x2160 | 60 | 15000k | 4K production |

### Generation Parameters

- **num_inference_steps** (default: 20-25): Number of denoising steps. Higher = better quality but slower.
- **guidance_scale** (default: 7.5): How closely to follow the prompt. Range: 1-20.
- **duration_per_frame** (default: 3.0): How long each scene lasts in seconds.

## Requirements

### For AI-Generated Videos

**Hardware Requirements:**
- **Stable Diffusion 1.5**: 4+ GB GPU memory, 8+ GB RAM
- **SDXL**: 8+ GB GPU memory, 16+ GB RAM
- **CPU-only**: Possible but very slow

**Alternative: API-Based Generation:**
- Set `OPENAI_API_KEY` environment variable for DALL-E
- Requires OpenAI API subscription

### For Placeholder Videos

No special requirements - runs on any system.

## Performance Considerations

### Generation Time Estimates

**With GPU (NVIDIA RTX 3090 / AMD MI25):**
- DRAFT (480p): ~5-10 seconds per frame
- STANDARD (720p): ~10-15 seconds per frame
- HIGH (1080p): ~15-25 seconds per frame

**First Run:**
- Add 5-10 minutes for model download (4-7 GB)

**Optimization Tips:**
1. Use lower `num_inference_steps` (15-20) for faster generation
2. Start with DRAFT quality for testing
3. Generate fewer scenes initially
4. Use placeholder mode for previews

## Error Handling

The system automatically handles common issues:

1. **No GPU Available**: Uses CPU (slower) or fails gracefully
2. **Insufficient Memory**: Falls back to smaller models or placeholders
3. **Model Download Fails**: Uses cached models or placeholders
4. **Generation Timeout**: Returns placeholder for that frame
5. **Invalid Parameters**: Uses safe defaults

## Testing

Run the test suite:

```bash
# Test AI video frame generation
pytest tests/unit/test_ai_video_frame_generation.py -v

# Test all video features
pytest tests/unit/test_video_processing.py tests/unit/test_ai_models_video.py tests/unit/test_ai_video_frame_generation.py -v
```

Run the demo:

```bash
# Compare placeholder vs AI generation
python demo_ai_video_generation.py

# Original video features demo
python demo_video_features.py
```

## API Reference

### VideoProcessingService._generate_single_frame()

```python
async def _generate_single_frame(
    prompt: str,
    quality: VideoQuality,
    frame_index: int = 0,
    use_ai_generation: bool = True,
    ai_model_manager: AIModelManager = None,
    num_inference_steps: int = 20,
    guidance_scale: float = 7.5,
    **kwargs
) -> np.ndarray
```

Generates a single video frame.

**Parameters:**
- `prompt`: Text description of the scene
- `quality`: Video quality preset
- `frame_index`: Index of frame in sequence (for logging)
- `use_ai_generation`: Whether to use AI (True) or placeholder (False)
- `ai_model_manager`: AIModelManager instance (required if use_ai_generation=True)
- `num_inference_steps`: Number of denoising steps
- `guidance_scale`: Prompt adherence strength

**Returns:** Numpy array (height, width, 3) in BGR format

### AIModelManager.generate_video()

```python
async def generate_video(
    prompt: Union[str, List[str]],
    video_type: str = "multi_frame",
    quality: str = "high",
    transition: str = "crossfade",
    duration_per_frame: float = 3.0,
    use_ai_generation: bool = True,
    **kwargs
) -> Dict[str, Any]
```

Generates a video using frame-by-frame generation.

**Parameters:**
- `prompt`: Single prompt or list of prompts for scenes
- `video_type`: Type of video generation (default: "multi_frame")
- `quality`: Quality preset ("draft", "standard", "high", "premium")
- `transition`: Transition type between scenes
- `duration_per_frame`: Duration of each scene in seconds
- `use_ai_generation`: Use AI (True) or placeholders (False)

**Returns:** Dictionary with video metadata and file path

## Examples

See the following files for complete examples:

- `demo_ai_video_generation.py` - AI generation showcase
- `demo_video_features.py` - Original video features
- `tests/unit/test_ai_video_frame_generation.py` - Test examples

## Troubleshooting

### "No image generation models available"

**Solution**: Install a model or set API keys:

```bash
# For local generation (requires GPU)
export HF_TOKEN=your_huggingface_token
# Model will auto-download on first use

# For API-based generation
export OPENAI_API_KEY=your_openai_key
```

### "AI frame generation failed"

**Cause**: GPU memory exhausted or model loading failed

**Solution**:
1. Use lower quality preset (DRAFT or STANDARD)
2. Reduce num_inference_steps
3. Enable placeholder mode temporarily

### Slow generation times

**Solutions**:
1. Reduce `num_inference_steps` (try 15-20)
2. Use lower quality preset
3. Ensure GPU is being used (check `ai_manager.gpu_type`)
4. Use placeholder mode for previews

## Future Enhancements

Planned features:
- [ ] Batch frame generation for better GPU utilization
- [ ] Video-to-video generation (modify existing videos)
- [ ] ControlNet support for consistent character appearance
- [ ] Audio synchronization with AI-generated frames
- [ ] Real-time preview while generating
- [ ] Progress callbacks and cancellation support

## Related Documentation

- [Video Features Documentation](VIDEO_FEATURES.md)
- [AI Models Setup](../AI_MODELS_SETUP_ENDPOINT_VERIFICATION.md)
- [Local Image Generation](../LOCAL_IMAGE_GENERATION.md)
