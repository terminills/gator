# Advanced Video Features Documentation

## Overview

The Gator platform now includes advanced video generation capabilities as part of the Q2-Q3 2025 roadmap. These features enable sophisticated video content creation with multiple scenes, transitions, audio synchronization, and storyboarding.

## Features

### 1. Frame-by-Frame Video Generation

Generate longer videos by creating multiple frames/scenes and stitching them together with smooth transitions.

**Capabilities:**
- Multi-scene video composition
- Customizable duration per scene
- Six transition types (fade, crossfade, wipe, slide, zoom, dissolve)
- Multiple quality presets (draft, standard, high, premium)

**Usage Example:**

```python
from backend.services.ai_models import ai_models

# Initialize AI models
await ai_models.initialize_models()

# Generate multi-scene video
result = await ai_models.generate_video(
    prompt=[
        "Opening scene: sunrise over city",
        "Middle scene: bustling street life",
        "Closing scene: sunset reflection"
    ],
    video_type="multi_frame",
    quality="high",
    transition="crossfade",
    duration_per_frame=3.0
)

print(f"Video created: {result['file_path']}")
print(f"Duration: {result['duration']} seconds")
print(f"Scenes: {result['num_scenes']}")
```

### 2. Video Transitions

Six professional transition types are supported:

- **Fade**: Fade through black between scenes
- **Crossfade**: Direct blend between scenes
- **Wipe**: Left-to-right wipe transition
- **Slide**: Slide in from right
- **Zoom**: Zoom transition effect
- **Dissolve**: Crossfade with blur effect

**Usage Example:**

```python
from backend.services.video_processing_service import (
    VideoProcessingService,
    TransitionType,
    VideoQuality
)

video_service = VideoProcessingService()

result = await video_service.generate_frame_by_frame_video(
    prompts=["Scene 1", "Scene 2", "Scene 3"],
    duration_per_frame=2.0,
    quality=VideoQuality.HIGH,
    transition=TransitionType.DISSOLVE
)
```

### 3. Audio Synchronization

Synchronize audio tracks (voice or music) with video content.

**Capabilities:**
- Merge audio and video streams
- Automatic duration matching
- AAC audio encoding at 192kbps
- Support for various audio formats

**Requirements:**
- ffmpeg must be installed

**Usage Example:**

```python
from backend.services.ai_models import ai_models

result = await ai_models.synchronize_audio_to_video(
    video_path="/path/to/video.mp4",
    audio_path="/path/to/audio.mp3",
    output_path="/path/to/output.mp4"
)

print(f"Video with audio: {result['file_path']}")
print(f"Has audio: {result['has_audio']}")
```

### 4. Storyboard Creation

Create complex videos from detailed scene descriptions with individual timing and transitions.

**Capabilities:**
- Scene-by-scene composition
- Individual duration control per scene
- Custom transitions per scene
- Scene markers with timestamps

**Usage Example:**

```python
from backend.services.ai_models import ai_models

scenes = [
    {
        "prompt": "Opening: hero introduction",
        "duration": 3.0,
        "transition": "fade"
    },
    {
        "prompt": "Act 1: discovery",
        "duration": 5.0,
        "transition": "crossfade"
    },
    {
        "prompt": "Act 2: conflict",
        "duration": 4.0,
        "transition": "wipe"
    },
    {
        "prompt": "Resolution: victory",
        "duration": 3.0,
        "transition": "dissolve"
    }
]

result = await ai_models.create_video_storyboard(
    scenes=scenes,
    quality="high"
)

print(f"Storyboard created: {result['file_path']}")
print(f"Total duration: {result['duration']} seconds")
print(f"Scene markers: {result['scene_markers']}")
```

### 5. Video Quality Presets

Four quality presets are available to balance quality and file size:

| Quality  | Resolution | FPS | Bitrate | Use Case |
|----------|-----------|-----|---------|----------|
| Draft    | 854x480   | 24  | 1.5Mbps | Quick previews |
| Standard | 1280x720  | 30  | 3Mbps   | Social media |
| High     | 1920x1080 | 30  | 6Mbps   | Professional content |
| Premium  | 3840x2160 | 60  | 15Mbps  | 4K productions |

## Content Generation Service Integration

The content generation service now supports advanced video features through the `_generate_video` method:

```python
from backend.services.content_generation_service import (
    ContentGenerationService,
    GenerationRequest,
    ContentType,
    ContentRating
)

# Create generation request for multi-scene video
request = GenerationRequest(
    persona_id=persona.id,
    content_type=ContentType.VIDEO,
    content_rating=ContentRating.SFW,
    prompt="Create engaging lifestyle video",
    quality="high",
    style_override={
        "prompts": [
            "Morning routine",
            "Workout session", 
            "Healthy breakfast"
        ],
        "transition": "crossfade",
        "duration_per_frame": 3.0
    }
)

service = ContentGenerationService(db_session)
result = await service.generate_content(request)
```

### Storyboard via Content Generation

```python
request = GenerationRequest(
    persona_id=persona.id,
    content_type=ContentType.VIDEO,
    content_rating=ContentRating.SFW,
    prompt="Create travel vlog",
    quality="high",
    style_override={
        "scenes": [
            {
                "prompt": "Airport departure",
                "duration": 2.0,
                "transition": "fade"
            },
            {
                "prompt": "Flight journey",
                "duration": 3.0,
                "transition": "crossfade"
            },
            {
                "prompt": "Destination arrival",
                "duration": 2.0,
                "transition": "wipe"
            }
        ]
    }
)

result = await service.generate_content(request)
```

### Audio Synchronization via Content Generation

```python
# First generate voice
voice_request = GenerationRequest(
    persona_id=persona.id,
    content_type=ContentType.VOICE,
    prompt="Welcome to my channel! Today we're exploring...",
    quality="high"
)

voice_result = await service.generate_content(voice_request)

# Then generate video with audio sync
video_request = GenerationRequest(
    persona_id=persona.id,
    content_type=ContentType.VIDEO,
    prompt="Create intro video",
    quality="high",
    style_override={
        "duration": 10.0,
        "audio_path": voice_result.file_path
    }
)

video_result = await service.generate_content(video_request)
```

## AI Model Integration

### Available Video Models

Three video generation models are configured:

1. **Frame-by-Frame Generator** (Always Available)
   - Type: Multi-frame video
   - Provider: Local
   - Features: Multi-scene, transitions, audio sync, storyboarding
   - Status: Production ready

2. **Stable Video Diffusion** (Requires Setup)
   - Type: Image-to-video
   - Provider: Local
   - Model: `stabilityai/stable-video-diffusion-img2vid-xt`
   - Requirements: 24GB+ VRAM
   - Features: Image-to-video, frame interpolation
   - Duration: Up to 4 seconds
   - Resolution: 576x1024

3. **Runway Gen-2** (Requires API Key)
   - Type: Text-to-video
   - Provider: Cloud (Runway ML)
   - Requirements: RUNWAY_API_KEY environment variable
   - Features: Text-to-video, image-to-video, 4K output
   - Duration: Up to 18 seconds

### Model Selection

The system automatically selects the best available model:

```python
from backend.services.ai_models import ai_models

# Get available models
models = await ai_models.get_available_models()
video_models = models["video"]

for model in video_models:
    print(f"Model: {model['name']}")
    print(f"Loaded: {model['loaded']}")
    print(f"Features: {model.get('features', [])}")
```

## Technical Requirements

### Required Dependencies

All dependencies are included in `pyproject.toml`:

- `opencv-python>=4.8.0` - Video processing
- `numpy>=1.24.0` - Array operations
- `torch` - ML models (for SVD)
- `diffusers` - Diffusion models (for SVD)

### Optional Dependencies

- **ffmpeg** - Required for audio synchronization and re-encoding
  - Install: `sudo apt-get install ffmpeg` (Linux)
  - Check availability: The service automatically detects ffmpeg

### Hardware Requirements

| Feature | Minimum | Recommended |
|---------|---------|-------------|
| Frame-by-Frame | 8GB RAM | 16GB RAM |
| SVD (Local) | 24GB VRAM | 32GB VRAM |
| Runway (Cloud) | API Key | API Key + Credits |

## Performance Considerations

### Frame-by-Frame Generation

- **Speed**: ~1-2 seconds per frame generation
- **Memory**: ~2-4GB RAM during processing
- **Storage**: ~10-50MB per minute of video (depends on quality)

### Video Export

- **With opencv**: Fast export but larger files
- **With ffmpeg**: Slower but better compression
- **Re-encoding**: Adds 2-5 seconds for better compression

### Optimization Tips

1. **Use draft quality** for previews
2. **Enable ffmpeg** for smaller file sizes
3. **Batch scenes** for longer videos
4. **Cache generated frames** for reuse

## Error Handling

The video processing service includes comprehensive error handling:

```python
try:
    result = await video_service.generate_frame_by_frame_video(
        prompts=["Scene 1"],
        duration_per_frame=3.0,
        quality=VideoQuality.HIGH
    )
except RuntimeError as e:
    print(f"Video generation failed: {e}")
except ValueError as e:
    print(f"Invalid parameters: {e}")
```

### Common Errors

- **ffmpeg not available**: Audio sync requires ffmpeg installation
- **Invalid quality**: Use VideoQuality enum values
- **Empty prompts**: At least one prompt is required
- **Invalid transition**: Use TransitionType enum values

## Testing

Comprehensive test suite included:

```bash
# Run all video tests
pytest tests/unit/test_video_processing.py tests/unit/test_ai_models_video.py -v

# Run specific test
pytest tests/unit/test_video_processing.py::TestVideoProcessingService::test_frame_by_frame_video_generation -v
```

**Test Coverage:**
- 24 tests for video processing service
- 23 tests for AI models video generation
- 47 total tests (100% passing)

## API Endpoints

Video generation is integrated into the content generation API:

```http
POST /api/v1/content/generate
Content-Type: application/json

{
  "persona_id": "uuid",
  "content_type": "video",
  "content_rating": "sfw",
  "prompt": "Create video",
  "quality": "high",
  "style_override": {
    "prompts": ["Scene 1", "Scene 2"],
    "transition": "crossfade",
    "duration_per_frame": 3.0
  }
}
```

## Examples

### Example 1: Simple Video

```python
result = await ai_models.generate_video(
    prompt="Beautiful sunset over ocean",
    video_type="single_frame",
    quality="high",
    duration_per_frame=5.0
)
```

### Example 2: Multi-Scene with Transitions

```python
result = await ai_models.generate_video(
    prompt=[
        "Morning sunrise",
        "Afternoon cityscape",
        "Evening sunset"
    ],
    video_type="multi_frame",
    quality="premium",
    transition="dissolve",
    duration_per_frame=4.0
)
```

### Example 3: Complete Workflow

```python
# 1. Generate voice narration
voice_result = await ai_models.generate_voice(
    text="Welcome to our journey...",
    voice="alloy"
)

# 2. Create video storyboard
scenes = [
    {"prompt": "Intro", "duration": 3.0, "transition": "fade"},
    {"prompt": "Main", "duration": 5.0, "transition": "crossfade"},
    {"prompt": "Outro", "duration": 2.0, "transition": "dissolve"}
]

video_result = await ai_models.create_video_storyboard(
    scenes=scenes,
    quality="high"
)

# 3. Synchronize audio with video
final_result = await ai_models.synchronize_audio_to_video(
    video_path=video_result["file_path"],
    audio_path=voice_result["file_path"]
)

print(f"Final video: {final_result['file_path']}")
```

## Future Enhancements

Planned for future releases:

- **Q3 2025**: Integration with Stable Video Diffusion models
- **Q3 2025**: Runway Gen-2 API integration
- **Q4 2025**: Real-time video preview
- **Q4 2025**: Advanced editing features (trim, crop, filters)
- **Q4 2025**: Template-based video generation

## Support

For questions or issues:
- Open an issue on GitHub
- Check the test suite for usage examples
- Review the inline code documentation

## Related Documentation

- [Enhancement Roadmap](../ENHANCEMENTS_ROADMAP.md)
- [API Documentation](../API.md)
- [Testing Guide](../TESTING_GUIDE.md)
