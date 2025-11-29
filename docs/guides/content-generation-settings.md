# Content Generation with Persona Settings

## Overview

Content generation in the Gator AI platform now uses persona-specific settings instead of hardcoded values. This allows each persona to have their own preferred resolution, quality, style, and model configurations.

## New Persona Fields

### Content Generation Preferences

Each persona can now be configured with the following content generation settings:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `default_image_resolution` | string | `1024x1024` | Default resolution for image generation |
| `default_video_resolution` | string | `1920x1080` | Default resolution for video generation |
| `post_style` | string | `casual` | Post style preference |
| `video_types` | array | `[]` | Preferred video types |
| `nsfw_model_preference` | string | `null` | Preferred NSFW model |
| `generation_quality` | string | `standard` | Default quality level |

### Image Resolution Options

- `512x512` - Square Draft
- `1024x1024` - Square HD (default)
- `2048x2048` - Square Ultra HD
- `720x1280` - Portrait 720p
- `1080x1920` - Portrait 1080p
- `1280x720` - Landscape 720p
- `1920x1080` - Landscape 1080p

### Video Resolution Options

- `1280x720` - 720p HD
- `1920x1080` - 1080p Full HD (default)
- `2560x1440` - 1440p 2K
- `3840x2160` - 2160p 4K

### Generation Quality Levels

- `draft` - Fast, lower quality
- `standard` - Balanced (default)
- `hd` - High detail
- `premium` - Best quality, slower

### Post Style Options

- `casual` (default)
- `professional`
- `artistic`
- `provocative`
- `playful`
- `elegant`
- `edgy`

### Video Types

- `short_clip` - 15-30 second clips
- `story` - Instagram/Snapchat style stories
- `reel` - TikTok/Instagram Reels
- `long_form` - YouTube/extended content
- `tutorial` - How-to/educational content

### NSFW Model Support

The platform now supports multiple NSFW models that can be selected per persona:

1. **flux-nsfw-highress** - `CultriX/flux-nsfw-highress`
   - High resolution NSFW content generation
   - Model: `nsfw-highress.safetensors`

2. **darkblueaphrodite** - `stablediffusionapi/darkblueaphrodite-nsfw-he`
   - NSFW Hentai style content
   - Specialized for anime/illustrated NSFW content

3. **modifier_sexual_coaching** - `DervlexVenice/modifier_sexual_coaching_nai_vpred_illustrious-style-illustrious`
   - Illustrious style NSFW content
   - Model: `Modifier_Sexual_Coaching_NAI_VPRED_ILLUSTRIOUS_1056564.safetensors`

4. **eye_contact_blowjob** - `DervlexVenice/eye_contact_blowjob-action-flux`
   - Flux action-specific NSFW content
   - Model: `Eye_contact_blowjob_1488724.safetensors`

## Using Persona Settings in Content Generation

### Single Content Generation

When generating content for a specific persona, the system automatically uses that persona's settings:

```bash
POST /api/v1/content/generate
{
  "persona_id": "92198e46-f9c2-48db-bce9-50ff104a685c",
  "content_type": "image"
}
```

The persona's settings will be applied:
- Resolution: Uses `default_image_resolution`
- Quality: Uses `generation_quality`
- Content Rating: Uses `default_content_rating`
- NSFW Model: Uses `nsfw_model_preference` (if NSFW content)

### Batch Content Generation

When generating content for all personas, each persona uses their own settings:

```bash
POST /api/v1/content/generate/all?content_type=image
```

Each persona will:
- Use their own `default_image_resolution`
- Use their own `generation_quality`
- Use their own `default_content_rating`
- Apply their own `nsfw_model_preference` if applicable

### Admin UI

The admin panel provides an intuitive interface for:

1. **Persona Editor** - Configure all generation preferences per persona
   - Located at: `/admin/personas?action=edit&id={persona_id}`
   - New "Content Generation Preferences" section
   - All settings configurable through dropdowns and checkboxes

2. **Content Generation Dialog** - Shows persona settings preview
   - Located at: `/admin/content`
   - Select persona before generating
   - Preview shows all applicable settings
   - No more hardcoded quality or resolution values

## Content Filtering

Content filtering is enforced based on persona settings:

1. **Content Rating Validation**
   - Generated content must match persona's `default_content_rating`
   - Or be within `allowed_content_ratings`
   - Platform restrictions (`platform_restrictions`) are enforced

2. **NSFW Model Application**
   - NSFW model preference only applied when `content_rating == "nsfw"`
   - Automatically selected based on persona configuration
   - Falls back to default model if no preference set

3. **Platform-Specific Filtering**
   - Each platform can have specific restrictions
   - Example: `{"instagram": "sfw_only", "onlyfans": "both"}`
   - Content generation respects these restrictions

## Example Persona Configuration

```json
{
  "name": "Professional Fashion Model",
  "appearance": "Elegant woman with professional styling",
  "personality": "Sophisticated, confident, professional",
  "content_themes": ["fashion", "lifestyle", "beauty"],
  "default_content_rating": "moderate",
  "allowed_content_ratings": ["sfw", "moderate"],
  "platform_restrictions": {
    "instagram": "sfw_only",
    "twitter": "moderate_allowed"
  },
  "default_image_resolution": "2048x2048",
  "default_video_resolution": "1920x1080",
  "post_style": "professional",
  "video_types": ["reel", "story"],
  "generation_quality": "hd",
  "image_style": "photorealistic"
}
```

## Example NSFW Persona Configuration

```json
{
  "name": "NSFW Content Creator",
  "appearance": "Attractive woman with artistic styling",
  "personality": "Bold, playful, confident",
  "content_themes": ["adult", "artistic"],
  "default_content_rating": "nsfw",
  "allowed_content_ratings": ["sfw", "moderate", "nsfw"],
  "platform_restrictions": {
    "instagram": "sfw_only",
    "onlyfans": "both",
    "twitter": "moderate_allowed"
  },
  "default_image_resolution": "2048x2048",
  "default_video_resolution": "3840x2160",
  "post_style": "provocative",
  "video_types": ["short_clip", "reel"],
  "nsfw_model_preference": "flux-nsfw-highress",
  "generation_quality": "hd",
  "image_style": "photorealistic"
}
```

## Migration

The database migration `migrate_add_content_generation_prefs.py` adds the new columns to existing personas tables:

```bash
python migrate_add_content_generation_prefs.py
```

All existing personas will get default values:
- `default_image_resolution`: `1024x1024`
- `default_video_resolution`: `1920x1080`
- `post_style`: `casual`
- `video_types`: `[]`
- `nsfw_model_preference`: `null`
- `generation_quality`: `standard`

## Testing

### Test Persona Creation

```bash
curl -X POST http://localhost:8000/api/v1/personas/ \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Test Persona",
    "appearance": "Professional appearance",
    "personality": "Confident and creative",
    "content_themes": ["technology"],
    "default_content_rating": "sfw",
    "allowed_content_ratings": ["sfw"],
    "default_image_resolution": "2048x2048",
    "default_video_resolution": "1920x1080",
    "post_style": "professional",
    "video_types": ["reel", "story"],
    "generation_quality": "hd",
    "image_style": "photorealistic"
  }'
```

### Verify Settings

```bash
curl http://localhost:8000/api/v1/personas/{persona_id}
```

The response should include all content generation preferences.

## Security

### NSFW Model Security

1. **Model Validation**: Only whitelisted NSFW models are allowed
2. **Content Rating Check**: NSFW models only applied when `content_rating == "nsfw"`
3. **Platform Filtering**: Platform-specific restrictions enforced
4. **Allowed Ratings**: Content must be in persona's `allowed_content_ratings`

### Code Scanning

All changes have been scanned with CodeQL:
- ✅ No security vulnerabilities detected
- ✅ No SQL injection risks
- ✅ Proper input validation in place
- ✅ Safe model preference handling

## Benefits

1. **Flexibility**: Each persona can have unique generation settings
2. **Consistency**: Personas maintain their visual style and quality
3. **Efficiency**: No need to specify settings for each generation
4. **Safety**: Built-in content filtering and validation
5. **Scalability**: Batch generation uses per-persona settings automatically

## Future Enhancements

Potential future improvements:

1. **Advanced Model Selection**: Support for custom model weights
2. **Style Templates**: Predefined style configurations
3. **A/B Testing**: Test different settings per persona
4. **Dynamic Quality**: Adjust quality based on platform and context
5. **Model Performance Tracking**: Monitor which models work best per persona
