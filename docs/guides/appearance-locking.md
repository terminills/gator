# Visual Consistency & Appearance Locking Feature

## Overview

This feature implements a robust visual consistency system for AI personas by providing a mechanism to lock and enforce a baseline physical appearance. This ensures near-perfect visual consistency across all generated content through the use of both detailed textual descriptions and reference images.

## Problem Statement

Relying solely on text prompts for content generation often leads to visual drift - where the same AI persona looks slightly different across different generated images. This inconsistency breaks immersion and makes it difficult to maintain a recognizable brand identity for AI influencers.

## Solution

We've added three new fields to the `personas` table that work together to create a visual consistency cache:

1. **`base_appearance_description`** (TEXT): A detailed, approved baseline appearance description used as the foundation for all content generation
2. **`base_image_path`** (VARCHAR(500)): Path to a single, approved reference image that serves as the visual anchor
3. **`appearance_locked`** (BOOLEAN): A flag that enables the consistency logic and prevents accidental overwrites

## Implementation Details

### Database Schema Changes

```sql
-- New columns added to personas table
ALTER TABLE personas ADD COLUMN base_appearance_description TEXT;
ALTER TABLE personas ADD COLUMN base_image_path VARCHAR(500);
ALTER TABLE personas ADD COLUMN appearance_locked BOOLEAN DEFAULT FALSE;
CREATE INDEX ix_personas_appearance_locked ON personas (appearance_locked);
```

### API Schema Updates

#### PersonaCreate
```python
class PersonaCreate(BaseModel):
    # ... existing fields ...
    base_appearance_description: Optional[str] = Field(
        default=None,
        max_length=5000,
        description="Detailed baseline appearance description for visual consistency"
    )
    base_image_path: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Path to reference image for visual consistency"
    )
    appearance_locked: bool = Field(
        default=False,
        description="When True, locks appearance and enables visual consistency features"
    )
```

#### PersonaUpdate
```python
class PersonaUpdate(BaseModel):
    # ... existing fields ...
    base_appearance_description: Optional[str] = Field(None, max_length=5000)
    base_image_path: Optional[str] = Field(None, max_length=500)
    appearance_locked: Optional[bool] = None
```

#### PersonaResponse
```python
class PersonaResponse(BaseModel):
    # ... existing fields ...
    base_appearance_description: Optional[str] = None
    base_image_path: Optional[str] = None
    appearance_locked: bool = False
```

### Content Generation Integration

The `ContentGenerationService` has been updated to use the visual consistency cache when `appearance_locked` is `True`:

#### Prompt Generation (`_generate_prompt`)
```python
if persona.appearance_locked and persona.base_appearance_description:
    base_prompt = f"{persona.base_appearance_description}, {persona.personality}"
    logger.info(f"Using locked base appearance for persona {persona.id}")
else:
    base_prompt = f"{persona.appearance}, {persona.personality}"
```

#### Image Generation (`_generate_image`)
```python
if persona.appearance_locked and persona.base_image_path:
    generation_params["reference_image_path"] = persona.base_image_path
    generation_params["use_controlnet"] = True
    logger.info(f"Using visual reference for consistency: {persona.base_image_path}")
```

#### Text Generation (`_generate_text`)
```python
appearance_desc = (
    persona.base_appearance_description
    if persona.appearance_locked and persona.base_appearance_description
    else persona.appearance
)
```

#### Fallback Text Templates (`_create_enhanced_fallback_text`)
The fallback text generation templates have been enhanced to work with appearance locking:

```python
# Uses base_appearance_description when locked for consistency
appearance_desc = (
    persona.base_appearance_description
    if persona.appearance_locked and persona.base_appearance_description
    else persona.appearance
)

# Extracts visual cues from appearance for personalized templates
appearance_keywords = appearance_desc.lower() if appearance_desc else ""
is_visual_locked = persona.appearance_locked and persona.base_appearance_description

# Adds subtle appearance context to templates when locked
if is_visual_locked:
    if "professional" in appearance_keywords:
        appearance_context = " (staying true to my professional image)"
    elif "creative" in appearance_keywords:
        appearance_context = " (expressing my creative side)"
```

This ensures that even fallback text generation maintains consistency with the persona's locked visual identity.

## Usage Examples

### Creating a Persona with Appearance Locking

```python
from backend.models.persona import PersonaCreate, ContentRating

persona_data = PersonaCreate(
    name="Emma - AI Fashion Influencer",
    appearance="Young professional woman with long blonde hair",
    personality="Creative, innovative, passionate about fashion",
    content_themes=["fashion", "style", "trends"],
    base_appearance_description=(
        "A 28-year-old professional woman with long, wavy blonde hair "
        "cascading past her shoulders. Striking blue eyes with subtle "
        "makeup emphasizing natural beauty. Fair complexion with warm "
        "undertones. Modern business casual attire. High-resolution "
        "portrait photography style with professional lighting."
    ),
    base_image_path="/models/base_images/emma_reference_001.jpg",
    appearance_locked=True
)
```

### Updating an Existing Persona to Enable Locking

```python
from backend.models.persona import PersonaUpdate

update_data = PersonaUpdate(
    base_appearance_description=(
        "Detailed baseline appearance with specific features..."
    ),
    base_image_path="/models/base_images/persona_ref.jpg",
    appearance_locked=True
)

updated_persona = await persona_service.update_persona(persona_id, update_data)
```

### Checking if a Persona Has Appearance Locking

```python
persona = await persona_service.get_persona(persona_id)

if persona.appearance_locked:
    print(f"✅ Appearance is locked for {persona.name}")
    if persona.base_image_path:
        print(f"📷 Reference image: {persona.base_image_path}")
    if persona.base_appearance_description:
        print(f"📝 Base description: {persona.base_appearance_description[:50]}...")
```

## Migration

For existing databases, run the migration script:

```bash
python migrate_add_appearance_locking.py
```

This script:
- Detects the database type (SQLite or PostgreSQL)
- Checks if columns already exist
- Adds missing columns safely
- Creates necessary indexes
- Provides clear status messages

## Testing

### Unit Tests

Run the comprehensive unit tests:
```bash
python -m pytest tests/unit/test_appearance_locking.py -v
```

Tests cover:
- Creating personas with and without locking
- Updating personas to enable/disable locking
- Field validation and max length enforcement
- Optional field handling

### Integration Tests

Run the full integration test:
```bash
python test_appearance_locking.py
```

This demonstrates:
- Creating personas with different locking configurations
- Updating locking status
- Verifying data persistence
- Listing personas with locking status

## Best Practices

### When to Use Appearance Locking

✅ **Use appearance locking when:**
- You need consistent visual identity across all content
- Building a recognizable AI influencer brand
- Working with professional photoshoots or curated images
- Generating content for commercial/marketing purposes

❌ **Don't use appearance locking when:**
- Experimenting with different looks
- Creating diverse content with varied styles
- Testing persona concepts
- You want maximum creative flexibility

### Creating Effective Base Descriptions

A good `base_appearance_description` should include:

1. **Age and basic demographics**: "28-year-old professional woman"
2. **Physical characteristics**: Hair color, length, style; eye color; skin tone
3. **Facial features**: Distinctive features that make the persona unique
4. **Typical attire**: Style of clothing commonly worn
5. **Photography style**: Lighting, focus, composition preferences
6. **Quality indicators**: "high-resolution", "professional", "studio lighting"

Example:
```python
base_appearance_description = """
A 35-year-old Asian woman with shoulder-length black hair styled in loose waves.
Deep brown eyes with subtle makeup emphasizing natural features. Warm skin tone
with healthy complexion. Wearing contemporary business attire in neutral colors -
typically cream, grey, or navy blazers with simple accessories. Confident posture
with approachable expression. Professional studio photography with soft focus
background, natural lighting from the side, high-resolution portrait style.
"""
```

### Managing Reference Images

**File Organization:**
```
/models/
  └── base_images/
      ├── persona_001_reference.jpg
      ├── persona_002_reference.jpg
      └── ...
```

**File Requirements:**
- High resolution (1024x1024 or higher recommended)
- Clear, well-lit subject
- Neutral background
- Professional quality
- Consistent with base_appearance_description

**Storage Considerations:**
- Store images on accessible file system or cloud storage
- Use consistent naming conventions
- Version control for reference images (e.g., `_v1`, `_v2`)
- Backup reference images securely

## Advanced Usage

### Conditional Locking

You can lock appearance for specific content types while allowing flexibility for others:

```python
# In your content generation logic
if content_type == ContentType.IMAGE and persona.appearance_locked:
    # Use strict visual consistency
    use_reference_image = True
elif content_type == ContentType.TEXT:
    # Text can be more flexible
    use_reference_image = False
```

### Versioning Locked Appearances

When you need to update a locked persona's appearance:

```python
# 1. Create new version of reference image
new_ref_path = f"/models/base_images/{persona_id}_v2.jpg"

# 2. Update with new baseline
update = PersonaUpdate(
    base_appearance_description="Updated detailed description...",
    base_image_path=new_ref_path,
    appearance_locked=True
)

# 3. Archive old version for history
await archive_persona_version(persona_id, old_ref_path)
```

### Unlocking for Experimentation

Temporarily unlock appearance for testing:

```python
# Unlock
await persona_service.update_persona(
    persona_id,
    PersonaUpdate(appearance_locked=False)
)

# ... experiment with new looks ...

# Re-lock with original settings
await persona_service.update_persona(
    persona_id,
    PersonaUpdate(appearance_locked=True)
)
```

## Technical Notes

### Performance Considerations

- The `appearance_locked` field is indexed for fast filtering
- Reference images should be preprocessed and cached for optimal performance
- Base descriptions are stored as TEXT for flexibility

### GPU Pipeline Integration

When using the reference image with GPU pipelines:

```python
# The ContentGenerationService passes these parameters to the AI models
generation_params = {
    "prompt": enhanced_prompt,
    "reference_image_path": persona.base_image_path,
    "use_controlnet": True,  # For advanced image conditioning
    # ... other params
}
```

Your AI model service should handle:
- Loading the reference image
- Applying ControlNet or similar conditioning
- Ensuring visual consistency across generations

### Security Considerations

- Validate `base_image_path` to prevent directory traversal attacks
- Sanitize file paths before storage
- Implement access controls for reference images
- Consider encryption for sensitive persona data

## Troubleshooting

### Common Issues

**Q: Reference image not being used in generation**
- Verify `appearance_locked` is `True`
- Check that `base_image_path` points to existing file
- Ensure AI model service supports reference images
- Check logs for image loading errors

**Q: Visual consistency not as expected**
- Review base_appearance_description for specificity
- Ensure reference image quality is high
- Verify ControlNet or conditioning model is enabled
- Consider adjusting generation parameters

**Q: Cannot update appearance when locked**
- This is intended behavior - use `PersonaUpdate` to explicitly change locked fields
- To make changes, update `base_appearance_description` instead of `appearance`
- Or temporarily unlock, make changes, then re-lock

## Future Enhancements

Potential improvements for this feature:

1. **Multi-reference support**: Multiple reference images for different angles/poses
2. **Automatic consistency scoring**: AI-based validation of generated content consistency
3. **Face recognition integration**: Verify generated images match reference
4. **Template system**: Pre-defined appearance templates for quick setup
5. **Version history**: Track changes to locked appearances over time
6. **A/B testing**: Compare consistency metrics between locked and unlocked generations

## See Also

- [Persona Management API Documentation](../docs/api/personas.md)
- [Content Generation Guide](../docs/content-generation.md)
- [Best Practices for AI Influencers](../BEST_PRACTICES.md)
