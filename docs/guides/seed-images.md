# Seed Image Generation Workflow

## Overview

The Seed Image Generation Workflow provides a complete system for creating, managing, and approving baseline reference images for AI personas. This enables consistent visual appearance across all content generated for a persona.

## Components

### 1. Database Schema

#### BaseImageStatus Enum
Tracks the approval state of persona base images:
- `PENDING_UPLOAD` - No image yet, awaiting upload or generation
- `DRAFT` - Image exists but is not approved
- `APPROVED` - Image is final baseline, appearance locked
- `REJECTED` - Image was rejected, needs replacement

#### Database Fields
Added to `PersonaModel`:
- `base_image_status` (VARCHAR(20), default='pending_upload') - Status in approval workflow
- Indexed for query performance

### 2. API Endpoints

All endpoints are under `/api/v1/personas/{persona_id}/seed-image/`:

#### POST `/upload`
**Method 1: User Upload**
- Accepts image file upload (PNG, JPG, WEBP)
- Max size: 10MB
- Sets status to `DRAFT`
- Saves to `/opt/gator/data/models/base_images/`

#### POST `/generate-cloud`
**Method 2: DALL-E Cloud Generation**
- Uses OpenAI DALL-E 3 API
- Requires `OPENAI_API_KEY` configuration
- Generates high-quality reference image from persona's appearance description
- Quality: HD (1024x1024)
- Sets status to `DRAFT`

#### POST `/generate-local`
**Method 3: Local Stable Diffusion**
- Uses local MI25/ROCm hardware
- Supports ControlNet for refining existing drafts
- High-resolution generation (1024x1024, 50 inference steps)
- Cost-effective alternative to cloud generation
- Sets status to `DRAFT`

#### POST `/approve`
**Baseline Approval**
- Approves the baseline image
- Sets status to `APPROVED`
- Automatically sets `appearance_locked = True`
- Enables visual consistency for all future content

### 3. Service Methods

#### PersonaService

**`_save_image_to_disk(persona_id, image_data, filename=None)`**
- Saves image bytes to disk
- Creates directory structure if needed
- Returns path to saved file

**`approve_baseline(persona_id)`**
- Validates persona has base image
- Sets status to APPROVED
- Locks appearance for consistency
- Returns updated persona

#### AIModelManager

**`_generate_reference_image_openai(appearance_prompt, personality_context=None, **kwargs)`**
- Generates high-quality reference with DALL-E 3
- Uses HD quality setting for baseline images
- Includes personality context in prompt
- Returns image data with metadata

**`_generate_reference_image_local(appearance_prompt, personality_context=None, reference_image_path=None, **kwargs)`**
- Generates reference using local Stable Diffusion
- Supports ControlNet for draft refinement
- High-resolution parameters (50 steps)
- Cost-effective for iteration

**`_get_best_local_image_model()`**
- Selects best available local image model
- Prefers SDXL models for quality
- Raises exception if no models available

## Workflow Examples

### Complete Workflow: Upload → Approve

```bash
# 1. Create persona
curl -X POST http://localhost:8000/api/v1/personas/ \
  -H "Content-Type: application/json" \
  -d '{
    "name": "AI Influencer Sarah",
    "appearance": "Professional woman in her 30s",
    "personality": "Confident tech expert",
    "base_appearance_description": "Professional woman in her early 30s, shoulder-length dark hair, warm smile, modern business casual attire"
  }'
# Returns: persona_id, base_image_status: "pending_upload"

# 2. Upload base image
curl -X POST http://localhost:8000/api/v1/personas/{persona_id}/seed-image/upload \
  -F "file=@reference_image.png"
# Returns: base_image_status: "draft"

# 3. Review the image, then approve
curl -X POST http://localhost:8000/api/v1/personas/{persona_id}/seed-image/approve
# Returns: base_image_status: "approved", appearance_locked: true
```

### Cloud Generation Workflow

```bash
# 1. Create persona with detailed appearance
curl -X POST http://localhost:8000/api/v1/personas/ \
  -H "Content-Type: application/json" \
  -d '{
    "name": "AI Model Emma",
    "base_appearance_description": "Athletic woman in her mid-20s, long blonde hair, bright blue eyes, confident expression, wearing fashionable streetwear"
  }'

# 2. Generate with DALL-E 3
curl -X POST http://localhost:8000/api/v1/personas/{persona_id}/seed-image/generate-cloud
# Takes 10-30 seconds, returns draft image

# 3. If satisfied, approve
curl -X POST http://localhost:8000/api/v1/personas/{persona_id}/seed-image/approve
```

### Local Generation with Iteration

```bash
# 1. Generate initial draft locally
curl -X POST http://localhost:8000/api/v1/personas/{persona_id}/seed-image/generate-local

# 2. Review draft - if not satisfied, regenerate (uses ControlNet with previous as reference)
curl -X POST http://localhost:8000/api/v1/personas/{persona_id}/seed-image/generate-local

# 3. Once satisfied, approve
curl -X POST http://localhost:8000/api/v1/personas/{persona_id}/seed-image/approve
```

### Rejection and Replacement

```bash
# If image is not acceptable, update status
curl -X PUT http://localhost:8000/api/v1/personas/{persona_id} \
  -H "Content-Type: application/json" \
  -d '{"base_image_status": "rejected"}'

# Then upload or generate a new one
curl -X POST http://localhost:8000/api/v1/personas/{persona_id}/seed-image/upload \
  -F "file=@new_image.png"

# Approve when ready
curl -X POST http://localhost:8000/api/v1/personas/{persona_id}/seed-image/approve
```

## Configuration

### Environment Variables

```bash
# Required for cloud generation
OPENAI_API_KEY=your_openai_api_key

# Image storage location (default shown)
BASE_IMAGES_DIR=/opt/gator/data/models/base_images
```

### Storage

Base images are stored at:
```
/opt/gator/data/models/base_images/
  ├── persona_{uuid}_uploaded.{ext}     # User uploads
  ├── persona_{uuid}_dalle3.png         # DALL-E generations
  └── persona_{uuid}_local.png          # Local generations
```

## Visual Consistency Integration

Once a baseline image is approved:
- `appearance_locked` is automatically set to `True`
- All future content generation for this persona will use the base image
- Image is passed as reference to:
  - Local Stable Diffusion (via ControlNet/IP-Adapter)
  - DALL-E 3 (via prompt enhancement for consistency)

## Error Handling

### Common Errors

**400 Bad Request - No Image**
```json
{"detail": "Cannot approve baseline: persona does not have a base image"}
```
Solution: Upload or generate an image first.

**400 Bad Request - Invalid File Type**
```json
{"detail": "Invalid file type. Allowed: image/png, image/jpeg, image/jpg, image/webp"}
```
Solution: Convert image to supported format.

**400 Bad Request - File Too Large**
```json
{"detail": "File too large. Maximum size is 10MB"}
```
Solution: Compress or resize image.

**400 Bad Request - No OpenAI API Key**
```json
{"detail": "DALL-E 3 is not available. Check OPENAI_API_KEY configuration"}
```
Solution: Set `OPENAI_API_KEY` environment variable.

**400 Bad Request - No Local Models**
```json
{"detail": "No local image generation models available"}
```
Solution: Install and initialize local Stable Diffusion models.

## Testing

### Unit Tests
```bash
python -m pytest tests/unit/test_seed_image_workflow.py -v
```

Tests cover:
- BaseImageStatus enum values
- Persona creation with status field
- Image upload simulation
- Approval workflow validation
- Error conditions

### Integration Tests
```bash
python test_seed_image_integration.py
```

Tests complete workflow:
1. Create persona
2. Add base image
3. Update status through states
4. Approve baseline
5. Verify appearance locking
6. Test error conditions

### API Tests
```bash
# Start server
cd src && python -m backend.api.main

# Run manual API tests (see examples above)
```

## Migration

### Database Migration
```bash
python migrate_add_base_image_status.py
```

Adds:
- `base_image_status` column to `personas` table
- Index on `base_image_status` for performance
- Default value `pending_upload` for existing personas

### Backwards Compatibility
- Existing personas get `base_image_status = "pending_upload"`
- Existing `appearance_locked` and `base_image_path` fields are preserved
- No breaking changes to existing API endpoints

## Performance Considerations

### Image Generation Times
- DALL-E 3: 10-30 seconds
- Local SD (CPU): 60-120 seconds
- Local SD (MI25 GPU): 5-15 seconds

### Storage Requirements
- Average base image: 1-3 MB
- 1000 personas: ~2-3 GB storage

### API Rate Limits
- DALL-E 3: OpenAI rate limits apply
- Local generation: Limited by hardware

## Security Considerations

### File Upload Security
- File type validation (whitelist)
- File size limits (10MB max)
- Filename sanitization
- Stored outside web root

### Access Control
- TODO: Add authentication/authorization
- Current: All endpoints are public

### Input Validation
- Persona ID validation (UUID format)
- Image format validation
- Prompt length limits

## Future Enhancements

### Planned Features
1. Batch generation (generate multiple candidates)
2. Side-by-side comparison UI
3. Version history for base images
4. Auto-approval based on quality metrics
5. Integration with content moderation
6. Support for video seed clips

### API Improvements
1. Async generation with webhooks
2. Progress tracking for long operations
3. Image preview/thumbnail generation
4. Bulk operations for multiple personas

## Troubleshooting

### Image Not Saving
Check directory permissions:
```bash
mkdir -p /opt/gator/data/models/base_images
chmod 755 /opt/gator/data/models/base_images
```

### DALL-E Generation Fails
Verify API key:
```bash
echo $OPENAI_API_KEY
# Should print your API key
```

Test API access:
```bash
curl https://api.openai.com/v1/models \
  -H "Authorization: Bearer $OPENAI_API_KEY"
```

### Local Generation Fails
Check model availability:
```python
from backend.services.ai_models import AIModelManager
manager = AIModelManager()
await manager.initialize_models()
print(manager.available_models["image"])
```

### Approval Fails
Verify image exists:
```sql
SELECT id, name, base_image_path, base_image_status 
FROM personas 
WHERE id = 'your-persona-id';
```

## References

- [DALL-E 3 API Documentation](https://platform.openai.com/docs/guides/images)
- [Stable Diffusion Documentation](https://github.com/Stability-AI/stablediffusion)
- [ControlNet Paper](https://arxiv.org/abs/2302.05543)
- [IP-Adapter Documentation](https://github.com/tencent-ailab/IP-Adapter)
