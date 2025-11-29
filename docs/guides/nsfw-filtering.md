# Per-Site NSFW Filtering Enhancement

## Overview

This enhancement allows personas to have customized content rating policies on a per-platform basis. This enables the same platform (e.g., Instagram) to have different NSFW policies for different personas based on their specific needs and permissions.

## Problem Statement

Previously, the system used global platform policies that applied to all personas uniformly:
- Instagram: SFW and MODERATE only (NSFW blocked)
- Facebook: SFW only
- OnlyFans: All ratings allowed
- etc.

This rigid approach didn't account for:
1. **Authorized NSFW content**: Some personas may have special permission to post NSFW content on typically restrictive platforms
2. **Brand safety**: Some personas may want to be more restrictive than platform defaults (e.g., SFW-only on OnlyFans)
3. **Multi-platform strategies**: Different content strategies for different platforms

## Solution

The enhancement uses the existing `platform_restrictions` field in the `PersonaModel` to override global platform policies on a per-persona basis.

### Data Structure

```python
persona.platform_restrictions = {
    "instagram": "both",              # Allow all content (SFW, MODERATE, NSFW)
    "facebook": "moderate_allowed",   # Allow SFW and MODERATE
    "twitter": "sfw_only",            # Only allow SFW
}
```

### Supported Restriction Values

- `"sfw_only"`: Only SFW content allowed
- `"moderate_allowed"`: SFW and MODERATE content allowed  
- `"both"` or `"all"`: All content types allowed (SFW, MODERATE, NSFW)

If a platform is not specified in `platform_restrictions`, it uses the global default policy.

## Implementation Details

### Modified Files

1. **`src/backend/models/persona.py`**
   - Added `MODERATE` to `ContentRating` enum for consistency with `content.py`

2. **`src/backend/services/content_generation_service.py`**
   - Updated `ContentModerationService.platform_content_filter()` to accept optional `persona_platform_restrictions` parameter
   - Modified logic to check persona restrictions before falling back to global policies
   - Updated `ContentGenerationService._create_platform_adaptations()` to pass persona and its restrictions to the filter

3. **`tests/unit/test_content_generation_enhancements.py`**
   - Added 5 new test cases covering persona override scenarios
   - Updated existing tests to pass persona parameter
   - Added `platform_restrictions` to mock persona fixture

### Code Flow

```python
# When generating content
1. ContentGenerationService.generate_content() retrieves the persona
2. Calls _create_platform_adaptations(persona, content_data, rating, platforms)
3. For each platform:
   - Calls platform_content_filter(rating, platform, persona.platform_restrictions)
   - If persona has restriction for this platform, use it
   - Otherwise, fall back to global platform policy
4. Return platform-specific adaptations (approved/blocked)
```

## Usage Examples

### Example 1: NSFW Influencer on Instagram

```python
from backend.models.persona import PersonaCreate, ContentRating

# Create persona that can post NSFW on Instagram
persona = PersonaCreate(
    name="NSFW Influencer Sarah",
    appearance="...",
    personality="...",
    allowed_content_ratings=[ContentRating.SFW, ContentRating.NSFW],
    platform_restrictions={
        "instagram": "both"  # Override: allow NSFW on Instagram
    }
)
```

Result: NSFW content allowed on Instagram for this persona only.

### Example 2: Family-Friendly Creator on OnlyFans

```python
# Create SFW-only creator even on adult platforms
persona = PersonaCreate(
    name="Family Content Creator",
    appearance="...",
    personality="...",
    allowed_content_ratings=[ContentRating.SFW],
    platform_restrictions={
        "onlyfans": "sfw_only",  # More restrictive than default
        "patreon": "sfw_only"
    }
)
```

Result: Only SFW content allowed on OnlyFans for this persona.

### Example 3: Mixed Content Strategy

```python
# Different strategies for different platforms
persona = PersonaCreate(
    name="Strategic Multi-Platform Creator",
    appearance="...",
    personality="...",
    allowed_content_ratings=[ContentRating.SFW, ContentRating.MODERATE, ContentRating.NSFW],
    platform_restrictions={
        "instagram": "both",           # Allow promotional NSFW
        "facebook": "moderate_allowed", # Artistic content only
        "twitter": "sfw_only"          # Keep it clean
    }
)
```

Result: Each platform has custom content policy for this persona.

## API Integration

### Creating Personas with Restrictions

```python
POST /api/v1/personas/
{
    "name": "NSFW Influencer",
    "appearance": "...",
    "personality": "...",
    "allowed_content_ratings": ["sfw", "nsfw"],
    "platform_restrictions": {
        "instagram": "both",
        "facebook": "sfw_only"
    }
}
```

### Updating Persona Restrictions

```python
PUT /api/v1/personas/{persona_id}
{
    "platform_restrictions": {
        "instagram": "both",
        "twitter": "moderate_allowed"
    }
}
```

### Content Generation with Platform Filtering

```python
from backend.services.content_generation_service import ContentGenerationService

service = ContentGenerationService(db_session)
result = await service.generate_content(GenerationRequest(
    persona_id=persona.id,
    content_type=ContentType.IMAGE,
    content_rating=ContentRating.NSFW,
    target_platforms=["instagram", "onlyfans"]
))

# Automatic filtering based on persona.platform_restrictions
# - Instagram: Checks persona restrictions, may allow NSFW
# - OnlyFans: Uses global policy, allows NSFW
```

## Testing

### Unit Tests

Located in `tests/unit/test_content_generation_enhancements.py`:

1. `test_platform_content_filter_with_persona_restrictions_sfw_only`
   - Tests SFW-only override on Instagram
   
2. `test_platform_content_filter_with_persona_restrictions_moderate_allowed`
   - Tests moderate-allowed override on Facebook
   
3. `test_platform_content_filter_with_persona_restrictions_all_content`
   - Tests full NSFW override on Instagram
   
4. `test_platform_content_filter_persona_restrictions_fallback`
   - Tests fallback to global policies for unspecified platforms
   
5. `test_nsfw_allowed_with_persona_override`
   - Integration test for NSFW on Instagram with override
   
6. `test_mixed_platform_restrictions`
   - Complex scenario with different restrictions per platform

### Running Tests

```bash
# Run all content generation tests
python -m pytest tests/unit/test_content_generation_enhancements.py -v

# Run specific test
python -m pytest tests/unit/test_content_generation_enhancements.py::TestContentModerationService::test_platform_content_filter_with_persona_restrictions_all_content -v

# Run validation script (no dependencies needed)
python validate_nsfw_filtering.py
```

### Example Output

```
example_per_site_nsfw_filtering.py - Documentation and usage examples
validate_nsfw_filtering.py - Standalone validation script
```

## Validation

A standalone validation script is provided that tests the implementation without requiring database setup:

```bash
python validate_nsfw_filtering.py
```

Expected output:
```
======================================================================
VALIDATING PER-SITE NSFW FILTERING IMPLEMENTATION
======================================================================

Test 1: Default global policies (no overrides)
----------------------------------------------------------------------
  ✓ PASS: SFW on Instagram (expected=True, got=True)
  ✓ PASS: NSFW blocked on Instagram (expected=False, got=False)
  ...

======================================================================
RESULTS: 16 passed, 0 failed
======================================================================

✓ VALIDATION SUCCESSFUL - All tests passed!
```

## Backward Compatibility

- **Fully backward compatible**: Existing personas without `platform_restrictions` continue to use global policies
- **Default behavior unchanged**: Empty or missing `platform_restrictions` falls back to global policies
- **No database migration required**: `platform_restrictions` field already exists in PersonaModel

## Security Considerations

1. **Authorization**: Ensure API endpoints validate that users have permission to set NSFW overrides
2. **Audit logging**: Log when persona restrictions override global policies
3. **Content moderation**: NSFW content should still go through standard moderation pipelines
4. **Platform compliance**: Overrides should only be used when legally and contractually permitted

## Future Enhancements

1. **Validation rules**: Add validation to ensure restriction values are valid
2. **Default restrictions**: Allow setting default restrictions at organization level
3. **Time-based restrictions**: Support temporary overrides (e.g., promotional periods)
4. **Analytics**: Track content distribution across platforms and ratings
5. **UI support**: Admin interface for managing persona restrictions

## References

- Issue: Enhancement - Allow nsfw generation with a per site filter
- PR: [Link to PR]
- Example: `example_per_site_nsfw_filtering.py`
- Validation: `validate_nsfw_filtering.py`
- Tests: `tests/unit/test_content_generation_enhancements.py`
