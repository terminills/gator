#!/usr/bin/env python3
"""
Example: Per-Site NSFW Filtering with Persona Overrides

This example demonstrates the new per-site NSFW filtering feature that allows
personas to have different content rating policies for different platforms.

NOTE: This is a documentation example. To run it, you need to:
1. Install dependencies: pip install -e .
2. Run from the repository root with: PYTHONPATH=src python example_per_site_nsfw_filtering.py
"""

# Conceptual example showing how the feature works
print("=" * 70)
print("Per-Site NSFW Filtering Example")
print("=" * 70)
print()

print("""
FEATURE OVERVIEW:
-----------------
The platform_restrictions field in PersonaModel allows per-persona, per-site
content filtering. This enables different personas to have different NSFW
policies on the same platform.

CONFIGURATION FORMAT:
---------------------
persona.platform_restrictions = {
    "instagram": "both",              # Allow all content (SFW, MODERATE, NSFW)
    "facebook": "moderate_allowed",   # Allow SFW and MODERATE
    "twitter": "sfw_only",            # Only allow SFW
}

SUPPORTED VALUES:
-----------------
- "sfw_only": Only SFW content allowed
- "moderate_allowed": SFW and MODERATE content allowed
- "both" or "all": All content types allowed (SFW, MODERATE, NSFW)

If a platform is not specified in restrictions, it uses the global default policy.

EXAMPLE USE CASES:
------------------

1. NSFW INFLUENCER ON INSTAGRAM
   Problem: Instagram normally blocks NSFW, but this persona has permission
   Solution: persona.platform_restrictions = {"instagram": "both"}
   Result: NSFW content allowed on Instagram for this persona only

2. FAMILY-FRIENDLY CREATOR ON ONLYFANS
   Problem: OnlyFans allows NSFW by default, but this creator is SFW-only
   Solution: persona.platform_restrictions = {"onlyfans": "sfw_only"}
   Result: Only SFW content allowed for this persona on OnlyFans

3. MIXED CONTENT STRATEGY
   Problem: Different content types for different platforms
   Solution: persona.platform_restrictions = {
       "instagram": "both",           # Allow promotional NSFW
       "facebook": "moderate_allowed", # Artistic content only
       "twitter": "sfw_only"          # Keep it clean
   }
   Result: Each platform has its own content policy for this persona

CODE INTEGRATION:
-----------------

# When creating a persona with custom platform restrictions:
from backend.models.persona import PersonaCreate, ContentRating

persona_data = PersonaCreate(
    name="NSFW Influencer Sarah",
    appearance="...",
    personality="...",
    allowed_content_ratings=[ContentRating.SFW, ContentRating.NSFW],
    platform_restrictions={
        "instagram": "both",  # Override: allow NSFW on Instagram
        "facebook": "sfw_only"  # More restrictive than default
    }
)

# When generating content:
from backend.services.content_generation_service import ContentGenerationService

service = ContentGenerationService(db_session)
result = await service.generate_content(GenerationRequest(
    persona_id=persona.id,
    content_type=ContentType.IMAGE,
    content_rating=ContentRating.NSFW,
    target_platforms=["instagram", "onlyfans"]
))

# The service will automatically:
# 1. Check persona.platform_restrictions for Instagram
# 2. See "both" allows NSFW
# 3. Allow content on Instagram (normally blocked)
# 4. Allow content on OnlyFans (global policy)

TESTING:
--------
See tests/unit/test_content_generation_enhancements.py for comprehensive tests:
- test_platform_content_filter_with_persona_restrictions_sfw_only
- test_platform_content_filter_with_persona_restrictions_moderate_allowed  
- test_platform_content_filter_with_persona_restrictions_all_content
- test_nsfw_allowed_with_persona_override
- test_mixed_platform_restrictions

""")

print("=" * 70)
print("Implementation is complete and tested!")
print("=" * 70)
