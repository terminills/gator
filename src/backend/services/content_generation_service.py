"""
Content Generation Service

Handles AI-powered content generation including image and text creation
using integrated AI models like Stable Diffusion and language models.
"""

import asyncio
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config.logging import get_logger
from backend.models.acd import ACDContextModel, AIComplexity, AIConfidence, AIState
from backend.models.content import (
    ContentModel,
    ContentRating,
    ContentResponse,
    ContentType,
    GenerationRequest,
    ModerationStatus,
)
from backend.models.persona import PersonaModel
from backend.utils.paths import get_paths

# --- MODULE-LEVEL IMPORTS TO PREVENT SCOPED FAILURE ---
# These imports are moved up from within the generation methods to ensure
# that any ImportError due to missing dependencies manifests immediately
# at application startup, rather than being silently caught and masked as
# a generic content generation failure at runtime.
from backend.services.ai_models import ai_models
from backend.services.prompt_generation_service import get_prompt_service
from backend.services.template_service import TemplateService
from backend.services.video_processing_service import (
    VideoQuality,
)
from backend.utils.acd_integration import ACDContextManager

# --------------------------------------------------------

logger = get_logger(__name__)


class ContentModerationService:
    """Service for content moderation and rating classification."""

    @staticmethod
    async def is_nsfw_filter_disabled(
        content_type: Optional[str] = None,
        db_session=None,
    ) -> bool:
        """
        Check if NSFW filtering is disabled for a content type.

        For private server mode, NSFW filtering is disabled by default.

        Args:
            content_type: Optional content type (text, chat, voice, image, video)
            db_session: Optional database session for settings lookup

        Returns:
            bool: True if NSFW filtering is disabled, False otherwise
        """
        # If we have a database session, check the settings
        if db_session:
            try:
                from backend.services.settings_service import SettingsService

                settings_service = SettingsService(db_session)

                # First check global setting
                global_setting = await settings_service.get_setting(
                    "nsfw_filter_disabled_global"
                )
                if global_setting and global_setting.value:
                    return True

                # Check content-type specific setting
                if content_type:
                    content_type_key = f"nsfw_filter_disabled_{content_type.lower()}"
                    type_setting = await settings_service.get_setting(content_type_key)
                    if type_setting and type_setting.value:
                        return True

            except Exception as e:
                logger.warning(f"Error checking NSFW filter settings: {e}")
                # Default to disabled for private server
                return True

        # Default: NSFW filtering is disabled for private server
        return True

    @staticmethod
    def analyze_content_rating(prompt: str, persona_rating: str) -> ContentRating:
        """
        Analyze content to determine appropriate rating.

        This is informational - helps tag content appropriately for platform filtering.
        Does NOT block content generation. All ratings can be generated.

        Note: This is a placeholder - in production, this would use ML models.
        """
        # If no prompt provided, return SFW as default
        if prompt is None or not prompt:
            return ContentRating.SFW

        # Simple keyword-based analysis
        nsfw_keywords = [
            "sexy",
            "nude",
            "naked",
            "adult",
            "erotic",
            "sensual",
            "lingerie",
            "bikini",
            "provocative",
            "intimate",
        ]

        prompt_lower = prompt.lower()

        # If persona allows NSFW and prompt contains NSFW keywords
        if persona_rating in ["nsfw", "both"] and any(
            keyword in prompt_lower for keyword in nsfw_keywords
        ):
            return ContentRating.NSFW
        elif any(
            keyword in prompt_lower for keyword in ["suggestive", "flirty", "romantic"]
        ):
            return ContentRating.MODERATE

        return ContentRating.SFW

    @staticmethod
    async def platform_content_filter(
        content_rating: ContentRating,
        target_platform: str,
        persona_platform_restrictions: Optional[Dict[str, str]] = None,
        platform_policy_service=None,
        db_session=None,
    ) -> bool:
        """
        Check if content is appropriate for target platform.

        Uses database-driven platform policies instead of hardcoded rules.
        This allows platform rules to be updated dynamically without code changes.

        For private server mode, NSFW filtering is disabled by default.

        Args:
            content_rating: The content rating to check (SFW, MODERATE, NSFW)
            target_platform: The target platform name (e.g., "instagram", "onlyfans")
            persona_platform_restrictions: Optional persona-specific platform restrictions.
                Format: {"instagram": "sfw_only", "onlyfans": "both", "twitter": "moderate_allowed"}
                Supported values: "sfw_only", "moderate_allowed", "both" (all ratings)
            platform_policy_service: Optional PlatformPolicyService instance for database lookup
            db_session: Optional database session for checking NSFW filter settings

        Returns:
            bool: True if content is allowed, False otherwise
        """
        # Check if NSFW filtering is disabled globally (private server mode)
        if await ContentModerationService.is_nsfw_filter_disabled(
            db_session=db_session
        ):
            # All content is allowed when NSFW filtering is disabled
            return True

        platform_lower = target_platform.lower()

        # Check persona-specific restrictions first (per-site override)
        if (
            persona_platform_restrictions
            and platform_lower in persona_platform_restrictions
        ):
            restriction = persona_platform_restrictions[platform_lower].lower()

            if restriction == "sfw_only":
                return content_rating == ContentRating.SFW
            elif restriction == "moderate_allowed":
                return content_rating in [ContentRating.SFW, ContentRating.MODERATE]
            elif restriction == "both" or restriction == "all":
                # Allow all content types for this persona on this platform
                return True
            # If unrecognized restriction, fall through to platform policies

        # Use platform policy service if provided
        if platform_policy_service:
            try:
                return await platform_policy_service.check_content_allowed(
                    platform_lower, content_rating
                )
            except Exception as e:
                logger.warning(
                    f"Failed to check platform policy for {platform_lower}: {e}. "
                    f"Falling back to safe default."
                )
                # Fallback to safe default if database lookup fails
                return content_rating == ContentRating.SFW

        # Fallback: if no platform policy service provided, default to safe
        # This should not happen in production but provides backwards compatibility
        logger.warning(
            f"No platform policy service provided for {platform_lower}. "
            f"Defaulting to SFW-only. Please initialize platform policies."
        )
        return content_rating == ContentRating.SFW


class ServiceGenerationRequest(BaseModel):
    """Request for content generation - service-specific version with required persona_id."""

    persona_id: UUID
    content_type: ContentType  # 'image', 'video', 'audio', 'voice', 'text'
    content_rating: Optional[ContentRating] = (
        None  # Use persona's default if not specified
    )
    prompt: Optional[str] = None
    style_override: Optional[Dict[str, Any]] = None
    quality: str = "high"  # 'draft', 'standard', 'high', 'premium'
    target_platforms: Optional[List[str]] = None


class ContentGenerationService:
    """
    Service for AI-powered content generation.

    Integrates with AI models to generate images, videos, and text content
    based on persona configurations and current trends.
    """

    def __init__(
        self, db_session: AsyncSession, content_dir: Optional[str] = None
    ):
        """
        Initialize content generation service.

        Args:
            db_session: Database session for persistence
            content_dir: Directory to store generated content files (uses centralized path if not provided)
        """
        self.db = db_session

        # Use centralized paths if not explicitly provided
        if content_dir is None:
            paths = get_paths()
            self.content_dir = paths.generated_content_dir
        else:
            self.content_dir = Path(content_dir)

        self.content_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories for different content types
        (self.content_dir / "images").mkdir(exist_ok=True)
        (self.content_dir / "videos").mkdir(exist_ok=True)
        (self.content_dir / "audio").mkdir(exist_ok=True)
        (self.content_dir / "voice").mkdir(exist_ok=True)
        (self.content_dir / "text").mkdir(exist_ok=True)

        self.moderation_service = ContentModerationService()
        self.template_service = TemplateService()

        # Import here to avoid circular dependency
        from backend.services.platform_policy_service import PlatformPolicyService

        self.platform_policy_service = PlatformPolicyService(db_session)

    async def generate_content(self, request: GenerationRequest) -> ContentResponse:
        """
        Generate content based on request parameters.

        Args:
            request: Content generation request

        Returns:
            ContentResponse: Generated content metadata

        Raises:
            ValueError: If persona not found or generation fails
        """
        logger.info("=" * 80)
        logger.info("🚀 CONTENT GENERATION REQUEST RECEIVED")
        logger.info(f"   Content type: {request.content_type.value}")
        logger.info(f"   Quality: {request.quality}")
        logger.info(
            f"   Rating: {request.content_rating.value if request.content_rating else 'use persona default'}"
        )

        try:
            # Get persona data - if no persona_id provided, use first available persona
            if request.persona_id is None:
                logger.info("   No persona specified, searching for default...")
                persona = await self._get_first_persona()
                if not persona:
                    logger.error("   ❌ No personas available in database")
                    raise ValueError(
                        "No personas available. Please create a persona first."
                    )
                request.persona_id = persona.id
                logger.info(
                    f"   ✓ Using default persona: {persona.name} ({persona.id})"
                )
            else:
                logger.info(f"   Loading persona: {request.persona_id}")
                persona = await self._get_persona(request.persona_id)
                if not persona:
                    logger.error(f"   ❌ Persona not found: {request.persona_id}")
                    raise ValueError(f"Persona not found: {request.persona_id}")
                logger.info(f"   ✓ Persona loaded: {persona.name}")

            # Use persona's default content rating if not specified
            if request.content_rating is None:
                logger.info(
                    "   No content_rating in request, using persona default..."
                )
                logger.info(
                    f"   Persona default_content_rating: {persona.default_content_rating}"
                )
                logger.info(
                    f"   Persona allowed_content_ratings: {getattr(persona, 'allowed_content_ratings', [])}"
                )

                # Use persona's default, or randomly select from allowed ratings if no default is set
                persona_rating = persona.default_content_rating
                if not persona_rating:
                    # If no default, randomly select from allowed ratings for variety
                    allowed_ratings = getattr(persona, "allowed_content_ratings", [])
                    if allowed_ratings:
                        persona_rating = random.choice(allowed_ratings)
                        logger.info(
                            f"   No default set, randomly selected from allowed: {persona_rating}"
                        )
                    else:
                        # Last resort: randomly pick from all available ratings
                        # This should never happen if persona is properly configured
                        persona_rating = random.choice([r.value for r in ContentRating])
                        logger.warning(
                            f"Persona {persona.name} has no content ratings configured. "
                            f"Randomly selected: {persona_rating}"
                        )
                request.content_rating = ContentRating(persona_rating)
                logger.info(
                    f"   ✓ Using content rating from persona: {request.content_rating.value}"
                )
            else:
                logger.info(
                    f"   Using explicit content_rating from request: {request.content_rating.value}"
                )

            # Generate prompt if not provided
            # Note: For IMAGE type, we skip this and let the image generation
            # service use the advanced prompt generation service instead
            if not request.prompt and request.content_type != ContentType.IMAGE:
                logger.info("   Generating AI prompt based on persona...")
                request.prompt = await self._generate_prompt(
                    persona, request.content_type, request.content_rating
                )
                logger.info(f"   ✓ Generated prompt: {request.prompt[:80]}...")

            # Note: We do NOT validate content rating against persona settings here
            # Content can be generated with any rating - the platform filtering will
            # determine which social media sites it can be posted to
            # The persona's allowed_content_ratings is only used to select a preferred
            # rating when none is specified, not to block generation

            # Analyze and adjust content rating based on prompt
            analyzed_rating = self.moderation_service.analyze_content_rating(
                request.prompt, persona.default_content_rating
            )
            if analyzed_rating != request.content_rating:
                logger.info(
                    f"   📊 Content rating adjusted from {request.content_rating} to {analyzed_rating}"
                )
                request.content_rating = analyzed_rating

            logger.info("-" * 80)
            # Generate content based on type
            if request.content_type == ContentType.IMAGE:
                content_data = await self._generate_image(persona, request)
            elif request.content_type == ContentType.VIDEO:
                content_data = await self._generate_video(persona, request)
            elif request.content_type == ContentType.AUDIO:
                content_data = await self._generate_audio(persona, request)
            elif request.content_type == ContentType.VOICE:
                content_data = await self._generate_voice(persona, request)
            elif request.content_type == ContentType.TEXT:
                content_data = await self._generate_text(persona, request)
            else:
                raise ValueError(f"Unsupported content type: {request.content_type}")

            logger.info("-" * 80)
            logger.info("   💾 Saving content record to database...")

            # Apply platform-specific adaptations
            platform_adaptations = await self._create_platform_adaptations(
                persona,
                content_data,
                request.content_rating,
                request.target_platforms or [],
            )

            if request.target_platforms:
                logger.info(
                    f"   ✓ Platform adaptations: {', '.join(request.target_platforms)}"
                )

            # Store content metadata in database
            content_record = await self._save_content_record(
                persona, request, content_data, platform_adaptations
            )
            logger.info(f"   ✓ Content saved with ID: {content_record.id}")

            # Update persona generation count
            await self._increment_persona_count(persona.id)

            logger.info("=" * 80)
            logger.info("✅ CONTENT GENERATION COMPLETE")
            logger.info(f"   Content ID: {content_record.id}")
            logger.info(f"   Type: {request.content_type.value}")
            logger.info(f"   Persona: {persona.name}")
            logger.info("=" * 80)

            return content_record

        except Exception as e:
            logger.error("=" * 80)
            logger.error("❌ CONTENT GENERATION FAILED")
            logger.error(f"   Error: {str(e)}")
            logger.error(f"   Persona ID: {request.persona_id}")
            logger.error(f"   Content type: {request.content_type}")
            logger.error("=" * 80)
            raise ValueError(f"Content generation failed: {str(e)}")

    async def generate_content_for_all_personas(
        self,
        content_type: ContentType = ContentType.IMAGE,
        quality: Optional[str] = "standard",
        content_rating: Optional[ContentRating] = None,  # Use persona defaults
    ) -> Dict[str, Any]:
        """
        Generate content for all active personas.

        Args:
            content_type: Type of content to generate
            quality: Quality level (standard or hd), None to use persona defaults
            content_rating: Content rating filter, None to use persona defaults

        Returns:
            Dict with generation results and statistics
        """
        logger.info("=" * 80)
        logger.info("🚀 BATCH CONTENT GENERATION FOR ALL PERSONAS")
        logger.info(f"   Content type: {content_type.value}")
        logger.info(
            f"   Quality: {quality if quality is not None else 'persona defaults'}"
        )
        logger.info(
            f"   Rating: {content_rating.value if content_rating is not None else 'persona defaults'}"
        )
        logger.info("=" * 80)

        # Get all active personas
        stmt = (
            select(PersonaModel)
            .where(PersonaModel.is_active.is_(True))
            .order_by(PersonaModel.created_at.asc())
        )
        result = await self.db.execute(stmt)
        personas = result.scalars().all()

        if not personas:
            logger.warning("No active personas found for batch generation")
            return {
                "status": "error",
                "message": "No active personas found",
                "generated": 0,
                "failed": 0,
                "results": [],
            }

        logger.info(f"Found {len(personas)} active personas")

        # Generate content for each persona
        results = []
        generated_count = 0
        failed_count = 0

        for persona in personas:
            try:
                logger.info(
                    f"Generating content for persona: {persona.name} ({persona.id})"
                )

                # Use provided values or fall back to persona defaults
                effective_quality = (
                    quality
                    if quality is not None
                    else (
                        persona.image_quality
                        if hasattr(persona, "image_quality")
                        else "standard"
                    )
                )
                effective_rating = (
                    content_rating
                    if content_rating is not None
                    else (
                        ContentRating(persona.default_content_rating)
                        if hasattr(persona, "default_content_rating")
                        and persona.default_content_rating
                        else ContentRating.SFW
                    )
                )

                request = GenerationRequest(
                    persona_id=persona.id,
                    content_type=content_type,
                    quality=effective_quality,
                    content_rating=effective_rating,
                    prompt=None,  # Will be auto-generated
                )

                content = await self.generate_content(request)

                results.append(
                    {
                        "persona_id": str(persona.id),
                        "persona_name": persona.name,
                        "content_id": str(content.id),
                        "status": "success",
                    }
                )
                generated_count += 1
                logger.info(f"✓ Generated content for {persona.name}")

            except Exception as e:
                logger.error(
                    f"✗ Failed to generate content for {persona.name}: {str(e)}"
                )
                results.append(
                    {
                        "persona_id": str(persona.id),
                        "persona_name": persona.name,
                        "status": "failed",
                        "error": str(e),
                    }
                )
                failed_count += 1

        logger.info("=" * 80)
        logger.info("✅ BATCH CONTENT GENERATION COMPLETE")
        logger.info(f"   Total personas: {len(personas)}")
        logger.info(f"   Generated: {generated_count}")
        logger.info(f"   Failed: {failed_count}")
        logger.info("=" * 80)

        return {
            "status": "completed",
            "total_personas": len(personas),
            "generated": generated_count,
            "failed": failed_count,
            "results": results,
        }

    async def get_content(self, content_id: UUID) -> Optional[ContentResponse]:
        """Get content record by ID."""
        try:
            stmt = select(ContentModel).where(ContentModel.id == content_id)
            result = await self.db.execute(stmt)
            content = result.scalar_one_or_none()

            if content:
                return ContentResponse.model_validate(content)
            return None

        except Exception as e:
            logger.error(
                f"Error retrieving content error={str(e)} content_id={content_id}"
            )
            return None

    async def list_persona_content(
        self, persona_id: UUID, limit: int = 50
    ) -> List[ContentResponse]:
        """List content generated for a specific persona."""
        try:
            stmt = (
                select(ContentModel)
                .where(ContentModel.persona_id == persona_id)
                .where(ContentModel.is_deleted.is_(False))
                .order_by(ContentModel.created_at.desc())
                .limit(limit)
            )

            result = await self.db.execute(stmt)
            contents = result.scalars().all()

            return [ContentResponse.model_validate(content) for content in contents]

        except Exception as e:
            logger.error(
                f"Error listing persona content error={str(e)} persona_id={persona_id}"
            )
            return []

    async def list_all_content(
        self, limit: int = 50, offset: int = 0
    ) -> List[ContentResponse]:
        """List all generated content across all personas."""
        try:
            stmt = (
                select(ContentModel)
                .where(ContentModel.is_deleted.is_(False))
                .order_by(ContentModel.created_at.desc())
                .limit(limit)
                .offset(offset)
            )

            result = await self.db.execute(stmt)
            contents = result.scalars().all()

            return [ContentResponse.model_validate(content) for content in contents]

        except Exception as e:
            logger.error(f"Error listing all content error={str(e)}")
            return []

    async def _get_persona(self, persona_id: UUID) -> Optional[PersonaModel]:
        """Retrieve persona from database."""
        stmt = select(PersonaModel).where(
            PersonaModel.id == persona_id, PersonaModel.is_active.is_(True)
        )
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    async def _get_first_persona(self) -> Optional[PersonaModel]:
        """Retrieve first available active persona from database."""
        stmt = (
            select(PersonaModel)
            .where(PersonaModel.is_active.is_(True))
            .order_by(PersonaModel.created_at.asc())
            .limit(1)
        )
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    async def _get_trending_topics_from_feeds(
        self, persona: PersonaModel, limit: int = 3
    ) -> List[str]:
        """
        Fetch trending topics from RSS feeds assigned to persona.

        Args:
            persona: Persona to fetch feeds for
            limit: Maximum number of topics to return

        Returns:
            List of trending topic strings
        """
        try:
            from backend.models.feed import (
                FeedItemModel,
                PersonaFeedModel,
            )

            # Get RSS feeds assigned to this persona
            stmt = (
                select(PersonaFeedModel)
                .where(PersonaFeedModel.persona_id == persona.id)
                .where(PersonaFeedModel.is_active.is_(True))
                .limit(5)
            )
            result = await self.db.execute(stmt)
            persona_feeds = result.scalars().all()

            if not persona_feeds:
                logger.info(f"No RSS feeds assigned to persona {persona.name}")
                return []

            # Get recent feed items from the last 24 hours
            feed_ids = [pf.feed_id for pf in persona_feeds]
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=24)

            stmt = (
                select(FeedItemModel)
                .where(FeedItemModel.feed_id.in_(feed_ids))
                .where(FeedItemModel.published_date >= cutoff_time)
                .order_by(FeedItemModel.published_date.desc())
                .limit(limit * 2)  # Get more to filter
            )
            result = await self.db.execute(stmt)
            feed_items = result.scalars().all()

            if not feed_items:
                logger.info(f"No recent feed items for persona {persona.name}")
                return []

            # Extract topics from titles
            topics = []
            for item in feed_items[:limit]:
                # Clean and shorten title for use in prompt
                topic = item.title[:100] if item.title else ""
                if topic:
                    topics.append(topic)

            logger.info(
                f"Found {len(topics)} trending topics from RSS feeds for persona {persona.name}"
            )
            return topics

        except Exception as e:
            logger.warning(f"Failed to fetch trending topics from RSS feeds: {str(e)}")
            return []

    async def _generate_prompt(
        self,
        persona: PersonaModel,
        content_type: ContentType,
        content_rating: ContentRating,
    ) -> str:
        """
        Generate AI prompt based on persona characteristics and content rating.
        Uses base_appearance_description if appearance_locked is True for consistency.
        Enriched with trending topics from RSS feeds when available.
        """
        # Fetch trending topics from RSS feeds
        trending_topics = await self._get_trending_topics_from_feeds(persona)
        trending_context = ""
        if trending_topics:
            # Add trending context to prompt
            topics_text = ", ".join(trending_topics[:2])  # Use top 2 topics
            trending_context = f"related to current trending topics: {topics_text}, "
            logger.info(f"Enriching prompt with trending topics: {topics_text[:100]}")

        # Use base appearance if locked, otherwise use standard appearance
        if persona.appearance_locked and persona.base_appearance_description:
            base_prompt = (
                f"{persona.base_appearance_description}, {persona.personality}"
            )
            logger.info(f"Using locked base appearance for persona {persona.id}")
        else:
            base_prompt = f"{persona.appearance}, {persona.personality}"

        # Add content rating modifiers
        rating_modifiers = {
            ContentRating.SFW: "safe for work, family-friendly, appropriate for all audiences",
            ContentRating.MODERATE: "tasteful, artistic, suitable for mature audiences",
            ContentRating.NSFW: "adult content, explicit, 18+ only",
        }

        rating_modifier = rating_modifiers.get(content_rating, "")

        if content_type == ContentType.IMAGE:
            style_info = persona.style_preferences.get("visual_style", "realistic")
            lighting = persona.style_preferences.get("lighting", "natural")
            # Enhanced photorealistic prompt with detailed appearance and trending context
            prompt = f"Professional high-resolution portrait photograph of {base_prompt}, {trending_context}{style_info} style, {lighting} lighting, photorealistic, ultra detailed, 8k quality, sharp focus, professional photography, {rating_modifier}"
        elif content_type == ContentType.VIDEO:
            prompt = f"Short video featuring {base_prompt}, engaging and dynamic, {rating_modifier}"
        elif content_type == ContentType.AUDIO:
            prompt = f"Audio content with {base_prompt}, engaging background music, {rating_modifier}"
        elif content_type == ContentType.VOICE:
            voice_style = persona.style_preferences.get("voice_style", "natural")
            prompt = f"Voice narration by {base_prompt}, {voice_style} tone, clear speech, {rating_modifier}"
        else:  # text
            themes = (
                ", ".join(persona.content_themes[:3])
                if persona.content_themes
                else "general topics"
            )
            prompt = f"Write engaging social media content about {themes} in the style of {persona.personality}, {rating_modifier}"

        return prompt

    async def _check_content_rating_preference(
        self, persona: PersonaModel, requested_rating: ContentRating
    ) -> bool:
        """
        Check if the requested content rating matches persona's preferences.

        NOTE: This is informational only - it does NOT block content generation.
        Content can be generated with any rating. The persona's allowed_content_ratings
        indicates which ratings the persona PREFERS to generate, not which are blocked.
        Platform filtering will determine where content can be posted.

        Returns:
            True if rating matches persona preferences, False if outside preferences
        """
        # Get allowed ratings from persona, default to ['sfw'] if empty
        allowed_ratings = getattr(persona, "allowed_content_ratings", ["sfw"])
        if not allowed_ratings:  # If empty list, default to sfw
            allowed_ratings = ["sfw"]

        if isinstance(allowed_ratings, str):
            allowed_ratings = [allowed_ratings]

        # Ensure the persona's default_content_rating is always in allowed list
        # This handles cases where database has inconsistent data
        default_rating = getattr(persona, "default_content_rating", "sfw")
        if default_rating and default_rating.lower() not in [
            r.lower() for r in allowed_ratings
        ]:
            # Add default to allowed list to fix inconsistency
            allowed_ratings.append(default_rating)
            logger.debug(
                f"Persona {persona.name} has inconsistent rating config: default '{default_rating}' "
                f"not in allowed {allowed_ratings}. This is informational only."
            )

        # Check if requested rating matches preferences
        return requested_rating.value in [r.lower() for r in allowed_ratings]

    async def _generate_image(
        self, persona: PersonaModel, request: GenerationRequest
    ) -> Dict[str, Any]:
        """
        Generate image using AI model with ACD context tracking.

        Integrated with real AI models including OpenAI DALL-E and Stable Diffusion.
        Uses base_image_path for visual consistency when appearance_locked is True.
        Tracks generation context via ACD for learning and debugging.
        """
        # Log the generation attempt
        logger.info(
            f"Starting image generation for persona {persona.name} ({persona.id})",
            extra={
                "persona_id": str(persona.id),
                "persona_name": persona.name,
                "prompt": request.prompt,
                "quality": request.quality,
                "content_rating": request.content_rating.value,
                "appearance_locked": persona.appearance_locked,
            },
        )

        # Determine complexity based on quality and locked appearance
        complexity = AIComplexity.LOW
        if request.quality == "hd":
            complexity = AIComplexity.HIGH
        elif persona.appearance_locked:
            complexity = AIComplexity.MEDIUM

        # Create ACD context for tracking
        initial_context = {
            "prompt": request.prompt,
            "persona_id": str(persona.id),
            "quality": request.quality,
            "content_rating": request.content_rating.value,
            "appearance_locked": persona.appearance_locked,
        }

        # Note: content_id will be None here since content doesn't exist yet
        # We'll link it after content creation
        async with ACDContextManager(
            self.db,
            phase="IMAGE_GENERATION",
            note=f"Generating image for persona {persona.name}",
            complexity=complexity,
            initial_context=initial_context,
        ) as acd:
            try:
                # Ensure AI models are initialized
                if not ai_models.models_loaded:
                    await ai_models.initialize_models()

                # Use persona's generation settings instead of hardcoded values
                # Quality: use persona's generation_quality if not overridden by request
                quality = (
                    request.quality if request.quality else persona.generation_quality
                )
                if quality not in ["draft", "standard", "hd", "premium"]:
                    quality = persona.generation_quality or "standard"

                # Resolution: use persona's default_image_resolution
                size = persona.default_image_resolution or "1024x1024"
                # Override with higher resolution for HD quality if not explicitly set
                if quality == "hd" and size == "1024x1024":
                    size = "2048x2048"
                elif quality == "premium":
                    size = "2048x2048"

                logger.info(
                    f"   Image generation: quality={quality}, resolution={size}"
                )
                logger.info(
                    f"   Persona defaults: quality={persona.generation_quality}, resolution={persona.default_image_resolution}"
                )

                # Generate enhanced prompt using AI (llama.cpp) or templates
                # This creates detailed prompts that can exceed 77 tokens when using SDXL
                # Pass database session to enable RSS content integration
                prompt_service = get_prompt_service(db_session=self.db)
                prompt_data = await prompt_service.generate_image_prompt(
                    persona=persona,
                    context=request.prompt,  # User's request becomes context
                    content_rating=request.content_rating,
                    rss_content=None,  # Will be auto-fetched from DB if available
                    image_style=persona.image_style,
                    use_ai=True,  # Enable AI-powered prompt generation
                )

                logger.info(
                    f"Generated prompt ({prompt_data['word_count']} words, source: {prompt_data['source']})"
                )
                logger.info(f"Prompt preview: {prompt_data['prompt'][:150]}...")

                generation_params = {
                    "prompt": prompt_data["prompt"],
                    "negative_prompt": prompt_data["negative_prompt"],
                    "size": size,
                    "quality": quality,
                }

                # Add persona's image model preference if set
                # This takes highest priority for model selection
                if persona.image_model_preference:
                    generation_params["image_model_pref"] = (
                        persona.image_model_preference
                    )
                    logger.info(
                        f"   Using persona's preferred image model: {persona.image_model_preference}"
                    )

                # Add NSFW model preference if applicable
                if (
                    request.content_rating == ContentRating.NSFW
                    and persona.nsfw_model_preference
                ):
                    generation_params["nsfw_model"] = persona.nsfw_model_preference
                    logger.info(f"   Using NSFW model: {persona.nsfw_model_preference}")

                # Add visual consistency parameters if appearance is locked
                if persona.appearance_locked and persona.base_image_path:
                    generation_params["reference_image_path"] = persona.base_image_path
                    generation_params["use_controlnet"] = True

                    # Check if RSS content was used for reaction prompt
                    rss_used = (
                        "rss" in prompt_data.get("prompt", "").lower()
                        or prompt_data.get("word_count", 0) > 50
                    )

                    logger.info(
                        f"✓ Using base image for visual consistency: {persona.base_image_path}"
                    )
                    if rss_used:
                        logger.info(
                            "✓ Base image will be used with RSS-inspired reaction prompt"
                        )

                    await acd.set_metadata(
                        {
                            **initial_context,
                            "using_reference": True,
                            "reference_path": persona.base_image_path,
                            "prompt_source": prompt_data["source"],
                            "prompt_word_count": prompt_data["word_count"],
                            "rss_reaction": rss_used,
                        }
                    )

                # Generate image using AI model with enhanced prompt
                image_result = await ai_models.generate_image(**generation_params)

                # Validate image result
                if not image_result:
                    raise ValueError("Image generation returned None result")

                if "image_data" not in image_result:
                    raise ValueError(
                        f"Image generation returned invalid result: missing 'image_data' key. Keys present: {list(image_result.keys())}"
                    )

                if not image_result["image_data"]:
                    raise ValueError("Image generation returned empty image_data")

                # Save the generated image
                filename = f"image_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                file_path = self.content_dir / "images" / filename

                # Write image data to file
                with open(file_path, "wb") as f:
                    f.write(image_result["image_data"])

                # Store relative path for web serving (relative to /content mount point)
                relative_path = f"images/{filename}"

                result_data = {
                    "file_path": relative_path,
                    "file_size": len(image_result["image_data"]),
                    "width": image_result.get("width", 1024),
                    "height": image_result.get("height", 1024),
                    "format": image_result.get("format", "PNG"),
                    "content_rating": request.content_rating.value,
                    "model": image_result.get("model", "unknown"),
                    "provider": image_result.get("provider", "unknown"),
                    "acd_context_id": acd.context_id,  # Link to ACD context
                }

                # Add benchmark data for feedback tracking
                if "benchmark_data" in image_result:
                    result_data["benchmark_data"] = image_result["benchmark_data"]
                    result_data["generation_time_seconds"] = image_result.get(
                        "generation_time_seconds"
                    )
                    result_data["total_time_seconds"] = image_result.get(
                        "total_time_seconds"
                    )

                # Update ACD with successful generation details
                await acd.set_confidence(AIConfidence.CONFIDENT)
                await acd.set_metadata(
                    {
                        **initial_context,
                        "model_used": image_result.get("model", "unknown"),
                        "provider": image_result.get("provider", "unknown"),
                        "file_size": len(image_result["image_data"]),
                        "success": True,
                    }
                )

                return result_data

            except Exception as e:
                # Mark ACD as failed
                await acd.set_confidence(AIConfidence.UNCERTAIN)
                await acd.set_state(AIState.FAILED)
                await acd.set_metadata(
                    {
                        **initial_context,
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "failed": True,
                    }
                )

                # Log the failure comprehensively
                logger.error(
                    f"Image generation failed for persona {persona.id}: {str(e)}",
                    extra={
                        "persona_id": str(persona.id),
                        "persona_name": persona.name,
                        "prompt": request.prompt,
                        "quality": request.quality,
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "acd_context_id": str(acd.context_id),
                    },
                )

                # Re-raise the exception instead of creating placeholder
                raise ValueError(f"Image generation failed: {str(e)}") from e

    async def _generate_video(
        self, persona: PersonaModel, request: GenerationRequest
    ) -> Dict[str, Any]:
        """
        Generate video using AI model with advanced features.

        This implementation supports:
        - Frame-by-frame generation for longer videos
        - Audio synchronization with voice synthesis
        - Video editing and transitions
        - Scene composition and storyboarding

        Q2-Q3 2025 advanced video features implementation.
        """
        # Log the generation attempt
        logger.info(
            f"Starting video generation for persona {persona.name} ({persona.id})",
            extra={
                "persona_id": str(persona.id),
                "persona_name": persona.name,
                "prompt": request.prompt,
                "quality": request.quality,
                "content_rating": request.content_rating.value,
            },
        )

        try:
            # Ensure AI models are initialized
            if not ai_models.models_loaded:
                await ai_models.initialize_models()

            # Get video generation parameters from persona settings
            quality = request.quality or persona.generation_quality or "standard"
            VideoQuality(quality)

            # Use persona's default video resolution
            resolution = persona.default_video_resolution or "1920x1080"
            logger.info(
                f"   Video generation: quality={quality}, resolution={resolution}"
            )
            logger.info(f"   Persona video preferences: {persona.video_types}")

            # Check if this is a multi-scene video (storyboard)
            if request.style_override and "scenes" in request.style_override:
                # Storyboard generation
                scenes = request.style_override["scenes"]
                logger.info(f"Generating storyboard with {len(scenes)} scenes")

                video_result = await ai_models.create_video_storyboard(
                    scenes=scenes, quality=quality, resolution=resolution
                )

            elif request.style_override and "prompts" in request.style_override:
                # Multi-frame generation
                prompts = request.style_override["prompts"]
                transition = request.style_override.get("transition", "crossfade")
                duration_per_frame = request.style_override.get(
                    "duration_per_frame", 3.0
                )

                logger.info(f"Generating multi-frame video with {len(prompts)} frames")

                video_result = await ai_models.generate_video(
                    prompt=prompts,
                    video_type="multi_frame",
                    quality=quality,
                    resolution=resolution,
                    transition=transition,
                    duration_per_frame=duration_per_frame,
                )

            else:
                # Single frame video generation
                logger.info(f"Generating single-frame video: {request.prompt[:50]}...")

                video_result = await ai_models.generate_video(
                    prompt=request.prompt,
                    video_type="single_frame",
                    quality=quality,
                    resolution=resolution,
                    duration_per_frame=(
                        request.style_override.get("duration", 4.0)
                        if request.style_override
                        else 4.0
                    ),
                )

            # If audio sync is requested, add audio track
            if request.style_override and "audio_path" in request.style_override:
                audio_path = request.style_override["audio_path"]
                logger.info(f"Synchronizing audio: {audio_path}")

                video_result = await ai_models.synchronize_audio_to_video(
                    video_path=video_result["file_path"], audio_path=audio_path
                )

            # Save the video file to content directory
            filename = f"video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            file_path = self.content_dir / "videos" / filename

            # Move/copy generated video to content directory
            import shutil

            if not Path(video_result["file_path"]).exists():
                # Fail if video file doesn't exist - no placeholders!
                raise ValueError(
                    f"Video generation failed: output file not found at {video_result['file_path']}"
                )

            shutil.copy2(video_result["file_path"], file_path)

            return {
                "file_path": str(file_path),
                "file_size": file_path.stat().st_size,
                "duration": video_result.get("duration", 15.0),
                "resolution": video_result.get("resolution", resolution),
                "format": video_result.get("format", "MP4"),
                "quality": quality,
                "content_rating": request.content_rating.value,
                "fps": video_result.get("fps", 30),
                "has_audio": video_result.get("has_audio", False),
                "num_scenes": video_result.get("num_scenes", 1),
                "transition_type": video_result.get("transition_type"),
                "model": video_result.get("model", "frame-by-frame-generator"),
            }

        except Exception as e:
            # Log the failure comprehensively
            logger.error(
                f"Video generation failed for persona {persona.id}: {str(e)}",
                extra={
                    "persona_id": str(persona.id),
                    "persona_name": persona.name,
                    "prompt": request.prompt,
                    "quality": request.quality,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )

            # Re-raise the exception instead of creating placeholder
            raise ValueError(f"Video generation failed: {str(e)}") from e

    async def _generate_audio(
        self, persona: PersonaModel, request: GenerationRequest
    ) -> Dict[str, Any]:
        """
        Generate audio content using AI model.

        Requires integration with audio generation models like MusicLM or AudioCraft.
        """
        # Log the attempt
        logger.error(
            f"Audio generation not implemented for persona {persona.id}",
            extra={
                "persona_id": str(persona.id),
                "persona_name": persona.name,
                "prompt": request.prompt,
                "quality": request.quality,
            },
        )

        # Audio generation not yet implemented - fail properly
        raise NotImplementedError(
            "Audio generation requires AI model integration (MusicLM, AudioCraft, or similar). "
            "Please configure audio generation models to use this feature."
        )

    async def _generate_voice(
        self, persona: PersonaModel, request: GenerationRequest
    ) -> Dict[str, Any]:
        """
        Generate voice content using AI model.

        Integrated with real AI models including ElevenLabs and OpenAI TTS.
        """
        # Log the generation attempt
        logger.info(
            f"Starting voice generation for persona {persona.name} ({persona.id})",
            extra={
                "persona_id": str(persona.id),
                "persona_name": persona.name,
                "text": request.prompt,
                "quality": request.quality,
                "voice_settings": {
                    "voice_id": persona.style_preferences.get("voice_id", "default"),
                    "voice_style": persona.style_preferences.get(
                        "voice_style", "alloy"
                    ),
                },
            },
        )

        try:
            # Ensure AI models are initialized
            if not ai_models.models_loaded:
                await ai_models.initialize_models()

            # Get voice settings from persona preferences
            voice_settings = {
                "voice_id": persona.style_preferences.get("voice_id", "default"),
                "voice": persona.style_preferences.get("voice_style", "alloy"),
                "stability": persona.style_preferences.get("voice_stability", 0.5),
                "similarity_boost": persona.style_preferences.get(
                    "voice_similarity", 0.5
                ),
            }

            # Generate voice using AI model
            voice_result = await ai_models.generate_voice(
                text=request.prompt, **voice_settings
            )

            # Save the generated voice file
            format_ext = voice_result.get("format", "MP3").lower()
            filename = f"voice_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{format_ext}"
            file_path = self.content_dir / "voice" / filename

            # Write audio data to file
            with open(file_path, "wb") as f:
                f.write(voice_result["audio_data"])

            return {
                "file_path": str(file_path),
                "file_size": len(voice_result["audio_data"]),
                "duration": len(request.prompt.split()) * 0.6,  # Rough estimate
                "format": voice_result.get("format", "MP3"),
                "sample_rate": "44.1kHz",
                "voice_characteristics": {
                    "voice_id": voice_result.get(
                        "voice_id", voice_settings["voice_id"]
                    ),
                    "voice": voice_result.get("voice", voice_settings["voice"]),
                    "provider": voice_result.get("provider", "unknown"),
                },
                "content_rating": request.content_rating.value,
            }

        except Exception as e:
            # Log the failure comprehensively
            logger.error(
                f"Voice generation failed for persona {persona.id}: {str(e)}",
                extra={
                    "persona_id": str(persona.id),
                    "persona_name": persona.name,
                    "text": request.prompt,
                    "voice_settings": {
                        "voice_id": persona.style_preferences.get(
                            "voice_id", "default"
                        ),
                        "voice_style": persona.style_preferences.get(
                            "voice_style", "alloy"
                        ),
                    },
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )

            # Re-raise the exception instead of creating placeholder
            raise ValueError(f"Voice generation failed: {str(e)}") from e

    async def _generate_text(
        self, persona: PersonaModel, request: GenerationRequest
    ) -> Dict[str, Any]:
        """
        Generate text using AI model with ACD context tracking.

        First attempts real AI generation, falls back to smart template-based generation.
        Uses base_appearance_description for consistency when appearance_locked is True.
        Tracks generation context via ACD for learning and debugging.
        """
        # Determine complexity based on quality and content themes
        complexity = AIComplexity.LOW
        if request.quality in ["high", "premium"]:
            complexity = AIComplexity.MEDIUM
        if persona.appearance_locked:
            complexity = AIComplexity.MEDIUM

        # Create ACD context for tracking
        initial_context = {
            "prompt": request.prompt,
            "persona_id": str(persona.id),
            "quality": request.quality,
            "content_rating": request.content_rating.value,
            "content_themes": (
                persona.content_themes[:3] if persona.content_themes else []
            ),
        }

        async with ACDContextManager(
            self.db,
            phase="TEXT_GENERATION",
            note=f"Generating text for persona {persona.name}",
            complexity=complexity,
            initial_context=initial_context,
        ) as acd:
            try:
                # Ensure AI models are initialized
                if not ai_models.models_loaded:
                    await ai_models.initialize_models()

                # Use base appearance if locked for consistency
                appearance_desc = (
                    persona.base_appearance_description
                    if persona.appearance_locked and persona.base_appearance_description
                    else persona.appearance
                )

                # Enhanced prompt with persona context
                enhanced_prompt = f"""You are {persona.name}, an AI influencer with the following characteristics:

Appearance: {appearance_desc}
Personality: {persona.personality}
Content Themes: {', '.join(persona.content_themes[:3]) if persona.content_themes else 'general topics'}

Create engaging social media content that matches this persona. The content should be:
- Authentic to the personality described
- Relevant to the content themes
- Appropriate for {request.content_rating.value} rating
- Engaging and shareable

Original request: {request.prompt}

Generate the social media content now:"""

                # Try AI generation first
                generated_text = await ai_models.generate_text(
                    prompt=enhanced_prompt,
                    max_tokens=500 if request.quality == "draft" else 800,
                    temperature=0.7 if request.quality in ["standard", "high"] else 0.5,
                )

                # Clean up and format the text
                generated_text = generated_text.strip()

                # Save the generated text
                filename = f"text_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                file_path = self.content_dir / "text" / filename

                file_path.write_text(generated_text, encoding="utf-8")

                # Update ACD with successful generation details
                await acd.set_confidence(AIConfidence.CONFIDENT)
                await acd.set_metadata(
                    {
                        **initial_context,
                        "word_count": len(generated_text.split()),
                        "character_count": len(generated_text),
                        "ai_generated": True,
                        "success": True,
                    }
                )

                return {
                    "file_path": str(file_path),
                    "file_size": file_path.stat().st_size,
                    "word_count": len(generated_text.split()),
                    "character_count": len(generated_text),
                    "text_preview": (
                        generated_text[:200] + "..."
                        if len(generated_text) > 200
                        else generated_text
                    ),
                    "content_rating": request.content_rating.value,
                    "full_text": generated_text,
                    "ai_generated": True,
                    "acd_context_id": acd.context_id,  # Link to ACD context
                }

            except Exception as e:
                # Mark ACD as using fallback (not failed, since we have a working fallback)
                await acd.set_confidence(AIConfidence.UNCERTAIN)
                await acd.set_state(AIState.DONE)  # Still completed, just with fallback
                await acd.set_metadata(
                    {
                        **initial_context,
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "using_fallback": True,
                        "fallback_method": "template_based",
                    }
                )

                # Enhanced fallback generation using persona characteristics
                logger.warning(
                    "⚠️  AI text generation unavailable, using fallback method"
                )
                logger.warning(f"   Reason: {str(e)}")
                logger.warning("   Fallback: Template-based generation")
                logger.info("   🔄 Generating content using template fallback...")

                await asyncio.sleep(0.05)  # Simulate processing time

                # Create more sophisticated fallback content based on persona and prompt
                generated_text = await self._create_enhanced_fallback_text(
                    persona, request
                )
                logger.info(
                    f"   ✓ Fallback content generated: {len(generated_text)} characters"
                )

                filename = (
                    f"text_fallback_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                )
                file_path = self.content_dir / "text" / filename

                file_path.write_text(generated_text, encoding="utf-8")

                # Update ACD with fallback details
                await acd.set_metadata(
                    {
                        **initial_context,
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "using_fallback": True,
                        "fallback_method": "template_based",
                        "word_count": len(generated_text.split()),
                        "character_count": len(generated_text),
                    }
                )

                return {
                    "file_path": str(file_path),
                    "file_size": file_path.stat().st_size,
                    "word_count": len(generated_text.split()),
                    "character_count": len(generated_text),
                    "text_preview": (
                        generated_text[:200] + "..."
                        if len(generated_text) > 200
                        else generated_text
                    ),
                    "content_rating": request.content_rating.value,
                    "full_text": generated_text,
                    "ai_generated": False,
                    "template_based": True,
                    "fallback": True,
                    "fallback_reason": str(e),
                    "acd_context_id": acd.context_id,  # Link to ACD context even on fallback
                }

    async def _create_enhanced_fallback_text(
        self, persona: PersonaModel, request: GenerationRequest
    ) -> str:
        """
        Create enhanced fallback text using persona characteristics and prompt analysis.

        Delegates to TemplateService for sophisticated template-based generation.
        Uses base_appearance_description when appearance_locked is True for consistency.
        Leverages style_preferences for sophisticated content styling and tone.

        Note: Eagerly accesses all persona attributes in async context to prevent
        greenlet_spawn errors when passing to synchronous template service.
        """
        # Eagerly load all persona attributes we need to prevent lazy loading issues
        # in the synchronous template service
        persona_data = {
            "appearance": persona.appearance,
            "base_appearance_description": persona.base_appearance_description,
            "appearance_locked": persona.appearance_locked,
            "personality": persona.personality,
            "content_themes": persona.content_themes if persona.content_themes else [],
            "style_preferences": (
                persona.style_preferences if persona.style_preferences else {}
            ),
            "name": persona.name,
        }

        # Pass extracted data instead of the model to avoid lazy loading
        return self.template_service.generate_fallback_text_from_data(
            persona_data=persona_data,
            prompt=request.prompt,
            content_rating=request.content_rating.value,
        )

    async def _create_platform_adaptations(
        self,
        persona: PersonaModel,
        content_data: Dict[str, Any],
        content_rating: ContentRating,
        target_platforms: List[str],
    ) -> Dict[str, Any]:
        """
        Create platform-specific adaptations for content.

        Uses persona's platform_restrictions to override global platform policies.
        This allows per-persona, per-site NSFW content filtering.
        """
        adaptations = {}

        for platform in target_platforms:
            platform_lower = platform.lower()

            # Check if content rating is appropriate for platform
            # Pass persona's platform_restrictions to enable per-site filtering
            # Uses database-driven platform policies
            if not await self.moderation_service.platform_content_filter(
                content_rating,
                platform_lower,
                persona.platform_restrictions,
                self.platform_policy_service,
            ):
                adaptations[platform_lower] = {
                    "status": "blocked",
                    "reason": f"Content rating {content_rating.value} not allowed on {platform}",
                }
                continue

            # Platform-specific adaptations
            adaptation = {"status": "approved", "modified_for_platform": False}

            # Instagram adaptations
            if platform_lower == "instagram":
                if (
                    content_data.get("width", 0) > 0
                    and content_data.get("height", 0) > 0
                ):
                    # Square crop for Instagram
                    adaptation["crop_ratio"] = "1:1"
                    adaptation["modified_for_platform"] = True

            # Facebook adaptations
            elif platform_lower == "facebook":
                if content_rating in [ContentRating.NSFW, ContentRating.MODERATE]:
                    adaptation["content_warning"] = True
                    adaptation["modified_for_platform"] = True

            # Twitter adaptations
            elif platform_lower == "twitter":
                if content_rating == ContentRating.NSFW:
                    adaptation["sensitive_content_flag"] = True
                    adaptation["modified_for_platform"] = True

            adaptations[platform_lower] = adaptation

        return adaptations

    async def _save_content_record(
        self,
        persona: PersonaModel,
        request: GenerationRequest,
        content_data: Dict[str, Any],
        platform_adaptations: Dict[str, Any],
    ) -> ContentResponse:
        """Save content record to database and link ACD context."""
        # Convert any UUID objects to strings for JSON serialization
        serializable_content_data = {}
        for key, value in content_data.items():
            if isinstance(value, UUID):
                serializable_content_data[key] = str(value)
            else:
                serializable_content_data[key] = value

        # Generate description - handle None prompt for IMAGE type
        if request.prompt:
            description = f"AI-generated {request.content_type.value} using prompt: {request.prompt[:100]}..."
        else:
            # For IMAGE type, prompt might be None as it's generated internally
            description = (
                f"AI-generated {request.content_type.value} for {persona.name}"
            )

        content = ContentModel(
            persona_id=persona.id,
            content_type=request.content_type.value,
            title=f"Generated {request.content_type.value} for {persona.name}",
            description=description,
            content_rating=request.content_rating.value,
            file_path=content_data.get("file_path"),
            file_size=content_data.get("file_size"),
            generation_params={
                "prompt": request.prompt,
                "quality": request.quality,
                "style_override": request.style_override,
                "target_platforms": request.target_platforms,
                **serializable_content_data,
            },
            platform_adaptations=platform_adaptations,
            quality_score=85,  # Placeholder scoring
            moderation_status=ModerationStatus.PENDING.value,
        )

        self.db.add(content)
        await self.db.commit()
        await self.db.refresh(content)

        # Link ACD context back to content now that content exists
        acd_context_id = content_data.get("acd_context_id")
        if acd_context_id:
            try:
                from backend.models.acd import ACDContextUpdate
                from backend.services.acd_service import ACDService

                acd_service = ACDService(self.db)

                # Convert string to UUID if needed
                if isinstance(acd_context_id, str):
                    from uuid import UUID as UUIDType

                    acd_context_id = UUIDType(acd_context_id)

                # Update ACD context with content_id
                await acd_service.update_context(
                    acd_context_id,
                    ACDContextUpdate(ai_note=f"Content created: {content.id}"),
                )

                # Also update the content_id field in ACD context directly
                stmt = select(ACDContextModel).where(
                    ACDContextModel.id == acd_context_id
                )
                result = await self.db.execute(stmt)
                acd_context = result.scalar_one_or_none()
                if acd_context:
                    acd_context.content_id = content.id
                    await self.db.commit()
                    logger.info(
                        f"Linked ACD context {acd_context_id} to content {content.id}"
                    )

            except Exception as e:
                logger.warning(f"Failed to link ACD context to content: {str(e)}")
                # Don't fail content creation if ACD linking fails

        return ContentResponse.model_validate(content)

    async def _increment_persona_count(self, persona_id: UUID) -> None:
        """Increment generation count for persona."""
        stmt = select(PersonaModel).where(PersonaModel.id == persona_id)
        result = await self.db.execute(stmt)
        persona = result.scalar_one_or_none()

        if persona:
            persona.generation_count += 1
            await self.db.commit()

    async def delete_content(self, content_id: UUID) -> bool:
        """
        Soft delete content by ID.

        Args:
            content_id: Content identifier to delete

        Returns:
            bool: True if content was deleted, False if not found
        """
        try:
            stmt = select(ContentModel).where(ContentModel.id == content_id)
            result = await self.db.execute(stmt)
            content = result.scalar_one_or_none()

            if not content:
                return False

            # Soft delete - mark as deleted instead of removing from database
            content.is_deleted = True
            content.deleted_at = datetime.now()

            await self.db.commit()
            logger.info(f"Content soft deleted content_id={content_id}")
            return True

        except Exception as e:
            logger.error(
                f"Error deleting content error={str(e)} content_id={content_id}"
            )
            await self.db.rollback()
            return False
