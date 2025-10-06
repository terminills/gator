"""
Content Generation Service

Handles AI-powered content generation including image and text creation
using integrated AI models like Stable Diffusion and language models.
"""

import asyncio
import os
from typing import Dict, List, Optional, Any
from datetime import datetime
from uuid import UUID
import json
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel

from backend.models.persona import PersonaModel
from backend.models.content import (
    ContentModel,
    ContentCreate,
    ContentResponse,
    GenerationRequest,
    ContentType,
    ContentRating,
    ModerationStatus,
)
from backend.config.logging import get_logger

logger = get_logger(__name__)


class ContentModerationService:
    """Service for content moderation and rating classification."""

    @staticmethod
    def analyze_content_rating(prompt: str, persona_rating: str) -> ContentRating:
        """
        Analyze content to determine appropriate rating.
        This is a placeholder - in production, this would use ML models.
        """
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
    def platform_content_filter(
        content_rating: ContentRating, target_platform: str
    ) -> bool:
        """Check if content is appropriate for target platform."""
        platform_policies = {
            "instagram": [ContentRating.SFW, ContentRating.MODERATE],
            "facebook": [ContentRating.SFW],
            "twitter": [ContentRating.SFW, ContentRating.MODERATE, ContentRating.NSFW],
            "onlyfans": [ContentRating.SFW, ContentRating.MODERATE, ContentRating.NSFW],
            "patreon": [ContentRating.SFW, ContentRating.MODERATE, ContentRating.NSFW],
            "discord": [ContentRating.SFW, ContentRating.MODERATE],
        }

        allowed_ratings = platform_policies.get(
            target_platform.lower(), [ContentRating.SFW]
        )
        return content_rating in allowed_ratings


class GenerationRequest(BaseModel):
    """Request for content generation."""

    persona_id: UUID
    content_type: ContentType  # 'image', 'video', 'audio', 'voice', 'text'
    content_rating: ContentRating = ContentRating.SFW
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
        self, db_session: AsyncSession, content_dir: str = "generated_content"
    ):
        """
        Initialize content generation service.

        Args:
            db_session: Database session for persistence
            content_dir: Directory to store generated content files
        """
        self.db = db_session
        self.content_dir = Path(content_dir)
        self.content_dir.mkdir(exist_ok=True)

        # Create subdirectories for different content types
        (self.content_dir / "images").mkdir(exist_ok=True)
        (self.content_dir / "videos").mkdir(exist_ok=True)
        (self.content_dir / "audio").mkdir(exist_ok=True)
        (self.content_dir / "voice").mkdir(exist_ok=True)
        (self.content_dir / "text").mkdir(exist_ok=True)

        self.moderation_service = ContentModerationService()

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
        try:
            # Get persona data
            persona = await self._get_persona(request.persona_id)
            if not persona:
                raise ValueError(f"Persona not found: {request.persona_id}")

            # Generate prompt if not provided
            if not request.prompt:
                request.prompt = await self._generate_prompt(
                    persona, request.content_type, request.content_rating
                )

            # Validate content rating against persona settings
            if not await self._validate_content_rating(persona, request.content_rating):
                raise ValueError(
                    f"Content rating {request.content_rating} not allowed for persona {persona.name}"
                )

            # Analyze and adjust content rating based on prompt
            analyzed_rating = self.moderation_service.analyze_content_rating(
                request.prompt, persona.default_content_rating
            )
            if analyzed_rating != request.content_rating:
                logger.info(
                    f"Content rating adjusted from {request.content_rating} to {analyzed_rating}"
                )
                request.content_rating = analyzed_rating

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

            # Apply platform-specific adaptations
            platform_adaptations = await self._create_platform_adaptations(
                content_data, request.content_rating, request.target_platforms or []
            )

            # Store content metadata in database
            content_record = await self._save_content_record(
                persona, request, content_data, platform_adaptations
            )

            # Update persona generation count
            await self._increment_persona_count(persona.id)

            logger.info(
                f"Content generated successfully content_id={content_record.id}"
            )

            return content_record

        except Exception as e:
            logger.error(
                f"Content generation failed error={str(e)} persona_id={request.persona_id} content_type={request.content_type}"
            )
            raise ValueError(f"Content generation failed: {str(e)}")

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
                .where(ContentModel.is_deleted == False)
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
                .where(ContentModel.is_deleted == False)
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
            PersonaModel.id == persona_id, PersonaModel.is_active == True
        )
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    async def _generate_prompt(
        self,
        persona: PersonaModel,
        content_type: ContentType,
        content_rating: ContentRating,
    ) -> str:
        """
        Generate AI prompt based on persona characteristics and content rating.
        Uses base_appearance_description if appearance_locked is True for consistency.
        """
        # Use base appearance if locked, otherwise use standard appearance
        if persona.appearance_locked and persona.base_appearance_description:
            base_prompt = f"{persona.base_appearance_description}, {persona.personality}"
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
            prompt = f"Portrait photo of {base_prompt}, {style_info} style, {lighting} lighting, high quality, {rating_modifier}"
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

    async def _validate_content_rating(
        self, persona: PersonaModel, requested_rating: ContentRating
    ) -> bool:
        """Validate that the requested content rating is allowed for this persona."""
        # Get allowed ratings from persona, default to ['sfw'] if empty
        allowed_ratings = getattr(persona, "allowed_content_ratings", ["sfw"])
        if not allowed_ratings:  # If empty list, default to sfw
            allowed_ratings = ["sfw"]

        if isinstance(allowed_ratings, str):
            allowed_ratings = [allowed_ratings]

        # Check if requested rating is allowed
        return requested_rating.value in [r.lower() for r in allowed_ratings]

    async def _generate_image(
        self, persona: PersonaModel, request: GenerationRequest
    ) -> Dict[str, Any]:
        """
        Generate image using AI model.

        Integrated with real AI models including OpenAI DALL-E and Stable Diffusion.
        Uses base_image_path for visual consistency when appearance_locked is True.
        """
        try:
            from backend.services.ai_models import ai_models

            # Ensure AI models are initialized
            if not ai_models.models_loaded:
                await ai_models.initialize_models()

            # Prepare generation parameters
            generation_params = {
                "prompt": request.prompt,
                "size": "1024x1024",
                "quality": (
                    request.quality
                    if request.quality in ["standard", "hd"]
                    else "standard"
                ),
            }

            # Add visual consistency parameters if appearance is locked
            if persona.appearance_locked and persona.base_image_path:
                generation_params["reference_image_path"] = persona.base_image_path
                generation_params["use_controlnet"] = True
                logger.info(
                    f"Using visual reference for consistency: {persona.base_image_path}"
                )

            # Generate image using AI model
            image_result = await ai_models.generate_image(**generation_params)

            # Save the generated image
            filename = f"image_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            file_path = self.content_dir / "images" / filename

            # Write image data to file
            with open(file_path, "wb") as f:
                f.write(image_result["image_data"])

            return {
                "file_path": str(file_path),
                "file_size": len(image_result["image_data"]),
                "width": image_result.get("width", 1024),
                "height": image_result.get("height", 1024),
                "format": image_result.get("format", "PNG"),
                "content_rating": request.content_rating.value,
                "model": image_result.get("model", "unknown"),
                "provider": image_result.get("provider", "unknown"),
            }

        except Exception as e:
            # Fallback to placeholder if AI generation fails
            logger.warning(f"AI image generation failed, using placeholder: {str(e)}")
            await asyncio.sleep(0.1)  # Simulate processing time

            filename = (
                f"image_placeholder_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            )
            file_path = self.content_dir / "images" / filename

            # Create placeholder image file with error info
            placeholder_content = f"# Placeholder for generated image content\n# Original prompt: {request.prompt}\n# Error: {str(e)}"
            file_path.write_text(placeholder_content)

            return {
                "file_path": str(file_path),
                "file_size": file_path.stat().st_size,
                "width": 1024,
                "height": 1024,
                "format": "PLACEHOLDER",
                "content_rating": request.content_rating.value,
                "error": str(e),
                "fallback": True,
            }

    async def _generate_video(
        self, persona: PersonaModel, request: GenerationRequest
    ) -> Dict[str, Any]:
        """
        Generate video using AI model.

        This is a placeholder implementation. In production, this would
        integrate with video generation models.
        """
        # Simulate video generation
        await asyncio.sleep(0.2)  # Simulate processing time

        filename = f"video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        file_path = self.content_dir / "videos" / filename

        # Create placeholder video file
        file_path.write_text("# Placeholder for generated video content")

        return {
            "file_path": str(file_path),
            "file_size": file_path.stat().st_size,
            "duration": 15.0,
            "resolution": "1920x1080",
            "format": "MP4",
            "content_rating": request.content_rating.value,
        }

    async def _generate_audio(
        self, persona: PersonaModel, request: GenerationRequest
    ) -> Dict[str, Any]:
        """
        Generate audio content using AI model.

        This is a placeholder implementation. In production, this would
        integrate with audio generation models like MusicLM or AudioCraft.
        """
        # Simulate audio generation
        await asyncio.sleep(0.3)  # Simulate processing time

        filename = f"audio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3"
        file_path = self.content_dir / "audio" / filename

        # Create placeholder audio file
        file_path.write_text("# Placeholder for generated audio content")

        return {
            "file_path": str(file_path),
            "file_size": file_path.stat().st_size,
            "duration": 30.0,
            "format": "MP3",
            "bitrate": "320kbps",
            "content_rating": request.content_rating.value,
        }

    async def _generate_voice(
        self, persona: PersonaModel, request: GenerationRequest
    ) -> Dict[str, Any]:
        """
        Generate voice content using AI model.

        Integrated with real AI models including ElevenLabs and OpenAI TTS.
        """
        try:
            from backend.services.ai_models import ai_models

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
            # Fallback to placeholder if AI generation fails
            logger.warning(f"AI voice generation failed, using placeholder: {str(e)}")
            await asyncio.sleep(0.4)  # Simulate processing time

            filename = (
                f"voice_placeholder_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            )
            file_path = self.content_dir / "voice" / filename

            # Create placeholder voice file with error info
            placeholder_content = f"# Placeholder for generated voice content\n# Text: {request.prompt}\n# Error: {str(e)}"
            file_path.write_text(placeholder_content)

            voice_characteristics = {
                "voice_id": persona.style_preferences.get("voice_id", "default"),
                "pitch": persona.style_preferences.get("voice_pitch", "medium"),
                "speed": persona.style_preferences.get("voice_speed", "normal"),
                "emotion": persona.style_preferences.get("voice_emotion", "neutral"),
            }

            return {
                "file_path": str(file_path),
                "file_size": file_path.stat().st_size,
                "duration": len(request.prompt.split()) * 0.6,  # Rough estimate
                "format": "PLACEHOLDER",
                "sample_rate": "44.1kHz",
                "voice_characteristics": voice_characteristics,
                "content_rating": request.content_rating.value,
                "error": str(e),
                "fallback": True,
            }
        """
        Generate video using AI model.
        
        This is a placeholder implementation. In a production system, this would
        integrate with video generation models.
        """
        # Simulate video generation
        await asyncio.sleep(0.2)  # Simulate processing time

        filename = f"video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        file_path = self.content_dir / "videos" / filename

        # Create placeholder video file
        file_path.write_text("# Placeholder for generated video content")

        return {
            "file_path": str(file_path),
            "file_size": file_path.stat().st_size,
            "duration": 15.0,
            "resolution": "1920x1080",
            "format": "MP4",
            "content_rating": request.content_rating.value,
        }

    async def _generate_text(
        self, persona: PersonaModel, request: GenerationRequest
    ) -> Dict[str, Any]:
        """
        Generate text using AI model.

        First attempts real AI generation, falls back to smart template-based generation.
        Uses base_appearance_description for consistency when appearance_locked is True.
        """
        try:
            from backend.services.ai_models import ai_models

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
            }

        except Exception as e:
            # Enhanced fallback generation using persona characteristics
            logger.warning(
                f"AI text generation failed, using enhanced fallback: {str(e)}"
            )
            await asyncio.sleep(0.05)  # Simulate processing time

            # Create more sophisticated fallback content based on persona and prompt
            generated_text = await self._create_enhanced_fallback_text(persona, request)

            filename = f"text_enhanced_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            file_path = self.content_dir / "text" / filename

            file_path.write_text(generated_text, encoding="utf-8")

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
                "fallback": True,
                "fallback_reason": str(e),
            }

    async def _create_enhanced_fallback_text(
        self, persona: PersonaModel, request: GenerationRequest
    ) -> str:
        """
        Create enhanced fallback text using persona characteristics and prompt analysis.
        
        Uses base_appearance_description when appearance_locked is True for consistency.
        """
        # Extract key elements - use locked appearance if available
        appearance_desc = (
            persona.base_appearance_description
            if persona.appearance_locked and persona.base_appearance_description
            else persona.appearance
        )
        
        personality_traits = persona.personality.split(", ")[:3]
        themes = (
            persona.content_themes[:3]
            if persona.content_themes
            else ["lifestyle", "thoughts"]
        )
        prompt_keywords = (
            request.prompt.lower().split() if request.prompt else ["content"]
        )

        # Determine content style based on personality
        if any(
            trait.lower() in ["creative", "artistic", "innovative"]
            for trait in personality_traits
        ):
            style = "creative"
        elif any(
            trait.lower() in ["professional", "business", "corporate"]
            for trait in personality_traits
        ):
            style = "professional"
        elif any(
            trait.lower() in ["tech", "technology", "analytical"]
            for trait in personality_traits
        ):
            style = "tech"
        else:
            style = "casual"

        # Extract visual/appearance cues for more personalized templates
        # This helps maintain consistency with the persona's visual identity
        appearance_keywords = appearance_desc.lower() if appearance_desc else ""
        is_visual_locked = persona.appearance_locked and persona.base_appearance_description
        
        # Add appearance context hint if locked (for consistency)
        appearance_context = ""
        if is_visual_locked:
            # Extract key appearance features for subtle context
            if "professional" in appearance_keywords:
                appearance_context = " (staying true to my professional image)"
            elif "creative" in appearance_keywords or "artistic" in appearance_keywords:
                appearance_context = " (expressing my creative side)"
            elif "casual" in appearance_keywords or "relaxed" in appearance_keywords:
                appearance_context = " (keeping it authentic and real)"

        # Generate content based on style and themes
        if style == "creative":
            templates = [
                f"🎨 Exploring the intersection of {themes[0]} and creativity today{appearance_context}. There's something magical about how innovation sparks when we blend different perspectives. What inspires your creative process? #creativity #{themes[0].replace(' ', '')} #inspiration",
                f"✨ Just had a breakthrough moment thinking about {themes[0]}{appearance_context}. Sometimes the best ideas come when we least expect them. The creative journey is all about embracing those unexpected connections. Share your latest 'aha' moment! 💡",
                f"🚀 Passionate about {themes[0]} and the endless possibilities it brings{appearance_context}. Every challenge is just a canvas waiting for the right creative solution. What problem are you solving creatively today? #innovation #{themes[0].replace(' ', '')}",
            ]
        elif style == "professional":
            templates = [
                f"Reflecting on the latest developments in {themes[0]}{appearance_context}. The landscape continues to evolve rapidly, and staying ahead requires continuous learning and adaptation. Key insights from today's analysis: strategic thinking remains paramount. Thoughts? #leadership #{themes[0].replace(' ', '')}",
                f"Professional insight{appearance_context}: {themes[0]} is reshaping how we approach business strategy. Organizations that embrace this transformation will gain significant competitive advantages. What trends are you monitoring in your industry? #business #strategy",
                f"Executive perspective on {themes[0]}{appearance_context}: Success in today's market requires both vision and execution. The companies thriving are those that balance innovation with operational excellence. How is your organization adapting?",
            ]
        elif style == "tech":
            templates = [
                f"🔧 Diving deep into {themes[0]} today{appearance_context}. The technical implications are fascinating - we're seeing unprecedented innovation in this space. For developers and tech enthusiasts: the future is being built now. What's on your tech radar? #technology #{themes[0].replace(' ', '')} #innovation",
                f"💻 Just analyzed the latest {themes[0]} developments{appearance_context}. The algorithmic approaches being implemented are genuinely impressive. Technical breakdown: efficiency gains are substantial. Fellow engineers - what are your thoughts on the current implementation patterns?",
                f"⚡ {themes[0]} technology stack evolution{appearance_context}: From proof-of-concept to production-ready solutions, the journey has been remarkable. System architecture considerations continue to be crucial. What technical challenges are you solving? #engineering #tech",
            ]
        else:  # casual
            templates = [
                f"💭 Had some interesting thoughts about {themes[0]} today{appearance_context}. It's amazing how much this topic touches our daily lives without us even realizing it. What's your take on this? Would love to hear different perspectives! #{themes[0].replace(' ', '')} #thoughts",
                f"🌟 Something about {themes[0]} just clicked for me today{appearance_context}. Sometimes the simplest insights are the most powerful. Life's full of these little learning moments. What did you discover today? #learning #growth",
                f"✌️ Quick reflection on {themes[0]}{appearance_context} - there's so much depth here that we often overlook. Taking time to really think about these things makes such a difference. Anyone else find themselves going down these thought rabbit holes? 😄",
            ]

        # Select template and customize based on prompt keywords
        import random

        selected_template = random.choice(templates)

        # If prompt contains specific keywords, try to incorporate them
        if any(
            keyword in ["trends", "future", "upcoming"] for keyword in prompt_keywords
        ):
            selected_template = selected_template.replace(
                "today", "for the future"
            ).replace("Today's", "Upcoming")
        elif any(
            keyword in ["analysis", "study", "research"] for keyword in prompt_keywords
        ):
            selected_template = selected_template.replace(
                "thoughts", "analysis"
            ).replace("thinking", "researching")

        return selected_template

    async def _create_platform_adaptations(
        self,
        content_data: Dict[str, Any],
        content_rating: ContentRating,
        target_platforms: List[str],
    ) -> Dict[str, Any]:
        """Create platform-specific adaptations for content."""
        adaptations = {}

        for platform in target_platforms:
            platform_lower = platform.lower()

            # Check if content rating is appropriate for platform
            if not self.moderation_service.platform_content_filter(
                content_rating, platform_lower
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
        """Save content record to database."""
        content = ContentModel(
            persona_id=persona.id,
            content_type=request.content_type.value,
            title=f"Generated {request.content_type.value} for {persona.name}",
            description=f"AI-generated {request.content_type.value} using prompt: {request.prompt[:100]}...",
            content_rating=request.content_rating.value,
            file_path=content_data.get("file_path"),
            file_size=content_data.get("file_size"),
            generation_params={
                "prompt": request.prompt,
                "quality": request.quality,
                "style_override": request.style_override,
                "target_platforms": request.target_platforms,
                **content_data,
            },
            platform_adaptations=platform_adaptations,
            quality_score=85,  # Placeholder scoring
            moderation_status=ModerationStatus.PENDING.value,
        )

        self.db.add(content)
        await self.db.commit()
        await self.db.refresh(content)

        return ContentResponse.model_validate(content)

    async def _increment_persona_count(self, persona_id: UUID) -> None:
        """Increment generation count for persona."""
        stmt = select(PersonaModel).where(PersonaModel.id == persona_id)
        result = await self.db.execute(stmt)
        persona = result.scalar_one_or_none()

        if persona:
            persona.generation_count += 1
            await self.db.commit()
