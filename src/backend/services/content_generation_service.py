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
    ContentModel, ContentCreate, ContentResponse, 
    GenerationRequest, ContentType, ContentRating, ModerationStatus
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
            "sexy", "nude", "naked", "adult", "erotic", "sensual", 
            "lingerie", "bikini", "provocative", "intimate"
        ]
        
        prompt_lower = prompt.lower()
        
        # If persona allows NSFW and prompt contains NSFW keywords
        if persona_rating in ["nsfw", "both"] and any(keyword in prompt_lower for keyword in nsfw_keywords):
            return ContentRating.NSFW
        elif any(keyword in prompt_lower for keyword in ["suggestive", "flirty", "romantic"]):
            return ContentRating.MODERATE
        
        return ContentRating.SFW
    
    @staticmethod
    def platform_content_filter(content_rating: ContentRating, target_platform: str) -> bool:
        """Check if content is appropriate for target platform."""
        platform_policies = {
            "instagram": [ContentRating.SFW, ContentRating.MODERATE],
            "facebook": [ContentRating.SFW],
            "twitter": [ContentRating.SFW, ContentRating.MODERATE, ContentRating.NSFW],
            "onlyfans": [ContentRating.SFW, ContentRating.MODERATE, ContentRating.NSFW],
            "patreon": [ContentRating.SFW, ContentRating.MODERATE, ContentRating.NSFW],
            "discord": [ContentRating.SFW, ContentRating.MODERATE],
        }
        
        allowed_ratings = platform_policies.get(target_platform.lower(), [ContentRating.SFW])
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
    
    def __init__(self, db_session: AsyncSession, content_dir: str = "generated_content"):
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
                request.prompt = await self._generate_prompt(persona, request.content_type, request.content_rating)
            
            # Validate content rating against persona settings
            if not await self._validate_content_rating(persona, request.content_rating):
                raise ValueError(f"Content rating {request.content_rating} not allowed for persona {persona.name}")
            
            # Analyze and adjust content rating based on prompt
            analyzed_rating = self.moderation_service.analyze_content_rating(
                request.prompt, 
                persona.default_content_rating
            )
            if analyzed_rating != request.content_rating:
                logger.info(f"Content rating adjusted from {request.content_rating} to {analyzed_rating}")
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
            
            logger.info("Content generated successfully", extra={
                       "content_id": content_record.id, 
                       "persona_id": persona.id,
                       "content_type": request.content_type
            })
            
            return content_record
            
        except Exception as e:
            logger.error("Content generation failed", extra={
                        "error": str(e), 
                        "persona_id": request.persona_id,
                        "content_type": request.content_type
            })
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
            logger.error("Error retrieving content", extra={"error": str(e), "content_id": content_id})
            return None
    
    async def list_persona_content(self, persona_id: UUID, limit: int = 50) -> List[ContentResponse]:
        """List content generated for a specific persona."""
        try:
            stmt = (select(ContentModel)
                   .where(ContentModel.persona_id == persona_id)
                   .order_by(ContentModel.created_at.desc())
                   .limit(limit))
            
            result = await self.db.execute(stmt)
            contents = result.scalars().all()
            
            return [ContentResponse.model_validate(content) for content in contents]
            
        except Exception as e:
            logger.error("Error listing persona content", extra={"error": str(e), "persona_id": persona_id})
            return []
    
    async def _get_persona(self, persona_id: UUID) -> Optional[PersonaModel]:
        """Retrieve persona from database."""
        stmt = select(PersonaModel).where(
            PersonaModel.id == persona_id,
            PersonaModel.is_active == True
        )
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()
    
    async def _generate_prompt(self, persona: PersonaModel, content_type: ContentType, content_rating: ContentRating) -> str:
        """
        Generate AI prompt based on persona characteristics and content rating.
        """
        base_prompt = f"{persona.appearance}, {persona.personality}"
        
        # Add content rating modifiers
        rating_modifiers = {
            ContentRating.SFW: "safe for work, family-friendly, appropriate for all audiences",
            ContentRating.MODERATE: "tasteful, artistic, suitable for mature audiences",
            ContentRating.NSFW: "adult content, explicit, 18+ only"
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
            themes = ", ".join(persona.content_themes[:3]) if persona.content_themes else "general topics"
            prompt = f"Write engaging social media content about {themes} in the style of {persona.personality}, {rating_modifier}"
        
        return prompt
    
    async def _validate_content_rating(self, persona: PersonaModel, requested_rating: ContentRating) -> bool:
        """Validate that the requested content rating is allowed for this persona."""
        allowed_ratings = getattr(persona, 'allowed_content_ratings', ['sfw'])
        if isinstance(allowed_ratings, str):
            allowed_ratings = [allowed_ratings]
        
        return requested_rating.value in [r.lower() for r in allowed_ratings]
    
    async def _generate_image(self, persona: PersonaModel, request: GenerationRequest) -> Dict[str, Any]:
        """
        Generate image using AI model.
        
        This is a placeholder implementation. In a production system, this would
        integrate with Stable Diffusion or similar models.
        """
        # Simulate image generation
        await asyncio.sleep(0.1)  # Simulate processing time
        
        filename = f"image_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        file_path = self.content_dir / "images" / filename
        
        # Create placeholder image file
        file_path.write_text("# Placeholder for generated image content")
        
        return {
            "file_path": str(file_path),
            "file_size": file_path.stat().st_size,
            "width": 1024,
            "height": 1024,
            "format": "JPEG",
            "content_rating": request.content_rating.value
        }
    
    async def _generate_video(self, persona: PersonaModel, request: GenerationRequest) -> Dict[str, Any]:
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
            "content_rating": request.content_rating.value
        }
    
    async def _generate_audio(self, persona: PersonaModel, request: GenerationRequest) -> Dict[str, Any]:
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
            "content_rating": request.content_rating.value
        }
    
    async def _generate_voice(self, persona: PersonaModel, request: GenerationRequest) -> Dict[str, Any]:
        """
        Generate voice content using AI model.
        
        This is a placeholder implementation. In production, this would
        integrate with voice synthesis models like ElevenLabs or Tortoise TTS.
        """
        # Simulate voice generation
        await asyncio.sleep(0.4)  # Simulate processing time
        
        filename = f"voice_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        file_path = self.content_dir / "voice" / filename
        
        # Create placeholder voice file
        file_path.write_text("# Placeholder for generated voice content")
        
        voice_characteristics = {
            "voice_id": persona.style_preferences.get("voice_id", "default"),
            "pitch": persona.style_preferences.get("voice_pitch", "medium"),
            "speed": persona.style_preferences.get("voice_speed", "normal"),
            "emotion": persona.style_preferences.get("voice_emotion", "neutral")
        }
        
        return {
            "file_path": str(file_path),
            "file_size": file_path.stat().st_size,
            "duration": len(request.prompt.split()) * 0.6,  # Rough estimate: 0.6 seconds per word
            "format": "WAV",
            "sample_rate": "44.1kHz",
            "voice_characteristics": voice_characteristics,
            "content_rating": request.content_rating.value
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
            "content_rating": request.content_rating.value
        }
    
    async def _generate_text(self, persona: PersonaModel, request: GenerationRequest) -> Dict[str, Any]:
        """
        Generate text using AI model.
        
        This is a placeholder implementation. In a production system, this would
        integrate with language models like GPT or Claude.
        """
        # Simulate text generation
        await asyncio.sleep(0.05)  # Simulate processing time
        
        # Generate sample text based on persona
        personality_traits = persona.personality.split(", ")[:2]
        themes = persona.content_themes[:2] if persona.content_themes else ["lifestyle"]
        
        sample_texts = [
            f"Excited to share my thoughts on {themes[0]}! As someone who's {personality_traits[0]}, I believe...",
            f"Today's focus: {themes[0]}. Being {personality_traits[0]} has taught me that...",
            f"Quick thoughts on {themes[0] if themes else 'today'}: Life is about...",
        ]
        
        generated_text = sample_texts[0]  # In real system, this would be AI-generated
        
        filename = f"text_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        file_path = self.content_dir / "text" / filename
        
        file_path.write_text(generated_text)
        
        return {
            "file_path": str(file_path),
            "file_size": file_path.stat().st_size,
            "word_count": len(generated_text.split()),
            "character_count": len(generated_text),
            "text_preview": generated_text[:200] + "..." if len(generated_text) > 200 else generated_text,
            "content_rating": request.content_rating.value
        }
    
    async def _create_platform_adaptations(
        self, 
        content_data: Dict[str, Any], 
        content_rating: ContentRating,
        target_platforms: List[str]
    ) -> Dict[str, Any]:
        """Create platform-specific adaptations for content."""
        adaptations = {}
        
        for platform in target_platforms:
            platform_lower = platform.lower()
            
            # Check if content rating is appropriate for platform
            if not self.moderation_service.platform_content_filter(content_rating, platform_lower):
                adaptations[platform_lower] = {
                    "status": "blocked",
                    "reason": f"Content rating {content_rating.value} not allowed on {platform}"
                }
                continue
            
            # Platform-specific adaptations
            adaptation = {
                "status": "approved",
                "modified_for_platform": False
            }
            
            # Instagram adaptations
            if platform_lower == "instagram":
                if content_data.get("width", 0) > 0 and content_data.get("height", 0) > 0:
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
        platform_adaptations: Dict[str, Any]
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
                **content_data
            },
            platform_adaptations=platform_adaptations,
            quality_score=85,  # Placeholder scoring
            moderation_status=ModerationStatus.PENDING.value
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