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
from backend.models.content import ContentModel, ContentCreate, ContentResponse
from backend.config.logging import get_logger

logger = get_logger(__name__)


class GenerationRequest(BaseModel):
    """Request for content generation."""
    persona_id: UUID
    content_type: str  # 'image', 'video', 'text'
    prompt: Optional[str] = None
    style_override: Optional[Dict[str, Any]] = None
    quality: str = "high"  # 'draft', 'standard', 'high'


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
        (self.content_dir / "text").mkdir(exist_ok=True)
        
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
                request.prompt = await self._generate_prompt(persona, request.content_type)
                
            # Generate content based on type
            if request.content_type == "image":
                content_data = await self._generate_image(persona, request)
            elif request.content_type == "video":
                content_data = await self._generate_video(persona, request)
            elif request.content_type == "text":
                content_data = await self._generate_text(persona, request)
            else:
                raise ValueError(f"Unsupported content type: {request.content_type}")
            
            # Store content metadata in database
            content_record = await self._save_content_record(persona, request, content_data)
            
            # Update persona generation count
            await self._increment_persona_count(persona.id)
            
            logger.info("Content generated successfully", 
                       content_id=content_record.id, 
                       persona_id=persona.id,
                       content_type=request.content_type)
            
            return content_record
            
        except Exception as e:
            logger.error("Content generation failed", 
                        error=str(e), 
                        persona_id=request.persona_id,
                        content_type=request.content_type)
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
            logger.error("Error retrieving content", error=str(e), content_id=content_id)
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
            logger.error("Error listing persona content", error=str(e), persona_id=persona_id)
            return []
    
    async def _get_persona(self, persona_id: UUID) -> Optional[PersonaModel]:
        """Retrieve persona from database."""
        stmt = select(PersonaModel).where(
            PersonaModel.id == persona_id,
            PersonaModel.is_active == True
        )
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()
    
    async def _generate_prompt(self, persona: PersonaModel, content_type: str) -> str:
        """
        Generate AI prompt based on persona characteristics.
        
        This is a placeholder implementation. In a real system, this would
        use a sophisticated prompt generation system.
        """
        base_prompt = f"{persona.appearance}, {persona.personality}"
        
        if content_type == "image":
            style_info = persona.style_preferences.get("visual_style", "realistic")
            lighting = persona.style_preferences.get("lighting", "natural")
            prompt = f"Portrait photo of {base_prompt}, {style_info} style, {lighting} lighting, high quality"
        elif content_type == "video":
            prompt = f"Short video featuring {base_prompt}, engaging and dynamic"
        else:  # text
            themes = ", ".join(persona.content_themes[:3]) if persona.content_themes else "general topics"
            prompt = f"Write engaging social media content about {themes} in the style of {persona.personality}"
        
        return prompt
    
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
            "format": "JPEG"
        }
    
    async def _generate_video(self, persona: PersonaModel, request: GenerationRequest) -> Dict[str, Any]:
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
            "format": "MP4"
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
            "text_preview": generated_text[:200] + "..." if len(generated_text) > 200 else generated_text
        }
    
    async def _save_content_record(self, persona: PersonaModel, request: GenerationRequest, content_data: Dict[str, Any]) -> ContentResponse:
        """Save content record to database."""
        content = ContentModel(
            persona_id=persona.id,
            content_type=request.content_type,
            title=f"Generated {request.content_type} for {persona.name}",
            description=f"AI-generated {request.content_type} using prompt: {request.prompt[:100]}...",
            file_path=content_data.get("file_path"),
            file_size=content_data.get("file_size"),
            generation_params={
                "prompt": request.prompt,
                "quality": request.quality,
                "style_override": request.style_override,
                **content_data
            },
            quality_score=85,  # Placeholder scoring
            moderation_status="pending"
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