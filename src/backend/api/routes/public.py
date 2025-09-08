"""
Public View API Routes

Provides public-facing endpoints for viewing AI influencer content
without requiring authentication. Designed for public consumption.
"""

from typing import List, Optional, Dict, Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status, Query, Path
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from backend.database.connection import get_db_session
from backend.models.persona import PersonaModel, PersonaResponse
from backend.models.content import ContentModel, ContentResponse
from backend.config.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(
    prefix="/public",
    tags=["public"],
    responses={404: {"description": "Resource not found"}},
)


@router.get("/personas", response_model=List[Dict[str, Any]])
async def list_public_personas(
    limit: int = Query(default=10, ge=1, le=50, description="Maximum personas to return"),
    db: AsyncSession = Depends(get_db_session),
):
    """
    List publicly available AI personas.
    
    Returns basic information about active AI personas for public viewing.
    Personal details and generation metadata are filtered for privacy.
    
    Args:
        limit: Maximum number of personas to return
        db: Database session
    
    Returns:
        List of public persona information
    """
    try:
        # Get active personas
        stmt = (select(PersonaModel)
               .where(PersonaModel.is_active == True)
               .order_by(PersonaModel.generation_count.desc())
               .limit(limit))
        
        result = await db.execute(stmt)
        personas = result.scalars().all()
        
        # Filter for public consumption
        public_personas = []
        for persona in personas:
            public_personas.append({
                "id": persona.id,
                "name": persona.name,
                "bio": f"{persona.personality[:100]}...",
                "themes": persona.content_themes[:5],  # Limit themes
                "content_count": persona.generation_count,
                "style": persona.style_preferences.get("visual_style", "realistic") if persona.style_preferences else "realistic"
            })
        
        logger.info("Public personas listed", count=len(public_personas))
        return public_personas
        
    except Exception as e:
        logger.error("Error listing public personas", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve personas"
        )


@router.get("/personas/{persona_id}", response_model=Dict[str, Any])
async def get_public_persona(
    persona_id: UUID = Path(..., description="Persona unique identifier"),
    db: AsyncSession = Depends(get_db_session),
):
    """
    Get detailed public information about specific persona.
    
    Args:
        persona_id: Persona identifier
        db: Database session
    
    Returns:
        Public persona details
    
    Raises:
        404: Persona not found or not public
    """
    try:
        # Get persona
        stmt = (select(PersonaModel)
               .where(PersonaModel.id == persona_id)
               .where(PersonaModel.is_active == True))
        
        result = await db.execute(stmt)
        persona = result.scalar_one_or_none()
        
        if not persona:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Persona not found"
            )
        
        # Build public persona profile
        public_persona = {
            "id": persona.id,
            "name": persona.name,
            "bio": persona.personality,
            "appearance": persona.appearance,
            "themes": persona.content_themes,
            "style_preferences": {
                "visual_style": persona.style_preferences.get("visual_style", "realistic"),
                "color_palette": persona.style_preferences.get("color_palette", "natural"),
                "lighting": persona.style_preferences.get("lighting", "natural")
            } if persona.style_preferences else {},
            "statistics": {
                "content_created": persona.generation_count,
                "active_since": persona.created_at.isoformat(),
                "last_updated": persona.updated_at.isoformat() if persona.updated_at else None
            }
        }
        
        logger.info("Public persona viewed", persona_id=persona_id)
        return public_persona
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error retrieving public persona", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve persona"
        )


@router.get("/personas/{persona_id}/gallery", response_model=List[Dict[str, Any]])
async def get_persona_gallery(
    persona_id: UUID = Path(..., description="Persona unique identifier"),
    content_type: Optional[str] = Query(None, regex="^(image|video|text)$", description="Filter by content type"),
    limit: int = Query(default=20, ge=1, le=100, description="Maximum items to return"),
    offset: int = Query(default=0, ge=0, description="Number of items to skip"),
    db: AsyncSession = Depends(get_db_session),
):
    """
    Get public gallery of content for specific persona.
    
    Returns approved, publicly viewable content created by the AI persona.
    
    Args:
        persona_id: Persona identifier
        content_type: Optional filter by content type
        limit: Maximum number of items to return
        offset: Number of items to skip for pagination
        db: Database session
    
    Returns:
        List of public content items
    
    Raises:
        404: Persona not found
    """
    try:
        # Verify persona exists and is active
        persona_stmt = (select(PersonaModel)
                       .where(PersonaModel.id == persona_id)
                       .where(PersonaModel.is_active == True))
        
        persona_result = await db.execute(persona_stmt)
        persona = persona_result.scalar_one_or_none()
        
        if not persona:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Persona not found"
            )
        
        # Build content query
        stmt = (select(ContentModel)
               .where(ContentModel.persona_id == persona_id)
               .where(ContentModel.moderation_status == "approved")
               .where(ContentModel.is_published == True))
        
        if content_type:
            stmt = stmt.where(ContentModel.content_type == content_type)
        
        stmt = (stmt.order_by(ContentModel.created_at.desc())
               .offset(offset)
               .limit(limit))
        
        result = await db.execute(stmt)
        content_items = result.scalars().all()
        
        # Build public gallery items
        gallery_items = []
        for item in content_items:
            gallery_item = {
                "id": item.id,
                "type": item.content_type,
                "title": item.title,
                "description": item.description,
                "created_at": item.created_at.isoformat(),
                "quality_score": item.quality_score,
                "preview": {
                    "file_size": item.file_size,
                    **item.generation_params
                }
            }
            
            # Add type-specific metadata
            if item.content_type == "text" and item.generation_params:
                gallery_item["preview"]["text_preview"] = item.generation_params.get("text_preview")
            elif item.content_type in ["image", "video"] and item.generation_params:
                gallery_item["preview"]["dimensions"] = item.generation_params.get("width", "unknown")
            
            gallery_items.append(gallery_item)
        
        logger.info("Persona gallery viewed", 
                   persona_id=persona_id, 
                   content_type=content_type,
                   item_count=len(gallery_items))
        
        return gallery_items
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error retrieving persona gallery", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve gallery"
        )


@router.get("/content/{content_id}", response_model=Dict[str, Any])
async def get_public_content(
    content_id: UUID = Path(..., description="Content unique identifier"),
    db: AsyncSession = Depends(get_db_session),
):
    """
    Get specific public content item.
    
    Args:
        content_id: Content identifier
        db: Database session
    
    Returns:
        Public content details
    
    Raises:
        404: Content not found or not public
    """
    try:
        # Get content with persona info
        stmt = (select(ContentModel, PersonaModel)
               .join(PersonaModel, ContentModel.persona_id == PersonaModel.id)
               .where(ContentModel.id == content_id)
               .where(ContentModel.moderation_status == "approved")
               .where(ContentModel.is_published == True)
               .where(PersonaModel.is_active == True))
        
        result = await db.execute(stmt)
        row = result.first()
        
        if not row:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Content not found"
            )
        
        content, persona = row
        
        # Build public content details
        public_content = {
            "id": content.id,
            "type": content.content_type,
            "title": content.title,
            "description": content.description,
            "created_at": content.created_at.isoformat(),
            "quality_score": content.quality_score,
            "persona": {
                "id": persona.id,
                "name": persona.name,
                "style": persona.style_preferences.get("visual_style", "realistic") if persona.style_preferences else "realistic"
            },
            "metadata": {
                "file_size": content.file_size,
                **{k: v for k, v in content.generation_params.items() 
                   if k not in ["prompt"]}  # Exclude internal prompt details
            }
        }
        
        logger.info("Public content viewed", 
                   content_id=content_id,
                   persona_id=persona.id)
        
        return public_content
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error retrieving public content", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve content"
        )


@router.get("/stats", response_model=Dict[str, Any])
async def get_platform_stats(
    db: AsyncSession = Depends(get_db_session),
):
    """
    Get public platform statistics.
    
    Returns general statistics about the AI influencer platform
    for public viewing without revealing sensitive details.
    
    Args:
        db: Database session
    
    Returns:
        Platform statistics
    """
    try:
        # Get persona count
        persona_stmt = select(PersonaModel).where(PersonaModel.is_active == True)
        persona_result = await db.execute(persona_stmt)
        persona_count = len(persona_result.scalars().all())
        
        # Get content count
        content_stmt = (select(ContentModel)
                       .where(ContentModel.moderation_status == "approved")
                       .where(ContentModel.is_published == True))
        content_result = await db.execute(content_stmt)
        content_items = content_result.scalars().all()
        content_count = len(content_items)
        
        # Content type breakdown
        content_types = {}
        for item in content_items:
            content_types[item.content_type] = content_types.get(item.content_type, 0) + 1
        
        stats = {
            "active_personas": persona_count,
            "published_content": content_count,
            "content_breakdown": content_types,
            "platform_version": "1.0",
            "last_updated": "2024-01-01T00:00:00Z"  # TODO: Use actual timestamp
        }
        
        logger.info("Platform stats viewed", stats=stats)
        return stats
        
    except Exception as e:
        logger.error("Error retrieving platform stats", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve statistics"
        )