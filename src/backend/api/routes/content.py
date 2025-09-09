"""
Content Generation API Routes

Handles content generation requests and management.
"""

from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database.connection import get_db_session
from backend.services.content_generation_service import ContentGenerationService
from backend.models.content import ContentResponse, ContentCreate, GenerationRequest
from backend.config.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(
    prefix="/api/v1/content",
    tags=["content"],
    responses={404: {"description": "Content not found"}},
)


def get_content_service(
    db: AsyncSession = Depends(get_db_session)
) -> ContentGenerationService:
    """Dependency injection for ContentGenerationService."""
    return ContentGenerationService(db)


@router.post("/generate", status_code=status.HTTP_202_ACCEPTED)
async def generate_content(
    request: Optional[GenerationRequest] = None,
    content_service: ContentGenerationService = Depends(get_content_service),
):
    """
    Generate new AI content.
    
    Creates new content (image, video, or text) based on persona configuration
    and optional prompt or style overrides.
    
    Args:
        request: Content generation parameters
        content_service: Injected content generation service
    
    Returns:
        Placeholder response until implemented
    
    Raises:
        400: Invalid request parameters
        404: Persona not found
        500: Generation failed
    """
    # Placeholder implementation
    return {
        "status": "placeholder - content generation not yet implemented", 
        "message": "Content generation will be processed when implemented",
        "request_received": request is not None
    }


@router.get("/{content_id}", response_model=ContentResponse)
async def get_content(
    content_id: UUID,
    content_service: ContentGenerationService = Depends(get_content_service),
):
    """
    Get content by ID.
    
    Args:
        content_id: Unique content identifier
        content_service: Injected content generation service
    
    Returns:
        ContentResponse: Content metadata
    
    Raises:
        404: Content not found
    """
    content = await content_service.get_content(content_id)
    if not content:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Content not found"
        )
    return content


@router.get("/persona/{persona_id}", response_model=List[ContentResponse])
async def list_persona_content(
    persona_id: UUID,
    limit: int = Query(default=50, ge=1, le=100, description="Maximum items to return"),
    content_service: ContentGenerationService = Depends(get_content_service),
):
    """
    List content for specific persona.
    
    Args:
        persona_id: Persona identifier
        limit: Maximum number of items to return
        content_service: Injected content generation service
    
    Returns:
        List[ContentResponse]: List of content items
    """
    return await content_service.list_persona_content(persona_id, limit)


@router.get("/", status_code=status.HTTP_200_OK)
async def list_all_content(
    limit: int = Query(default=50, ge=1, le=100, description="Maximum items to return"),
    content_service: ContentGenerationService = Depends(get_content_service),
):
    """
    List all generated content.
    
    Args:
        limit: Maximum number of items to return
        content_service: Injected content generation service
    
    Returns:
        Placeholder response until implemented
    """
    # Placeholder implementation
    return {
        "status": "placeholder - content listing not yet implemented",
        "message": "This endpoint will list generated content items",
        "limit": limit
    }