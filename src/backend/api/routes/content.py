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
    db: AsyncSession = Depends(get_db_session),
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
        Content generation status and details

    Raises:
        400: Invalid request parameters
        404: Persona not found
        500: Generation failed
    """
    try:
        if not request:
            # Provide a default generation request
            from backend.models.content import GenerationRequest, ContentType

            request = GenerationRequest(
                content_type=ContentType.IMAGE,
                prompt="AI generated content placeholder",
                persona_id=None,  # Will use default persona or create one
            )

        # Generate content using the service
        result = await content_service.generate_content(request)

        return {
            "status": "accepted",
            "message": "Content generation started",
            "content_id": str(result.id) if result else None,
            "generation_type": request.content_type,
            "prompt": (
                request.prompt[:100] + "..."
                if len(request.prompt) > 100
                else request.prompt
            ),
        }

    except ValueError as e:
        logger.warning(f"Invalid content generation request: {str(e)}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Content generation failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Content generation failed",
        )


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
            status_code=status.HTTP_404_NOT_FOUND, detail="Content not found"
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
        List of generated content items
    """
    try:
        # Get content from service
        content_list = await content_service.list_all_content(limit=limit)

        return {
            "status": "success",
            "content": (
                [content.model_dump() for content in content_list]
                if content_list
                else []
            ),
            "count": len(content_list) if content_list else 0,
            "limit": limit,
        }

    except Exception as e:
        logger.error(f"Failed to list content: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve content list",
        )


@router.delete("/{content_id}", status_code=status.HTTP_200_OK)
async def delete_content(
    content_id: UUID,
    content_service: ContentGenerationService = Depends(get_content_service),
):
    """
    Delete content by ID (soft delete).

    Args:
        content_id: Unique content identifier
        content_service: Injected content generation service

    Returns:
        Success message

    Raises:
        404: Content not found
        500: Delete failed
    """
    try:
        success = await content_service.delete_content(content_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Content not found"
            )
        return {
            "status": "success",
            "message": "Content deleted successfully",
            "content_id": str(content_id),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete content: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete content",
        )
