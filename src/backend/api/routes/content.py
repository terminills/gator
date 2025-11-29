"""
Content Generation API Routes

Handles content generation requests and management.
"""

from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config.logging import get_logger
from backend.database.connection import get_db_session
from backend.models.content import ContentCreate, ContentResponse, GenerationRequest
from backend.services.content_generation_service import ContentGenerationService

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
            from backend.models.content import ContentType, GenerationRequest

            request = GenerationRequest(
                content_type=ContentType.IMAGE,
                prompt="Create engaging social media content showcasing the persona's unique style and personality",
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


@router.post("/generate/all", status_code=status.HTTP_202_ACCEPTED)
async def generate_content_for_all_personas(
    content_type: str = Query(
        default="image", description="Type of content to generate"
    ),
    quality: Optional[str] = Query(
        default=None, description="Quality level override (None=use persona defaults)"
    ),
    content_rating: Optional[str] = Query(
        default=None, description="Content rating override (None=use persona defaults)"
    ),
    content_service: ContentGenerationService = Depends(get_content_service),
):
    """
    Generate content for all active personas.

    This endpoint creates content for every active persona in the system,
    making it easy to generate a batch of content with one request.

    When quality or content_rating are not specified (None), each persona
    will use their own default settings from their configuration.

    Args:
        content_type: Type of content (image, video, text, etc.)
        quality: Quality level override (None to use persona defaults)
        content_rating: Content rating override (None to use persona defaults)
        content_service: Injected content generation service

    Returns:
        Batch generation results with statistics

    Raises:
        400: Invalid parameters
        500: Generation failed
    """
    try:
        from backend.models.content import ContentRating, ContentType

        # Parse and validate parameters
        try:
            ct = ContentType(content_type.lower())
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid content_type: {content_type}. Must be one of: image, video, text, audio, voice",
            )

        # Parse content rating if provided, otherwise None means use persona defaults
        cr = None
        if content_rating is not None:
            try:
                cr = ContentRating(content_rating.lower())
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid content_rating: {content_rating}. Must be one of: sfw, moderate, nsfw",
                )

        # Generate content for all personas
        result = await content_service.generate_content_for_all_personas(
            content_type=ct,
            quality=quality,  # Can be None to use persona defaults
            content_rating=cr,  # Can be None to use persona defaults
        )

        # Handle case where result might not have total_personas (e.g., when no personas found)
        persona_count = result.get("total_personas", 0)

        return {
            "status": "accepted",
            "message": f"Batch content generation completed for {persona_count} personas",
            "details": result,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch content generation failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch content generation failed: {str(e)}",
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


@router.get("/{content_id}/status", status_code=status.HTTP_200_OK)
async def get_content_generation_status(
    content_id: UUID,
    content_service: ContentGenerationService = Depends(get_content_service),
    db: AsyncSession = Depends(get_db_session),
):
    """
    Get real-time generation status for content.

    Retrieves ACD context to show what AI agents are actually doing
    during content generation.

    Args:
        content_id: Unique content identifier
        content_service: Injected content generation service
        db: Database session

    Returns:
        Generation status with ACD context details

    Raises:
        404: Content not found
        500: Status retrieval failed
    """
    try:
        # Get content record
        content = await content_service.get_content(content_id)
        if not content:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Content not found"
            )

        # Extract ACD context ID from generation params
        acd_context_id = None
        if content.generation_params and isinstance(content.generation_params, dict):
            acd_context_id = content.generation_params.get("acd_context_id")

        # If no ACD context, return basic status
        if not acd_context_id:
            return {
                "status": "completed",
                "content_id": str(content_id),
                "message": "Content generation completed (no tracking context)",
                "has_acd_context": False,
            }

        # Get ACD context for detailed status
        from backend.services.acd_service import ACDService

        acd_service = ACDService(db)

        try:
            from uuid import UUID as UUIDType

            acd_context_id_uuid = (
                UUIDType(acd_context_id)
                if isinstance(acd_context_id, str)
                else acd_context_id
            )
            acd_context = await acd_service.get_context(acd_context_id_uuid)
        except Exception as e:
            logger.warning(f"Failed to parse ACD context ID: {str(e)}")
            acd_context = None

        if not acd_context:
            return {
                "status": "completed",
                "content_id": str(content_id),
                "message": "Content generation completed (context not found)",
                "has_acd_context": False,
            }

        # Return detailed status with ACD context
        return {
            "status": "tracked",
            "content_id": str(content_id),
            "has_acd_context": True,
            "acd_context": {
                "id": str(acd_context.id),
                "phase": acd_context.ai_phase,
                "state": acd_context.ai_state,
                "status": acd_context.ai_status,
                "confidence": acd_context.ai_confidence,
                "queue_status": acd_context.ai_queue_status,
                "queue_priority": acd_context.ai_queue_priority,
                "note": acd_context.ai_note,
                "assigned_to": acd_context.ai_assigned_to,
                "context": acd_context.ai_context,
                "metadata": acd_context.ai_metadata,
                "created_at": (
                    acd_context.created_at.isoformat()
                    if acd_context.created_at
                    else None
                ),
                "updated_at": (
                    acd_context.updated_at.isoformat()
                    if acd_context.updated_at
                    else None
                ),
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get content generation status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve generation status",
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
