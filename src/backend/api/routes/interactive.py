"""
Interactive Content API Routes

API endpoints for managing interactive content (polls, stories, Q&A).
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database.connection import get_db_session
from backend.models.interactive_content import (
    InteractiveContentCreate,
    InteractiveContentResponseCreate,
    InteractiveContentResponseSchema,
    InteractiveContentSchema,
    InteractiveContentStats,
    InteractiveContentStatus,
    InteractiveContentType,
    InteractiveContentUpdate,
)
from backend.services.interactive_content_service import InteractiveContentService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/interactive", tags=["interactive"])


@router.post(
    "/", response_model=InteractiveContentSchema, status_code=status.HTTP_201_CREATED
)
async def create_interactive_content(
    request: InteractiveContentCreate, session: AsyncSession = Depends(get_db_session)
):
    """
    Create new interactive content.

    Supports:
    - Polls with multiple choice options
    - Stories with 24-hour expiration
    - Q&A sessions
    - Quizzes

    Args:
        request: Content creation parameters

    Returns:
        Created interactive content
    """
    try:
        service = InteractiveContentService(session)
        content = await service.create_content(
            persona_id=request.persona_id,
            content_type=request.content_type,
            title=request.title,
            question=request.question,
            description=request.description,
            options=request.options,
            media_url=request.media_url,
            expires_at=request.expires_at,
        )

        logger.info(
            f"Created interactive content: id={content.id}, type={request.content_type}"
        )

        return InteractiveContentSchema(
            id=str(content.id),
            persona_id=str(content.persona_id),
            content_type=content.content_type,
            title=content.title,
            question=content.question,
            description=content.description,
            options=content.options,
            responses=content.responses,
            media_url=content.media_url,
            status=content.status,
            view_count=content.view_count,
            response_count=content.response_count,
            share_count=content.share_count,
            published_at=content.published_at,
            expires_at=content.expires_at,
            created_at=content.created_at,
            updated_at=content.updated_at,
        )

    except Exception as e:
        logger.error(f"Failed to create interactive content: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create interactive content: {str(e)}",
        )


@router.get("/{content_id}", response_model=InteractiveContentSchema)
async def get_interactive_content(
    content_id: str, session: AsyncSession = Depends(get_db_session)
):
    """
    Get interactive content by ID.

    Args:
        content_id: Content ID

    Returns:
        Interactive content details
    """
    try:
        service = InteractiveContentService(session)
        content = await service.get_content(content_id)

        if not content:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Interactive content not found: {content_id}",
            )

        # Increment view count
        await service.increment_view_count(content_id)

        return InteractiveContentSchema(
            id=str(content.id),
            persona_id=str(content.persona_id),
            content_type=content.content_type,
            title=content.title,
            question=content.question,
            description=content.description,
            options=content.options,
            responses=content.responses,
            media_url=content.media_url,
            status=content.status,
            view_count=content.view_count,
            response_count=content.response_count,
            share_count=content.share_count,
            published_at=content.published_at,
            expires_at=content.expires_at,
            created_at=content.created_at,
            updated_at=content.updated_at,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get interactive content: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get interactive content: {str(e)}",
        )


@router.get("/", response_model=List[InteractiveContentSchema])
async def list_interactive_content(
    persona_id: Optional[str] = Query(None, description="Filter by persona ID"),
    content_type: Optional[InteractiveContentType] = Query(
        None, description="Filter by content type"
    ),
    status: Optional[InteractiveContentStatus] = Query(
        None, description="Filter by status"
    ),
    include_expired: bool = Query(False, description="Include expired content"),
    limit: int = Query(50, ge=1, le=100, description="Maximum results"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    session: AsyncSession = Depends(get_db_session),
):
    """
    List interactive content with filters.

    Args:
        persona_id: Filter by persona
        content_type: Filter by type (poll, story, qna, quiz)
        status: Filter by status
        include_expired: Include expired content
        limit: Maximum results
        offset: Pagination offset

    Returns:
        List of interactive content
    """
    try:
        service = InteractiveContentService(session)
        contents = await service.list_content(
            persona_id=persona_id,
            content_type=content_type,
            status=status,
            include_expired=include_expired,
            limit=limit,
            offset=offset,
        )

        return [
            InteractiveContentSchema(
                id=str(content.id),
                persona_id=str(content.persona_id),
                content_type=content.content_type,
                title=content.title,
                question=content.question,
                description=content.description,
                options=content.options,
                responses=content.responses,
                media_url=content.media_url,
                status=content.status,
                view_count=content.view_count,
                response_count=content.response_count,
                share_count=content.share_count,
                published_at=content.published_at,
                expires_at=content.expires_at,
                created_at=content.created_at,
                updated_at=content.updated_at,
            )
            for content in contents
        ]

    except Exception as e:
        logger.error(f"Failed to list interactive content: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list interactive content: {str(e)}",
        )


@router.put("/{content_id}", response_model=InteractiveContentSchema)
async def update_interactive_content(
    content_id: str,
    request: InteractiveContentUpdate,
    session: AsyncSession = Depends(get_db_session),
):
    """
    Update interactive content.

    Args:
        content_id: Content ID
        request: Update parameters

    Returns:
        Updated interactive content
    """
    try:
        service = InteractiveContentService(session)

        updates = {}
        if request.title is not None:
            updates["title"] = request.title
        if request.question is not None:
            updates["question"] = request.question
        if request.description is not None:
            updates["description"] = request.description
        if request.options is not None:
            updates["options"] = request.options
        if request.media_url is not None:
            updates["media_url"] = request.media_url
        if request.status is not None:
            updates["status"] = request.status.value
        if request.expires_at is not None:
            updates["expires_at"] = request.expires_at

        content = await service.update_content(content_id, **updates)

        if not content:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Interactive content not found: {content_id}",
            )

        logger.info(f"Updated interactive content: id={content_id}")

        return InteractiveContentSchema(
            id=str(content.id),
            persona_id=str(content.persona_id),
            content_type=content.content_type,
            title=content.title,
            question=content.question,
            description=content.description,
            options=content.options,
            responses=content.responses,
            media_url=content.media_url,
            status=content.status,
            view_count=content.view_count,
            response_count=content.response_count,
            share_count=content.share_count,
            published_at=content.published_at,
            expires_at=content.expires_at,
            created_at=content.created_at,
            updated_at=content.updated_at,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update interactive content: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update interactive content: {str(e)}",
        )


@router.post("/{content_id}/publish", response_model=InteractiveContentSchema)
async def publish_interactive_content(
    content_id: str, session: AsyncSession = Depends(get_db_session)
):
    """
    Publish interactive content (change status to active).

    Args:
        content_id: Content ID

    Returns:
        Published interactive content
    """
    try:
        service = InteractiveContentService(session)
        content = await service.publish_content(content_id)

        if not content:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Interactive content not found: {content_id}",
            )

        logger.info(f"Published interactive content: id={content_id}")

        return InteractiveContentSchema(
            id=str(content.id),
            persona_id=str(content.persona_id),
            content_type=content.content_type,
            title=content.title,
            question=content.question,
            description=content.description,
            options=content.options,
            responses=content.responses,
            media_url=content.media_url,
            status=content.status,
            view_count=content.view_count,
            response_count=content.response_count,
            share_count=content.share_count,
            published_at=content.published_at,
            expires_at=content.expires_at,
            created_at=content.created_at,
            updated_at=content.updated_at,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to publish interactive content: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to publish interactive content: {str(e)}",
        )


@router.delete("/{content_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_interactive_content(
    content_id: str, session: AsyncSession = Depends(get_db_session)
):
    """
    Delete interactive content.

    Args:
        content_id: Content ID
    """
    try:
        service = InteractiveContentService(session)
        deleted = await service.delete_content(content_id)

        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Interactive content not found: {content_id}",
            )

        logger.info(f"Deleted interactive content: id={content_id}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete interactive content: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete interactive content: {str(e)}",
        )


@router.post(
    "/{content_id}/respond",
    response_model=InteractiveContentResponseSchema,
    status_code=status.HTTP_201_CREATED,
)
async def submit_response(
    content_id: str,
    request: InteractiveContentResponseCreate,
    session: AsyncSession = Depends(get_db_session),
):
    """
    Submit a response to interactive content.

    For polls: Include {"option_id": 1} in response_data
    For Q&A: Include {"answer": "text"} in response_data
    For quizzes: Include {"answers": [1, 3, 4]} in response_data

    Args:
        content_id: Content ID
        request: Response data

    Returns:
        Created response
    """
    try:
        service = InteractiveContentService(session)
        response = await service.submit_response(
            content_id=content_id,
            response_data=request.response_data,
            user_id=request.user_id,
        )

        logger.info(
            f"Submitted response to interactive content: content_id={content_id}"
        )

        return InteractiveContentResponseSchema(
            id=str(response.id),
            content_id=str(response.content_id),
            user_id=str(response.user_id) if response.user_id else None,
            response_data=response.response_data,
            user_identifier=response.user_identifier,
            created_at=response.created_at,
        )

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to submit response: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to submit response: {str(e)}",
        )


@router.get("/{content_id}/stats", response_model=InteractiveContentStats)
async def get_content_stats(
    content_id: str, session: AsyncSession = Depends(get_db_session)
):
    """
    Get statistics for interactive content.

    Provides:
    - View, response, and share counts
    - Response rate
    - Top options (for polls)

    Args:
        content_id: Content ID

    Returns:
        Content statistics
    """
    try:
        service = InteractiveContentService(session)
        stats = await service.get_content_stats(content_id)

        return InteractiveContentStats(**stats)

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to get content stats: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get content stats: {str(e)}",
        )


@router.post("/{content_id}/share", status_code=status.HTTP_200_OK)
async def share_content(
    content_id: str, session: AsyncSession = Depends(get_db_session)
):
    """
    Increment share count for content.

    Args:
        content_id: Content ID

    Returns:
        Success message
    """
    try:
        service = InteractiveContentService(session)
        await service.increment_share_count(content_id)

        return {"message": "Share count incremented"}

    except Exception as e:
        logger.error(f"Failed to increment share count: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to increment share count: {str(e)}",
        )


@router.get("/health", response_model=Dict[str, Any])
async def interactive_service_health():
    """
    Check interactive content service health.

    Returns:
        Service health status
    """
    return {
        "status": "healthy",
        "service": "interactive_content",
        "version": "1.0.0",
        "features": [
            "polls",
            "stories",
            "qna",
            "quizzes",
            "response_tracking",
            "analytics",
            "expiration",
        ],
    }
