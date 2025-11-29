"""
Scheduled Posts API Routes

API endpoints for managing scheduled social media posts.
Provides CRUD operations for scheduling, modifying, and cancelling posts.
"""

from datetime import datetime
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config.logging import get_logger
from backend.database.connection import get_db_session
from backend.models.scheduled_post import (
    ScheduledPostCreate,
    ScheduledPostListResponse,
    ScheduledPostResponse,
    ScheduledPostStats,
    ScheduledPostStatus,
    ScheduledPostUpdate,
)
from backend.services.scheduled_post_service import ScheduledPostService

logger = get_logger(__name__)

router = APIRouter(
    prefix="/api/v1/scheduled-posts",
    tags=["scheduled-posts"],
    responses={404: {"description": "Scheduled post not found"}},
)


def get_scheduled_post_service(
    db: AsyncSession = Depends(get_db_session),
) -> ScheduledPostService:
    """Dependency injection for ScheduledPostService."""
    return ScheduledPostService(db)


@router.post(
    "/",
    response_model=ScheduledPostResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_scheduled_post(
    data: ScheduledPostCreate,
    service: ScheduledPostService = Depends(get_scheduled_post_service),
):
    """
    Create a new scheduled post.

    Schedules content to be published to a social media platform at a specified
    future time. Creates a database record and schedules a Celery task.

    Args:
        data: Scheduled post creation data including content_id, platform,
              scheduled_at, and optional caption/hashtags

    Returns:
        Created scheduled post with ID and task information

    Raises:
        400: Invalid data (e.g., scheduled time in the past)
        500: Failed to create scheduled post
    """
    try:
        result = await service.create_scheduled_post(data)
        logger.info(
            f"Scheduled post created: id={result.id} "
            f"platform={data.platform} scheduled_at={data.scheduled_at}"
        )
        return result
    except ValueError as e:
        logger.warning(f"Validation error creating scheduled post: {str(e)}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to create scheduled post: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create scheduled post",
        )


@router.get("/", response_model=ScheduledPostListResponse)
async def list_scheduled_posts(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    status_filter: Optional[ScheduledPostStatus] = Query(
        None, alias="status", description="Filter by status"
    ),
    platform: Optional[str] = Query(None, description="Filter by platform"),
    persona_id: Optional[UUID] = Query(None, description="Filter by persona"),
    content_id: Optional[UUID] = Query(None, description="Filter by content"),
    from_date: Optional[datetime] = Query(None, description="Filter from date"),
    to_date: Optional[datetime] = Query(None, description="Filter to date"),
    service: ScheduledPostService = Depends(get_scheduled_post_service),
):
    """
    List scheduled posts with filtering and pagination.

    Retrieve scheduled posts with various filtering options and pagination support.

    Args:
        page: Page number (1-indexed)
        page_size: Number of items per page (max 100)
        status: Filter by post status (pending, published, failed, etc.)
        platform: Filter by social media platform
        persona_id: Filter by persona UUID
        content_id: Filter by content UUID
        from_date: Filter posts scheduled from this date
        to_date: Filter posts scheduled until this date

    Returns:
        Paginated list of scheduled posts with total count
    """
    try:
        result = await service.list_scheduled_posts(
            page=page,
            page_size=page_size,
            status=status_filter,
            platform=platform,
            persona_id=persona_id,
            content_id=content_id,
            from_date=from_date,
            to_date=to_date,
        )
        return result
    except Exception as e:
        logger.error(f"Failed to list scheduled posts: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list scheduled posts",
        )


@router.get("/stats", response_model=ScheduledPostStats)
async def get_scheduled_posts_stats(
    persona_id: Optional[UUID] = Query(None, description="Filter by persona"),
    time_window_hours: int = Query(
        24, ge=1, le=168, description="Time window for upcoming count"
    ),
    service: ScheduledPostService = Depends(get_scheduled_post_service),
):
    """
    Get scheduled post statistics.

    Returns aggregate statistics about scheduled posts including counts by status,
    platform distribution, and upcoming posts count.

    Args:
        persona_id: Optional filter by persona
        time_window_hours: Time window for counting upcoming posts (default 24h)

    Returns:
        Statistics including total counts, by-status breakdown, and distributions
    """
    try:
        result = await service.get_stats(
            persona_id=persona_id,
            time_window_hours=time_window_hours,
        )
        return result
    except Exception as e:
        logger.error(f"Failed to get scheduled post stats: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get statistics",
        )


@router.get("/upcoming")
async def get_upcoming_posts(
    hours: int = Query(24, ge=1, le=168, description="Time window in hours"),
    persona_id: Optional[UUID] = Query(None, description="Filter by persona"),
    limit: int = Query(50, ge=1, le=100, description="Maximum results"),
    service: ScheduledPostService = Depends(get_scheduled_post_service),
):
    """
    Get upcoming scheduled posts.

    Retrieve posts scheduled to be published within a time window.

    Args:
        hours: Time window in hours (default 24h, max 168h/1 week)
        persona_id: Optional filter by persona
        limit: Maximum number of results

    Returns:
        List of upcoming scheduled posts ordered by scheduled time
    """
    try:
        result = await service.get_upcoming_posts(
            hours=hours,
            persona_id=persona_id,
            limit=limit,
        )
        return {"posts": result, "count": len(result), "hours": hours}
    except Exception as e:
        logger.error(f"Failed to get upcoming posts: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get upcoming posts",
        )


@router.get("/{post_id}", response_model=ScheduledPostResponse)
async def get_scheduled_post(
    post_id: UUID,
    service: ScheduledPostService = Depends(get_scheduled_post_service),
):
    """
    Get a scheduled post by ID.

    Retrieve details of a specific scheduled post.

    Args:
        post_id: UUID of the scheduled post

    Returns:
        Scheduled post details

    Raises:
        404: Scheduled post not found
    """
    try:
        result = await service.get_scheduled_post(post_id)
        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Scheduled post {post_id} not found",
            )
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get scheduled post {post_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get scheduled post",
        )


@router.put("/{post_id}", response_model=ScheduledPostResponse)
async def update_scheduled_post(
    post_id: UUID,
    data: ScheduledPostUpdate,
    service: ScheduledPostService = Depends(get_scheduled_post_service),
):
    """
    Update a scheduled post.

    Modify a pending or paused scheduled post. If the scheduled time is changed,
    the Celery task will be rescheduled.

    Args:
        post_id: UUID of the scheduled post
        data: Update data (all fields optional)

    Returns:
        Updated scheduled post

    Raises:
        400: Cannot update post with current status
        404: Scheduled post not found
    """
    try:
        result = await service.update_scheduled_post(post_id, data)
        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Scheduled post {post_id} not found",
            )
        logger.info(f"Scheduled post updated: {post_id}")
        return result
    except HTTPException:
        raise
    except ValueError as e:
        logger.warning(f"Validation error updating scheduled post: {str(e)}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to update scheduled post {post_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update scheduled post",
        )


@router.post("/{post_id}/cancel", response_model=ScheduledPostResponse)
async def cancel_scheduled_post(
    post_id: UUID,
    service: ScheduledPostService = Depends(get_scheduled_post_service),
):
    """
    Cancel a scheduled post.

    Cancels a pending, paused, or processing scheduled post and revokes
    the associated Celery task.

    Args:
        post_id: UUID of the scheduled post

    Returns:
        Cancelled scheduled post

    Raises:
        400: Cannot cancel post with current status
        404: Scheduled post not found
    """
    try:
        result = await service.cancel_scheduled_post(post_id)
        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Scheduled post {post_id} not found",
            )
        logger.info(f"Scheduled post cancelled: {post_id}")
        return result
    except HTTPException:
        raise
    except ValueError as e:
        logger.warning(f"Cannot cancel scheduled post: {str(e)}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to cancel scheduled post {post_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to cancel scheduled post",
        )


@router.post("/{post_id}/pause", response_model=ScheduledPostResponse)
async def pause_scheduled_post(
    post_id: UUID,
    service: ScheduledPostService = Depends(get_scheduled_post_service),
):
    """
    Pause a scheduled post.

    Temporarily pauses a pending scheduled post. The post can be resumed later.
    The associated Celery task is revoked.

    Args:
        post_id: UUID of the scheduled post

    Returns:
        Paused scheduled post

    Raises:
        400: Cannot pause post with current status
        404: Scheduled post not found
    """
    try:
        result = await service.pause_scheduled_post(post_id)
        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Scheduled post {post_id} not found",
            )
        logger.info(f"Scheduled post paused: {post_id}")
        return result
    except HTTPException:
        raise
    except ValueError as e:
        logger.warning(f"Cannot pause scheduled post: {str(e)}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to pause scheduled post {post_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to pause scheduled post",
        )


@router.post("/{post_id}/resume", response_model=ScheduledPostResponse)
async def resume_scheduled_post(
    post_id: UUID,
    service: ScheduledPostService = Depends(get_scheduled_post_service),
):
    """
    Resume a paused scheduled post.

    Resumes a previously paused scheduled post. A new Celery task is scheduled.

    Args:
        post_id: UUID of the scheduled post

    Returns:
        Resumed scheduled post

    Raises:
        400: Cannot resume post (not paused or time passed)
        404: Scheduled post not found
    """
    try:
        result = await service.resume_scheduled_post(post_id)
        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Scheduled post {post_id} not found",
            )
        logger.info(f"Scheduled post resumed: {post_id}")
        return result
    except HTTPException:
        raise
    except ValueError as e:
        logger.warning(f"Cannot resume scheduled post: {str(e)}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to resume scheduled post {post_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to resume scheduled post",
        )


@router.post("/{post_id}/retry", response_model=ScheduledPostResponse)
async def retry_failed_post(
    post_id: UUID,
    service: ScheduledPostService = Depends(get_scheduled_post_service),
):
    """
    Retry a failed scheduled post.

    Retries publishing a failed scheduled post. Increments the retry count.

    Args:
        post_id: UUID of the scheduled post

    Returns:
        Retrying scheduled post

    Raises:
        400: Cannot retry (not failed or max retries exceeded)
        404: Scheduled post not found
    """
    try:
        result = await service.retry_failed_post(post_id)
        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Scheduled post {post_id} not found",
            )
        logger.info(f"Scheduled post retry initiated: {post_id}")
        return result
    except HTTPException:
        raise
    except ValueError as e:
        logger.warning(f"Cannot retry scheduled post: {str(e)}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to retry scheduled post {post_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retry scheduled post",
        )


@router.delete("/{post_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_scheduled_post(
    post_id: UUID,
    service: ScheduledPostService = Depends(get_scheduled_post_service),
):
    """
    Delete a scheduled post.

    Permanently deletes a cancelled scheduled post. Active posts must be
    cancelled first.

    Args:
        post_id: UUID of the scheduled post

    Raises:
        400: Post must be cancelled before deletion
        404: Scheduled post not found
    """
    try:
        # Get the post first
        post = await service.get_scheduled_post(post_id)
        if not post:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Scheduled post {post_id} not found",
            )

        # Only allow deletion of cancelled or published posts
        if post.status not in [
            ScheduledPostStatus.CANCELLED.value,
            ScheduledPostStatus.PUBLISHED.value,
        ]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Post must be cancelled or published before deletion",
            )

        # Delete from database
        from sqlalchemy import delete
        from backend.models.scheduled_post import ScheduledPostModel

        stmt = delete(ScheduledPostModel).where(ScheduledPostModel.id == post_id)
        await service.db.execute(stmt)
        await service.db.commit()

        logger.info(f"Scheduled post deleted: {post_id}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete scheduled post {post_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete scheduled post",
        )
