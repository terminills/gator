"""
Scheduled Post Service

Service layer for managing scheduled social media posts.
Provides CRUD operations and integrates with Celery for task scheduling.
"""

from datetime import datetime, timedelta
from math import ceil
from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlalchemy import and_, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config.logging import get_logger
from backend.models.scheduled_post import (
    ScheduledPostCreate,
    ScheduledPostListResponse,
    ScheduledPostModel,
    ScheduledPostResponse,
    ScheduledPostStats,
    ScheduledPostStatus,
    ScheduledPostUpdate,
)

logger = get_logger(__name__)


class ScheduledPostService:
    """
    Service for managing scheduled social media posts.

    Provides operations for creating, listing, updating, and cancelling
    scheduled posts, with integration to Celery for task execution.
    """

    def __init__(self, db_session: AsyncSession):
        """
        Initialize scheduled post service.

        Args:
            db_session: Database session for persistence
        """
        self.db = db_session

    async def create_scheduled_post(
        self, data: ScheduledPostCreate
    ) -> ScheduledPostResponse:
        """
        Create a new scheduled post and schedule the Celery task.

        Args:
            data: Scheduled post creation data

        Returns:
            Created scheduled post response

        Raises:
            ValueError: If scheduled_at is in the past
        """
        # Validate schedule time
        if data.scheduled_at <= datetime.now(data.scheduled_at.tzinfo):
            raise ValueError("Scheduled time must be in the future")

        # Create database record
        scheduled_post = ScheduledPostModel(
            content_id=data.content_id,
            persona_id=data.persona_id,
            platform=data.platform,
            scheduled_at=data.scheduled_at,
            caption=data.caption,
            hashtags=data.hashtags,
            platform_specific_data=data.platform_specific_data,
            acd_context_id=data.acd_context_id,
            priority=data.priority,
            is_recurring=data.is_recurring,
            recurrence_pattern=data.recurrence_pattern,
            notes=data.notes,
            max_retries=data.max_retries,
            status=ScheduledPostStatus.PENDING.value,
        )

        self.db.add(scheduled_post)
        await self.db.commit()
        await self.db.refresh(scheduled_post)

        # Schedule Celery task
        celery_task_id = await self._schedule_celery_task(scheduled_post)

        # Update with task ID
        scheduled_post.celery_task_id = celery_task_id
        await self.db.commit()
        await self.db.refresh(scheduled_post)

        logger.info(
            f"Scheduled post created: {scheduled_post.id} "
            f"scheduled_at={data.scheduled_at} platform={data.platform}"
        )

        return ScheduledPostResponse.model_validate(scheduled_post)

    async def get_scheduled_post(
        self, post_id: UUID
    ) -> Optional[ScheduledPostResponse]:
        """
        Get a scheduled post by ID.

        Args:
            post_id: UUID of the scheduled post

        Returns:
            Scheduled post response or None if not found
        """
        stmt = select(ScheduledPostModel).where(ScheduledPostModel.id == post_id)
        result = await self.db.execute(stmt)
        post = result.scalar_one_or_none()

        if post:
            return ScheduledPostResponse.model_validate(post)
        return None

    async def list_scheduled_posts(
        self,
        page: int = 1,
        page_size: int = 20,
        status: Optional[ScheduledPostStatus] = None,
        platform: Optional[str] = None,
        persona_id: Optional[UUID] = None,
        content_id: Optional[UUID] = None,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
    ) -> ScheduledPostListResponse:
        """
        List scheduled posts with filtering and pagination.

        Args:
            page: Page number (1-indexed)
            page_size: Number of items per page
            status: Filter by status
            platform: Filter by platform
            persona_id: Filter by persona
            content_id: Filter by content
            from_date: Filter posts scheduled from this date
            to_date: Filter posts scheduled until this date

        Returns:
            Paginated list of scheduled posts
        """
        # Build base query
        conditions = []

        if status:
            conditions.append(ScheduledPostModel.status == status.value)
        if platform:
            conditions.append(ScheduledPostModel.platform == platform)
        if persona_id:
            conditions.append(ScheduledPostModel.persona_id == persona_id)
        if content_id:
            conditions.append(ScheduledPostModel.content_id == content_id)
        if from_date:
            conditions.append(ScheduledPostModel.scheduled_at >= from_date)
        if to_date:
            conditions.append(ScheduledPostModel.scheduled_at <= to_date)

        # Count total
        count_stmt = select(func.count(ScheduledPostModel.id))
        if conditions:
            count_stmt = count_stmt.where(and_(*conditions))
        total_result = await self.db.execute(count_stmt)
        total = total_result.scalar()

        # Get paginated results
        offset = (page - 1) * page_size
        stmt = (
            select(ScheduledPostModel)
            .order_by(ScheduledPostModel.scheduled_at.asc())
            .offset(offset)
            .limit(page_size)
        )
        if conditions:
            stmt = stmt.where(and_(*conditions))

        result = await self.db.execute(stmt)
        posts = result.scalars().all()

        return ScheduledPostListResponse(
            posts=[ScheduledPostResponse.model_validate(p) for p in posts],
            total=total,
            page=page,
            page_size=page_size,
            total_pages=ceil(total / page_size) if total > 0 else 0,
        )

    async def update_scheduled_post(
        self, post_id: UUID, data: ScheduledPostUpdate
    ) -> Optional[ScheduledPostResponse]:
        """
        Update a scheduled post.

        Args:
            post_id: UUID of the scheduled post
            data: Update data

        Returns:
            Updated scheduled post response or None if not found

        Raises:
            ValueError: If trying to update a non-pending post or invalid status
        """
        stmt = select(ScheduledPostModel).where(ScheduledPostModel.id == post_id)
        result = await self.db.execute(stmt)
        post = result.scalar_one_or_none()

        if not post:
            return None

        # Only allow updates to pending or paused posts (for most fields)
        if post.status not in [
            ScheduledPostStatus.PENDING.value,
            ScheduledPostStatus.PAUSED.value,
        ]:
            # Allow status changes even for non-pending
            if data.status is None:
                raise ValueError(
                    f"Cannot update post with status: {post.status}. "
                    "Only pending or paused posts can be modified."
                )

        # Update fields if provided
        update_data = data.model_dump(exclude_unset=True)

        for field, value in update_data.items():
            if field == "status" and value:
                value = value.value if isinstance(value, ScheduledPostStatus) else value
            setattr(post, field, value)

        # If schedule time changed, reschedule Celery task
        if data.scheduled_at:
            await self._cancel_celery_task(post.celery_task_id)
            new_task_id = await self._schedule_celery_task(post)
            post.celery_task_id = new_task_id

        await self.db.commit()
        await self.db.refresh(post)

        logger.info(f"Scheduled post updated: {post_id}")
        return ScheduledPostResponse.model_validate(post)

    async def cancel_scheduled_post(self, post_id: UUID) -> Optional[ScheduledPostResponse]:
        """
        Cancel a scheduled post.

        Args:
            post_id: UUID of the scheduled post

        Returns:
            Cancelled scheduled post response or None if not found

        Raises:
            ValueError: If post cannot be cancelled
        """
        stmt = select(ScheduledPostModel).where(ScheduledPostModel.id == post_id)
        result = await self.db.execute(stmt)
        post = result.scalar_one_or_none()

        if not post:
            return None

        # Only allow cancellation of pending, paused, or processing posts
        if post.status not in [
            ScheduledPostStatus.PENDING.value,
            ScheduledPostStatus.PAUSED.value,
            ScheduledPostStatus.PROCESSING.value,
        ]:
            raise ValueError(
                f"Cannot cancel post with status: {post.status}. "
                "Only pending, paused, or processing posts can be cancelled."
            )

        # Cancel Celery task
        if post.celery_task_id:
            await self._cancel_celery_task(post.celery_task_id)

        # Update status
        post.status = ScheduledPostStatus.CANCELLED.value
        post.cancelled_at = datetime.utcnow()

        await self.db.commit()
        await self.db.refresh(post)

        logger.info(f"Scheduled post cancelled: {post_id}")
        return ScheduledPostResponse.model_validate(post)

    async def pause_scheduled_post(self, post_id: UUID) -> Optional[ScheduledPostResponse]:
        """
        Pause a scheduled post.

        Args:
            post_id: UUID of the scheduled post

        Returns:
            Paused scheduled post response or None if not found
        """
        stmt = select(ScheduledPostModel).where(ScheduledPostModel.id == post_id)
        result = await self.db.execute(stmt)
        post = result.scalar_one_or_none()

        if not post:
            return None

        if post.status != ScheduledPostStatus.PENDING.value:
            raise ValueError(f"Cannot pause post with status: {post.status}")

        # Cancel Celery task
        if post.celery_task_id:
            await self._cancel_celery_task(post.celery_task_id)

        post.status = ScheduledPostStatus.PAUSED.value
        post.celery_task_id = None

        await self.db.commit()
        await self.db.refresh(post)

        logger.info(f"Scheduled post paused: {post_id}")
        return ScheduledPostResponse.model_validate(post)

    async def resume_scheduled_post(self, post_id: UUID) -> Optional[ScheduledPostResponse]:
        """
        Resume a paused scheduled post.

        Args:
            post_id: UUID of the scheduled post

        Returns:
            Resumed scheduled post response or None if not found
        """
        stmt = select(ScheduledPostModel).where(ScheduledPostModel.id == post_id)
        result = await self.db.execute(stmt)
        post = result.scalar_one_or_none()

        if not post:
            return None

        if post.status != ScheduledPostStatus.PAUSED.value:
            raise ValueError(f"Cannot resume post with status: {post.status}")

        # If scheduled time is in the past, don't resume
        if post.scheduled_at <= datetime.utcnow():
            raise ValueError("Cannot resume: scheduled time has passed")

        # Schedule new Celery task
        task_id = await self._schedule_celery_task(post)

        post.status = ScheduledPostStatus.PENDING.value
        post.celery_task_id = task_id

        await self.db.commit()
        await self.db.refresh(post)

        logger.info(f"Scheduled post resumed: {post_id}")
        return ScheduledPostResponse.model_validate(post)

    async def retry_failed_post(self, post_id: UUID) -> Optional[ScheduledPostResponse]:
        """
        Retry a failed scheduled post.

        Args:
            post_id: UUID of the scheduled post

        Returns:
            Retried scheduled post response or None if not found
        """
        stmt = select(ScheduledPostModel).where(ScheduledPostModel.id == post_id)
        result = await self.db.execute(stmt)
        post = result.scalar_one_or_none()

        if not post:
            return None

        if post.status != ScheduledPostStatus.FAILED.value:
            raise ValueError(f"Cannot retry post with status: {post.status}")

        if post.retry_count >= post.max_retries:
            raise ValueError(f"Maximum retries ({post.max_retries}) exceeded")

        # Schedule for immediate execution
        post.scheduled_at = datetime.utcnow() + timedelta(seconds=30)
        task_id = await self._schedule_celery_task(post)

        post.status = ScheduledPostStatus.PENDING.value
        post.celery_task_id = task_id
        post.retry_count += 1
        post.error_message = None

        await self.db.commit()
        await self.db.refresh(post)

        logger.info(f"Scheduled post retry initiated: {post_id} (attempt {post.retry_count})")
        return ScheduledPostResponse.model_validate(post)

    async def get_stats(
        self,
        persona_id: Optional[UUID] = None,
        time_window_hours: int = 24,
    ) -> ScheduledPostStats:
        """
        Get statistics about scheduled posts.

        Args:
            persona_id: Optional filter by persona
            time_window_hours: Time window for upcoming posts count

        Returns:
            Scheduled post statistics
        """
        base_conditions = []
        if persona_id:
            base_conditions.append(ScheduledPostModel.persona_id == persona_id)

        # Count by status
        status_counts = {}
        for status in ScheduledPostStatus:
            stmt = select(func.count(ScheduledPostModel.id)).where(
                and_(
                    ScheduledPostModel.status == status.value,
                    *base_conditions,
                )
            )
            result = await self.db.execute(stmt)
            status_counts[status.value] = result.scalar()

        # Count upcoming in time window
        upcoming_cutoff = datetime.utcnow() + timedelta(hours=time_window_hours)
        stmt = select(func.count(ScheduledPostModel.id)).where(
            and_(
                ScheduledPostModel.status == ScheduledPostStatus.PENDING.value,
                ScheduledPostModel.scheduled_at <= upcoming_cutoff,
                *base_conditions,
            )
        )
        result = await self.db.execute(stmt)
        upcoming_24h = result.scalar()

        # Count by platform
        stmt = (
            select(
                ScheduledPostModel.platform,
                func.count(ScheduledPostModel.id).label("count"),
            )
            .group_by(ScheduledPostModel.platform)
        )
        if base_conditions:
            stmt = stmt.where(and_(*base_conditions))
        result = await self.db.execute(stmt)
        by_platform = {row.platform: row.count for row in result}

        # Count by persona (top 10)
        stmt = (
            select(
                ScheduledPostModel.persona_id,
                func.count(ScheduledPostModel.id).label("count"),
            )
            .group_by(ScheduledPostModel.persona_id)
            .order_by(func.count(ScheduledPostModel.id).desc())
            .limit(10)
        )
        result = await self.db.execute(stmt)
        by_persona = {str(row.persona_id): row.count for row in result}

        total = sum(status_counts.values())

        return ScheduledPostStats(
            total_scheduled=total,
            pending=status_counts.get(ScheduledPostStatus.PENDING.value, 0),
            processing=status_counts.get(ScheduledPostStatus.PROCESSING.value, 0),
            published=status_counts.get(ScheduledPostStatus.PUBLISHED.value, 0),
            failed=status_counts.get(ScheduledPostStatus.FAILED.value, 0),
            cancelled=status_counts.get(ScheduledPostStatus.CANCELLED.value, 0),
            paused=status_counts.get(ScheduledPostStatus.PAUSED.value, 0),
            upcoming_24h=upcoming_24h,
            by_platform=by_platform,
            by_persona=by_persona,
        )

    async def get_upcoming_posts(
        self,
        hours: int = 24,
        persona_id: Optional[UUID] = None,
        limit: int = 50,
    ) -> List[ScheduledPostResponse]:
        """
        Get upcoming scheduled posts.

        Args:
            hours: Time window in hours
            persona_id: Optional filter by persona
            limit: Maximum number of results

        Returns:
            List of upcoming scheduled posts
        """
        cutoff = datetime.utcnow() + timedelta(hours=hours)

        conditions = [
            ScheduledPostModel.status == ScheduledPostStatus.PENDING.value,
            ScheduledPostModel.scheduled_at <= cutoff,
        ]
        if persona_id:
            conditions.append(ScheduledPostModel.persona_id == persona_id)

        stmt = (
            select(ScheduledPostModel)
            .where(and_(*conditions))
            .order_by(ScheduledPostModel.scheduled_at.asc())
            .limit(limit)
        )

        result = await self.db.execute(stmt)
        posts = result.scalars().all()

        return [ScheduledPostResponse.model_validate(p) for p in posts]

    async def _schedule_celery_task(self, post: ScheduledPostModel) -> Optional[str]:
        """
        Schedule a Celery task for publishing.

        Args:
            post: Scheduled post model

        Returns:
            Celery task ID or None if scheduling failed
        """
        try:
            from backend.tasks.social_media_tasks import publish_scheduled_post

            # Prepare post data for task
            post_data = {
                "content_id": str(post.content_id),
                "platforms": [post.platform],
                "caption": post.caption,
                "hashtags": post.hashtags or [],
                "scheduled_post_id": str(post.id),
            }

            # Schedule with ETA
            task = publish_scheduled_post.apply_async(
                args=[str(post.id), post_data],
                eta=post.scheduled_at,
            )

            logger.info(
                f"Celery task scheduled: task_id={task.id} "
                f"scheduled_post_id={post.id} eta={post.scheduled_at}"
            )

            return task.id

        except Exception as e:
            logger.error(f"Failed to schedule Celery task: {str(e)}")
            return None

    async def _cancel_celery_task(self, task_id: Optional[str]) -> bool:
        """
        Cancel a Celery task.

        Args:
            task_id: Celery task ID

        Returns:
            True if cancelled successfully
        """
        if not task_id:
            return True

        try:
            from backend.celery_app import app

            app.control.revoke(task_id, terminate=True)

            logger.info(f"Celery task cancelled: {task_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to cancel Celery task {task_id}: {str(e)}")
            return False
