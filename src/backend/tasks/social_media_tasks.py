"""
Social Media Background Tasks

Celery tasks for scheduled content publishing and social media automation.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List

from celery import Task
from sqlalchemy import and_, select

from backend.celery_app import app
from backend.config.logging import get_logger
from backend.database.connection import database_manager
from backend.models.content import ContentModel
from backend.services.social_media_service import (
    PlatformType,
    PostRequest,
    SocialMediaService,
)

logger = get_logger(__name__)


class AsyncTask(Task):
    """Base task class that supports async operations."""

    def __call__(self, *args, **kwargs):
        """
        Execute task with async support.

        This method creates a new event loop and runs the task's run method,
        which should be an async function when using this base class.
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            # Check if the task's run method is async
            if asyncio.iscoroutinefunction(self.run):
                # If run is async, execute it directly
                return loop.run_until_complete(self.run(*args, **kwargs))
            else:
                # Otherwise, try to call run_async (for subclasses that override it)
                return loop.run_until_complete(self.run_async(*args, **kwargs))
        finally:
            loop.close()

    async def run_async(self, *args, **kwargs):
        """
        Override this method for async task implementation in subclasses.

        This is a fallback method for tasks that don't define their own
        async run method and need to implement custom async logic.
        """
        # Default implementation: no-op
        # Subclasses should override this if they need custom async logic
        return None


@app.task(
    base=AsyncTask,
    bind=True,
    name="backend.tasks.social_media_tasks.publish_scheduled_post",
)
async def publish_scheduled_post(
    self, schedule_id: str, post_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Publish a scheduled social media post.

    Args:
        schedule_id: Unique identifier for the scheduled post
        post_data: Post configuration including content_id, platforms, etc.

    Returns:
        Dictionary with publishing results
    """
    try:
        logger.info(f"Processing scheduled post: {schedule_id}")

        async with database_manager.get_session() as session:
            # Create social media service instance
            social_service = SocialMediaService(session)

            # Build PostRequest from post_data
            request = PostRequest(
                content_id=post_data["content_id"],
                platforms=[PlatformType(p) for p in post_data["platforms"]],
                caption=post_data.get("caption"),
                hashtags=post_data.get("hashtags", []),
                schedule_time=None,  # Publishing now
            )

            # Publish the content
            results = await social_service.publish_content(request)

            # Format results for task return
            result_data = {
                "schedule_id": schedule_id,
                "published_at": datetime.utcnow().isoformat(),
                "platforms": [r.platform for r in results],
                "statuses": [r.status for r in results],
                "post_ids": [r.post_id for r in results if r.post_id],
                "errors": [r.error_message for r in results if r.error_message],
            }

            logger.info(f"Scheduled post published successfully: {schedule_id}")
            return result_data

    except Exception as e:
        logger.error(f"Failed to publish scheduled post {schedule_id}: {str(e)}")
        return {"schedule_id": schedule_id, "error": str(e), "published_at": None}


@app.task(name="backend.tasks.social_media_tasks.process_scheduled_posts")
def process_scheduled_posts() -> Dict[str, Any]:
    """
    Process all posts scheduled for the current time.
    Runs every minute via Celery beat.

    Returns:
        Dictionary with processing statistics
    """
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(_process_scheduled_posts_async())
        loop.close()
        return result
    except Exception as e:
        logger.error(f"Error in process_scheduled_posts: {str(e)}")
        return {"error": str(e), "processed": 0}


async def _process_scheduled_posts_async() -> Dict[str, Any]:
    """Async implementation of scheduled post processing."""
    processed_count = 0
    error_count = 0

    try:
        async with database_manager.get_session() as session:
            # Query for scheduled posts that are due
            # Note: In a real implementation, you'd have a scheduled_posts table
            # For now, we'll check content with schedule metadata
            current_time = datetime.utcnow()

            # This is a placeholder - you'd query your scheduled_posts table
            stmt = select(ContentModel).where(
                and_(
                    ContentModel.metadata.contains({"scheduled": True}),
                    ContentModel.metadata["schedule_time"].astext.cast(datetime)
                    <= current_time,
                )
            )

            result = await session.execute(stmt)
            scheduled_content = result.scalars().all()

            logger.info(f"Found {len(scheduled_content)} scheduled posts to process")

            for content in scheduled_content:
                try:
                    # Extract schedule metadata
                    schedule_data = content.metadata.get("schedule_data", {})
                    schedule_id = content.metadata.get("schedule_id")

                    # Trigger the publishing task
                    publish_scheduled_post.delay(schedule_id, schedule_data)

                    # Mark as processed in metadata
                    content.metadata["scheduled"] = False
                    content.metadata["processed_at"] = datetime.utcnow().isoformat()

                    processed_count += 1

                except Exception as e:
                    logger.error(
                        f"Error processing scheduled content {content.id}: {str(e)}"
                    )
                    error_count += 1

            await session.commit()

        return {
            "processed": processed_count,
            "errors": error_count,
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"Error in _process_scheduled_posts_async: {str(e)}")
        return {
            "processed": processed_count,
            "errors": error_count + 1,
            "error": str(e),
        }


@app.task(name="backend.tasks.social_media_tasks.cleanup_old_tasks")
def cleanup_old_tasks() -> Dict[str, Any]:
    """
    Clean up old completed tasks and results.
    Runs daily via Celery beat.

    Returns:
        Dictionary with cleanup statistics
    """
    try:
        # Clean up task results older than 7 days
        cutoff_date = datetime.utcnow() - timedelta(days=7)

        # This would connect to your task result backend and clean up
        # For now, just log the cleanup attempt
        logger.info(f"Cleaning up tasks older than {cutoff_date}")

        return {
            "cleaned": 0,
            "cutoff_date": cutoff_date.isoformat(),
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"Error in cleanup_old_tasks: {str(e)}")
        return {"error": str(e), "cleaned": 0}


@app.task(name="backend.tasks.social_media_tasks.batch_publish_content")
def batch_publish_content(
    content_ids: List[str], platforms: List[str]
) -> Dict[str, Any]:
    """
    Batch publish multiple pieces of content to specified platforms.

    Args:
        content_ids: List of content IDs to publish
        platforms: List of platform names to publish to

    Returns:
        Dictionary with batch publishing results
    """
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(_batch_publish_async(content_ids, platforms))
        loop.close()
        return result
    except Exception as e:
        logger.error(f"Error in batch_publish_content: {str(e)}")
        return {"error": str(e), "published": 0}


async def _batch_publish_async(
    content_ids: List[str], platforms: List[str]
) -> Dict[str, Any]:
    """Async implementation of batch publishing."""
    published_count = 0
    failed_count = 0
    results = []

    try:
        async with database_manager.get_session() as session:
            social_service = SocialMediaService(session)

            for content_id in content_ids:
                try:
                    request = PostRequest(
                        content_id=content_id,
                        platforms=[PlatformType(p) for p in platforms],
                    )

                    post_results = await social_service.publish_content(request)

                    results.append(
                        {
                            "content_id": content_id,
                            "status": "success",
                            "results": [
                                {"platform": r.platform, "status": r.status}
                                for r in post_results
                            ],
                        }
                    )

                    published_count += 1

                except Exception as e:
                    logger.error(f"Failed to publish content {content_id}: {str(e)}")
                    results.append(
                        {"content_id": content_id, "status": "failed", "error": str(e)}
                    )
                    failed_count += 1

        return {
            "published": published_count,
            "failed": failed_count,
            "results": results,
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"Error in _batch_publish_async: {str(e)}")
        return {"published": published_count, "failed": failed_count, "error": str(e)}
