"""
RSS Feed Background Tasks

Handles periodic fetching and processing of RSS feeds.
"""

import asyncio
from celery import Task
from backend.celery_app import app
from backend.config.logging import get_logger

logger = get_logger(__name__)


class AsyncTask(Task):
    """Base task class that handles async functions."""

    def __call__(self, *args, **kwargs):
        """Execute the task, handling async functions."""
        result = self.run(*args, **kwargs)
        if asyncio.iscoroutine(result):
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(result)
        return result


@app.task(base=AsyncTask, name="backend.tasks.rss_feed_tasks.fetch_all_rss_feeds")
async def fetch_all_rss_feeds():
    """
    Fetch content from all active RSS feeds.
    
    This task runs periodically to update RSS feed content and populate
    the content suggestion system for personas.
    """
    try:
        from backend.database.connection import database_manager
        from backend.services.rss_ingestion_service import RSSIngestionService

        logger.info("Starting RSS feed fetch cycle")

        # Connect to database
        await database_manager.connect()

        try:
            # Get database session
            async with database_manager.get_session() as session:
                rss_service = RSSIngestionService(session)
                
                # Fetch all active feeds
                results = await rss_service.fetch_all_feeds()
                
                total_items = sum(results.values())
                logger.info(
                    f"RSS feed fetch completed: {len(results)} feeds, {total_items} new items",
                    extra={"feed_results": results}
                )
                
                return {
                    "success": True,
                    "feeds_processed": len(results),
                    "total_new_items": total_items,
                    "results": results,
                }

        finally:
            # Disconnect from database
            await database_manager.disconnect()

    except Exception as e:
        logger.error(f"RSS feed fetch task failed: {str(e)}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "feeds_processed": 0,
            "total_new_items": 0,
        }


@app.task(base=AsyncTask, name="backend.tasks.rss_feed_tasks.cleanup_old_feed_items")
async def cleanup_old_feed_items(days_to_keep: int = 30):
    """
    Clean up old RSS feed items to prevent database bloat.
    
    Args:
        days_to_keep: Number of days to keep feed items (default: 30)
    """
    try:
        from datetime import datetime, timezone, timedelta
        from backend.database.connection import database_manager
        from backend.models.feed import FeedItemModel
        from sqlalchemy import delete

        logger.info(f"Starting RSS feed cleanup (keeping last {days_to_keep} days)")

        # Connect to database
        await database_manager.connect()

        try:
            async with database_manager.get_session() as session:
                # Calculate cutoff date
                cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_to_keep)
                
                # Delete old items
                stmt = delete(FeedItemModel).where(
                    FeedItemModel.created_at < cutoff_date
                )
                result = await session.execute(stmt)
                await session.commit()
                
                deleted_count = result.rowcount
                logger.info(f"RSS feed cleanup completed: {deleted_count} items removed")
                
                return {
                    "success": True,
                    "items_deleted": deleted_count,
                    "cutoff_date": cutoff_date.isoformat(),
                }

        finally:
            await database_manager.disconnect()

    except Exception as e:
        logger.error(f"RSS feed cleanup task failed: {str(e)}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "items_deleted": 0,
        }


@app.task(name="backend.tasks.rss_feed_tasks.validate_feed_urls")
def validate_feed_urls():
    """
    Validate all RSS feed URLs and mark inactive ones.
    
    Checks each feed URL to ensure it's still accessible and working.
    """
    try:
        import httpx
        from backend.database.connection import database_manager
        from backend.models.feed import RSSFeedModel

        logger.info("Starting RSS feed URL validation")
        
        # This would need to be async, but keeping it simple for now
        # In production, this should be fully async
        
        return {
            "success": True,
            "message": "Feed validation not yet implemented",
        }

    except Exception as e:
        logger.error(f"RSS feed validation task failed: {str(e)}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
        }
