"""
RSS Feed Management API Routes

Handles RSS feed configuration, ingestion monitoring, and trend analysis.
"""

from typing import List, Dict, Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database.connection import get_db_session
from backend.services.rss_ingestion_service import RSSIngestionService
from backend.models.feed import (
    RSSFeedCreate,
    RSSFeedResponse,
    FeedItemResponse,
    PersonaFeedAssignment,
    PersonaFeedResponse,
    FeedsByTopicResponse,
)
from backend.config.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(
    prefix="/api/v1/feeds",
    tags=["rss-feeds"],
    responses={404: {"description": "Feed not found"}},
)


def get_rss_service(db: AsyncSession = Depends(get_db_session)) -> RSSIngestionService:
    """Dependency injection for RSSIngestionService."""
    return RSSIngestionService(db)


@router.post("/", response_model=RSSFeedResponse, status_code=status.HTTP_201_CREATED)
async def add_feed(
    feed_data: RSSFeedCreate,
    rss_service: RSSIngestionService = Depends(get_rss_service),
):
    """
    Add new RSS feed for monitoring.

    Validates the RSS feed URL and adds it to the monitoring system
    for regular content ingestion.

    Args:
        feed_data: RSS feed configuration
        rss_service: Injected RSS ingestion service

    Returns:
        RSSFeedResponse: Created feed record

    Raises:
        400: Invalid feed URL or duplicate feed
        500: Feed validation or creation failed
    """
    try:
        feed = await rss_service.add_feed(feed_data)
        logger.info(f"RSS feed added via API {feed.id}: {feed.url}")
        return feed

    except ValueError as e:
        logger.warning(f"RSS feed validation error: {str(e)}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"RSS feed creation failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to add RSS feed",
        )


@router.get("/", response_model=List[RSSFeedResponse])
async def list_feeds(
    active_only: bool = Query(default=True, description="Only return active feeds"),
    rss_service: RSSIngestionService = Depends(get_rss_service),
):
    """
    List all RSS feeds.

    Args:
        active_only: Filter to only active feeds
        rss_service: Injected RSS ingestion service

    Returns:
        List[RSSFeedResponse]: List of RSS feeds
    """
    return await rss_service.list_feeds(active_only)


@router.post("/fetch", response_model=Dict[str, Any])
async def fetch_all_feeds(
    rss_service: RSSIngestionService = Depends(get_rss_service),
):
    """
    Trigger manual fetch of all active RSS feeds.

    Forces an immediate update of content from all active RSS feeds.
    Useful for testing or when immediate content updates are needed.

    Args:
        rss_service: Injected RSS ingestion service

    Returns:
        Dict with fetch results including counts and recent items summary
    """
    try:
        results = await rss_service.fetch_all_feeds()
        logger.info(f"Manual RSS fetch completed results={results}")
        
        # Get summary of recently fetched items for display
        recent_items = await rss_service.get_recent_items(limit=10)
        
        # Calculate total items fetched
        total_items = sum(results.values())
        
        return {
            "status": "success",
            "feeds_fetched": len(results),
            "total_new_items": total_items,
            "results": results,
            "recent_items": [
                {
                    "title": item.title,
                    "feed_name": item.feed.name if hasattr(item, 'feed') and item.feed else "Unknown",
                    "published": item.published_date.isoformat() if item.published_date else None,
                    "url": item.url,
                }
                for item in recent_items
            ],
            "message": f"Successfully fetched {total_items} new items from {len(results)} feeds"
        }

    except Exception as e:
        logger.error(f"RSS fetch failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Feed fetch failed",
        )


@router.get("/trending", response_model=List[Dict[str, Any]])
async def get_trending_topics(
    limit: int = Query(default=20, ge=1, le=50, description="Maximum topics to return"),
    hours: int = Query(default=24, ge=1, le=168, description="Time window in hours"),
    rss_service: RSSIngestionService = Depends(get_rss_service),
):
    """
    Get trending topics from RSS feeds.

    Analyzes recent RSS feed content to identify trending topics
    and themes that can inform content generation.

    Args:
        limit: Maximum number of topics to return
        hours: Time window for trend analysis
        rss_service: Injected RSS ingestion service

    Returns:
        List[Dict]: Trending topics with metadata
    """
    try:
        topics = await rss_service.get_trending_topics(limit, hours)
        logger.info(f"Trending topics retrieved count={len(topics)}")
        return topics

    except Exception as e:
        logger.error(f"Trending topics analysis failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Trending topics analysis failed",
        )


@router.get("/suggestions/{persona_id}", response_model=List[FeedItemResponse])
async def get_content_suggestions(
    persona_id: UUID,
    limit: int = Query(
        default=10, ge=1, le=20, description="Maximum suggestions to return"
    ),
    rss_service: RSSIngestionService = Depends(get_rss_service),
):
    """
    Get content suggestions for specific persona.

    Analyzes RSS feed content from feeds assigned to the persona to find
    items relevant to their assigned topics for content generation inspiration.

    Args:
        persona_id: Persona identifier
        limit: Maximum suggestions to return
        rss_service: Injected RSS ingestion service

    Returns:
        List[FeedItemResponse]: Relevant feed items
    """
    try:
        suggestions = await rss_service.get_content_suggestions(persona_id, limit)
        logger.info(
            f"Content suggestions retrieved {persona_id} count={len(suggestions)}"
        )
        return suggestions

    except Exception as e:
        logger.error(f"Content suggestions failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Content suggestions failed",
        )


@router.post(
    "/personas/{persona_id}/feeds",
    response_model=PersonaFeedResponse,
    status_code=status.HTTP_201_CREATED,
)
async def assign_feed_to_persona(
    persona_id: UUID,
    assignment: PersonaFeedAssignment,
    rss_service: RSSIngestionService = Depends(get_rss_service),
):
    """
    Assign an RSS feed to a persona.

    Links an RSS feed to a persona for content suggestions. Can optionally
    specify topics to filter and priority for content selection.

    Args:
        persona_id: Persona identifier
        assignment: Feed assignment data (feed_id, topics, priority)
        rss_service: Injected RSS ingestion service

    Returns:
        PersonaFeedResponse: Assignment details
    """
    try:
        result = await rss_service.assign_feed_to_persona(persona_id, assignment)
        logger.info(f"Feed assigned to persona {persona_id}: {assignment.feed_id}")
        return result

    except ValueError as e:
        logger.warning(f"Feed assignment validation error: {str(e)}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Feed assignment failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Feed assignment failed",
        )


@router.get("/personas/{persona_id}/feeds", response_model=List[PersonaFeedResponse])
async def list_persona_feeds(
    persona_id: UUID,
    rss_service: RSSIngestionService = Depends(get_rss_service),
):
    """
    List all feeds assigned to a persona.

    Returns all RSS feeds assigned to the specified persona along with
    their topic filters and priority settings.

    Args:
        persona_id: Persona identifier
        rss_service: Injected RSS ingestion service

    Returns:
        List[PersonaFeedResponse]: List of feed assignments
    """
    try:
        feeds = await rss_service.list_persona_feeds(persona_id)
        logger.info(f"Listed persona feeds {persona_id} count={len(feeds)}")
        return feeds

    except Exception as e:
        logger.error(f"List persona feeds failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list persona feeds",
        )


@router.delete(
    "/personas/{persona_id}/feeds/{feed_id}", status_code=status.HTTP_204_NO_CONTENT
)
async def unassign_feed_from_persona(
    persona_id: UUID,
    feed_id: UUID,
    rss_service: RSSIngestionService = Depends(get_rss_service),
):
    """
    Remove feed assignment from persona.

    Unlinks an RSS feed from a persona, removing it from content suggestions.

    Args:
        persona_id: Persona identifier
        feed_id: Feed identifier
        rss_service: Injected RSS ingestion service

    Returns:
        204 No Content on success
    """
    try:
        success = await rss_service.unassign_feed_from_persona(persona_id, feed_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Feed assignment not found",
            )

        logger.info(f"Feed unassigned from persona {persona_id}: {feed_id}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unassign feed failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to unassign feed",
        )


@router.get("/by-topic/{topic}", response_model=List[RSSFeedResponse])
async def list_feeds_by_topic(
    topic: str,
    rss_service: RSSIngestionService = Depends(get_rss_service),
):
    """
    List feeds filtered by topic category.

    Returns all RSS feeds that have the specified topic in their categories.
    Useful for discovering feeds to assign to personas based on themes.

    Args:
        topic: Topic to filter by (e.g., "technology", "business")
        rss_service: Injected RSS ingestion service

    Returns:
        List[RSSFeedResponse]: Matching feeds
    """
    try:
        feeds = await rss_service.list_feeds_by_topic(topic)
        logger.info(f"Listed feeds by topic {topic} count={len(feeds)}")
        return feeds

    except Exception as e:
        logger.error(f"List feeds by topic failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list feeds by topic",
        )
