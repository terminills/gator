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
from backend.models.feed import RSSFeedCreate, RSSFeedResponse, FeedItemResponse
from backend.config.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(
    prefix="/api/v1/feeds",
    tags=["rss-feeds"],
    responses={404: {"description": "Feed not found"}},
)


def get_rss_service(
    db: AsyncSession = Depends(get_db_session)
) -> RSSIngestionService:
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
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"RSS feed creation failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to add RSS feed"
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


@router.post("/fetch", response_model=Dict[str, int])
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
        Dict[str, int]: Mapping of feed_id to number of new items fetched
    """
    try:
        results = await rss_service.fetch_all_feeds()
        logger.info(f"Manual RSS fetch completed results={results}")
        return results
        
    except Exception as e:
        logger.error(f"RSS fetch failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Feed fetch failed"
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
        logger.info(f"Trending topics retrieved count={len(topics}")
        return topics
        
    except Exception as e:
        logger.error(f"Trending topics analysis failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Trending topics analysis failed"
        )


@router.get("/suggestions/{persona_id}", response_model=List[FeedItemResponse])
async def get_content_suggestions(
    persona_id: UUID,
    limit: int = Query(default=10, ge=1, le=20, description="Maximum suggestions to return"),
    rss_service: RSSIngestionService = Depends(get_rss_service),
):
    """
    Get content suggestions for specific persona.
    
    Analyzes RSS feed content to find items relevant to the persona's
    themes and interests for content generation inspiration.
    
    Args:
        persona_id: Persona identifier
        limit: Maximum suggestions to return
        rss_service: Injected RSS ingestion service
    
    Returns:
        List[FeedItemResponse]: Relevant feed items
    """
    try:
        # Get persona themes (would need to fetch from persona service)
        # For now, using placeholder themes
        persona_themes = ["technology", "business", "innovation"]  # Placeholder
        
        suggestions = await rss_service.get_content_suggestions(persona_themes, limit)
        logger.info(f"Content suggestions retrieved {persona_id} count={len(suggestions}")
        return suggestions
        
    except Exception as e:
        logger.error(f"Content suggestions failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Content suggestions failed"
        )