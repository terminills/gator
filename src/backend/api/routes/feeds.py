"""
RSS Feed Management API Routes

Handles RSS feed configuration, ingestion monitoring, and trend analysis.
"""

from typing import Any, Dict, List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config.logging import get_logger
from backend.database.connection import get_db_session
from backend.models.feed import (
    FeedItemResponse,
    FeedsByTopicResponse,
    PersonaFeedAssignment,
    PersonaFeedResponse,
    RSSFeedCreate,
    RSSFeedResponse,
)
from backend.services.proactive_topics_service import get_proactive_topics_service
from backend.services.rss_ingestion_service import RSSIngestionService


# Pydantic models for proactive topics endpoints
class ProactiveTopicResponse(BaseModel):
    """Response model for a proactive topic."""

    title: str = Field(..., description="Topic title or headline")
    summary: str = Field(default="", description="Topic summary or description")
    source: str = Field(..., description="Source type: 'rss' or 'interests'")
    source_url: str = Field(
        default=None, description="URL of the source article if from RSS"
    )
    published_date: str = Field(
        default=None, description="Publication date if available"
    )
    relevance_score: int = Field(default=50, description="Relevance score 0-100")
    sentiment_score: float = Field(default=0.0, description="Sentiment score -1 to 1")
    categories: List[str] = Field(default_factory=list, description="Topic categories")
    keywords: List[str] = Field(default_factory=list, description="Extracted keywords")


class GenerateOpinionRequest(BaseModel):
    """Request model for generating a persona opinion."""

    topic: str = Field(
        ..., min_length=1, max_length=500, description="Topic title or headline"
    )
    topic_summary: str = Field(
        default=None, max_length=2000, description="Optional topic context"
    )
    use_ai: bool = Field(default=True, description="Whether to use AI generation")


class GenerateOpinionResponse(BaseModel):
    """Response model for a generated persona opinion."""

    success: bool = Field(..., description="Whether opinion generation succeeded")
    opinion: str = Field(default=None, description="Generated opinion text")
    topic: str = Field(..., description="The topic discussed")
    topic_summary: str = Field(default=None, description="Topic context provided")
    sentiment: Any = Field(
        default="neutral",
        description="Sentiment analysis result (category or detailed dict)",
    )
    persona_name: str = Field(default=None, description="Name of the persona")
    generated_at: str = Field(default=None, description="Generation timestamp")
    error: str = Field(default=None, description="Error message if failed")


class ProactivePostResponse(BaseModel):
    """Response model for a complete proactive post."""

    success: bool = Field(..., description="Whether post generation succeeded")
    topic: Dict[str, Any] = Field(default=None, description="Selected topic details")
    opinion: str = Field(default=None, description="Generated opinion text")
    sentiment: Any = Field(
        default="neutral",
        description="Sentiment analysis result (category or detailed dict)",
    )
    hashtags: List[str] = Field(default_factory=list, description="Generated hashtags")
    persona_name: str = Field(default=None, description="Name of the persona")
    source: str = Field(default=None, description="Topic source type")
    generated_at: str = Field(default=None, description="Generation timestamp")
    error: str = Field(default=None, description="Error message if failed")


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
                    "feed_name": (
                        item.feed.name
                        if hasattr(item, "feed") and item.feed
                        else "Unknown"
                    ),
                    "published": (
                        item.published_date.isoformat() if item.published_date else None
                    ),
                    "url": item.url,
                }
                for item in recent_items
            ],
            "message": f"Successfully fetched {total_items} new items from {len(results)} feeds",
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


# RSS-specific subrouter for frontend compatibility
# These endpoints match the frontend expectations at /api/v1/feeds/rss
@router.get("/rss", response_model=Dict[str, Any])
async def list_rss_feeds(
    active_only: bool = Query(default=True, description="Only return active feeds"),
    rss_service: RSSIngestionService = Depends(get_rss_service),
):
    """
    List all RSS feeds (frontend-compatible endpoint).

    This endpoint wraps the feeds list in a structure expected by the frontend.

    Args:
        active_only: Filter to only active feeds
        rss_service: Injected RSS ingestion service

    Returns:
        Dict containing feeds list
    """
    feeds = await rss_service.list_feeds(active_only)
    return {"feeds": feeds, "total": len(feeds)}


@router.post(
    "/rss", response_model=RSSFeedResponse, status_code=status.HTTP_201_CREATED
)
async def add_rss_feed(
    feed_data: RSSFeedCreate,
    rss_service: RSSIngestionService = Depends(get_rss_service),
):
    """
    Add new RSS feed (frontend-compatible endpoint).

    This endpoint accepts feed data and adds it to the monitoring system.

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
        logger.info(f"RSS feed added via /rss endpoint {feed.id}: {feed.url}")
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


@router.post("/rss/{feed_id}/refresh", response_model=Dict[str, Any])
async def refresh_rss_feed(
    feed_id: UUID,
    rss_service: RSSIngestionService = Depends(get_rss_service),
):
    """
    Refresh a specific RSS feed.

    Triggers immediate fetch of content from the specified RSS feed.

    Args:
        feed_id: Feed identifier
        rss_service: Injected RSS ingestion service

    Returns:
        Dict with refresh results

    Raises:
        404: Feed not found
        500: Feed refresh failed
    """
    try:
        result = await rss_service.fetch_feed(feed_id)
        logger.info(f"RSS feed refreshed {feed_id}: {result['new_items']} new items")
        return {
            "status": "success",
            "message": f"Feed refreshed successfully. {result['new_items']} new items.",
            "data": result,
        }

    except ValueError as e:
        logger.warning(f"RSS feed not found: {str(e)}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.error(f"RSS feed refresh failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to refresh RSS feed",
        )


@router.get("/rss/{feed_id}/items", response_model=List[FeedItemResponse])
async def get_feed_items(
    feed_id: UUID,
    limit: int = Query(default=50, ge=1, le=100, description="Maximum items to return"),
    page: int = Query(default=1, ge=1, description="Page number"),
    rss_service: RSSIngestionService = Depends(get_rss_service),
):
    """
    Get items from a specific RSS feed.

    Returns recent feed items with their metadata including title, description,
    published date, and categories for preview display.

    Args:
        feed_id: Feed identifier
        limit: Maximum items to return (1-100)
        page: Page number for pagination
        rss_service: Injected RSS ingestion service

    Returns:
        List[FeedItemResponse]: List of feed items

    Raises:
        404: Feed not found
        500: Failed to retrieve feed items
    """
    try:
        items = await rss_service.get_feed_items(feed_id, limit, page)
        logger.info(f"Retrieved {len(items)} items from feed {feed_id}")
        return items

    except ValueError as e:
        logger.warning(f"Feed not found: {str(e)}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to get feed items: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve feed items",
        )


@router.delete("/rss/{feed_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_rss_feed(
    feed_id: UUID,
    rss_service: RSSIngestionService = Depends(get_rss_service),
):
    """
    Delete an RSS feed.

    Soft deletes the feed by marking it as deleted.

    Args:
        feed_id: Feed identifier
        rss_service: Injected RSS ingestion service

    Returns:
        204 No Content on success

    Raises:
        404: Feed not found
        500: Feed deletion failed
    """
    try:
        await rss_service.delete_feed(feed_id)
        logger.info(f"RSS feed deleted {feed_id}")

    except ValueError as e:
        logger.warning(f"RSS feed not found: {str(e)}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.error(f"RSS feed deletion failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete RSS feed",
        )


# =============================================================================
# Proactive Topics Endpoints
# =============================================================================
# These endpoints enable personas to proactively share opinions on topics
# from RSS feeds or based on their interests, instead of waiting for user chat.


@router.get(
    "/proactive/{persona_id}/topics",
    response_model=List[ProactiveTopicResponse],
    tags=["proactive-content"],
)
async def get_proactive_topics(
    persona_id: UUID,
    limit: int = Query(default=5, ge=1, le=20, description="Maximum topics to return"),
    hours_window: int = Query(
        default=48, ge=1, le=168, description="Time window for RSS items in hours"
    ),
    db: AsyncSession = Depends(get_db_session),
):
    """
    Get proactive topics for a persona to discuss.

    This endpoint fetches topics from:
    1. RSS feeds assigned to the persona (primary source)
    2. Generated topics based on persona's content_themes (fallback if no feeds)

    Use this to get topics the persona can proactively share opinions on,
    instead of waiting for users to initiate conversations.

    Args:
        persona_id: UUID of the persona
        limit: Maximum topics to return (1-20)
        hours_window: Time window for RSS items (1-168 hours)
        db: Database session

    Returns:
        List[ProactiveTopicResponse]: Topics with metadata for discussion

    Raises:
        404: Persona not found
        500: Failed to retrieve topics
    """
    try:
        proactive_service = get_proactive_topics_service(db)
        topics = await proactive_service.get_proactive_topics(
            persona_id=persona_id,
            limit=limit,
            hours_window=hours_window,
        )

        logger.info(
            f"Retrieved {len(topics)} proactive topics for persona {persona_id}"
        )
        return topics

    except Exception as e:
        logger.error(f"Failed to get proactive topics for persona {persona_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve proactive topics: {str(e)}",
        )


@router.post(
    "/proactive/{persona_id}/opinion",
    response_model=GenerateOpinionResponse,
    tags=["proactive-content"],
)
async def generate_persona_opinion(
    persona_id: UUID,
    request: GenerateOpinionRequest,
    db: AsyncSession = Depends(get_db_session),
):
    """
    Generate a persona's opinion on a given topic.

    Uses the persona's personality, worldview, and voice to generate
    an authentic, character-consistent opinion on the provided topic.

    This enables personas to share proactive content based on:
    - RSS feed articles
    - Trending topics
    - Their interests/content themes

    Args:
        persona_id: UUID of the persona
        request: Topic details and generation options
        db: Database session

    Returns:
        GenerateOpinionResponse: Generated opinion with sentiment analysis

    Raises:
        404: Persona not found
        500: Opinion generation failed
    """
    try:
        proactive_service = get_proactive_topics_service(db)
        result = await proactive_service.generate_opinion(
            persona_id=persona_id,
            topic=request.topic,
            topic_summary=request.topic_summary,
            use_ai=request.use_ai,
        )

        if result.get("success"):
            logger.info(
                f"Generated opinion for persona {persona_id} on topic: {request.topic[:50]}..."
            )
        else:
            logger.warning(
                f"Opinion generation failed for persona {persona_id}: {result.get('error')}"
            )

        return GenerateOpinionResponse(**result)

    except Exception as e:
        logger.error(f"Failed to generate opinion for persona {persona_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate opinion: {str(e)}",
        )


@router.get(
    "/proactive/{persona_id}/post",
    response_model=ProactivePostResponse,
    tags=["proactive-content"],
)
async def get_proactive_post(
    persona_id: UUID,
    use_ai: bool = Query(
        default=True, description="Whether to use AI for opinion generation"
    ),
    db: AsyncSession = Depends(get_db_session),
):
    """
    Generate a complete proactive post for a persona.

    This is a convenience endpoint that combines:
    1. Topic selection (from RSS or interests)
    2. Opinion generation
    3. Hashtag generation

    Returns content ready for social media posting, enabling personas
    to be proactive instead of only responding to user conversations.

    Args:
        persona_id: UUID of the persona
        use_ai: Whether to use AI for generation
        db: Database session

    Returns:
        ProactivePostResponse: Complete post content with topic, opinion, hashtags

    Raises:
        404: Persona not found
        500: Post generation failed
    """
    try:
        proactive_service = get_proactive_topics_service(db)
        result = await proactive_service.get_proactive_post_content(
            persona_id=persona_id,
            use_ai=use_ai,
        )

        if result.get("success"):
            logger.info(f"Generated proactive post for persona {persona_id}")
        else:
            logger.warning(
                f"Proactive post generation failed for persona {persona_id}: {result.get('error')}"
            )

        return ProactivePostResponse(**result)

    except Exception as e:
        logger.error(f"Failed to generate proactive post for persona {persona_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate proactive post: {str(e)}",
        )
