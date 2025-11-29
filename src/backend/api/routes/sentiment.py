"""
Sentiment Analysis API Routes

Provides endpoints for social media sentiment analysis and insights.
"""

from datetime import timedelta
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config.logging import get_logger
from backend.database.connection import get_db_session
from backend.services.sentiment_analysis_service import SentimentAnalysisService

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1/sentiment", tags=["sentiment"])


class TextAnalysisRequest(BaseModel):
    """Request to analyze text sentiment."""

    text: str = Field(
        ..., description="Text content to analyze", min_length=1, max_length=10000
    )
    context: Optional[Dict[str, Any]] = Field(
        None, description="Optional context information"
    )


class CommentAnalysisRequest(BaseModel):
    """Request to analyze multiple comments."""

    comments: List[Dict[str, Any]] = Field(
        ..., description="List of comments to analyze"
    )
    persona_id: Optional[str] = Field(None, description="Optional persona ID filter")


class EngagementAnalysisRequest(BaseModel):
    """Request to analyze post engagement sentiment."""

    post_data: Dict[str, Any] = Field(..., description="Post content and metadata")
    engagement_data: Dict[str, Any] = Field(
        ..., description="Engagement metrics (likes, comments, etc.)"
    )


class CompetitorComparisonRequest(BaseModel):
    """Request to compare sentiment with competitors."""

    persona_id: str = Field(..., description="Your persona ID")
    competitor_ids: List[str] = Field(..., description="List of competitor persona IDs")


@router.post("/analyze-text", response_model=Dict[str, Any])
async def analyze_text_sentiment(
    request: TextAnalysisRequest, session: AsyncSession = Depends(get_db_session)
):
    """
    Analyze sentiment of a text string.

    Provides:
    - Sentiment score (-1.0 to 1.0)
    - Sentiment label (very_positive, positive, neutral, negative, very_negative)
    - Detected emotions
    - Key topics
    - Intent classification
    - Confidence score

    Args:
        request: Text analysis request with content and optional context

    Returns:
        Comprehensive sentiment analysis results
    """
    try:
        service = SentimentAnalysisService(session)
        result = await service.analyze_text(request.text, request.context)

        logger.info(
            f"Text sentiment analyzed: sentiment={result['sentiment_label']} confidence={result['confidence']}"
        )

        return result

    except Exception as e:
        logger.error(f"Failed to analyze text sentiment: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Sentiment analysis failed: {str(e)}",
        )


@router.post("/analyze-comments", response_model=Dict[str, Any])
async def analyze_social_comments(
    request: CommentAnalysisRequest, session: AsyncSession = Depends(get_db_session)
):
    """
    Analyze sentiment of multiple social media comments.

    Provides aggregated metrics:
    - Average sentiment score
    - Sentiment distribution (positive/negative/neutral ratios)
    - Emotion distribution
    - Key topics mentioned
    - Positive/negative ratios

    Args:
        request: Comments to analyze with optional persona filter

    Returns:
        Aggregated sentiment analysis for all comments
    """
    try:
        service = SentimentAnalysisService(session)
        result = await service.analyze_social_comments(
            request.comments, request.persona_id
        )

        logger.info(
            f"Comments analyzed: count={result['total_comments']} sentiment={result.get('overall_sentiment')}"
        )

        return result

    except Exception as e:
        logger.error(f"Failed to analyze comments: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Comment analysis failed: {str(e)}",
        )


@router.post("/analyze-engagement", response_model=Dict[str, Any])
async def analyze_post_engagement(
    request: EngagementAnalysisRequest, session: AsyncSession = Depends(get_db_session)
):
    """
    Analyze sentiment based on post content and engagement metrics.

    Combines:
    - Post content sentiment
    - Comment sentiment
    - Engagement metrics (likes, shares, etc.)
    - Overall sentiment score

    Args:
        request: Post data and engagement metrics

    Returns:
        Combined sentiment and engagement analysis
    """
    try:
        service = SentimentAnalysisService(session)
        result = await service.analyze_engagement_sentiment(
            request.post_data, request.engagement_data
        )

        logger.info(
            f"Engagement analyzed: overall_sentiment={result['overall_sentiment']}"
        )

        return result

    except Exception as e:
        logger.error(f"Failed to analyze engagement: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Engagement analysis failed: {str(e)}",
        )


@router.get("/trends/{persona_id}", response_model=Dict[str, Any])
async def get_sentiment_trends(
    persona_id: str,
    days: int = Query(30, ge=1, le=365, description="Number of days to analyze"),
    session: AsyncSession = Depends(get_db_session),
):
    """
    Track sentiment trends over time for a persona.

    Provides:
    - Sentiment trend direction (improving/declining/stable)
    - Average sentiment over period
    - Sentiment volatility
    - Emotion trends
    - Engagement correlation
    - Content strategy recommendations

    Args:
        persona_id: Persona identifier
        days: Time range in days (default: 30)

    Returns:
        Sentiment trend analysis and recommendations
    """
    try:
        service = SentimentAnalysisService(session)
        result = await service.track_sentiment_trends(persona_id, timedelta(days=days))

        logger.info(
            f"Sentiment trends retrieved: persona_id={persona_id} trend={result['sentiment_trend']['direction']}"
        )

        return result

    except Exception as e:
        logger.error(f"Failed to get sentiment trends: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Trend analysis failed: {str(e)}",
        )


@router.post("/compare-competitors", response_model=Dict[str, Any])
async def compare_competitor_sentiment(
    request: CompetitorComparisonRequest,
    session: AsyncSession = Depends(get_db_session),
):
    """
    Compare sentiment metrics with competitors.

    Provides:
    - Your sentiment vs competitor average
    - Engagement rate comparison
    - Positive ratio comparison
    - Comparative insights and recommendations

    Args:
        request: Persona ID and competitor IDs to compare

    Returns:
        Comparative sentiment analysis
    """
    try:
        service = SentimentAnalysisService(session)
        result = await service.compare_competitor_sentiment(
            request.persona_id, request.competitor_ids
        )

        logger.info(
            f"Competitor sentiment compared: persona_id={request.persona_id} competitors={len(request.competitor_ids)}"
        )

        return result

    except Exception as e:
        logger.error(f"Failed to compare competitor sentiment: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Competitor comparison failed: {str(e)}",
        )


@router.get("/health", response_model=Dict[str, Any])
async def sentiment_service_health():
    """
    Check sentiment analysis service health.

    Returns:
        Service health status
    """
    return {
        "status": "healthy",
        "service": "sentiment_analysis",
        "version": "1.0.0",
        "features": [
            "text_sentiment_analysis",
            "comment_aggregation",
            "engagement_analysis",
            "trend_tracking",
            "competitor_comparison",
            "emotion_detection",
            "intent_classification",
        ],
    }
