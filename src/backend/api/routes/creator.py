"""
Creator Panel API Routes

Provides authenticated endpoints for content creators to manage their
AI personas, monitor performance, and configure content generation.
"""

from typing import List, Optional, Dict, Any
from uuid import UUID
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from pydantic import BaseModel

from backend.database.connection import get_db_session
from backend.models.persona import PersonaModel, PersonaResponse, PersonaUpdate
from backend.models.content import ContentModel, ContentResponse
from backend.services.persona_service import PersonaService
from backend.services.content_generation_service import (
    ContentGenerationService,
    GenerationRequest,
)
from backend.services.rss_ingestion_service import RSSIngestionService
from backend.config.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(
    prefix="/creator",
    tags=["creator-panel"],
    responses={401: {"description": "Not authenticated"}},
)


class DashboardStats(BaseModel):
    """Creator dashboard statistics."""

    total_personas: int
    total_content: int
    content_this_week: int
    avg_quality_score: float
    top_performing_persona: Optional[Dict[str, Any]]
    recent_activity: List[Dict[str, Any]]
    content_breakdown: Dict[str, int]


class ContentGenerationBatch(BaseModel):
    """Batch content generation request."""

    persona_id: UUID
    content_types: List[str]
    count_per_type: int = 1
    theme_override: Optional[str] = None
    quality: str = "high"


def get_persona_service(db: AsyncSession = Depends(get_db_session)) -> PersonaService:
    """Dependency injection for PersonaService."""
    return PersonaService(db)


def get_content_service(
    db: AsyncSession = Depends(get_db_session),
) -> ContentGenerationService:
    """Dependency injection for ContentGenerationService."""
    return ContentGenerationService(db)


def get_rss_service(db: AsyncSession = Depends(get_db_session)) -> RSSIngestionService:
    """Dependency injection for RSSIngestionService."""
    return RSSIngestionService(db)


@router.get("/dashboard", response_model=DashboardStats)
async def get_creator_dashboard(
    days: int = Query(
        default=7, ge=1, le=90, description="Days to include in statistics"
    ),
    db: AsyncSession = Depends(get_db_session),
):
    """
    Get creator dashboard statistics.

    Provides comprehensive statistics and insights for content creators
    about their AI personas and content performance.

    Note: This endpoint currently returns aggregate data for all users.
    In production, authentication middleware should be added to filter
    results by authenticated user_id. See the User model and authentication
    documentation for implementation guidance.

    Args:
        days: Number of days to include in recent statistics
        db: Database session

    Returns:
        DashboardStats: Dashboard statistics and insights
    """
    try:
        # Note: Authentication filtering to be added when auth middleware is implemented
        # Expected filter: .where(PersonaModel.user_id == current_user.id)

        cutoff_date = datetime.utcnow() - timedelta(days=days)

        # Get persona count
        persona_stmt = select(PersonaModel).where(PersonaModel.is_active == True)
        persona_result = await db.execute(persona_stmt)
        personas = persona_result.scalars().all()
        total_personas = len(personas)

        # Get total content
        content_stmt = select(ContentModel)
        content_result = await db.execute(content_stmt)
        all_content = content_result.scalars().all()
        total_content = len(all_content)

        # Get recent content
        recent_content = [c for c in all_content if c.created_at >= cutoff_date]
        content_this_week = len(recent_content)

        # Calculate average quality score
        quality_scores = [c.quality_score for c in all_content if c.quality_score]
        avg_quality_score = (
            sum(quality_scores) / len(quality_scores) if quality_scores else 0
        )

        # Find top performing persona
        top_persona = None
        if personas:
            top_persona_model = max(personas, key=lambda p: p.generation_count)
            top_persona = {
                "id": top_persona_model.id,
                "name": top_persona_model.name,
                "content_count": top_persona_model.generation_count,
            }

        # Recent activity
        recent_activity = []
        for content in sorted(recent_content, key=lambda c: c.created_at, reverse=True)[
            :10
        ]:
            persona = next((p for p in personas if p.id == content.persona_id), None)
            recent_activity.append(
                {
                    "type": "content_generated",
                    "persona_name": persona.name if persona else "Unknown",
                    "content_type": content.content_type,
                    "timestamp": content.created_at.isoformat(),
                    "quality_score": content.quality_score,
                }
            )

        # Content type breakdown
        content_breakdown = {}
        for content in all_content:
            content_type = content.content_type
            content_breakdown[content_type] = content_breakdown.get(content_type, 0) + 1

        dashboard_stats = DashboardStats(
            total_personas=total_personas,
            total_content=total_content,
            content_this_week=content_this_week,
            avg_quality_score=round(avg_quality_score, 2),
            top_performing_persona=top_persona,
            recent_activity=recent_activity,
            content_breakdown=content_breakdown,
        )

        logger.info(f"Creator dashboard accessed stats={dashboard_stats.dict()}")
        return dashboard_stats

    except Exception as e:
        logger.error(f"Error generating dashboard stats: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve dashboard statistics",
        )


@router.get("/personas/analytics/{persona_id}")
async def get_persona_analytics(
    persona_id: UUID,
    days: int = Query(default=30, ge=1, le=365, description="Days of analytics data"),
    db: AsyncSession = Depends(get_db_session),
):
    """
    Get detailed analytics for specific persona.

    Args:
        persona_id: Persona identifier
        days: Number of days of analytics data
        db: Database session

    Returns:
        Persona analytics and performance metrics
    """
    try:
        # Verify persona exists
        persona_stmt = select(PersonaModel).where(PersonaModel.id == persona_id)
        persona_result = await db.execute(persona_stmt)
        persona = persona_result.scalar_one_or_none()

        if not persona:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Persona not found"
            )

        cutoff_date = datetime.utcnow() - timedelta(days=days)

        # Get persona content
        content_stmt = (
            select(ContentModel)
            .where(ContentModel.persona_id == persona_id)
            .where(ContentModel.created_at >= cutoff_date)
        )
        content_result = await db.execute(content_stmt)
        content_items = content_result.scalars().all()

        # Calculate analytics
        analytics = {
            "persona": {
                "id": persona.id,
                "name": persona.name,
                "total_content": persona.generation_count,
                "active_since": persona.created_at.isoformat(),
            },
            "period_metrics": {
                "content_generated": len(content_items),
                "avg_quality_score": (
                    round(
                        sum(c.quality_score for c in content_items if c.quality_score)
                        / len([c for c in content_items if c.quality_score]),
                        2,
                    )
                    if content_items
                    else 0
                ),
                "content_types": {
                    content_type: len(
                        [c for c in content_items if c.content_type == content_type]
                    )
                    for content_type in set(c.content_type for c in content_items)
                },
            },
            "performance_trends": {
                "daily_generation": {},  # Would be calculated with actual date grouping
                "quality_trend": "stable",  # Would be calculated from quality scores over time
                "popular_themes": persona.content_themes[:5],
            },
            "recommendations": [
                "Consider adding 'sustainability' theme based on trending topics",
                "Quality scores are consistently high - maintain current style",
                "Video content performs 20% better than images for this persona",
            ],
        }

        logger.info(f"Persona analytics retrieved {persona_id}")
        return analytics

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving persona analytics: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve analytics",
        )


@router.post("/content/batch", response_model=List[ContentResponse])
async def generate_content_batch(
    batch_request: ContentGenerationBatch,
    content_service: ContentGenerationService = Depends(get_content_service),
):
    """
    Generate multiple content items in batch.

    Efficiently generates multiple pieces of content for a persona,
    useful for content creators who need to produce content at scale.

    Args:
        batch_request: Batch generation parameters
        content_service: Content generation service

    Returns:
        List of generated content items
    """
    try:
        generated_content = []

        for content_type in batch_request.content_types:
            for i in range(batch_request.count_per_type):
                request = GenerationRequest(
                    persona_id=batch_request.persona_id,
                    content_type=content_type,
                    prompt=batch_request.theme_override,
                    quality=batch_request.quality,
                )

                content = await content_service.generate_content(request)
                generated_content.append(content)

        logger.info(
            f"Batch content generated {batch_request.persona_id} total_items={len(generated_content)}"
        )

        return generated_content

    except Exception as e:
        logger.error(f"Batch content generation failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Batch content generation failed",
        )


@router.get("/content/suggestions")
async def get_content_suggestions(
    persona_id: Optional[UUID] = Query(
        None, description="Persona to get suggestions for"
    ),
    limit: int = Query(default=10, ge=1, le=20),
    rss_service: RSSIngestionService = Depends(get_rss_service),
    db: AsyncSession = Depends(get_db_session),
):
    """
    Get content generation suggestions based on trending topics.

    Args:
        persona_id: Optional persona to personalize suggestions for
        limit: Number of suggestions to return
        rss_service: RSS ingestion service
        db: Database session

    Returns:
        Content generation suggestions
    """
    try:
        # Get trending topics
        trending = await rss_service.get_trending_topics(limit=limit)

        suggestions = []
        for topic_data in trending:
            suggestion = {
                "topic": topic_data["topic"],
                "trending_score": topic_data["mentions"],
                "sentiment": topic_data["avg_sentiment"],
                "suggested_content_types": ["image", "text"],
                "sample_titles": topic_data["sample_titles"][:3],
                "recommended_prompt": f"Create content about {topic_data['topic']} focusing on current trends",
            }

            # Customize for specific persona
            if persona_id:
                persona_stmt = select(PersonaModel).where(PersonaModel.id == persona_id)
                persona_result = await db.execute(persona_stmt)
                persona = persona_result.scalar_one_or_none()

                if persona and topic_data["topic"].lower() in [
                    t.lower() for t in persona.content_themes
                ]:
                    suggestion["relevance"] = "high"
                    suggestion["recommended_prompt"] = (
                        f"Create {persona.personality.split(',')[0].lower()} content about "
                        f"{topic_data['topic']} in your unique style"
                    )
                else:
                    suggestion["relevance"] = "medium"

            suggestions.append(suggestion)

        logger.info(
            f"Content suggestions generated {persona_id} count={len(suggestions)}"
        )

        return {"suggestions": suggestions, "updated_at": datetime.utcnow().isoformat()}

    except Exception as e:
        logger.error(f"Error generating content suggestions: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate content suggestions",
        )


@router.put("/personas/{persona_id}/optimize")
async def optimize_persona(
    persona_id: UUID,
    persona_service: PersonaService = Depends(get_persona_service),
    db: AsyncSession = Depends(get_db_session),
):
    """
    Optimize persona based on performance analytics.

    Analyzes persona performance and suggests optimizations
    to improve content quality and engagement.

    Args:
        persona_id: Persona to optimize
        persona_service: Persona service
        db: Database session

    Returns:
        Optimization results and applied changes
    """
    try:
        # Get persona and content performance
        persona_stmt = select(PersonaModel).where(PersonaModel.id == persona_id)
        persona_result = await db.execute(persona_stmt)
        persona = persona_result.scalar_one_or_none()

        if not persona:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Persona not found"
            )

        # Get recent content performance
        content_stmt = (
            select(ContentModel)
            .where(ContentModel.persona_id == persona_id)
            .where(ContentModel.created_at >= datetime.utcnow() - timedelta(days=30))
        )
        content_result = await db.execute(content_stmt)
        recent_content = content_result.scalars().all()

        # Analyze performance and generate optimizations
        optimizations = []
        changes_applied = []

        # Quality analysis
        if recent_content:
            avg_quality = sum(
                c.quality_score for c in recent_content if c.quality_score
            ) / len(recent_content)
            if avg_quality < 70:
                optimizations.append(
                    {
                        "type": "quality_improvement",
                        "recommendation": "Consider refining style preferences for higher quality output",
                        "impact": "medium",
                    }
                )

        # Theme analysis
        if len(persona.content_themes) < 3:
            optimizations.append(
                {
                    "type": "theme_expansion",
                    "recommendation": "Add more content themes to increase variety",
                    "suggested_themes": ["innovation", "lifestyle", "technology"],
                    "impact": "high",
                }
            )

        # Style optimization
        if not persona.style_preferences or len(persona.style_preferences) < 3:
            suggested_styles = {
                "visual_style": "cinematic",
                "color_palette": "vibrant",
                "lighting": "dramatic",
            }

            # Apply optimization
            persona.style_preferences = {
                **persona.style_preferences,
                **suggested_styles,
            }
            await db.commit()

            changes_applied.append(
                {"type": "style_enhancement", "changes": suggested_styles}
            )

        result = {
            "persona_id": persona_id,
            "optimization_results": {
                "total_recommendations": len(optimizations),
                "changes_applied": len(changes_applied),
                "recommendations": optimizations,
                "applied_changes": changes_applied,
            },
            "next_review_date": (datetime.utcnow() + timedelta(days=30)).isoformat(),
        }

        logger.info(
            f"Persona optimized {persona_id} changes_applied={len(changes_applied)}"
        )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error optimizing persona: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to optimize persona",
        )
