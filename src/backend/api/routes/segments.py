"""
Audience Segmentation API Routes

API endpoints for managing audience segments and personalized content.
"""

import logging
from typing import Dict, List, Optional, Any

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database.connection import get_db_session
from backend.services.audience_segment_service import AudienceSegmentService
from backend.models.audience_segment import (
    AudienceSegmentCreate,
    AudienceSegmentUpdate,
    AudienceSegmentSchema,
    PersonalizedContentCreate,
    PersonalizedContentSchema,
    SegmentAnalytics,
    PersonalizationRecommendation,
    SegmentStatus,
    PersonalizationStrategy,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/segments", tags=["segments"])


@router.post(
    "/", response_model=AudienceSegmentSchema, status_code=status.HTTP_201_CREATED
)
async def create_segment(
    request: AudienceSegmentCreate, session: AsyncSession = Depends(get_db_session)
):
    """
    Create new audience segment.

    Segments allow you to group users based on criteria like:
    - Demographics (age, location, interests)
    - Behavior (engagement level, purchase history)
    - Engagement patterns (active times, content preferences)

    Args:
        request: Segment creation parameters

    Returns:
        Created segment
    """
    try:
        service = AudienceSegmentService(session)
        segment = await service.create_segment(
            persona_id=request.persona_id,
            segment_name=request.segment_name,
            criteria=request.criteria,
            description=request.description,
            strategy=request.strategy,
        )

        logger.info(
            f"Created audience segment: id={segment.id}, name={request.segment_name}"
        )

        return AudienceSegmentSchema(
            id=str(segment.id),
            persona_id=str(segment.persona_id),
            segment_name=segment.segment_name,
            description=segment.description,
            criteria=segment.criteria,
            strategy=segment.strategy,
            status=segment.status,
            performance_metrics=segment.performance_metrics,
            estimated_size=segment.estimated_size,
            member_count=segment.member_count,
            created_at=segment.created_at,
            updated_at=segment.updated_at,
            last_analyzed_at=segment.last_analyzed_at,
        )

    except Exception as e:
        logger.error(f"Failed to create segment: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create segment: {str(e)}",
        )


@router.get("/{segment_id}", response_model=AudienceSegmentSchema)
async def get_segment(segment_id: str, session: AsyncSession = Depends(get_db_session)):
    """Get segment by ID."""
    try:
        service = AudienceSegmentService(session)
        segment = await service.get_segment(segment_id)

        if not segment:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Segment not found: {segment_id}",
            )

        return AudienceSegmentSchema(
            id=str(segment.id),
            persona_id=str(segment.persona_id),
            segment_name=segment.segment_name,
            description=segment.description,
            criteria=segment.criteria,
            strategy=segment.strategy,
            status=segment.status,
            performance_metrics=segment.performance_metrics,
            estimated_size=segment.estimated_size,
            member_count=segment.member_count,
            created_at=segment.created_at,
            updated_at=segment.updated_at,
            last_analyzed_at=segment.last_analyzed_at,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get segment: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get segment: {str(e)}",
        )


@router.get("/", response_model=List[AudienceSegmentSchema])
async def list_segments(
    persona_id: Optional[str] = Query(None, description="Filter by persona ID"),
    status: Optional[SegmentStatus] = Query(None, description="Filter by status"),
    strategy: Optional[PersonalizationStrategy] = Query(
        None, description="Filter by strategy"
    ),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    session: AsyncSession = Depends(get_db_session),
):
    """List audience segments with filters."""
    try:
        service = AudienceSegmentService(session)
        segments = await service.list_segments(
            persona_id=persona_id,
            status=status,
            strategy=strategy,
            limit=limit,
            offset=offset,
        )

        return [
            AudienceSegmentSchema(
                id=str(seg.id),
                persona_id=str(seg.persona_id),
                segment_name=seg.segment_name,
                description=seg.description,
                criteria=seg.criteria,
                strategy=seg.strategy,
                status=seg.status,
                performance_metrics=seg.performance_metrics,
                estimated_size=seg.estimated_size,
                member_count=seg.member_count,
                created_at=seg.created_at,
                updated_at=seg.updated_at,
                last_analyzed_at=seg.last_analyzed_at,
            )
            for seg in segments
        ]

    except Exception as e:
        logger.error(f"Failed to list segments: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list segments: {str(e)}",
        )


@router.put("/{segment_id}", response_model=AudienceSegmentSchema)
async def update_segment(
    segment_id: str,
    request: AudienceSegmentUpdate,
    session: AsyncSession = Depends(get_db_session),
):
    """Update audience segment."""
    try:
        service = AudienceSegmentService(session)

        updates = {}
        if request.segment_name is not None:
            updates["segment_name"] = request.segment_name
        if request.description is not None:
            updates["description"] = request.description
        if request.criteria is not None:
            updates["criteria"] = request.criteria
        if request.strategy is not None:
            updates["strategy"] = request.strategy.value
        if request.status is not None:
            updates["status"] = request.status.value
        if request.performance_metrics is not None:
            updates["performance_metrics"] = request.performance_metrics

        segment = await service.update_segment(segment_id, **updates)

        if not segment:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Segment not found: {segment_id}",
            )

        return AudienceSegmentSchema(
            id=str(segment.id),
            persona_id=str(segment.persona_id),
            segment_name=segment.segment_name,
            description=segment.description,
            criteria=segment.criteria,
            strategy=segment.strategy,
            status=segment.status,
            performance_metrics=segment.performance_metrics,
            estimated_size=segment.estimated_size,
            member_count=segment.member_count,
            created_at=segment.created_at,
            updated_at=segment.updated_at,
            last_analyzed_at=segment.last_analyzed_at,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update segment: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update segment: {str(e)}",
        )


@router.delete("/{segment_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_segment(
    segment_id: str, session: AsyncSession = Depends(get_db_session)
):
    """Delete audience segment."""
    try:
        service = AudienceSegmentService(session)
        deleted = await service.delete_segment(segment_id)

        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Segment not found: {segment_id}",
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete segment: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete segment: {str(e)}",
        )


@router.post("/{segment_id}/members/{user_id}", status_code=status.HTTP_201_CREATED)
async def add_member(
    segment_id: str,
    user_id: str,
    confidence_score: float = Query(1.0, ge=0.0, le=1.0),
    session: AsyncSession = Depends(get_db_session),
):
    """Add user to segment."""
    try:
        service = AudienceSegmentService(session)
        await service.add_member_to_segment(
            segment_id=segment_id,
            user_id=user_id,
            confidence_score=confidence_score,
        )

        return {"message": "Member added to segment"}

    except Exception as e:
        logger.error(f"Failed to add member: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to add member: {str(e)}",
        )


@router.delete(
    "/{segment_id}/members/{user_id}", status_code=status.HTTP_204_NO_CONTENT
)
async def remove_member(
    segment_id: str, user_id: str, session: AsyncSession = Depends(get_db_session)
):
    """Remove user from segment."""
    try:
        service = AudienceSegmentService(session)
        removed = await service.remove_member_from_segment(segment_id, user_id)

        if not removed:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Member not found in segment",
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to remove member: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to remove member: {str(e)}",
        )


@router.post(
    "/personalized",
    response_model=PersonalizedContentSchema,
    status_code=status.HTTP_201_CREATED,
)
async def create_personalized_content(
    request: PersonalizedContentCreate, session: AsyncSession = Depends(get_db_session)
):
    """
    Create personalized content for a segment.

    Links content to a specific segment for targeted delivery.
    Supports A/B testing with variant_id and is_control flags.
    """
    try:
        service = AudienceSegmentService(session)
        personalized = await service.create_personalized_content(
            content_id=request.content_id,
            segment_id=request.segment_id,
            variant_id=request.variant_id,
            is_control=request.is_control,
        )

        return PersonalizedContentSchema(
            id=str(personalized.id),
            content_id=str(personalized.content_id),
            segment_id=str(personalized.segment_id),
            performance=personalized.performance,
            variant_id=personalized.variant_id,
            is_control=personalized.is_control,
            view_count=personalized.view_count,
            engagement_count=personalized.engagement_count,
            conversion_count=personalized.conversion_count,
            engagement_rate=personalized.engagement_rate,
            published_at=personalized.published_at,
            created_at=personalized.created_at,
            updated_at=personalized.updated_at,
        )

    except Exception as e:
        logger.error(f"Failed to create personalized content: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create personalized content: {str(e)}",
        )


@router.get("/{segment_id}/analytics", response_model=SegmentAnalytics)
async def get_segment_analytics(
    segment_id: str, session: AsyncSession = Depends(get_db_session)
):
    """
    Get analytics for a segment.

    Provides:
    - Member count and performance metrics
    - Top performing content
    - Engagement trends
    - Personalization recommendations
    """
    try:
        service = AudienceSegmentService(session)
        analytics = await service.get_segment_analytics(segment_id)

        return SegmentAnalytics(**analytics)

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to get segment analytics: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get segment analytics: {str(e)}",
        )


@router.post("/{segment_id}/analyze", status_code=status.HTTP_200_OK)
async def analyze_segment(
    segment_id: str, session: AsyncSession = Depends(get_db_session)
):
    """
    Run analysis on segment and update metrics.

    Updates segment performance metrics based on content performance.
    """
    try:
        service = AudienceSegmentService(session)
        await service.analyze_segment(segment_id)

        return {"message": "Segment analysis complete"}

    except Exception as e:
        logger.error(f"Failed to analyze segment: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to analyze segment: {str(e)}",
        )


@router.get("/health", response_model=Dict[str, Any])
async def segmentation_service_health():
    """Check audience segmentation service health."""
    return {
        "status": "healthy",
        "service": "audience_segmentation",
        "version": "1.0.0",
        "features": [
            "demographic_segmentation",
            "behavioral_segmentation",
            "engagement_segmentation",
            "personalized_content",
            "ab_testing",
            "analytics",
            "recommendations",
        ],
    }
