"""
Audience Segmentation Service

Business logic for managing audience segments and personalized content.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from uuid import UUID

from sqlalchemy import select, func, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models.audience_segment import (
    AudienceSegmentModel,
    PersonalizedContentModel,
    SegmentMemberModel,
    SegmentStatus,
    PersonalizationStrategy,
)

logger = logging.getLogger(__name__)


class AudienceSegmentService:
    """Service for managing audience segments and personalization."""

    def __init__(self, db_session: AsyncSession):
        """
        Initialize audience segment service.

        Args:
            db_session: Database session for persistence
        """
        self.db = db_session

    async def create_segment(
        self,
        persona_id: str,
        segment_name: str,
        criteria: Dict[str, Any],
        description: Optional[str] = None,
        strategy: PersonalizationStrategy = PersonalizationStrategy.HYBRID,
    ) -> AudienceSegmentModel:
        """
        Create new audience segment.

        Args:
            persona_id: Persona ID
            segment_name: Segment name
            criteria: Segmentation criteria
            description: Segment description
            strategy: Personalization strategy

        Returns:
            Created segment
        """
        segment = AudienceSegmentModel(
            persona_id=UUID(persona_id),
            segment_name=segment_name,
            description=description,
            criteria=criteria,
            strategy=strategy.value,
            status=SegmentStatus.ACTIVE.value,
        )

        self.db.add(segment)
        await self.db.commit()
        await self.db.refresh(segment)

        logger.info(f"Created audience segment: id={segment.id}, name={segment_name}")
        return segment

    async def get_segment(self, segment_id: str) -> Optional[AudienceSegmentModel]:
        """
        Get segment by ID.

        Args:
            segment_id: Segment ID

        Returns:
            Segment or None
        """
        result = await self.db.execute(
            select(AudienceSegmentModel).where(
                AudienceSegmentModel.id == UUID(segment_id)
            )
        )
        return result.scalar_one_or_none()

    async def list_segments(
        self,
        persona_id: Optional[str] = None,
        status: Optional[SegmentStatus] = None,
        strategy: Optional[PersonalizationStrategy] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[AudienceSegmentModel]:
        """
        List audience segments with filters.

        Args:
            persona_id: Filter by persona ID
            status: Filter by status
            strategy: Filter by strategy
            limit: Maximum results
            offset: Pagination offset

        Returns:
            List of segments
        """
        query = select(AudienceSegmentModel)

        filters = []
        if persona_id:
            filters.append(AudienceSegmentModel.persona_id == UUID(persona_id))
        if status:
            filters.append(AudienceSegmentModel.status == status.value)
        if strategy:
            filters.append(AudienceSegmentModel.strategy == strategy.value)

        if filters:
            query = query.where(and_(*filters))

        query = query.order_by(AudienceSegmentModel.created_at.desc())
        query = query.limit(limit).offset(offset)

        result = await self.db.execute(query)
        return result.scalars().all()

    async def update_segment(
        self, segment_id: str, **updates
    ) -> Optional[AudienceSegmentModel]:
        """
        Update audience segment.

        Args:
            segment_id: Segment ID
            **updates: Fields to update

        Returns:
            Updated segment or None
        """
        segment = await self.get_segment(segment_id)
        if not segment:
            return None

        for key, value in updates.items():
            if hasattr(segment, key) and value is not None:
                setattr(segment, key, value)

        await self.db.commit()
        await self.db.refresh(segment)

        logger.info(f"Updated audience segment: id={segment_id}")
        return segment

    async def delete_segment(self, segment_id: str) -> bool:
        """
        Delete audience segment.

        Args:
            segment_id: Segment ID

        Returns:
            True if deleted, False otherwise
        """
        segment = await self.get_segment(segment_id)
        if not segment:
            return False

        await self.db.delete(segment)
        await self.db.commit()

        logger.info(f"Deleted audience segment: id={segment_id}")
        return True

    async def add_member_to_segment(
        self,
        segment_id: str,
        user_id: str,
        confidence_score: float = 1.0,
        assignment_reason: Optional[Dict[str, Any]] = None,
    ) -> SegmentMemberModel:
        """
        Add user to segment.

        Args:
            segment_id: Segment ID
            user_id: User ID
            confidence_score: Confidence in assignment (0.0-1.0)
            assignment_reason: Reason for assignment

        Returns:
            Created membership
        """
        member = SegmentMemberModel(
            segment_id=UUID(segment_id),
            user_id=UUID(user_id),
            confidence_score=confidence_score,
            assignment_reason=assignment_reason,
        )

        self.db.add(member)

        # Update segment member count
        segment = await self.get_segment(segment_id)
        if segment:
            segment.member_count += 1

        await self.db.commit()
        await self.db.refresh(member)

        logger.info(
            f"Added member to segment: segment_id={segment_id}, user_id={user_id}"
        )
        return member

    async def remove_member_from_segment(
        self,
        segment_id: str,
        user_id: str,
    ) -> bool:
        """
        Remove user from segment.

        Args:
            segment_id: Segment ID
            user_id: User ID

        Returns:
            True if removed, False otherwise
        """
        result = await self.db.execute(
            select(SegmentMemberModel).where(
                and_(
                    SegmentMemberModel.segment_id == UUID(segment_id),
                    SegmentMemberModel.user_id == UUID(user_id),
                )
            )
        )
        member = result.scalar_one_or_none()

        if not member:
            return False

        await self.db.delete(member)

        # Update segment member count
        segment = await self.get_segment(segment_id)
        if segment and segment.member_count > 0:
            segment.member_count -= 1

        await self.db.commit()

        logger.info(
            f"Removed member from segment: segment_id={segment_id}, user_id={user_id}"
        )
        return True

    async def get_segment_members(
        self,
        segment_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> List[SegmentMemberModel]:
        """
        Get members of a segment.

        Args:
            segment_id: Segment ID
            limit: Maximum results
            offset: Pagination offset

        Returns:
            List of segment members
        """
        query = select(SegmentMemberModel).where(
            SegmentMemberModel.segment_id == UUID(segment_id)
        )
        query = query.order_by(SegmentMemberModel.joined_at.desc())
        query = query.limit(limit).offset(offset)

        result = await self.db.execute(query)
        return result.scalars().all()

    async def create_personalized_content(
        self,
        content_id: str,
        segment_id: str,
        variant_id: Optional[str] = None,
        is_control: bool = False,
    ) -> PersonalizedContentModel:
        """
        Create personalized content mapping.

        Args:
            content_id: Content ID
            segment_id: Segment ID
            variant_id: A/B test variant ID
            is_control: Is control group

        Returns:
            Created personalized content
        """
        personalized = PersonalizedContentModel(
            content_id=UUID(content_id),
            segment_id=UUID(segment_id),
            variant_id=variant_id,
            is_control=is_control,
        )

        self.db.add(personalized)
        await self.db.commit()
        await self.db.refresh(personalized)

        logger.info(
            f"Created personalized content: content_id={content_id}, segment_id={segment_id}"
        )
        return personalized

    async def get_personalized_content(
        self,
        segment_id: Optional[str] = None,
        content_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[PersonalizedContentModel]:
        """
        Get personalized content mappings.

        Args:
            segment_id: Filter by segment ID
            content_id: Filter by content ID
            limit: Maximum results
            offset: Pagination offset

        Returns:
            List of personalized content
        """
        query = select(PersonalizedContentModel)

        filters = []
        if segment_id:
            filters.append(PersonalizedContentModel.segment_id == UUID(segment_id))
        if content_id:
            filters.append(PersonalizedContentModel.content_id == UUID(content_id))

        if filters:
            query = query.where(and_(*filters))

        query = query.order_by(PersonalizedContentModel.created_at.desc())
        query = query.limit(limit).offset(offset)

        result = await self.db.execute(query)
        return result.scalars().all()

    async def update_content_performance(
        self,
        personalized_content_id: str,
        performance_update: Dict[str, Any],
    ) -> Optional[PersonalizedContentModel]:
        """
        Update performance metrics for personalized content.

        Args:
            personalized_content_id: Personalized content ID
            performance_update: Performance metrics to update

        Returns:
            Updated personalized content or None
        """
        result = await self.db.execute(
            select(PersonalizedContentModel).where(
                PersonalizedContentModel.id == UUID(personalized_content_id)
            )
        )
        personalized = result.scalar_one_or_none()

        if not personalized:
            return None

        # Update performance metrics
        if personalized.performance:
            personalized.performance.update(performance_update)
        else:
            personalized.performance = performance_update

        # Update individual metrics
        if "views" in performance_update:
            personalized.view_count = performance_update["views"]
        if "engagement" in performance_update:
            personalized.engagement_count = performance_update["engagement"]
        if "conversions" in performance_update:
            personalized.conversion_count = performance_update["conversions"]

        # Calculate engagement rate
        if personalized.view_count > 0:
            personalized.engagement_rate = round(
                (personalized.engagement_count / personalized.view_count) * 100, 2
            )

        await self.db.commit()
        await self.db.refresh(personalized)

        return personalized

    async def get_segment_analytics(self, segment_id: str) -> Dict[str, Any]:
        """
        Get analytics for a segment.

        Args:
            segment_id: Segment ID

        Returns:
            Analytics dictionary
        """
        segment = await self.get_segment(segment_id)
        if not segment:
            raise ValueError(f"Segment not found: {segment_id}")

        # Get personalized content for this segment
        personalized_content = await self.get_personalized_content(
            segment_id=segment_id, limit=100
        )

        # Calculate aggregate metrics
        total_views = sum(pc.view_count for pc in personalized_content)
        total_engagement = sum(pc.engagement_count for pc in personalized_content)
        total_conversions = sum(pc.conversion_count for pc in personalized_content)

        avg_engagement_rate = 0.0
        if personalized_content:
            avg_engagement_rate = round(
                sum(pc.engagement_rate for pc in personalized_content)
                / len(personalized_content),
                2,
            )

        # Find top performing content
        top_performing = sorted(
            personalized_content, key=lambda x: x.engagement_rate, reverse=True
        )[:5]

        return {
            "segment_id": str(segment.id),
            "segment_name": segment.segment_name,
            "member_count": segment.member_count,
            "performance_summary": {
                "total_views": total_views,
                "total_engagement": total_engagement,
                "total_conversions": total_conversions,
                "avg_engagement_rate": avg_engagement_rate,
            },
            "top_performing_content": [
                {
                    "content_id": str(pc.content_id),
                    "engagement_rate": pc.engagement_rate,
                    "views": pc.view_count,
                }
                for pc in top_performing
            ],
            "engagement_trends": segment.performance_metrics or {},
            "recommendations": self._generate_recommendations(
                segment, personalized_content
            ),
        }

    def _generate_recommendations(
        self,
        segment: AudienceSegmentModel,
        personalized_content: List[PersonalizedContentModel],
    ) -> List[str]:
        """Generate recommendations based on segment performance."""
        recommendations = []

        if not personalized_content:
            recommendations.append(
                "Start creating personalized content for this segment"
            )
            return recommendations

        avg_engagement = sum(pc.engagement_rate for pc in personalized_content) / len(
            personalized_content
        )

        if avg_engagement < 10:
            recommendations.append(
                "Engagement is low. Consider refining segment criteria"
            )
            recommendations.append("Test different content types and formats")
        elif avg_engagement < 25:
            recommendations.append(
                "Engagement is moderate. A/B test content variations"
            )
        else:
            recommendations.append(
                "Engagement is strong. Scale up content for this segment"
            )

        if segment.member_count < 100:
            recommendations.append(
                "Segment size is small. Consider broadening criteria"
            )

        return recommendations

    async def analyze_segment(self, segment_id: str) -> None:
        """
        Run analysis on segment and update metrics.

        Args:
            segment_id: Segment ID
        """
        segment = await self.get_segment(segment_id)
        if not segment:
            return

        # Get analytics
        analytics = await self.get_segment_analytics(segment_id)

        # Update segment performance metrics
        segment.performance_metrics = analytics["performance_summary"]
        segment.last_analyzed_at = datetime.utcnow()

        await self.db.commit()

        logger.info(f"Analyzed segment: id={segment_id}")
