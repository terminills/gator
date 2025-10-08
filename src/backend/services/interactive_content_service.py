"""
Interactive Content Service

Business logic for managing interactive content (polls, stories, Q&A).
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from uuid import UUID

from sqlalchemy import select, func, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models.interactive_content import (
    InteractiveContentModel,
    InteractiveContentResponse,
    InteractiveContentType,
    InteractiveContentStatus,
)

logger = logging.getLogger(__name__)


class InteractiveContentService:
    """Service for managing interactive content."""

    def __init__(self, db_session: AsyncSession):
        """
        Initialize interactive content service.

        Args:
            db_session: Database session for persistence
        """
        self.db = db_session

    async def create_content(
        self,
        persona_id: str,
        content_type: InteractiveContentType,
        title: Optional[str] = None,
        question: Optional[str] = None,
        description: Optional[str] = None,
        options: Optional[List[Dict[str, Any]]] = None,
        media_url: Optional[str] = None,
        expires_at: Optional[datetime] = None,
    ) -> InteractiveContentModel:
        """
        Create new interactive content.

        Args:
            persona_id: Persona ID
            content_type: Type of content (poll, story, qna, quiz)
            title: Content title
            question: Question text
            description: Detailed description
            options: Options for polls/quizzes
            media_url: URL for media content
            expires_at: Expiration timestamp

        Returns:
            Created interactive content
        """
        # Validate options for polls
        if content_type == InteractiveContentType.POLL and options:
            validated_options = []
            for idx, option in enumerate(options):
                validated_options.append(
                    {
                        "id": idx + 1,
                        "text": option.get("text", f"Option {idx + 1}"),
                        "votes": 0,
                        "percentage": 0.0,
                    }
                )
            options = validated_options

        # Set default expiration for stories (24 hours)
        if content_type == InteractiveContentType.STORY and not expires_at:
            expires_at = datetime.utcnow() + timedelta(hours=24)

        content = InteractiveContentModel(
            persona_id=UUID(persona_id),
            content_type=content_type.value,
            title=title,
            question=question,
            description=description,
            options=options,
            media_url=media_url,
            status=InteractiveContentStatus.DRAFT.value,
            expires_at=expires_at,
        )

        self.db.add(content)
        await self.db.commit()
        await self.db.refresh(content)

        logger.info(
            f"Created interactive content: id={content.id}, type={content_type}"
        )
        return content

    async def get_content(self, content_id: str) -> Optional[InteractiveContentModel]:
        """
        Get interactive content by ID.

        Args:
            content_id: Content ID

        Returns:
            Interactive content or None
        """
        result = await self.db.execute(
            select(InteractiveContentModel).where(
                InteractiveContentModel.id == UUID(content_id)
            )
        )
        return result.scalar_one_or_none()

    async def list_content(
        self,
        persona_id: Optional[str] = None,
        content_type: Optional[InteractiveContentType] = None,
        status: Optional[InteractiveContentStatus] = None,
        include_expired: bool = False,
        limit: int = 50,
        offset: int = 0,
    ) -> List[InteractiveContentModel]:
        """
        List interactive content with filters.

        Args:
            persona_id: Filter by persona ID
            content_type: Filter by content type
            status: Filter by status
            include_expired: Include expired content
            limit: Maximum results
            offset: Pagination offset

        Returns:
            List of interactive content
        """
        query = select(InteractiveContentModel)

        filters = []
        if persona_id:
            filters.append(InteractiveContentModel.persona_id == UUID(persona_id))
        if content_type:
            filters.append(InteractiveContentModel.content_type == content_type.value)
        if status:
            filters.append(InteractiveContentModel.status == status.value)
        if not include_expired:
            filters.append(
                or_(
                    InteractiveContentModel.expires_at.is_(None),
                    InteractiveContentModel.expires_at > datetime.utcnow(),
                )
            )

        if filters:
            query = query.where(and_(*filters))

        query = query.order_by(InteractiveContentModel.created_at.desc())
        query = query.limit(limit).offset(offset)

        result = await self.db.execute(query)
        return result.scalars().all()

    async def update_content(
        self, content_id: str, **updates
    ) -> Optional[InteractiveContentModel]:
        """
        Update interactive content.

        Args:
            content_id: Content ID
            **updates: Fields to update

        Returns:
            Updated content or None
        """
        content = await self.get_content(content_id)
        if not content:
            return None

        for key, value in updates.items():
            if hasattr(content, key) and value is not None:
                setattr(content, key, value)

        await self.db.commit()
        await self.db.refresh(content)

        logger.info(f"Updated interactive content: id={content_id}")
        return content

    async def publish_content(
        self, content_id: str
    ) -> Optional[InteractiveContentModel]:
        """
        Publish interactive content (change status to active).

        Args:
            content_id: Content ID

        Returns:
            Published content or None
        """
        return await self.update_content(
            content_id,
            status=InteractiveContentStatus.ACTIVE.value,
            published_at=datetime.utcnow(),
        )

    async def delete_content(self, content_id: str) -> bool:
        """
        Delete interactive content.

        Args:
            content_id: Content ID

        Returns:
            True if deleted, False otherwise
        """
        content = await self.get_content(content_id)
        if not content:
            return False

        await self.db.delete(content)
        await self.db.commit()

        logger.info(f"Deleted interactive content: id={content_id}")
        return True

    async def submit_response(
        self,
        content_id: str,
        response_data: Dict[str, Any],
        user_id: Optional[str] = None,
        user_identifier: Optional[str] = None,
    ) -> InteractiveContentResponse:
        """
        Submit a response to interactive content.

        Args:
            content_id: Content ID
            response_data: Response data
            user_id: User ID (for authenticated users)
            user_identifier: Anonymous identifier

        Returns:
            Created response
        """
        # Get content and update vote count for polls
        content = await self.get_content(content_id)
        if not content:
            raise ValueError(f"Content not found: {content_id}")

        # For polls, update vote counts
        if (
            content.content_type == InteractiveContentType.POLL.value
            and "option_id" in response_data
        ):
            option_id = response_data["option_id"]
            if content.options:
                updated_options = []
                total_votes = 0

                for option in content.options:
                    if option["id"] == option_id:
                        option["votes"] = option.get("votes", 0) + 1
                    total_votes += option.get("votes", 0)
                    updated_options.append(option)

                # Calculate percentages
                for option in updated_options:
                    if total_votes > 0:
                        option["percentage"] = round(
                            (option["votes"] / total_votes) * 100, 2
                        )

                content.options = updated_options

        # Increment response count
        content.response_count += 1

        # Create response record
        response = InteractiveContentResponse(
            content_id=UUID(content_id),
            user_id=UUID(user_id) if user_id else None,
            response_data=response_data,
            user_identifier=user_identifier,
        )

        self.db.add(response)
        await self.db.commit()
        await self.db.refresh(response)

        logger.info(f"Submitted response to content: content_id={content_id}")
        return response

    async def get_content_stats(self, content_id: str) -> Dict[str, Any]:
        """
        Get statistics for interactive content.

        Args:
            content_id: Content ID

        Returns:
            Statistics dictionary
        """
        content = await self.get_content(content_id)
        if not content:
            raise ValueError(f"Content not found: {content_id}")

        # Get response count
        response_count_query = select(func.count()).where(
            InteractiveContentResponse.content_id == UUID(content_id)
        )
        result = await self.db.execute(response_count_query)
        response_count = result.scalar()

        # Calculate response rate
        response_rate = 0.0
        if content.view_count > 0:
            response_rate = round((response_count / content.view_count) * 100, 2)

        stats = {
            "content_id": str(content.id),
            "content_type": content.content_type,
            "total_views": content.view_count,
            "total_responses": response_count,
            "total_shares": content.share_count,
            "response_rate": response_rate,
        }

        # Add poll-specific stats
        if (
            content.content_type == InteractiveContentType.POLL.value
            and content.options
        ):
            stats["top_options"] = sorted(
                content.options, key=lambda x: x.get("votes", 0), reverse=True
            )[:3]

        return stats

    async def increment_view_count(self, content_id: str) -> None:
        """
        Increment view count for content.

        Args:
            content_id: Content ID
        """
        content = await self.get_content(content_id)
        if content:
            content.view_count += 1
            await self.db.commit()

    async def increment_share_count(self, content_id: str) -> None:
        """
        Increment share count for content.

        Args:
            content_id: Content ID
        """
        content = await self.get_content(content_id)
        if content:
            content.share_count += 1
            await self.db.commit()

    async def expire_old_content(
        self, content_type: Optional[InteractiveContentType] = None
    ) -> int:
        """
        Mark expired content as expired.

        Args:
            content_type: Optional filter by content type

        Returns:
            Number of items expired
        """
        query = select(InteractiveContentModel).where(
            and_(
                InteractiveContentModel.status == InteractiveContentStatus.ACTIVE.value,
                InteractiveContentModel.expires_at.isnot(None),
                InteractiveContentModel.expires_at <= datetime.utcnow(),
            )
        )

        if content_type:
            query = query.where(
                InteractiveContentModel.content_type == content_type.value
            )

        result = await self.db.execute(query)
        expired_items = result.scalars().all()

        for item in expired_items:
            item.status = InteractiveContentStatus.EXPIRED.value

        if expired_items:
            await self.db.commit()

        logger.info(f"Expired {len(expired_items)} interactive content items")
        return len(expired_items)
