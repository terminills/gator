"""
Content Moderation Pipeline Service

Provides a comprehensive content moderation system with support for:
- Text content analysis (toxicity, hate speech, adult content)
- Image content analysis (nudity, violence, graphic content)
- Video content analysis (frame-by-frame analysis)
- ML model integration for advanced moderation
- Human review queue for uncertain cases
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel
from sqlalchemy import Column, DateTime, Float, Integer, String, Text, Boolean
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.dialects.sqlite import JSON

from backend.config.logging import get_logger
from backend.database.connection import Base

logger = get_logger(__name__)


class ModerationCategory(str, Enum):
    """Categories of content that can be flagged."""

    ADULT = "adult"
    VIOLENCE = "violence"
    HATE_SPEECH = "hate_speech"
    HARASSMENT = "harassment"
    SELF_HARM = "self_harm"
    SPAM = "spam"
    MISINFORMATION = "misinformation"
    COPYRIGHT = "copyright"
    PERSONAL_INFO = "personal_info"
    SAFE = "safe"


class ModerationAction(str, Enum):
    """Actions that can be taken on flagged content."""

    APPROVE = "approve"
    REJECT = "reject"
    FLAG_FOR_REVIEW = "flag_for_review"
    WARN = "warn"
    REMOVE = "remove"
    RESTRICT = "restrict"


class ModerationSeverity(str, Enum):
    """Severity levels for flagged content."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ContentType(str, Enum):
    """Types of content that can be moderated."""

    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"


class ModerationQueueModel(Base):
    """SQLAlchemy model for moderation review queue."""

    __tablename__ = "moderation_queue"

    id = Column(String(36), primary_key=True)
    content_id = Column(String(36), nullable=False, index=True)
    content_type = Column(String(50), nullable=False)
    persona_id = Column(String(36), nullable=True, index=True)
    flagged_categories = Column(JSON, default=list)
    severity = Column(String(20), nullable=False)
    confidence_score = Column(Float, nullable=False)
    auto_action = Column(String(50), nullable=True)
    manual_action = Column(String(50), nullable=True)
    reviewer_id = Column(String(36), nullable=True)
    review_notes = Column(Text, nullable=True)
    model_predictions = Column(JSON, default=dict)
    content_preview = Column(Text, nullable=True)
    status = Column(String(20), default="pending", index=True)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    reviewed_at = Column(DateTime(timezone=True), nullable=True)


class ModerationResult(BaseModel):
    """Result of content moderation analysis."""

    content_id: str
    content_type: ContentType
    categories: List[ModerationCategory]
    severity: ModerationSeverity
    confidence: float
    recommended_action: ModerationAction
    details: Dict[str, Any] = {}
    requires_human_review: bool = False
    model_predictions: Dict[str, float] = {}


class ModerationConfig(BaseModel):
    """Configuration for moderation thresholds."""

    auto_approve_threshold: float = 0.9
    auto_reject_threshold: float = 0.1
    human_review_threshold: float = 0.7
    enabled_categories: List[ModerationCategory] = field(
        default_factory=lambda: list(ModerationCategory)
    )
    max_queue_size: int = 1000


class TextAnalysisResult(BaseModel):
    """Result of text content analysis."""

    toxicity_score: float = 0.0
    hate_speech_score: float = 0.0
    adult_score: float = 0.0
    violence_score: float = 0.0
    spam_score: float = 0.0
    categories: List[ModerationCategory] = []


class ImageAnalysisResult(BaseModel):
    """Result of image content analysis."""

    nudity_score: float = 0.0
    violence_score: float = 0.0
    graphic_score: float = 0.0
    safe_score: float = 1.0
    categories: List[ModerationCategory] = []
    detected_objects: List[str] = []


class ContentModerationPipeline:
    """
    Comprehensive content moderation pipeline.

    Provides multi-stage moderation for various content types with
    support for ML model integration and human review workflow.
    """

    def __init__(
        self,
        db_session: AsyncSession,
        config: Optional[ModerationConfig] = None,
    ):
        """
        Initialize the moderation pipeline.

        Args:
            db_session: Database session for persistence
            config: Optional moderation configuration
        """
        self.db = db_session
        self.config = config or ModerationConfig()
        self._ml_models_loaded = False

    async def moderate_content(
        self,
        content_id: str,
        content_type: ContentType,
        content_data: Any,
        persona_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ModerationResult:
        """
        Run content through the moderation pipeline.

        Args:
            content_id: Unique identifier for the content
            content_type: Type of content (text, image, video, audio)
            content_data: The content to moderate (text string, image path, etc.)
            persona_id: Optional persona ID for context
            metadata: Optional additional metadata

        Returns:
            ModerationResult with analysis and recommended action
        """
        logger.info(
            f"Starting moderation for content_id={content_id} type={content_type}"
        )

        # Analyze content based on type
        if content_type == ContentType.TEXT:
            result = await self._analyze_text(content_id, content_data, metadata)
        elif content_type == ContentType.IMAGE:
            result = await self._analyze_image(content_id, content_data, metadata)
        elif content_type == ContentType.VIDEO:
            result = await self._analyze_video(content_id, content_data, metadata)
        elif content_type == ContentType.AUDIO:
            result = await self._analyze_audio(content_id, content_data, metadata)
        else:
            result = ModerationResult(
                content_id=content_id,
                content_type=content_type,
                categories=[ModerationCategory.SAFE],
                severity=ModerationSeverity.LOW,
                confidence=1.0,
                recommended_action=ModerationAction.APPROVE,
            )

        # Determine if human review is needed
        result.requires_human_review = self._needs_human_review(result)

        # Queue for human review if needed
        if result.requires_human_review:
            await self._add_to_review_queue(result, persona_id, content_data)

        logger.info(
            f"Moderation complete for content_id={content_id} "
            f"action={result.recommended_action} categories={result.categories}"
        )

        return result

    async def _analyze_text(
        self,
        content_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ModerationResult:
        """
        Analyze text content for moderation.

        Uses keyword-based analysis with ML model integration points.
        """
        analysis = TextAnalysisResult()
        categories = []
        text_lower = text.lower() if text else ""

        # Toxicity/harassment keywords
        toxicity_keywords = [
            "hate", "kill", "die", "stupid", "idiot", "loser",
            "ugly", "worthless", "disgusting"
        ]
        if any(kw in text_lower for kw in toxicity_keywords):
            analysis.toxicity_score = 0.7
            categories.append(ModerationCategory.HARASSMENT)

        # Hate speech keywords
        hate_keywords = [
            "racist", "bigot", "supremacist", "extremist"
        ]
        if any(kw in text_lower for kw in hate_keywords):
            analysis.hate_speech_score = 0.8
            categories.append(ModerationCategory.HATE_SPEECH)

        # Adult content keywords
        adult_keywords = [
            "nude", "naked", "xxx", "porn", "explicit", "nsfw"
        ]
        if any(kw in text_lower for kw in adult_keywords):
            analysis.adult_score = 0.9
            categories.append(ModerationCategory.ADULT)

        # Violence keywords
        violence_keywords = [
            "blood", "gore", "murder", "assault", "weapon"
        ]
        if any(kw in text_lower for kw in violence_keywords):
            analysis.violence_score = 0.6
            categories.append(ModerationCategory.VIOLENCE)

        # Spam indicators
        spam_indicators = ["buy now", "click here", "free money", "winner"]
        if any(indicator in text_lower for indicator in spam_indicators):
            analysis.spam_score = 0.8
            categories.append(ModerationCategory.SPAM)

        # Calculate overall severity and confidence
        max_score = max(
            analysis.toxicity_score,
            analysis.hate_speech_score,
            analysis.adult_score,
            analysis.violence_score,
            analysis.spam_score,
        )

        if not categories:
            categories.append(ModerationCategory.SAFE)
            severity = ModerationSeverity.LOW
            action = ModerationAction.APPROVE
        elif max_score >= 0.8:
            severity = ModerationSeverity.HIGH
            action = ModerationAction.REJECT
        elif max_score >= 0.5:
            severity = ModerationSeverity.MEDIUM
            action = ModerationAction.FLAG_FOR_REVIEW
        else:
            severity = ModerationSeverity.LOW
            action = ModerationAction.WARN

        return ModerationResult(
            content_id=content_id,
            content_type=ContentType.TEXT,
            categories=categories,
            severity=severity,
            confidence=max(0.5, 1.0 - (max_score * 0.3)),  # Confidence inverse of max score
            recommended_action=action,
            details={"analysis": analysis.model_dump()},
            model_predictions={
                "toxicity": analysis.toxicity_score,
                "hate_speech": analysis.hate_speech_score,
                "adult": analysis.adult_score,
                "violence": analysis.violence_score,
                "spam": analysis.spam_score,
            },
        )

    async def _analyze_image(
        self,
        content_id: str,
        image_path: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ModerationResult:
        """
        Analyze image content for moderation.

        Uses placeholder analysis - can be integrated with ML models like:
        - Google Cloud Vision SafeSearch
        - AWS Rekognition Content Moderation
        - Custom CLIP/ViT models for NSFW detection
        """
        analysis = ImageAnalysisResult()

        # Placeholder - in production, integrate with ML model
        # For now, analyze filename/metadata for keywords
        if image_path:
            path_lower = image_path.lower()

            if any(kw in path_lower for kw in ["nsfw", "nude", "adult"]):
                analysis.nudity_score = 0.8
                analysis.categories.append(ModerationCategory.ADULT)

            if any(kw in path_lower for kw in ["violence", "gore", "blood"]):
                analysis.violence_score = 0.7
                analysis.categories.append(ModerationCategory.VIOLENCE)

        # If no flags, mark as safe
        if not analysis.categories:
            analysis.categories.append(ModerationCategory.SAFE)
            analysis.safe_score = 0.9

        max_score = max(
            analysis.nudity_score,
            analysis.violence_score,
            analysis.graphic_score,
        )

        if max_score >= 0.7:
            severity = ModerationSeverity.HIGH
            action = ModerationAction.FLAG_FOR_REVIEW
        elif max_score >= 0.4:
            severity = ModerationSeverity.MEDIUM
            action = ModerationAction.FLAG_FOR_REVIEW
        else:
            severity = ModerationSeverity.LOW
            action = ModerationAction.APPROVE

        return ModerationResult(
            content_id=content_id,
            content_type=ContentType.IMAGE,
            categories=analysis.categories,
            severity=severity,
            confidence=analysis.safe_score if action == ModerationAction.APPROVE else 0.6,
            recommended_action=action,
            details={"analysis": analysis.model_dump()},
            model_predictions={
                "nudity": analysis.nudity_score,
                "violence": analysis.violence_score,
                "graphic": analysis.graphic_score,
                "safe": analysis.safe_score,
            },
        )

    async def _analyze_video(
        self,
        content_id: str,
        video_path: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ModerationResult:
        """
        Analyze video content for moderation.

        In production, this would:
        1. Extract key frames
        2. Analyze each frame with image moderation
        3. Aggregate results
        4. Optionally analyze audio track
        """
        # Placeholder - analyze video path for keywords
        categories = []
        max_score = 0.0

        if video_path:
            path_lower = video_path.lower()

            if any(kw in path_lower for kw in ["nsfw", "adult", "explicit"]):
                categories.append(ModerationCategory.ADULT)
                max_score = 0.8

            if any(kw in path_lower for kw in ["violence", "fight", "blood"]):
                categories.append(ModerationCategory.VIOLENCE)
                max_score = max(max_score, 0.7)

        if not categories:
            categories.append(ModerationCategory.SAFE)

        if max_score >= 0.7:
            severity = ModerationSeverity.HIGH
            action = ModerationAction.FLAG_FOR_REVIEW
        else:
            severity = ModerationSeverity.LOW
            action = ModerationAction.APPROVE

        return ModerationResult(
            content_id=content_id,
            content_type=ContentType.VIDEO,
            categories=categories,
            severity=severity,
            confidence=0.7 if action == ModerationAction.APPROVE else 0.5,
            recommended_action=action,
            details={"note": "Video analysis placeholder - integrate ML for production"},
        )

    async def _analyze_audio(
        self,
        content_id: str,
        audio_path: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ModerationResult:
        """
        Analyze audio content for moderation.

        In production, this would:
        1. Transcribe audio to text
        2. Run text moderation on transcript
        3. Analyze audio features for profanity detection
        """
        # Placeholder - return safe by default
        return ModerationResult(
            content_id=content_id,
            content_type=ContentType.AUDIO,
            categories=[ModerationCategory.SAFE],
            severity=ModerationSeverity.LOW,
            confidence=0.6,
            recommended_action=ModerationAction.APPROVE,
            details={"note": "Audio analysis placeholder - integrate STT for production"},
        )

    def _needs_human_review(self, result: ModerationResult) -> bool:
        """Determine if content needs human review."""
        # Auto-approve very confident safe content
        if (
            ModerationCategory.SAFE in result.categories
            and result.confidence >= self.config.auto_approve_threshold
        ):
            return False

        # Auto-reject very confident violations
        if (
            result.severity == ModerationSeverity.CRITICAL
            and result.confidence >= self.config.auto_reject_threshold
        ):
            return False

        # Flag uncertain content for review
        if result.confidence < self.config.human_review_threshold:
            return True

        # Flag medium/high severity for review
        if result.severity in [ModerationSeverity.MEDIUM, ModerationSeverity.HIGH]:
            return True

        return False

    async def _add_to_review_queue(
        self,
        result: ModerationResult,
        persona_id: Optional[str],
        content_preview: Any,
    ) -> str:
        """Add content to human review queue."""
        import json

        queue_item = ModerationQueueModel(
            id=str(uuid4()),
            content_id=result.content_id,
            content_type=result.content_type.value,
            persona_id=persona_id,
            flagged_categories=[c.value for c in result.categories],
            severity=result.severity.value,
            confidence_score=result.confidence,
            auto_action=result.recommended_action.value,
            model_predictions=result.model_predictions,
            content_preview=str(content_preview)[:500] if content_preview else None,
            status="pending",
        )

        self.db.add(queue_item)
        await self.db.commit()

        logger.info(f"Added content_id={result.content_id} to moderation queue")

        return queue_item.id

    async def get_review_queue(
        self,
        status: str = "pending",
        limit: int = 50,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Get items from the moderation review queue."""
        from sqlalchemy import select

        stmt = (
            select(ModerationQueueModel)
            .where(ModerationQueueModel.status == status)
            .order_by(ModerationQueueModel.created_at.desc())
            .limit(limit)
            .offset(offset)
        )

        result = await self.db.execute(stmt)
        items = result.scalars().all()

        return [
            {
                "id": item.id,
                "content_id": item.content_id,
                "content_type": item.content_type,
                "persona_id": item.persona_id,
                "flagged_categories": item.flagged_categories,
                "severity": item.severity,
                "confidence_score": item.confidence_score,
                "auto_action": item.auto_action,
                "content_preview": item.content_preview,
                "created_at": item.created_at.isoformat() if item.created_at else None,
            }
            for item in items
        ]

    async def review_content(
        self,
        queue_item_id: str,
        action: ModerationAction,
        reviewer_id: str,
        notes: Optional[str] = None,
    ) -> bool:
        """
        Submit human review for queued content.

        Args:
            queue_item_id: ID of the queue item
            action: The moderation action to take
            reviewer_id: ID of the reviewer
            notes: Optional review notes

        Returns:
            True if review was recorded successfully
        """
        from sqlalchemy import select

        stmt = select(ModerationQueueModel).where(
            ModerationQueueModel.id == queue_item_id
        )
        result = await self.db.execute(stmt)
        item = result.scalar_one_or_none()

        if not item:
            return False

        item.manual_action = action.value
        item.reviewer_id = reviewer_id
        item.review_notes = notes
        item.reviewed_at = datetime.utcnow()
        item.status = "reviewed"

        await self.db.commit()

        logger.info(
            f"Review submitted for queue_item={queue_item_id} "
            f"action={action} reviewer={reviewer_id}"
        )

        return True

    async def get_moderation_stats(self) -> Dict[str, Any]:
        """Get moderation statistics."""
        from sqlalchemy import func, select

        # Total items
        total_stmt = select(func.count(ModerationQueueModel.id))
        total_result = await self.db.execute(total_stmt)
        total_count = total_result.scalar() or 0

        # Pending items
        pending_stmt = select(func.count(ModerationQueueModel.id)).where(
            ModerationQueueModel.status == "pending"
        )
        pending_result = await self.db.execute(pending_stmt)
        pending_count = pending_result.scalar() or 0

        # Reviewed items
        reviewed_count = total_count - pending_count

        # Average confidence
        conf_stmt = select(func.avg(ModerationQueueModel.confidence_score))
        conf_result = await self.db.execute(conf_stmt)
        avg_confidence = conf_result.scalar() or 0

        return {
            "total_items": total_count,
            "pending_review": pending_count,
            "reviewed": reviewed_count,
            "average_confidence": round(avg_confidence, 2),
            "queue_health": "healthy" if pending_count < self.config.max_queue_size else "backlogged",
        }
