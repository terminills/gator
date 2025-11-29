"""
Social Media Post Models

Database and API models for tracking social media posts and their engagement metrics.
Enables learning from real-time social media interactions.
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func

from backend.database.connection import Base


class SocialPlatform(str, Enum):
    """Social media platforms."""

    INSTAGRAM = "instagram"
    FACEBOOK = "facebook"
    TWITTER = "twitter"
    TIKTOK = "tiktok"
    LINKEDIN = "linkedin"
    CUSTOM = "custom"


class PostStatus(str, Enum):
    """Post status."""

    DRAFT = "draft"
    SCHEDULED = "scheduled"
    PUBLISHED = "published"
    FAILED = "failed"
    DELETED = "deleted"


class SocialMediaPostModel(Base):
    """
    SQLAlchemy model for social media posts.

    Tracks posts published to social media platforms and their engagement metrics.
    Links to content and ACD context for learning loop integration.
    """

    __tablename__ = "social_media_posts"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)

    # References
    content_id = Column(
        UUID(as_uuid=True),
        ForeignKey("content.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    persona_id = Column(
        UUID(as_uuid=True),
        ForeignKey("personas.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    acd_context_id = Column(
        UUID(as_uuid=True),
        ForeignKey("acd_contexts.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    # Post details
    platform = Column(String(50), nullable=False, index=True)
    platform_post_id = Column(String(255), nullable=True, index=True)
    platform_url = Column(Text, nullable=True)
    status = Column(String(20), nullable=False, default="draft", index=True)

    # Content
    caption = Column(Text, nullable=True)
    hashtags = Column(JSON, nullable=True)  # List of hashtags
    platform_specific_data = Column(JSON, nullable=True)  # Platform-specific metadata

    # Timing
    scheduled_at = Column(DateTime(timezone=True), nullable=True, index=True)
    published_at = Column(DateTime(timezone=True), nullable=True, index=True)
    last_metrics_update = Column(DateTime(timezone=True), nullable=True)

    # Engagement metrics
    likes_count = Column(Integer, default=0, nullable=False)
    comments_count = Column(Integer, default=0, nullable=False)
    shares_count = Column(Integer, default=0, nullable=False)
    saves_count = Column(Integer, default=0, nullable=False)
    impressions = Column(Integer, default=0, nullable=False)
    reach = Column(Integer, default=0, nullable=False)
    engagement_rate = Column(Float, nullable=True)  # Calculated field

    # Advanced metrics
    video_views = Column(Integer, default=0, nullable=False)
    video_completion_rate = Column(Float, nullable=True)
    click_through_rate = Column(Float, nullable=True)

    # User interaction filtering
    filtered_metrics = Column(
        JSON, nullable=True
    )  # Metrics after filtering out bots/other personas
    bot_interaction_count = Column(Integer, default=0, nullable=False)
    persona_interaction_count = Column(
        Integer, default=0, nullable=False
    )  # Other AI personas
    genuine_user_count = Column(Integer, default=0, nullable=False)

    # Detailed engagement data
    top_comments = Column(JSON, nullable=True)  # Top comments with sentiment
    engagement_timeline = Column(JSON, nullable=True)  # Hour-by-hour engagement
    demographic_insights = Column(
        JSON, nullable=True
    )  # Age, gender, location breakdown

    # Performance tracking
    compared_to_average = Column(Float, nullable=True)  # Performance vs persona average
    performance_percentile = Column(Integer, nullable=True)  # 0-100 percentile ranking

    # Errors
    error_message = Column(Text, nullable=True)
    retry_count = Column(Integer, default=0, nullable=False)

    # Timestamps
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True
    )
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


# Pydantic models for API
class SocialMediaPostCreate(BaseModel):
    """API model for creating social media posts."""

    content_id: uuid.UUID
    persona_id: uuid.UUID
    platform: SocialPlatform
    caption: Optional[str] = None
    hashtags: List[str] = []
    scheduled_at: Optional[datetime] = None
    platform_specific_data: Optional[Dict[str, Any]] = None
    acd_context_id: Optional[uuid.UUID] = None


class SocialMediaPostUpdate(BaseModel):
    """API model for updating social media posts."""

    status: Optional[PostStatus] = None
    platform_post_id: Optional[str] = None
    platform_url: Optional[str] = None
    published_at: Optional[datetime] = None
    likes_count: Optional[int] = None
    comments_count: Optional[int] = None
    shares_count: Optional[int] = None
    saves_count: Optional[int] = None
    impressions: Optional[int] = None
    reach: Optional[int] = None
    video_views: Optional[int] = None
    error_message: Optional[str] = None


class EngagementMetrics(BaseModel):
    """Engagement metrics update from social platforms."""

    likes_count: int = 0
    comments_count: int = 0
    shares_count: int = 0
    saves_count: int = 0
    impressions: int = 0
    reach: int = 0
    video_views: int = 0
    video_completion_rate: Optional[float] = None
    click_through_rate: Optional[float] = None

    # Filtered metrics
    bot_interaction_count: int = 0
    persona_interaction_count: int = 0
    genuine_user_count: int = 0

    # Additional data
    top_comments: Optional[List[Dict[str, Any]]] = None
    engagement_timeline: Optional[Dict[str, Any]] = None
    demographic_insights: Optional[Dict[str, Any]] = None


class SocialMediaPostResponse(BaseModel):
    """API response model for social media posts."""

    id: uuid.UUID
    content_id: uuid.UUID
    persona_id: uuid.UUID
    acd_context_id: Optional[uuid.UUID]
    platform: str
    platform_post_id: Optional[str]
    platform_url: Optional[str]
    status: str
    caption: Optional[str]
    hashtags: Optional[List[str]]
    scheduled_at: Optional[datetime]
    published_at: Optional[datetime]
    last_metrics_update: Optional[datetime]

    # Metrics
    likes_count: int
    comments_count: int
    shares_count: int
    saves_count: int
    impressions: int
    reach: int
    engagement_rate: Optional[float]
    video_views: int

    # Filtered metrics
    bot_interaction_count: int
    persona_interaction_count: int
    genuine_user_count: int

    # Performance
    compared_to_average: Optional[float]
    performance_percentile: Optional[int]

    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class EngagementAnalysis(BaseModel):
    """Analysis of engagement metrics for learning."""

    post_id: uuid.UUID
    platform: SocialPlatform
    total_engagement: int
    genuine_engagement: int  # After filtering bots/personas
    engagement_rate: float
    performance_vs_average: float
    top_performing_elements: List[str]  # What worked well (hashtags, timing, etc.)
    sentiment_analysis: Dict[str, float]  # Positive, negative, neutral percentages
    best_performing_time: Optional[str]  # When engagement peaked
    recommendations: List[str]  # AI-generated recommendations
