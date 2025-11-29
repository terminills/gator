"""
Scheduled Post Models

Database and API models for managing scheduled social media posts.
Provides persistent storage for scheduled posts with status tracking and modification support.
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
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func

from backend.database.connection import Base


class ScheduledPostStatus(str, Enum):
    """Status of a scheduled post."""

    PENDING = "pending"  # Waiting to be published
    PROCESSING = "processing"  # Currently being processed
    PUBLISHED = "published"  # Successfully published
    FAILED = "failed"  # Publishing failed
    CANCELLED = "cancelled"  # Cancelled by user
    PAUSED = "paused"  # Temporarily paused


class ScheduledPostModel(Base):
    """
    SQLAlchemy model for scheduled posts.

    Provides persistent storage for scheduled social media posts,
    enabling listing, modification, and cancellation of scheduled content.
    """

    __tablename__ = "scheduled_posts"

    # Indexes for common queries
    __table_args__ = (
        Index("ix_scheduled_posts_status_scheduled", "status", "scheduled_at"),
        Index("ix_scheduled_posts_persona_status", "persona_id", "status"),
        Index("ix_scheduled_posts_content_status", "content_id", "status"),
    )

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
    )

    # Scheduling details
    platform = Column(String(50), nullable=False, index=True)
    scheduled_at = Column(DateTime(timezone=True), nullable=False, index=True)
    status = Column(
        String(20), nullable=False, default=ScheduledPostStatus.PENDING.value
    )

    # Celery task tracking
    celery_task_id = Column(String(255), nullable=True, index=True)

    # Post content
    caption = Column(Text, nullable=True)
    hashtags = Column(JSON, nullable=True)  # List of hashtags
    platform_specific_data = Column(JSON, nullable=True)  # Platform-specific settings

    # Publishing results
    published_at = Column(DateTime(timezone=True), nullable=True)
    platform_post_id = Column(String(255), nullable=True)
    platform_url = Column(Text, nullable=True)

    # Error handling
    error_message = Column(Text, nullable=True)
    retry_count = Column(Integer, default=0, nullable=False)
    max_retries = Column(Integer, default=3, nullable=False)

    # Metadata
    priority = Column(Integer, default=0, nullable=False)  # Higher = more priority
    is_recurring = Column(Boolean, default=False, nullable=False)
    recurrence_pattern = Column(JSON, nullable=True)  # For recurring posts
    notes = Column(Text, nullable=True)  # User notes about this scheduled post

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
    cancelled_at = Column(DateTime(timezone=True), nullable=True)


# Pydantic models for API


class ScheduledPostCreate(BaseModel):
    """API model for creating a scheduled post."""

    content_id: uuid.UUID
    persona_id: uuid.UUID
    platform: str
    scheduled_at: datetime
    caption: Optional[str] = None
    hashtags: List[str] = Field(default_factory=list)
    platform_specific_data: Optional[Dict[str, Any]] = None
    acd_context_id: Optional[uuid.UUID] = None
    priority: int = Field(default=0, ge=-10, le=10)
    is_recurring: bool = False
    recurrence_pattern: Optional[Dict[str, Any]] = None
    notes: Optional[str] = None
    max_retries: int = Field(default=3, ge=0, le=10)


class ScheduledPostUpdate(BaseModel):
    """API model for updating a scheduled post."""

    scheduled_at: Optional[datetime] = None
    caption: Optional[str] = None
    hashtags: Optional[List[str]] = None
    platform_specific_data: Optional[Dict[str, Any]] = None
    priority: Optional[int] = Field(default=None, ge=-10, le=10)
    is_recurring: Optional[bool] = None
    recurrence_pattern: Optional[Dict[str, Any]] = None
    notes: Optional[str] = None
    status: Optional[ScheduledPostStatus] = None


class ScheduledPostResponse(BaseModel):
    """API response model for scheduled posts."""

    id: uuid.UUID
    content_id: uuid.UUID
    persona_id: uuid.UUID
    acd_context_id: Optional[uuid.UUID] = None

    # Scheduling details
    platform: str
    scheduled_at: datetime
    status: str
    celery_task_id: Optional[str] = None

    # Post content
    caption: Optional[str] = None
    hashtags: Optional[List[str]] = None
    platform_specific_data: Optional[Dict[str, Any]] = None

    # Publishing results
    published_at: Optional[datetime] = None
    platform_post_id: Optional[str] = None
    platform_url: Optional[str] = None

    # Error handling
    error_message: Optional[str] = None
    retry_count: int
    max_retries: int

    # Metadata
    priority: int
    is_recurring: bool
    recurrence_pattern: Optional[Dict[str, Any]] = None
    notes: Optional[str] = None

    # Timestamps
    created_at: datetime
    updated_at: datetime
    cancelled_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


class ScheduledPostListResponse(BaseModel):
    """Response for listing scheduled posts with pagination."""

    posts: List[ScheduledPostResponse]
    total: int
    page: int
    page_size: int
    total_pages: int


class ScheduledPostStats(BaseModel):
    """Statistics about scheduled posts."""

    total_scheduled: int
    pending: int
    processing: int
    published: int
    failed: int
    cancelled: int
    paused: int
    upcoming_24h: int
    by_platform: Dict[str, int]
    by_persona: Dict[str, int]
