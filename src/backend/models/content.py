"""
Content Models

Database and API models for AI-generated content management.
"""

import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum

from pydantic import BaseModel, Field
from sqlalchemy import (
    Column,
    String,
    DateTime,
    Boolean,
    ForeignKey,
    Text,
    JSON,
    Integer,
    Float,
    Enum as SQLEnum,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

from backend.database.connection import Base


class ContentType(str, Enum):
    """Types of content that can be generated."""

    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    VOICE = "voice"
    TEXT = "text"


class ContentRating(str, Enum):
    """Content rating classification."""

    SFW = "sfw"
    MODERATE = "moderate"
    NSFW = "nsfw"


class ModerationStatus(str, Enum):
    """Content moderation status."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    FLAGGED = "flagged"


class ContentModel(Base):
    """
    SQLAlchemy model for generated content.

    Represents AI-generated content items including images, videos, audio, and text
    created by persona-based AI models.
    """

    __tablename__ = "content"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)

    # Foreign keys
    persona_id = Column(
        UUID(as_uuid=True),
        ForeignKey("personas.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Content metadata
    content_type = Column(String(20), nullable=False, index=True)
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    content_rating = Column(String(20), nullable=False, default="sfw", index=True)

    # File information
    file_path = Column(String(500), nullable=True)
    file_size = Column(Integer, nullable=True)  # Size in bytes

    # Generation parameters
    generation_params = Column(JSON, nullable=True, default=dict)

    # Platform-specific adaptations
    platform_adaptations = Column(JSON, nullable=True, default=dict)

    # Quality and moderation
    quality_score = Column(Float, nullable=True)  # 0-100 quality score
    moderation_status = Column(
        String(20), nullable=False, default="pending", index=True
    )
    moderation_notes = Column(Text, nullable=True)

    # Status flags
    is_published = Column(Boolean, default=False, index=True)
    is_deleted = Column(Boolean, default=False, index=True)

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
    published_at = Column(DateTime(timezone=True), nullable=True)
    deleted_at = Column(DateTime(timezone=True), nullable=True)

    # Relationships
    persona = relationship("PersonaModel", back_populates="content")


class ContentCreate(BaseModel):
    """API model for creating new content."""

    persona_id: uuid.UUID = Field(description="ID of the persona generating content")
    content_type: ContentType = Field(description="Type of content to generate")
    title: str = Field(min_length=1, max_length=255, description="Content title")
    description: Optional[str] = Field(default=None, description="Content description")
    content_rating: ContentRating = Field(
        default=ContentRating.SFW, description="Content rating"
    )
    generation_params: Optional[Dict[str, Any]] = Field(
        default={}, description="Generation parameters"
    )
    target_platforms: Optional[List[str]] = Field(
        default=[], description="Target platforms"
    )


class ContentUpdate(BaseModel):
    """API model for updating existing content."""

    title: Optional[str] = Field(default=None, min_length=1, max_length=255)
    description: Optional[str] = Field(default=None)
    content_rating: Optional[ContentRating] = Field(default=None)
    moderation_status: Optional[ModerationStatus] = Field(default=None)
    moderation_notes: Optional[str] = Field(default=None)
    is_published: Optional[bool] = Field(default=None)
    is_deleted: Optional[bool] = Field(default=None)


class ContentResponse(BaseModel):
    """API model for content responses."""

    id: uuid.UUID
    persona_id: uuid.UUID
    content_type: str
    title: str
    description: Optional[str]
    content_rating: str
    file_path: Optional[str]
    file_size: Optional[int]
    generation_params: Dict[str, Any]
    platform_adaptations: Dict[str, Any]
    quality_score: Optional[float]
    moderation_status: str
    moderation_notes: Optional[str]
    is_published: bool
    is_deleted: bool
    created_at: datetime
    updated_at: datetime
    published_at: Optional[datetime]
    deleted_at: Optional[datetime]

    model_config = {"from_attributes": True}


class GenerationRequest(BaseModel):
    """Request for content generation."""

    persona_id: Optional[uuid.UUID] = None
    content_type: ContentType
    content_rating: ContentRating = ContentRating.SFW
    prompt: Optional[str] = None
    style_override: Optional[Dict[str, Any]] = None
    quality: str = "high"  # 'draft', 'standard', 'high', 'premium'
    target_platforms: Optional[List[str]] = None


class ContentStats(BaseModel):
    """Statistics for content generation."""

    total_content: int = 0
    by_type: Dict[str, int] = {}
    by_rating: Dict[str, int] = {}
    by_status: Dict[str, int] = {}
    total_file_size: int = 0  # Total size in bytes
    avg_quality_score: Optional[float] = None


class ContentListResponse(BaseModel):
    """Response for content list with pagination info."""

    items: List[ContentResponse]
    total: int
    page: int = 1
    page_size: int = 50
    stats: Optional[ContentStats] = None
