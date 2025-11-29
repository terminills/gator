"""
Audience Segmentation Models

Database and API models for audience segmentation and personalized content.
"""

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator
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
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from backend.database.connection import Base


class SegmentStatus(str, Enum):
    """Status of audience segment."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    ARCHIVED = "archived"


class PersonalizationStrategy(str, Enum):
    """Strategy for content personalization."""

    DEMOGRAPHIC = "demographic"
    BEHAVIORAL = "behavioral"
    ENGAGEMENT = "engagement"
    HYBRID = "hybrid"


class AudienceSegmentModel(Base):
    """
    SQLAlchemy model for audience segments.

    Represents a group of users with similar characteristics for targeted content.
    """

    __tablename__ = "audience_segments"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    persona_id = Column(
        UUID(as_uuid=True), ForeignKey("personas.id"), nullable=False, index=True
    )
    segment_name = Column(String(100), nullable=False, index=True)
    description = Column(Text, nullable=True)

    # Segmentation criteria
    criteria = Column(JSON, nullable=False)  # Flexible criteria definition
    # Example: {"age_range": [18, 35], "interests": ["tech", "fitness"], "engagement_level": "high"}

    strategy = Column(String(20), nullable=False, default="hybrid")
    status = Column(String(20), nullable=False, default="active", index=True)

    # Performance metrics
    performance_metrics = Column(JSON, nullable=True)
    # Example: {"avg_engagement": 0.35, "conversion_rate": 0.08, "avg_view_duration": 45.2}

    # Size and composition
    estimated_size = Column(Integer, default=0)
    member_count = Column(Integer, default=0)

    # Timing
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(
        DateTime, default=func.now(), onupdate=func.now(), nullable=False
    )
    last_analyzed_at = Column(DateTime, nullable=True)

    # Relationships
    # persona = relationship("PersonaModel", back_populates="audience_segments")

    def __repr__(self):
        return f"<AudienceSegment(id={self.id}, name={self.segment_name}, size={self.member_count})>"


class PersonalizedContentModel(Base):
    """
    SQLAlchemy model for personalized content mapping.

    Links content to specific audience segments with performance tracking.
    """

    __tablename__ = "personalized_content"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    content_id = Column(
        UUID(as_uuid=True), ForeignKey("content.id"), nullable=False, index=True
    )
    segment_id = Column(
        UUID(as_uuid=True),
        ForeignKey("audience_segments.id"),
        nullable=False,
        index=True,
    )

    # Performance tracking
    performance = Column(JSON, nullable=True)
    # Example: {"views": 1250, "likes": 89, "shares": 23, "comments": 45, "engagement_rate": 0.126}

    # A/B testing
    variant_id = Column(String(50), nullable=True)  # For A/B test variants
    is_control = Column(Boolean, default=False)

    # Metrics
    view_count = Column(Integer, default=0)
    engagement_count = Column(Integer, default=0)
    conversion_count = Column(Integer, default=0)
    engagement_rate = Column(Float, default=0.0)

    # Timing
    published_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(
        DateTime, default=func.now(), onupdate=func.now(), nullable=False
    )

    # Relationships
    # content = relationship("ContentModel", back_populates="personalized_versions")
    # segment = relationship("AudienceSegmentModel", back_populates="personalized_content")

    def __repr__(self):
        return f"<PersonalizedContent(id={self.id}, content_id={self.content_id}, segment_id={self.segment_id})>"


class SegmentMemberModel(Base):
    """
    SQLAlchemy model for segment membership.

    Tracks which users belong to which segments.
    """

    __tablename__ = "segment_members"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    segment_id = Column(
        UUID(as_uuid=True),
        ForeignKey("audience_segments.id"),
        nullable=False,
        index=True,
    )
    user_id = Column(
        UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True
    )

    # Membership details
    confidence_score = Column(
        Float, default=1.0
    )  # How confident we are in this assignment
    assignment_reason = Column(
        JSON, nullable=True
    )  # Why user was assigned to this segment

    # Timing
    joined_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(
        DateTime, default=func.now(), onupdate=func.now(), nullable=False
    )

    # Relationships
    # segment = relationship("AudienceSegmentModel", back_populates="members")
    # user = relationship("UserModel", back_populates="segment_memberships")

    def __repr__(self):
        return f"<SegmentMember(segment_id={self.segment_id}, user_id={self.user_id})>"


# Pydantic models for API


class AudienceSegmentCreate(BaseModel):
    """Request model for creating an audience segment."""

    persona_id: str = Field(..., description="Persona ID for this segment")
    segment_name: str = Field(..., max_length=100, description="Segment name")
    description: Optional[str] = Field(None, description="Segment description")
    criteria: Dict[str, Any] = Field(..., description="Segmentation criteria")
    strategy: PersonalizationStrategy = Field(
        PersonalizationStrategy.HYBRID, description="Personalization strategy"
    )

    @field_validator("persona_id")
    @classmethod
    def validate_persona_id(cls, v):
        """Validate persona_id is a valid UUID string."""
        try:
            uuid.UUID(v)
        except (ValueError, AttributeError):
            raise ValueError("persona_id must be a valid UUID string")
        return v


class AudienceSegmentUpdate(BaseModel):
    """Request model for updating an audience segment."""

    segment_name: Optional[str] = Field(None, max_length=100)
    description: Optional[str] = Field(None)
    criteria: Optional[Dict[str, Any]] = Field(None)
    strategy: Optional[PersonalizationStrategy] = Field(None)
    status: Optional[SegmentStatus] = Field(None)
    performance_metrics: Optional[Dict[str, Any]] = Field(None)


class AudienceSegmentSchema(BaseModel):
    """Response model for audience segment."""

    id: str
    persona_id: str
    segment_name: str
    description: Optional[str] = None
    criteria: Dict[str, Any]
    strategy: str
    status: str
    performance_metrics: Optional[Dict[str, Any]] = None
    estimated_size: int
    member_count: int
    created_at: datetime
    updated_at: datetime
    last_analyzed_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class PersonalizedContentCreate(BaseModel):
    """Request model for creating personalized content."""

    content_id: str = Field(..., description="Content ID to personalize")
    segment_id: str = Field(..., description="Target segment ID")
    variant_id: Optional[str] = Field(
        None, max_length=50, description="A/B test variant identifier"
    )
    is_control: bool = Field(False, description="Is this the control group?")

    @field_validator("content_id", "segment_id")
    @classmethod
    def validate_uuid(cls, v):
        """Validate UUID string."""
        try:
            uuid.UUID(v)
        except (ValueError, AttributeError):
            raise ValueError("Must be a valid UUID string")
        return v


class PersonalizedContentSchema(BaseModel):
    """Response model for personalized content."""

    id: str
    content_id: str
    segment_id: str
    performance: Optional[Dict[str, Any]] = None
    variant_id: Optional[str] = None
    is_control: bool
    view_count: int
    engagement_count: int
    conversion_count: int
    engagement_rate: float
    published_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class SegmentAnalytics(BaseModel):
    """Analytics for an audience segment."""

    segment_id: str
    segment_name: str
    member_count: int
    performance_summary: Dict[str, Any]
    top_performing_content: List[Dict[str, Any]]
    engagement_trends: Dict[str, Any]
    recommendations: List[str]


class PersonalizationRecommendation(BaseModel):
    """Recommendation for content personalization."""

    segment_id: str
    segment_name: str
    recommended_content_types: List[str]
    recommended_topics: List[str]
    best_posting_times: List[str]
    expected_engagement: float
    confidence: float
