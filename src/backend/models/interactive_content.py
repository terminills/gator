"""
Interactive Content Models

Database and API models for interactive content features (polls, stories, Q&A).
"""

import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from enum import Enum

from pydantic import BaseModel, Field, field_validator
from sqlalchemy import (
    Column,
    String,
    DateTime,
    Boolean,
    Integer,
    Text,
    JSON,
    ForeignKey,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from backend.database.connection import Base


class InteractiveContentType(str, Enum):
    """Type of interactive content."""

    POLL = "poll"
    STORY = "story"
    QNA = "qna"
    QUIZ = "quiz"


class InteractiveContentStatus(str, Enum):
    """Status of interactive content."""

    DRAFT = "draft"
    ACTIVE = "active"
    EXPIRED = "expired"
    ARCHIVED = "archived"


class InteractiveContentModel(Base):
    """
    SQLAlchemy model for interactive content.

    Supports polls, stories, Q&A sessions, and other engagement features.
    """

    __tablename__ = "interactive_content"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    persona_id = Column(
        UUID(as_uuid=True), ForeignKey("personas.id"), nullable=False, index=True
    )
    content_type = Column(String(20), nullable=False, index=True)
    title = Column(String(200), nullable=True)
    question = Column(Text, nullable=True)
    description = Column(Text, nullable=True)
    options = Column(
        JSON, nullable=True
    )  # For polls: [{"id": 1, "text": "Option 1", "votes": 0}]
    responses = Column(JSON, nullable=True)  # For storing user responses
    media_url = Column(String(500), nullable=True)  # For stories/visual content
    status = Column(String(20), nullable=False, default="draft", index=True)

    # Engagement metrics
    view_count = Column(Integer, default=0)
    response_count = Column(Integer, default=0)
    share_count = Column(Integer, default=0)

    # Timing
    published_at = Column(DateTime, nullable=True)
    expires_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(
        DateTime, default=func.now(), onupdate=func.now(), nullable=False
    )

    # Relationships
    # persona = relationship("PersonaModel", back_populates="interactive_content")

    def __repr__(self):
        return f"<InteractiveContent(id={self.id}, type={self.content_type}, title={self.title})>"


class InteractiveContentResponse(Base):
    """
    SQLAlchemy model for responses to interactive content.

    Tracks individual user responses to polls, Q&A, etc.
    """

    __tablename__ = "interactive_content_responses"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    content_id = Column(
        UUID(as_uuid=True),
        ForeignKey("interactive_content.id"),
        nullable=False,
        index=True,
    )
    user_id = Column(
        UUID(as_uuid=True), ForeignKey("users.id"), nullable=True, index=True
    )  # Nullable for anonymous
    response_data = Column(
        JSON, nullable=False
    )  # Flexible structure for different content types
    user_identifier = Column(
        String(100), nullable=True
    )  # For anonymous tracking (hashed IP, session ID, etc.)
    created_at = Column(DateTime, default=func.now(), nullable=False)

    # Relationships
    # content = relationship("InteractiveContentModel", back_populates="responses")
    # user = relationship("UserModel", back_populates="interactive_responses")

    def __repr__(self):
        return (
            f"<InteractiveContentResponse(id={self.id}, content_id={self.content_id})>"
        )


# Pydantic models for API


class InteractiveContentCreate(BaseModel):
    """Request model for creating interactive content."""

    persona_id: str = Field(..., description="Persona ID associated with this content")
    content_type: InteractiveContentType = Field(
        ..., description="Type of interactive content"
    )
    title: Optional[str] = Field(None, max_length=200, description="Content title")
    question: Optional[str] = Field(None, description="Question text for polls/Q&A")
    description: Optional[str] = Field(None, description="Detailed description")
    options: Optional[List[Dict[str, Any]]] = Field(
        None, description="Options for polls"
    )
    media_url: Optional[str] = Field(
        None, max_length=500, description="Media URL for stories"
    )
    expires_at: Optional[datetime] = Field(None, description="Expiration timestamp")

    @field_validator("persona_id")
    @classmethod
    def validate_persona_id(cls, v):
        """Validate persona_id is a valid UUID string."""
        try:
            uuid.UUID(v)
        except (ValueError, AttributeError):
            raise ValueError("persona_id must be a valid UUID string")
        return v


class InteractiveContentUpdate(BaseModel):
    """Request model for updating interactive content."""

    title: Optional[str] = Field(None, max_length=200)
    question: Optional[str] = Field(None)
    description: Optional[str] = Field(None)
    options: Optional[List[Dict[str, Any]]] = Field(None)
    media_url: Optional[str] = Field(None, max_length=500)
    status: Optional[InteractiveContentStatus] = Field(None)
    expires_at: Optional[datetime] = Field(None)


class InteractiveContentSchema(BaseModel):
    """Response model for interactive content."""

    id: str
    persona_id: str
    content_type: str
    title: Optional[str] = None
    question: Optional[str] = None
    description: Optional[str] = None
    options: Optional[List[Dict[str, Any]]] = None
    responses: Optional[Dict[str, Any]] = None
    media_url: Optional[str] = None
    status: str
    view_count: int
    response_count: int
    share_count: int
    published_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class InteractiveContentResponseCreate(BaseModel):
    """Request model for submitting a response to interactive content."""

    content_id: str = Field(..., description="Interactive content ID")
    response_data: Dict[str, Any] = Field(
        ..., description="Response data (structure varies by content type)"
    )
    user_id: Optional[str] = Field(
        None, description="User ID (for authenticated responses)"
    )

    @field_validator("content_id", "user_id")
    @classmethod
    def validate_uuid(cls, v):
        """Validate UUID string."""
        if v is None:
            return v
        try:
            uuid.UUID(v)
        except (ValueError, AttributeError):
            raise ValueError("Must be a valid UUID string")
        return v


class InteractiveContentResponseSchema(BaseModel):
    """Response model for interactive content response."""

    id: str
    content_id: str
    user_id: Optional[str] = None
    response_data: Dict[str, Any]
    user_identifier: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True


class InteractiveContentStats(BaseModel):
    """Statistics for interactive content."""

    content_id: str
    content_type: str
    total_views: int
    total_responses: int
    total_shares: int
    response_rate: float  # responses / views
    top_options: Optional[List[Dict[str, Any]]] = None  # For polls
    recent_responses: Optional[List[Dict[str, Any]]] = None
    time_to_first_response: Optional[float] = None  # seconds
    average_response_time: Optional[float] = None  # seconds
