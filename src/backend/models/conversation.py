"""
Conversation Models

Database and API models for DM conversations between users and AI personas.
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
)
from sqlalchemy import Enum as SQLEnum
from sqlalchemy import (
    ForeignKey,
    Integer,
    String,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from backend.database.connection import Base


class ConversationStatus(str, Enum):
    """Possible conversation statuses."""

    ACTIVE = "active"
    PAUSED = "paused"  # User or persona temporarily unavailable
    ARCHIVED = "archived"  # Conversation ended
    BLOCKED = "blocked"  # User blocked by persona or vice versa


class ConversationModel(Base):
    """
    SQLAlchemy model for DM conversations.

    Represents a conversation thread between a user and an AI persona.
    Each conversation can contain multiple messages.
    """

    __tablename__ = "conversations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)

    # Foreign keys
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    persona_id = Column(
        UUID(as_uuid=True),
        ForeignKey("personas.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Conversation metadata
    title = Column(String(200), nullable=True)  # Optional conversation title
    status = Column(
        SQLEnum(ConversationStatus),
        default=ConversationStatus.ACTIVE,
        nullable=False,
        index=True,
    )

    # Queue management for round-robin system
    last_persona_message_at = Column(DateTime(timezone=True), nullable=True, index=True)
    queue_priority = Column(Integer, default=0, index=True)  # Higher = more priority

    # Statistics
    message_count = Column(Integer, default=0)
    ppv_offers_sent = Column(Integer, default=0)
    ppv_offers_accepted = Column(Integer, default=0)

    # Settings
    notifications_enabled = Column(Boolean, default=True)
    auto_responses_enabled = Column(Boolean, default=True)

    # Timestamps
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
    last_message_at = Column(DateTime(timezone=True), nullable=True, index=True)

    # Relationships (if using ORM relationships)
    # user = relationship("UserModel", back_populates="conversations")
    # persona = relationship("PersonaModel", back_populates="conversations")
    # messages = relationship("MessageModel", back_populates="conversation")


class ConversationCreate(BaseModel):
    """API model for creating new conversations."""

    user_id: uuid.UUID = Field(description="ID of the user starting the conversation")
    persona_id: uuid.UUID = Field(description="ID of the persona to chat with")
    title: Optional[str] = Field(
        None, max_length=200, description="Optional conversation title"
    )


class ConversationResponse(BaseModel):
    """API model for conversation responses."""

    id: uuid.UUID
    user_id: uuid.UUID
    persona_id: uuid.UUID
    title: Optional[str]
    status: ConversationStatus
    message_count: int
    ppv_offers_sent: int
    ppv_offers_accepted: int
    notifications_enabled: bool
    auto_responses_enabled: bool
    created_at: datetime
    updated_at: datetime
    last_message_at: Optional[datetime]
    last_persona_message_at: Optional[datetime]
    queue_priority: int

    model_config = {"from_attributes": True}
