"""
Message Models

Database and API models for individual messages in DM conversations.
"""

import uuid
from datetime import datetime
from typing import Optional, Dict, Any
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
    Enum as SQLEnum,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func

from backend.database.connection import Base


class MessageType(str, Enum):
    """Types of messages in conversations."""

    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    FILE = "file"
    PPV_OFFER = "ppv_offer"  # Pay-per-view offer
    SYSTEM = "system"  # System generated messages


class MessageSender(str, Enum):
    """Who sent the message."""

    USER = "user"
    PERSONA = "persona"
    SYSTEM = "system"


class MessageModel(Base):
    """
    SQLAlchemy model for individual messages.

    Represents a single message within a conversation between a user and persona.
    """

    __tablename__ = "messages"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)

    # Foreign keys
    conversation_id = Column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Message metadata
    sender = Column(SQLEnum(MessageSender), nullable=False, index=True)
    message_type = Column(
        SQLEnum(MessageType), default=MessageType.TEXT, nullable=False, index=True
    )

    # Message content
    content = Column(Text, nullable=False)
    media_urls = Column(
        JSON, nullable=True, default=list
    )  # URLs for images, videos, etc.
    message_metadata = Column(
        JSON, nullable=True, default=dict
    )  # Additional message metadata

    # Message status
    is_read = Column(Boolean, default=False, index=True)
    is_deleted = Column(Boolean, default=False, index=True)

    # PPV related fields (for PPV_OFFER messages)
    ppv_price = Column(String(20), nullable=True)  # e.g., "9.99" or "free"
    ppv_currency = Column(String(3), nullable=True, default="USD")
    ppv_offer_id = Column(
        UUID(as_uuid=True),
        ForeignKey("ppv_offers.id", ondelete="SET NULL"),
        nullable=True,
    )

    # AI generation metadata (for persona messages)
    generation_prompt = Column(Text, nullable=True)
    generation_model = Column(String(100), nullable=True)
    generation_tokens = Column(Integer, nullable=True)

    # Timestamps
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True
    )
    read_at = Column(DateTime(timezone=True), nullable=True)
    deleted_at = Column(DateTime(timezone=True), nullable=True)


class MessageCreate(BaseModel):
    """API model for creating new messages."""

    conversation_id: uuid.UUID = Field(description="ID of the conversation")
    sender: MessageSender = Field(description="Who is sending the message")
    message_type: MessageType = Field(
        default=MessageType.TEXT, description="Type of message"
    )
    content: str = Field(min_length=1, max_length=4000, description="Message content")
    media_urls: list[str] = Field(default=[], description="URLs for media attachments")
    message_metadata: Optional[Dict[str, Any]] = Field(
        default={}, description="Additional metadata"
    )


class MessageResponse(BaseModel):
    """API model for message responses."""

    id: uuid.UUID
    conversation_id: uuid.UUID
    sender: MessageSender
    message_type: MessageType
    content: str
    media_urls: list[str]
    message_metadata: Dict[str, Any]
    is_read: bool
    is_deleted: bool
    ppv_price: Optional[str]
    ppv_currency: Optional[str]
    ppv_offer_id: Optional[uuid.UUID]
    generation_model: Optional[str]
    created_at: datetime
    read_at: Optional[datetime]
    deleted_at: Optional[datetime]

    model_config = {"from_attributes": True}
