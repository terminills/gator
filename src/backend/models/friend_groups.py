"""
Friend Groups and Social Interaction Models

Database and API models for persona friend groups and social interactions.
Enables personas to form friend groups, interact with each other's content,
and participate in collaborative content like duets and reactions.
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
    Table,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

from backend.database.connection import Base


# Association table for many-to-many relationship between personas in groups
persona_group_members = Table(
    "persona_group_members",
    Base.metadata,
    Column(
        "group_id",
        UUID(as_uuid=True),
        ForeignKey("friend_groups.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column(
        "persona_id",
        UUID(as_uuid=True),
        ForeignKey("personas.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column(
        "joined_at", DateTime(timezone=True), server_default=func.now(), nullable=False
    ),
    Column("role", String(50), default="member"),  # member, admin, creator
)


class InteractionType(str, Enum):
    """Types of social interactions between personas."""

    LIKE = "like"
    COMMENT = "comment"
    SHARE = "share"
    DUET = "duet"
    REACTION = "reaction"
    REPOST = "repost"
    MENTION = "mention"


class FriendGroupModel(Base):
    """
    SQLAlchemy model for persona friend groups.

    Groups of personas that can interact with each other on social media,
    collaborate on content, and appear together in videos/reels.
    """

    __tablename__ = "friend_groups"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    name = Column(String(100), nullable=False, index=True)
    description = Column(Text, nullable=True)

    # Group settings
    is_active = Column(Boolean, default=True, index=True)
    allow_auto_interactions = Column(
        Boolean, default=True
    )  # Auto-generate interactions
    interaction_frequency = Column(String(20), default="normal")  # low, normal, high

    # Social media configuration
    shared_platforms = Column(
        JSON, nullable=False, default=list
    )  # Platforms where group is active
    interaction_rules = Column(
        JSON, nullable=False, default=dict
    )  # Rules for interactions

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

    # Relationships - personas in this group
    # personas = relationship(
    #     "PersonaModel",
    #     secondary=persona_group_members,
    #     back_populates="friend_groups"
    # )


class PersonaInteractionModel(Base):
    """
    SQLAlchemy model for tracking interactions between personas.

    Records when one persona interacts with another's content
    (likes, comments, duets, etc.).
    """

    __tablename__ = "persona_interactions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)

    # Who is interacting
    source_persona_id = Column(
        UUID(as_uuid=True),
        ForeignKey("personas.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # What content they're interacting with
    target_content_id = Column(
        UUID(as_uuid=True),
        ForeignKey("content.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Who created the target content
    target_persona_id = Column(
        UUID(as_uuid=True),
        ForeignKey("personas.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Type of interaction
    interaction_type = Column(String(20), nullable=False, index=True)

    # Interaction details
    comment_text = Column(Text, nullable=True)  # For comments
    reaction_content_id = Column(
        UUID(as_uuid=True),
        ForeignKey("content.id", ondelete="SET NULL"),
        nullable=True,
    )  # For duets/reactions

    # Metadata
    platform = Column(String(50), nullable=True)  # Where interaction occurred
    interaction_metadata = Column(JSON, nullable=True, default=dict)  # Additional data

    # Timestamps
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True
    )


class DuetRequestModel(Base):
    """
    SQLAlchemy model for duet/collaboration requests.

    Tracks requests for personas to collaborate on content,
    including reels, duets, and reaction videos.
    """

    __tablename__ = "duet_requests"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)

    # Original content
    original_content_id = Column(
        UUID(as_uuid=True),
        ForeignKey("content.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Personas involved
    original_persona_id = Column(
        UUID(as_uuid=True),
        ForeignKey("personas.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Duet configuration
    duet_type = Column(
        String(20), nullable=False, default="side_by_side"
    )  # side_by_side, reaction, overlay
    layout_config = Column(JSON, nullable=False, default=dict)  # Split screen settings

    # Participating personas (can be multiple for group reactions)
    participant_personas = Column(
        JSON, nullable=False, default=list
    )  # List of persona IDs

    # Status
    status = Column(
        String(20), default="pending", index=True
    )  # pending, in_progress, completed, failed

    # Result
    result_content_id = Column(
        UUID(as_uuid=True),
        ForeignKey("content.id", ondelete="SET NULL"),
        nullable=True,
    )  # Generated duet video

    # Timestamps
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True
    )
    completed_at = Column(DateTime(timezone=True), nullable=True)


# Pydantic models for API


class FriendGroupCreate(BaseModel):
    """API model for creating friend groups."""

    name: str = Field(min_length=1, max_length=100)
    description: Optional[str] = None
    persona_ids: List[uuid.UUID] = Field(
        default=[], description="Personas in this group"
    )
    shared_platforms: List[str] = Field(
        default=[], description="Social platforms for group"
    )
    allow_auto_interactions: bool = Field(default=True)
    interaction_frequency: str = Field(default="normal")


class FriendGroupUpdate(BaseModel):
    """API model for updating friend groups."""

    name: Optional[str] = Field(None, min_length=1, max_length=100)
    description: Optional[str] = None
    is_active: Optional[bool] = None
    allow_auto_interactions: Optional[bool] = None
    interaction_frequency: Optional[str] = None
    shared_platforms: Optional[List[str]] = None


class FriendGroupResponse(BaseModel):
    """API model for friend group responses."""

    id: uuid.UUID
    name: str
    description: Optional[str]
    is_active: bool
    allow_auto_interactions: bool
    interaction_frequency: str
    shared_platforms: List[str]
    member_count: int = 0
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class PersonaInteractionCreate(BaseModel):
    """API model for creating persona interactions."""

    source_persona_id: uuid.UUID
    target_content_id: uuid.UUID
    interaction_type: InteractionType
    comment_text: Optional[str] = None
    platform: Optional[str] = None


class PersonaInteractionResponse(BaseModel):
    """API model for interaction responses."""

    id: uuid.UUID
    source_persona_id: uuid.UUID
    target_content_id: uuid.UUID
    target_persona_id: uuid.UUID
    interaction_type: str
    comment_text: Optional[str]
    platform: Optional[str]
    created_at: datetime

    model_config = {"from_attributes": True}


class DuetRequestCreate(BaseModel):
    """API model for creating duet requests."""

    original_content_id: uuid.UUID
    participant_personas: List[uuid.UUID] = Field(
        min_length=1, description="Personas to react/duet"
    )
    duet_type: str = Field(default="side_by_side", description="Layout type")
    layout_config: Dict[str, Any] = Field(default={})


class DuetRequestResponse(BaseModel):
    """API model for duet request responses."""

    id: uuid.UUID
    original_content_id: uuid.UUID
    original_persona_id: uuid.UUID
    participant_personas: List[uuid.UUID]
    duet_type: str
    status: str
    result_content_id: Optional[uuid.UUID]
    created_at: datetime
    completed_at: Optional[datetime]

    model_config = {"from_attributes": True}
