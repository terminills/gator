"""
User Models

Database and API models for platform users who interact with AI personas.
"""

import uuid
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, EmailStr, Field, field_validator
from sqlalchemy import Boolean, Column, DateTime, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func

from backend.database.connection import Base


class UserModel(Base):
    """
    SQLAlchemy model for platform users.

    Represents users who interact with AI personas through direct messaging
    and may purchase premium content via PPV offers.
    """

    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    display_name = Column(String(100), nullable=True)
    profile_picture_url = Column(String(500), nullable=True)
    bio = Column(Text, nullable=True)

    # Account status
    is_active = Column(Boolean, default=True, index=True)
    is_verified = Column(Boolean, default=False)
    is_premium = Column(Boolean, default=False)

    # Messaging preferences
    receive_dm_notifications = Column(Boolean, default=True)
    allow_ppv_offers = Column(Boolean, default=True)

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
    last_active_at = Column(DateTime(timezone=True), nullable=True)


class UserCreate(BaseModel):
    """API model for creating new users."""

    username: str = Field(
        min_length=3,
        max_length=50,
        pattern=r"^[a-zA-Z0-9_]+$",
        description="Unique username (alphanumeric and underscores only)",
    )
    email: EmailStr = Field(description="User email address")
    display_name: Optional[str] = Field(
        None, max_length=100, description="Display name for the user"
    )
    profile_picture_url: Optional[str] = Field(
        None, max_length=500, description="URL to user profile picture"
    )
    bio: Optional[str] = Field(None, max_length=500, description="User bio/description")

    @field_validator("username")
    @classmethod
    def validate_username(cls, v: str) -> str:
        """Validate username format and content."""
        if not v.replace("_", "").isalnum():
            raise ValueError(
                "Username can only contain letters, numbers, and underscores"
            )
        return v.lower()


class UserUpdate(BaseModel):
    """API model for updating existing users."""

    display_name: Optional[str] = Field(None, max_length=100)
    profile_picture_url: Optional[str] = Field(None, max_length=500)
    bio: Optional[str] = Field(None, max_length=500)
    receive_dm_notifications: Optional[bool] = None
    allow_ppv_offers: Optional[bool] = None
    is_active: Optional[bool] = None


class UserResponse(BaseModel):
    """API model for user responses."""

    id: uuid.UUID
    username: str
    email: str
    display_name: Optional[str]
    profile_picture_url: Optional[str]
    bio: Optional[str]
    is_active: bool
    is_verified: bool
    is_premium: bool
    receive_dm_notifications: bool
    allow_ppv_offers: bool
    created_at: datetime
    updated_at: datetime
    last_active_at: Optional[datetime]

    model_config = {"from_attributes": True}
