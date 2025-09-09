"""
PPV Offer Models

Database and API models for Pay-Per-View offers sent during conversations.
"""

import uuid
from datetime import datetime
from typing import Optional, Dict, Any
from decimal import Decimal
from enum import Enum

from pydantic import BaseModel, Field, field_validator
from sqlalchemy import Column, String, DateTime, Boolean, ForeignKey, Text, JSON, DECIMAL, Integer, Enum as SQLEnum
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func

from backend.database.connection import Base


class PPVOfferStatus(str, Enum):
    """Status of PPV offers."""
    PENDING = "pending"      # Offer sent, awaiting response
    ACCEPTED = "accepted"    # User accepted and paid
    DECLINED = "declined"    # User declined the offer
    EXPIRED = "expired"      # Offer expired without response
    CANCELLED = "cancelled"  # Offer cancelled by system/persona


class PPVOfferType(str, Enum):
    """Types of PPV content offers."""
    CUSTOM_IMAGE = "custom_image"      # Custom generated image
    CUSTOM_VIDEO = "custom_video"      # Custom generated video
    EXCLUSIVE_CHAT = "exclusive_chat"   # Extended private chat session
    PHOTO_SET = "photo_set"            # Set of exclusive photos
    VOICE_MESSAGE = "voice_message"     # Personal voice message
    CUSTOM_CONTENT = "custom_content"   # Other custom content


class PPVOfferModel(Base):
    """
    SQLAlchemy model for PPV offers.
    
    Represents premium content offers sent by AI personas to users
    during conversations as upselling opportunities.
    """
    
    __tablename__ = "ppv_offers"
    
    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        index=True
    )
    
    # Foreign keys
    conversation_id = Column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    persona_id = Column(
        UUID(as_uuid=True),
        ForeignKey("personas.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    
    # Offer details
    title = Column(String(200), nullable=False)
    description = Column(Text, nullable=False)
    offer_type = Column(
        SQLEnum(PPVOfferType),
        nullable=False,
        index=True
    )
    status = Column(
        SQLEnum(PPVOfferStatus),
        default=PPVOfferStatus.PENDING,
        nullable=False,
        index=True
    )
    
    # Pricing
    price = Column(DECIMAL(10, 2), nullable=False)  # Price in decimal format
    currency = Column(String(3), default="USD", nullable=False)
    
    # Content preview/details
    preview_url = Column(String(500), nullable=True)  # URL to content preview
    content_metadata = Column(JSON, nullable=True, default=dict)  # Additional content details
    
    # Delivery information
    estimated_delivery_hours = Column(Integer, default=24)  # How long to deliver content
    delivery_instructions = Column(Text, nullable=True)
    
    # Expiration
    expires_at = Column(DateTime(timezone=True), nullable=True, index=True)
    
    # Timestamps
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False
    )
    accepted_at = Column(DateTime(timezone=True), nullable=True)
    delivered_at = Column(DateTime(timezone=True), nullable=True)


class PPVOfferCreate(BaseModel):
    """API model for creating new PPV offers."""
    
    conversation_id: uuid.UUID = Field(description="ID of the conversation")
    user_id: uuid.UUID = Field(description="ID of the user receiving the offer")
    persona_id: uuid.UUID = Field(description="ID of the persona making the offer")
    title: str = Field(min_length=5, max_length=200, description="Offer title")
    description: str = Field(min_length=10, max_length=2000, description="Offer description")
    offer_type: PPVOfferType = Field(description="Type of content being offered")
    price: Decimal = Field(gt=0, le=999.99, description="Offer price")
    currency: str = Field(default="USD", min_length=3, max_length=3, description="Currency code")
    preview_url: Optional[str] = Field(None, max_length=500, description="Preview content URL")
    content_metadata: Optional[Dict[str, Any]] = Field(default={}, description="Additional content metadata")
    estimated_delivery_hours: int = Field(default=24, ge=1, le=168, description="Delivery time in hours")
    delivery_instructions: Optional[str] = Field(None, max_length=1000, description="Delivery instructions")
    expires_at: Optional[datetime] = Field(None, description="When the offer expires")
    
    @field_validator("currency")
    @classmethod
    def validate_currency(cls, v: str) -> str:
        """Validate currency code."""
        valid_currencies = ["USD", "EUR", "GBP", "CAD", "AUD"]
        if v.upper() not in valid_currencies:
            raise ValueError(f"Currency must be one of: {', '.join(valid_currencies)}")
        return v.upper()


class PPVOfferResponse(BaseModel):
    """API model for PPV offer responses."""
    
    id: uuid.UUID
    conversation_id: uuid.UUID
    user_id: uuid.UUID
    persona_id: uuid.UUID
    title: str
    description: str
    offer_type: PPVOfferType
    status: PPVOfferStatus
    price: Decimal
    currency: str
    preview_url: Optional[str]
    content_metadata: Dict[str, Any]
    estimated_delivery_hours: int
    delivery_instructions: Optional[str]
    expires_at: Optional[datetime]
    created_at: datetime
    updated_at: datetime
    accepted_at: Optional[datetime]
    delivered_at: Optional[datetime]
    
    model_config = {"from_attributes": True}