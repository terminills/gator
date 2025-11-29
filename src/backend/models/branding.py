"""
Branding Models

Database models for site branding and customization.
Allows dynamic branding without restarting the application.
"""

import uuid
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field
from sqlalchemy import Boolean, Column, DateTime, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func

from backend.database.connection import Base


class BrandingModel(Base):
    """
    SQLAlchemy model for site branding configuration.

    Stores customizable branding that can be updated via UI without
    restarting the application. Each installation has one branding record.
    """

    __tablename__ = "branding"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)

    # Site identity
    site_name = Column(String(100), nullable=False, default="AI Content Platform")
    site_icon = Column(String(10), nullable=False, default="🤖")
    instance_name = Column(String(100), nullable=False, default="My AI Platform")
    site_tagline = Column(
        String(200), nullable=False, default="AI-Powered Content Generation"
    )

    # Theme colors
    primary_color = Column(String(7), nullable=False, default="#667eea")  # Hex color
    accent_color = Column(String(7), nullable=False, default="#10b981")  # Hex color

    # Logo and assets
    logo_url = Column(String(500), nullable=True)
    favicon_url = Column(String(500), nullable=True)

    # Additional customization
    custom_css = Column(Text, nullable=True)  # Custom CSS for advanced users

    # Metadata
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class BrandingCreate(BaseModel):
    """API model for creating branding configuration."""

    site_name: str = Field(
        default="AI Content Platform",
        min_length=1,
        max_length=100,
        description="Public-facing name of the site",
    )
    site_icon: str = Field(
        default="🤖", max_length=10, description="Icon/emoji for the site"
    )
    instance_name: str = Field(
        default="My AI Platform",
        min_length=1,
        max_length=100,
        description="Internal instance name",
    )
    site_tagline: str = Field(
        default="AI-Powered Content Generation",
        max_length=200,
        description="Site tagline/slogan",
    )
    primary_color: str = Field(
        default="#667eea",
        pattern=r"^#[0-9A-Fa-f]{6}$",
        description="Primary brand color (hex)",
    )
    accent_color: str = Field(
        default="#10b981",
        pattern=r"^#[0-9A-Fa-f]{6}$",
        description="Accent brand color (hex)",
    )
    logo_url: Optional[str] = Field(
        default=None, max_length=500, description="URL to custom logo image"
    )
    favicon_url: Optional[str] = Field(
        default=None, max_length=500, description="URL to custom favicon"
    )
    custom_css: Optional[str] = Field(
        default=None, description="Custom CSS for advanced customization"
    )


class BrandingUpdate(BaseModel):
    """API model for updating branding configuration."""

    site_name: Optional[str] = Field(None, min_length=1, max_length=100)
    site_icon: Optional[str] = Field(None, max_length=10)
    instance_name: Optional[str] = Field(None, min_length=1, max_length=100)
    site_tagline: Optional[str] = Field(None, max_length=200)
    primary_color: Optional[str] = Field(None, pattern=r"^#[0-9A-Fa-f]{6}$")
    accent_color: Optional[str] = Field(None, pattern=r"^#[0-9A-Fa-f]{6}$")
    logo_url: Optional[str] = Field(None, max_length=500)
    favicon_url: Optional[str] = Field(None, max_length=500)
    custom_css: Optional[str] = None


class BrandingResponse(BaseModel):
    """API response model for branding configuration."""

    id: str
    site_name: str
    site_icon: str
    instance_name: str
    site_tagline: str
    primary_color: str
    accent_color: str
    logo_url: Optional[str] = None
    favicon_url: Optional[str] = None
    custom_css: Optional[str] = None
    powered_by: str = "Gator AI Platform"
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True
