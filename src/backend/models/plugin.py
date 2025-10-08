"""
Plugin Registry Database Models

SQLAlchemy models for plugin registry and marketplace.
"""

from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from uuid import uuid4

from sqlalchemy import (
    Column,
    String,
    Text,
    Integer,
    Float,
    Boolean,
    JSON,
    DateTime,
    Enum as SQLEnum,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
from pydantic import BaseModel, Field

from backend.plugins import PluginType, PluginStatus

Base = declarative_base()


class PluginModel(Base):
    """Database model for plugins in the marketplace."""

    __tablename__ = "plugins"

    # Primary key
    id = Column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid4()).replace("-", ""),
    )

    # Basic info
    name = Column(String(200), nullable=False, index=True)
    slug = Column(String(200), unique=True, nullable=False, index=True)
    version = Column(String(50), nullable=False)
    author = Column(String(200), nullable=False)
    author_email = Column(String(200))
    description = Column(Text, nullable=False)

    # Classification
    plugin_type = Column(SQLEnum(PluginType), nullable=False, index=True)
    tags = Column(JSON, default=list)  # List of tags
    categories = Column(JSON, default=list)  # List of categories

    # URLs and links
    homepage = Column(String(500))
    repository = Column(String(500))
    documentation = Column(String(500))
    license = Column(String(100), default="MIT")

    # Installation and configuration
    package_url = Column(String(500))  # Download URL for plugin package
    install_command = Column(Text)  # Installation command/script
    permissions = Column(JSON, default=list)  # Required permissions
    dependencies = Column(JSON, default=dict)  # Plugin dependencies
    config_schema = Column(JSON)  # JSON schema for configuration

    # Marketplace metadata
    downloads = Column(Integer, default=0)
    rating = Column(Float, default=0.0)
    rating_count = Column(Integer, default=0)
    featured = Column(Boolean, default=False, index=True)
    verified = Column(Boolean, default=False)
    price = Column(Float, default=0.0)  # 0 for free plugins

    # Status
    status = Column(
        SQLEnum(PluginStatus), default=PluginStatus.INACTIVE, nullable=False
    )
    published_at = Column(DateTime(timezone=True))
    deprecated = Column(Boolean, default=False)
    deprecated_message = Column(Text)

    # Timestamps
    created_at = Column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    # Metadata storage (renamed from 'metadata' to avoid SQLAlchemy reserved name)
    plugin_metadata = Column(JSON, default=dict)


class PluginInstallation(Base):
    """Database model for installed plugins (per user/tenant)."""

    __tablename__ = "plugin_installations"

    # Primary key
    id = Column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid4()).replace("-", ""),
    )

    # Plugin reference
    plugin_id = Column(String(36), nullable=False, index=True)
    plugin_slug = Column(String(200), nullable=False, index=True)
    plugin_version = Column(String(50), nullable=False)

    # Installation details
    installed_by = Column(String(36))  # User ID
    tenant_id = Column(String(36), index=True)  # For multi-tenancy support

    # Configuration
    config = Column(JSON, default=dict)  # Plugin-specific configuration
    enabled = Column(Boolean, default=True)

    # Status tracking
    status = Column(
        SQLEnum(PluginStatus), default=PluginStatus.INSTALLED, nullable=False
    )
    error_message = Column(Text)  # Last error message

    # Usage stats
    last_used_at = Column(DateTime(timezone=True))
    usage_count = Column(Integer, default=0)

    # Timestamps
    installed_at = Column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    # Metadata storage (renamed from 'metadata' to avoid SQLAlchemy reserved name)
    installation_metadata = Column(JSON, default=dict)


class PluginReview(Base):
    """Database model for plugin reviews and ratings."""

    __tablename__ = "plugin_reviews"

    # Primary key
    id = Column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid4()).replace("-", ""),
    )

    # References
    plugin_id = Column(String(36), nullable=False, index=True)
    user_id = Column(String(36), nullable=False, index=True)

    # Review content
    rating = Column(Integer, nullable=False)  # 1-5 stars
    title = Column(String(200))
    review_text = Column(Text)

    # Helpfulness tracking
    helpful_count = Column(Integer, default=0)
    reported = Column(Boolean, default=False)

    # Timestamps
    created_at = Column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )


# Pydantic schemas for API


class PluginSchema(BaseModel):
    """Schema for plugin in API responses."""

    id: str
    name: str
    slug: str
    version: str
    author: str
    description: str
    plugin_type: PluginType
    tags: List[str] = Field(default_factory=list)
    homepage: Optional[str] = None
    repository: Optional[str] = None
    license: str = "MIT"
    downloads: int = 0
    rating: float = 0.0
    rating_count: int = 0
    featured: bool = False
    verified: bool = False
    price: float = 0.0
    status: PluginStatus
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class PluginInstallationSchema(BaseModel):
    """Schema for plugin installation in API responses."""

    id: str
    plugin_id: str
    plugin_slug: str
    plugin_version: str
    config: Dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True
    status: PluginStatus
    error_message: Optional[str] = None
    installed_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class PluginReviewSchema(BaseModel):
    """Schema for plugin review in API responses."""

    id: str
    plugin_id: str
    user_id: str
    rating: int
    title: Optional[str] = None
    review_text: Optional[str] = None
    helpful_count: int = 0
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class PluginInstallRequest(BaseModel):
    """Request schema for installing a plugin."""

    plugin_slug: str
    config: Optional[Dict[str, Any]] = Field(default_factory=dict)


class PluginUpdateRequest(BaseModel):
    """Request schema for updating plugin configuration."""

    config: Optional[Dict[str, Any]] = None
    enabled: Optional[bool] = None


class PluginReviewRequest(BaseModel):
    """Request schema for creating/updating a plugin review."""

    rating: int = Field(..., ge=1, le=5)
    title: Optional[str] = None
    review_text: Optional[str] = None
