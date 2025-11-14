"""
Platform Content Policy Models

Database and API models for managing platform-specific content policies.
Allows dynamic configuration of content rating rules for different social media platforms.
"""

import uuid
from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field
from sqlalchemy import Column, String, DateTime, Boolean, JSON, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func

from backend.database.connection import Base
from backend.models.content import ContentRating


class PlatformPolicyModel(Base):
    """
    SQLAlchemy model for platform content policies.
    
    Stores content rating rules for each social media platform.
    Allows dynamic updates without code changes when platform rules change.
    """
    
    __tablename__ = "platform_policies"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    
    # Platform identification
    platform_name = Column(String(100), nullable=False, unique=True, index=True)
    platform_display_name = Column(String(255), nullable=False)
    platform_url = Column(String(500), nullable=True)
    
    # Content policy configuration
    allowed_content_ratings = Column(
        JSON, 
        nullable=False, 
        default=list
    )  # List of allowed ContentRating values: ["sfw", "moderate", "nsfw"]
    
    requires_content_warning = Column(
        JSON,
        nullable=False,
        default=list
    )  # Ratings that require content warnings: ["moderate", "nsfw"]
    
    requires_age_verification = Column(Boolean, default=False, nullable=False)
    min_age_requirement = Column(String(10), nullable=True)  # e.g., "18+", "13+", "21+"
    
    # Additional metadata
    policy_description = Column(Text, nullable=True)
    policy_url = Column(String(500), nullable=True)  # Link to official platform policy
    last_verified = Column(DateTime(timezone=True), nullable=True)
    
    # Status flags
    is_active = Column(Boolean, default=True, index=True)
    is_verified = Column(Boolean, default=False)  # Has policy been verified recently
    
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


class PlatformPolicyCreate(BaseModel):
    """API model for creating platform policy."""
    
    platform_name: str = Field(
        min_length=1,
        max_length=100,
        description="Platform identifier (lowercase, no spaces)"
    )
    platform_display_name: str = Field(
        min_length=1,
        max_length=255,
        description="Platform display name"
    )
    platform_url: Optional[str] = Field(default=None, description="Platform website URL")
    allowed_content_ratings: List[str] = Field(
        default=["sfw"],
        description="List of allowed content ratings"
    )
    requires_content_warning: List[str] = Field(
        default=[],
        description="Ratings that require content warnings"
    )
    requires_age_verification: bool = Field(
        default=False,
        description="Whether platform requires age verification"
    )
    min_age_requirement: Optional[str] = Field(
        default=None,
        description="Minimum age requirement (e.g., '18+', '13+')"
    )
    policy_description: Optional[str] = Field(
        default=None,
        description="Description of platform's content policy"
    )
    policy_url: Optional[str] = Field(
        default=None,
        description="URL to official platform policy"
    )


class PlatformPolicyUpdate(BaseModel):
    """API model for updating platform policy."""
    
    platform_display_name: Optional[str] = Field(default=None)
    platform_url: Optional[str] = Field(default=None)
    allowed_content_ratings: Optional[List[str]] = Field(default=None)
    requires_content_warning: Optional[List[str]] = Field(default=None)
    requires_age_verification: Optional[bool] = Field(default=None)
    min_age_requirement: Optional[str] = Field(default=None)
    policy_description: Optional[str] = Field(default=None)
    policy_url: Optional[str] = Field(default=None)
    is_active: Optional[bool] = Field(default=None)
    is_verified: Optional[bool] = Field(default=None)


class PlatformPolicyResponse(BaseModel):
    """API model for platform policy responses."""
    
    id: uuid.UUID
    platform_name: str
    platform_display_name: str
    platform_url: Optional[str]
    allowed_content_ratings: List[str]
    requires_content_warning: List[str]
    requires_age_verification: bool
    min_age_requirement: Optional[str]
    policy_description: Optional[str]
    policy_url: Optional[str]
    last_verified: Optional[datetime]
    is_active: bool
    is_verified: bool
    created_at: datetime
    updated_at: datetime
    
    model_config = {"from_attributes": True}


# Default platform policies to seed the database
DEFAULT_PLATFORM_POLICIES = [
    {
        "platform_name": "instagram",
        "platform_display_name": "Instagram",
        "platform_url": "https://www.instagram.com",
        "allowed_content_ratings": ["sfw", "moderate"],
        "requires_content_warning": ["moderate"],
        "requires_age_verification": False,
        "min_age_requirement": "13+",
        "policy_description": "Instagram allows safe-for-work and moderate content. Sexually explicit content is prohibited.",
        "policy_url": "https://help.instagram.com/477434105621119",
    },
    {
        "platform_name": "facebook",
        "platform_display_name": "Facebook",
        "platform_url": "https://www.facebook.com",
        "allowed_content_ratings": ["sfw"],
        "requires_content_warning": [],
        "requires_age_verification": False,
        "min_age_requirement": "13+",
        "policy_description": "Facebook only allows safe-for-work content. Adult content is prohibited.",
        "policy_url": "https://transparency.fb.com/policies/community-standards/",
    },
    {
        "platform_name": "twitter",
        "platform_display_name": "X (Twitter)",
        "platform_url": "https://twitter.com",
        "allowed_content_ratings": ["sfw", "moderate", "nsfw"],
        "requires_content_warning": ["moderate", "nsfw"],
        "requires_age_verification": True,
        "min_age_requirement": "18+",
        "policy_description": "Twitter allows all content ratings with appropriate content warnings for sensitive content.",
        "policy_url": "https://help.twitter.com/en/rules-and-policies/media-policy",
    },
    {
        "platform_name": "onlyfans",
        "platform_display_name": "OnlyFans",
        "platform_url": "https://onlyfans.com",
        "allowed_content_ratings": ["sfw", "moderate", "nsfw"],
        "requires_content_warning": [],
        "requires_age_verification": True,
        "min_age_requirement": "18+",
        "policy_description": "OnlyFans allows all content ratings with proper age verification.",
        "policy_url": "https://onlyfans.com/terms",
    },
    {
        "platform_name": "patreon",
        "platform_display_name": "Patreon",
        "platform_url": "https://www.patreon.com",
        "allowed_content_ratings": ["sfw", "moderate", "nsfw"],
        "requires_content_warning": ["nsfw"],
        "requires_age_verification": True,
        "min_age_requirement": "18+",
        "policy_description": "Patreon allows all content ratings with appropriate labeling and age-gating.",
        "policy_url": "https://support.patreon.com/hc/en-us/articles/360000317931",
    },
    {
        "platform_name": "discord",
        "platform_display_name": "Discord",
        "platform_url": "https://discord.com",
        "allowed_content_ratings": ["sfw", "moderate"],
        "requires_content_warning": ["moderate"],
        "requires_age_verification": False,
        "min_age_requirement": "13+",
        "policy_description": "Discord allows safe-for-work and moderate content. NSFW content must be in age-restricted channels.",
        "policy_url": "https://discord.com/guidelines",
    },
    {
        "platform_name": "reddit",
        "platform_display_name": "Reddit",
        "platform_url": "https://www.reddit.com",
        "allowed_content_ratings": ["sfw", "moderate", "nsfw"],
        "requires_content_warning": ["nsfw"],
        "requires_age_verification": False,
        "min_age_requirement": "13+",
        "policy_description": "Reddit allows all content ratings with NSFW tagging for adult content.",
        "policy_url": "https://www.redditinc.com/policies/content-policy",
    },
    {
        "platform_name": "tiktok",
        "platform_display_name": "TikTok",
        "platform_url": "https://www.tiktok.com",
        "allowed_content_ratings": ["sfw"],
        "requires_content_warning": [],
        "requires_age_verification": False,
        "min_age_requirement": "13+",
        "policy_description": "TikTok only allows safe-for-work content. Sexually explicit content is prohibited.",
        "policy_url": "https://www.tiktok.com/community-guidelines",
    },
    {
        "platform_name": "youtube",
        "platform_display_name": "YouTube",
        "platform_url": "https://www.youtube.com",
        "allowed_content_ratings": ["sfw", "moderate"],
        "requires_content_warning": ["moderate"],
        "requires_age_verification": False,
        "min_age_requirement": "13+",
        "policy_description": "YouTube allows safe-for-work and moderate content. Age-restricted content requires viewer confirmation.",
        "policy_url": "https://www.youtube.com/howyoutubeworks/policies/community-guidelines/",
    },
    {
        "platform_name": "twitch",
        "platform_display_name": "Twitch",
        "platform_url": "https://www.twitch.tv",
        "allowed_content_ratings": ["sfw", "moderate"],
        "requires_content_warning": ["moderate"],
        "requires_age_verification": False,
        "min_age_requirement": "13+",
        "policy_description": "Twitch allows safe-for-work and moderate content with appropriate content labels.",
        "policy_url": "https://www.twitch.tv/p/en/legal/community-guidelines/",
    },
]
