"""
System Settings Models

Database models for all application configuration.
Replaces environment variables with database storage for dynamic updates.
"""

import uuid
from datetime import datetime
from typing import Optional, Dict, Any
from enum import Enum

from pydantic import BaseModel, Field
from sqlalchemy import Column, String, DateTime, Boolean, Integer, Float, Text, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func

from backend.database.connection import Base


class SettingCategory(str, Enum):
    """Categories for organizing settings."""
    AI_MODELS = "ai_models"
    SOCIAL_MEDIA = "social_media"
    STORAGE = "storage"
    SECURITY = "security"
    MONITORING = "monitoring"
    EMAIL = "email"
    DNS = "dns"
    CLOUD = "cloud"
    PERFORMANCE = "performance"
    CONTENT = "content"
    IPMI = "ipmi"


class SystemSettingModel(Base):
    """
    SQLAlchemy model for system settings.
    
    Stores all configuration in database instead of environment variables.
    Allows dynamic updates without application restarts.
    """

    __tablename__ = "system_settings"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    
    # Setting identification
    key = Column(String(100), nullable=False, unique=True, index=True)
    category = Column(String(50), nullable=False, index=True)
    
    # Setting value (stored as JSON for flexibility)
    value = Column(JSON, nullable=True)
    
    # Metadata
    description = Column(Text, nullable=True)
    is_sensitive = Column(Boolean, default=False, nullable=False)  # Encrypt if True
    is_active = Column(Boolean, default=True, nullable=False)
    
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


class SettingCreate(BaseModel):
    """API model for creating settings."""
    
    key: str = Field(
        min_length=1,
        max_length=100,
        description="Unique setting key (e.g., 'openai_api_key')"
    )
    category: SettingCategory = Field(
        description="Setting category for organization"
    )
    value: Any = Field(
        description="Setting value (can be string, number, dict, list)"
    )
    description: Optional[str] = Field(
        default=None,
        description="Human-readable description"
    )
    is_sensitive: bool = Field(
        default=False,
        description="If true, value will be encrypted"
    )


class SettingUpdate(BaseModel):
    """API model for updating settings."""
    
    value: Optional[Any] = None
    description: Optional[str] = None
    is_sensitive: Optional[bool] = None
    is_active: Optional[bool] = None


class SettingResponse(BaseModel):
    """API response model for settings."""
    
    id: str
    key: str
    category: str
    value: Any
    description: Optional[str]
    is_sensitive: bool
    is_active: bool
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# Default settings that should be created on first run
DEFAULT_SETTINGS = {
    # AI Models
    "ai_model_path": {
        "category": "ai_models",
        "value": "./models",
        "description": "Path to AI models directory",
        "is_sensitive": False,
    },
    "openai_api_key": {
        "category": "ai_models",
        "value": None,
        "description": "OpenAI API key for GPT models",
        "is_sensitive": True,
    },
    "anthropic_api_key": {
        "category": "ai_models",
        "value": None,
        "description": "Anthropic API key for Claude",
        "is_sensitive": True,
    },
    "elevenlabs_api_key": {
        "category": "ai_models",
        "value": None,
        "description": "ElevenLabs API key for voice synthesis",
        "is_sensitive": True,
    },
    "hugging_face_token": {
        "category": "ai_models",
        "value": None,
        "description": "Hugging Face API token",
        "is_sensitive": True,
    },
    "civitai_api_key": {
        "category": "ai_models",
        "value": None,
        "description": "CivitAI API key for downloading models",
        "is_sensitive": True,
    },
    "civitai_allow_nsfw": {
        "category": "ai_models",
        "value": False,
        "description": "Allow NSFW models from CivitAI",
        "is_sensitive": False,
    },
    "civitai_track_usage": {
        "category": "ai_models",
        "value": True,
        "description": "Track which models are from CivitAI for content attribution",
        "is_sensitive": False,
    },
    "prefer_ollama_for_gfx1030": {
        "category": "ai_models",
        "value": True,
        "description": "Use Ollama instead of vLLM for AMD gfx1030 GPUs (RX 6000 series)",
        "is_sensitive": False,
    },
    
    # Social Media
    "facebook_api_key": {
        "category": "social_media",
        "value": None,
        "description": "Facebook API key",
        "is_sensitive": True,
    },
    "facebook_api_secret": {
        "category": "social_media",
        "value": None,
        "description": "Facebook API secret",
        "is_sensitive": True,
    },
    "instagram_api_key": {
        "category": "social_media",
        "value": None,
        "description": "Instagram API key",
        "is_sensitive": True,
    },
    "instagram_api_secret": {
        "category": "social_media",
        "value": None,
        "description": "Instagram API secret",
        "is_sensitive": True,
    },
    "twitter_api_key": {
        "category": "social_media",
        "value": None,
        "description": "Twitter API key",
        "is_sensitive": True,
    },
    "twitter_api_secret": {
        "category": "social_media",
        "value": None,
        "description": "Twitter API secret",
        "is_sensitive": True,
    },
    
    # Storage
    "upload_path": {
        "category": "storage",
        "value": "./uploads",
        "description": "Directory for uploaded files",
        "is_sensitive": False,
    },
    "generated_content_path": {
        "category": "storage",
        "value": "./generated",
        "description": "Directory for generated content",
        "is_sensitive": False,
    },
    "max_file_size_mb": {
        "category": "storage",
        "value": 50,
        "description": "Maximum upload file size in MB",
        "is_sensitive": False,
    },
    
    # Performance
    "max_content_generation_concurrent": {
        "category": "performance",
        "value": 4,
        "description": "Maximum concurrent content generations",
        "is_sensitive": False,
    },
    "rate_limit_per_minute": {
        "category": "performance",
        "value": 100,
        "description": "API rate limit per minute",
        "is_sensitive": False,
    },
    
    # Content Moderation
    "nsfw_threshold": {
        "category": "content",
        "value": 0.8,
        "description": "NSFW detection threshold (0-1)",
        "is_sensitive": False,
    },
    "bias_threshold": {
        "category": "content",
        "value": 0.7,
        "description": "Bias detection threshold (0-1)",
        "is_sensitive": False,
    },
    "toxicity_threshold": {
        "category": "content",
        "value": 0.8,
        "description": "Toxicity detection threshold (0-1)",
        "is_sensitive": False,
    },
    
    # Email
    "smtp_host": {
        "category": "email",
        "value": None,
        "description": "SMTP server hostname",
        "is_sensitive": False,
    },
    "smtp_port": {
        "category": "email",
        "value": 587,
        "description": "SMTP server port",
        "is_sensitive": False,
    },
    "smtp_user": {
        "category": "email",
        "value": None,
        "description": "SMTP username",
        "is_sensitive": True,
    },
    "smtp_password": {
        "category": "email",
        "value": None,
        "description": "SMTP password",
        "is_sensitive": True,
    },
    
    # DNS Management
    "godaddy_api_key": {
        "category": "dns",
        "value": None,
        "description": "GoDaddy API key",
        "is_sensitive": True,
    },
    "godaddy_api_secret": {
        "category": "dns",
        "value": None,
        "description": "GoDaddy API secret",
        "is_sensitive": True,
    },
    "godaddy_environment": {
        "category": "dns",
        "value": "production",
        "description": "GoDaddy API environment",
        "is_sensitive": False,
    },
    
    # Cloud/AWS
    "aws_access_key_id": {
        "category": "cloud",
        "value": None,
        "description": "AWS access key ID",
        "is_sensitive": True,
    },
    "aws_secret_access_key": {
        "category": "cloud",
        "value": None,
        "description": "AWS secret access key",
        "is_sensitive": True,
    },
    "aws_region": {
        "category": "cloud",
        "value": "us-west-2",
        "description": "AWS region",
        "is_sensitive": False,
    },
    "aws_s3_bucket": {
        "category": "cloud",
        "value": None,
        "description": "AWS S3 bucket name",
        "is_sensitive": False,
    },
    
    # Monitoring
    "sentry_dsn": {
        "category": "monitoring",
        "value": None,
        "description": "Sentry DSN for error tracking",
        "is_sensitive": True,
    },
    
    # IPMI/BMC Configuration
    "ipmi_host": {
        "category": "ipmi",
        "value": None,
        "description": "BMC/XCC IP address or hostname for IPMI access",
        "is_sensitive": False,
    },
    "ipmi_username": {
        "category": "ipmi",
        "value": None,
        "description": "BMC/XCC username for IPMI authentication",
        "is_sensitive": True,
    },
    "ipmi_password": {
        "category": "ipmi",
        "value": None,
        "description": "BMC/XCC password for IPMI authentication",
        "is_sensitive": True,
    },
    "ipmi_interface": {
        "category": "ipmi",
        "value": "lanplus",
        "description": "IPMI interface type (lanplus recommended for remote access)",
        "is_sensitive": False,
    },
}
