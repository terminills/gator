"""
Application Settings Configuration

Centralized configuration management using Pydantic settings.
Follows best practices for environment-based configuration.
"""

import os
from typing import List, Optional
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings with environment variable support.
    
    All settings can be overridden via environment variables with
    the prefix 'GATOR_' (e.g., GATOR_DEBUG).
    """
    
    model_config = SettingsConfigDict(
        env_prefix="GATOR_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    
    # Application settings
    debug: bool = Field(default=True, description="Enable debug mode")
    environment: str = Field(default="development", description="Environment name")
    log_level: str = Field(default="INFO", description="Logging level")
    
    # Database configuration
    database_url: str = Field(
        default="sqlite:///./gator.db",
        description="Database connection URL"
    )
    
    # Security settings
    secret_key: str = Field(
        default="dev-secret-key-change-in-production",
        description="Secret key for JWT tokens"
    )
    jwt_algorithm: str = Field(default="HS256", description="JWT algorithm")
    jwt_expire_minutes: int = Field(default=30, description="JWT token expiration")
    
    # API security
    allowed_hosts: List[str] = Field(
        default=["localhost", "127.0.0.1", "0.0.0.0", "testserver"],
        description="Allowed hosts for security"
    )
    allowed_origins: List[str] = Field(
        default=["http://localhost:3000", "http://127.0.0.1:3000"],
        description="CORS allowed origins"
    )
    
    # AI Model configuration
    ai_model_path: Optional[str] = Field(
        default=None,
        description="Path to AI models directory"
    )
    openai_api_key: Optional[str] = Field(
        default=None,
        description="OpenAI API key"
    )
    anthropic_api_key: Optional[str] = Field(
        default=None,
        description="Anthropic Claude API key"
    )
    elevenlabs_api_key: Optional[str] = Field(
        default=None,
        description="ElevenLabs voice synthesis API key"
    )
    hugging_face_token: Optional[str] = Field(
        default=None,
        description="Hugging Face API token"
    )
    
    # Content generation settings
    max_content_generations_per_hour: int = Field(
        default=10,
        description="Rate limit for content generation"
    )
    
    # Social media API settings
    facebook_api_key: Optional[str] = Field(default=None)
    facebook_api_secret: Optional[str] = Field(default=None)
    instagram_api_key: Optional[str] = Field(default=None)
    instagram_api_secret: Optional[str] = Field(default=None)
    
    # DNS Management (GoDaddy)
    godaddy_api_key: Optional[str] = Field(
        default=None,
        description="GoDaddy API key for DNS management"
    )
    godaddy_api_secret: Optional[str] = Field(
        default=None,
        description="GoDaddy API secret for DNS management"
    )
    godaddy_environment: str = Field(
        default="production",
        description="GoDaddy API environment (production/ote)"
    )
    default_domain: Optional[str] = Field(
        default=None,
        description="Default domain for the platform"
    )
    
    # Monitoring and observability
    sentry_dsn: Optional[str] = Field(
        default=None,
        description="Sentry DSN for error tracking"
    )
    prometheus_enabled: bool = Field(
        default=False,
        description="Enable Prometheus metrics"
    )
    
    # Redis/Celery configuration
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL for Celery task queue"
    )
    celery_broker_url: Optional[str] = Field(
        default=None,
        description="Celery broker URL (defaults to redis_url if not set)"
    )
    celery_result_backend: Optional[str] = Field(
        default=None,
        description="Celery result backend URL (defaults to redis_url if not set)"
    )
    
    # Backup configuration
    backup_dir: str = Field(
        default="/backups",
        description="Directory for storing automated backups"
    )
    backup_retention_days: int = Field(
        default=30,
        description="Number of days to retain backups"
    )
    content_storage_path: str = Field(
        default="generated_content",
        description="Path to generated content directory"
    )


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    
    Uses LRU cache to avoid re-parsing environment variables
    on every request.
    
    Returns:
        Settings: Application settings instance
    """
    return Settings()