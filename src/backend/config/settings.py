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
    the prefix 'GATOR_' (e.g., GATOR_DATABASE_URL).
    """
    
    model_config = SettingsConfigDict(
        env_prefix="GATOR_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    
    # Application settings
    debug: bool = Field(default=False, description="Enable debug mode")
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
    hugging_face_token: Optional[str] = Field(
        default=None,
        description="Hugging Face API token"
    )
    
    # Content generation settings
    max_content_generations_per_hour: int = Field(
        default=10,
        description="Rate limit for content generation"
    )
    
    # Social media API settings (placeholder - will be expanded)
    facebook_api_key: Optional[str] = Field(default=None)
    facebook_api_secret: Optional[str] = Field(default=None)
    instagram_api_key: Optional[str] = Field(default=None)
    instagram_api_secret: Optional[str] = Field(default=None)
    
    # Monitoring and observability
    sentry_dsn: Optional[str] = Field(
        default=None,
        description="Sentry DSN for error tracking"
    )
    prometheus_enabled: bool = Field(
        default=False,
        description="Enable Prometheus metrics"
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