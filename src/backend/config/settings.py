"""
Application Settings Configuration

Centralized configuration management using Pydantic settings.
Follows best practices for environment-based configuration.
"""

from enum import Enum
from functools import lru_cache
from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class GPULoadBalanceStrategy(str, Enum):
    """GPU load balancing strategies."""

    MEMORY = "memory"  # Select GPU with most free memory
    ROUND_ROBIN = "round_robin"  # Rotate through GPUs
    LEAST_LOADED = "least_loaded"  # Select GPU with lowest utilization


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
        default="sqlite:///./gator.db", description="Database connection URL"
    )
    database_pool_size: int = Field(
        default=5, description="Database connection pool size (PostgreSQL only)"
    )
    database_max_overflow: int = Field(
        default=10,
        description="Max extra connections above pool_size (PostgreSQL only)",
    )
    database_pool_recycle: int = Field(
        default=3600,
        description="Seconds before recycling connections (PostgreSQL only)",
    )

    # Security settings
    secret_key: str = Field(
        default="dev-secret-key-change-in-production",
        description="Secret key for JWT tokens",
    )
    jwt_algorithm: str = Field(default="HS256", description="JWT algorithm")
    jwt_expire_minutes: int = Field(default=30, description="JWT token expiration")

    # API security
    allowed_hosts: List[str] = Field(
        default=["*"],
        description="Allowed hosts for security. Use ['*'] to allow all hosts (recommended for development/internal networks) or specify exact hosts for production",
    )
    allowed_origins: List[str] = Field(
        default=["*"],
        description="CORS allowed origins. Use ['*'] to allow all origins (recommended for development) or specify exact origins for production",
    )

    # AI Model configuration
    ai_model_path: Optional[str] = Field(
        default=None, description="Path to AI models directory"
    )
    default_text_model: str = Field(
        default="llama3:8b", description="Default text generation model"
    )
    default_image_model: str = Field(
        default="stabilityai/stable-diffusion-xl-base-1.0",
        description="Default image generation model",
    )
    default_voice_model: str = Field(
        default="eleven_monolingual_v1",
        description="Default voice synthesis model",
    )
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    anthropic_api_key: Optional[str] = Field(
        default=None, description="Anthropic Claude API key"
    )
    elevenlabs_api_key: Optional[str] = Field(
        default=None, description="ElevenLabs voice synthesis API key"
    )
    hugging_face_token: Optional[str] = Field(
        default=None, description="Hugging Face API token"
    )
    civitai_api_key: Optional[str] = Field(
        default=None, description="Civitai API key for model downloads"
    )
    ollama_base_url: str = Field(
        default="http://localhost:11434", description="Ollama service base URL"
    )
    comfyui_base_url: Optional[str] = Field(
        default=None, description="ComfyUI service base URL"
    )

    # GPU Configuration
    gpu_memory_threshold: float = Field(
        default=0.9,
        ge=0.5,
        le=1.0,
        description="GPU memory utilization threshold (0.5-1.0)",
    )
    enable_multi_gpu: bool = Field(
        default=True, description="Enable multi-GPU load balancing"
    )
    gpu_load_balance_strategy: GPULoadBalanceStrategy = Field(
        default=GPULoadBalanceStrategy.MEMORY,
        description="GPU load balancing strategy",
    )
    max_gpu_memory_gb: Optional[float] = Field(
        default=None, description="Maximum GPU memory to use (GB), None for auto"
    )

    # ACD (Autonomous Continuous Development) Configuration
    acd_enabled: bool = Field(
        default=True, description="Enable ACD system for context tracking"
    )
    acd_learning_rate: float = Field(
        default=0.1,
        ge=0.01,
        le=1.0,
        description="ACD learning rate for self-improvement",
    )
    acd_correlation_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum correlation score for pattern matching",
    )
    acd_memory_consolidation_hours: int = Field(
        default=24,
        ge=1,
        description="Hours between automatic memory consolidation",
    )

    # Content generation settings
    max_content_generations_per_hour: int = Field(
        default=10, description="Rate limit for content generation"
    )
    max_concurrent_generations: int = Field(
        default=3, description="Maximum concurrent content generations"
    )
    generation_timeout_seconds: int = Field(
        default=300, description="Timeout for content generation in seconds"
    )

    # Timeout Configuration (centralized for production hardening)
    ollama_connect_timeout: float = Field(
        default=5.0, description="Ollama connection timeout in seconds"
    )
    ollama_generate_timeout: float = Field(
        default=60.0, description="Ollama generation timeout in seconds"
    )
    subprocess_default_timeout: int = Field(
        default=30, description="Default subprocess timeout in seconds"
    )
    model_download_timeout: int = Field(
        default=300, description="Model download timeout in seconds"
    )
    gpu_detection_timeout: int = Field(
        default=5, description="GPU detection timeout in seconds"
    )
    http_client_timeout: float = Field(
        default=30.0, description="Default HTTP client timeout in seconds"
    )
    orchestrator_invoke_timeout: float = Field(
        default=5.0, description="ACD orchestrator invocation timeout in seconds"
    )

    # Circuit Breaker Configuration
    circuit_breaker_failure_threshold: int = Field(
        default=5, description="Number of failures before circuit breaker opens"
    )
    circuit_breaker_recovery_timeout: int = Field(
        default=60, description="Seconds before attempting recovery after circuit opens"
    )

    # Social media API settings (legacy)
    facebook_api_key: Optional[str] = Field(default=None)
    facebook_api_secret: Optional[str] = Field(default=None)
    instagram_api_key: Optional[str] = Field(default=None)
    instagram_api_secret: Optional[str] = Field(default=None)

    # OAuth configuration
    base_url: str = Field(
        default="http://localhost:8000",
        description="Base URL for OAuth callbacks",
    )
    instagram_client_id: Optional[str] = Field(
        default=None, description="Instagram OAuth client ID"
    )
    instagram_client_secret: Optional[str] = Field(
        default=None, description="Instagram OAuth client secret"
    )
    facebook_client_id: Optional[str] = Field(
        default=None, description="Facebook OAuth client ID"
    )
    facebook_client_secret: Optional[str] = Field(
        default=None, description="Facebook OAuth client secret"
    )
    twitter_client_id: Optional[str] = Field(
        default=None, description="Twitter/X OAuth 2.0 client ID"
    )
    twitter_client_secret: Optional[str] = Field(
        default=None, description="Twitter/X OAuth 2.0 client secret"
    )
    tiktok_client_id: Optional[str] = Field(
        default=None, description="TikTok OAuth client ID"
    )
    tiktok_client_secret: Optional[str] = Field(
        default=None, description="TikTok OAuth client secret"
    )
    linkedin_client_id: Optional[str] = Field(
        default=None, description="LinkedIn OAuth client ID"
    )
    linkedin_client_secret: Optional[str] = Field(
        default=None, description="LinkedIn OAuth client secret"
    )

    # DNS Management (GoDaddy)
    godaddy_api_key: Optional[str] = Field(
        default=None, description="GoDaddy API key for DNS management"
    )
    godaddy_api_secret: Optional[str] = Field(
        default=None, description="GoDaddy API secret for DNS management"
    )
    godaddy_environment: str = Field(
        default="production", description="GoDaddy API environment (production/ote)"
    )
    default_domain: Optional[str] = Field(
        default=None, description="Default domain for the platform"
    )

    # Monitoring and observability
    sentry_dsn: Optional[str] = Field(
        default=None, description="Sentry DSN for error tracking"
    )
    prometheus_enabled: bool = Field(
        default=False, description="Enable Prometheus metrics"
    )

    # Redis/Celery configuration
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL for Celery task queue",
    )
    celery_broker_url: Optional[str] = Field(
        default=None, description="Celery broker URL (defaults to redis_url if not set)"
    )
    celery_result_backend: Optional[str] = Field(
        default=None,
        description="Celery result backend URL (defaults to redis_url if not set)",
    )

    # Backup configuration
    backup_dir: str = Field(
        default="/backups", description="Directory for storing automated backups"
    )
    backup_retention_days: int = Field(
        default=30, description="Number of days to retain backups"
    )
    content_storage_path: str = Field(
        default="generated_content", description="Path to generated content directory"
    )

    # IPMI/BMC Configuration
    ipmi_host: Optional[str] = Field(
        default=None, description="BMC/XCC IP address or hostname for IPMI access"
    )
    ipmi_username: Optional[str] = Field(
        default=None, description="BMC/XCC username for IPMI authentication"
    )
    ipmi_password: Optional[str] = Field(
        default=None, description="BMC/XCC password for IPMI authentication"
    )
    ipmi_interface: str = Field(
        default="lanplus",
        description="IPMI interface type (lanplus recommended for remote access)",
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
