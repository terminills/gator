"""
Setup API Routes

Provides endpoints for initial system configuration through the admin panel.
"""

from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from backend.services.setup_service import SetupService, get_setup_service
from backend.config.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/setup", tags=["setup"])


class SetupConfigRequest(BaseModel):
    """Request to update system configuration."""

    # Database Configuration
    database_url: Optional[str] = Field(None, description="Database connection URL")
    database_test_url: Optional[str] = Field(None, description="Test database URL")
    redis_url: Optional[str] = Field(None, description="Redis connection URL")

    # AI Model Configuration
    ai_model_path: Optional[str] = Field(
        None, description="Path to AI models directory"
    )
    openai_api_key: Optional[str] = Field(None, description="OpenAI API key")
    anthropic_api_key: Optional[str] = Field(None, description="Anthropic API key")
    elevenlabs_api_key: Optional[str] = Field(None, description="ElevenLabs API key")
    hugging_face_token: Optional[str] = Field(None, description="Hugging Face token")
    stable_diffusion_model: Optional[str] = Field(
        None, description="Stable Diffusion model name"
    )
    content_moderation_model: Optional[str] = Field(
        None, description="Content moderation model"
    )

    # Social Media APIs
    facebook_api_key: Optional[str] = Field(None, description="Facebook API key")
    facebook_api_secret: Optional[str] = Field(None, description="Facebook API secret")
    instagram_api_key: Optional[str] = Field(None, description="Instagram API key")
    instagram_api_secret: Optional[str] = Field(
        None, description="Instagram API secret"
    )
    twitter_api_key: Optional[str] = Field(None, description="Twitter API key")
    twitter_api_secret: Optional[str] = Field(None, description="Twitter API secret")

    # Security Configuration
    secret_key: Optional[str] = Field(None, description="Application secret key")
    jwt_secret: Optional[str] = Field(None, description="JWT secret key")
    jwt_algorithm: Optional[str] = Field(None, description="JWT algorithm")
    jwt_expiration_hours: Optional[int] = Field(
        None, description="JWT expiration in hours"
    )
    encryption_key: Optional[str] = Field(None, description="Encryption key (base64)")

    # Application Settings
    debug: Optional[bool] = Field(None, description="Debug mode")
    environment: Optional[str] = Field(
        None, description="Environment (development/production)"
    )
    log_level: Optional[str] = Field(None, description="Logging level")
    api_version: Optional[str] = Field(None, description="API version")
    max_content_generation_concurrent: Optional[int] = Field(
        None, description="Max concurrent generations"
    )
    content_cache_ttl_seconds: Optional[int] = Field(
        None, description="Content cache TTL"
    )

    # File Storage
    upload_path: Optional[str] = Field(None, description="Upload directory path")
    generated_content_path: Optional[str] = Field(
        None, description="Generated content path"
    )
    max_file_size_mb: Optional[int] = Field(None, description="Maximum file size in MB")

    # Rate Limiting
    rate_limit_per_minute: Optional[int] = Field(
        None, description="Rate limit per minute"
    )
    rate_limit_burst: Optional[int] = Field(None, description="Rate limit burst")

    # Content Moderation Thresholds
    nsfw_threshold: Optional[float] = Field(
        None, description="NSFW detection threshold"
    )
    bias_threshold: Optional[float] = Field(
        None, description="Bias detection threshold"
    )
    toxicity_threshold: Optional[float] = Field(None, description="Toxicity threshold")

    # Infrastructure
    aws_access_key_id: Optional[str] = Field(None, description="AWS access key ID")
    aws_secret_access_key: Optional[str] = Field(
        None, description="AWS secret access key"
    )
    aws_region: Optional[str] = Field(None, description="AWS region")
    aws_s3_bucket: Optional[str] = Field(None, description="AWS S3 bucket name")

    # Monitoring and Logging
    sentry_dsn: Optional[str] = Field(None, description="Sentry DSN for error tracking")
    prometheus_endpoint: Optional[str] = Field(None, description="Prometheus endpoint")
    grafana_endpoint: Optional[str] = Field(None, description="Grafana endpoint")

    # Email Configuration
    smtp_host: Optional[str] = Field(None, description="SMTP server host")
    smtp_port: Optional[int] = Field(None, description="SMTP server port")
    smtp_user: Optional[str] = Field(None, description="SMTP username")
    smtp_password: Optional[str] = Field(None, description="SMTP password")

    # Social Media Webhooks
    facebook_webhook_verify_token: Optional[str] = Field(
        None, description="Facebook webhook token"
    )
    instagram_webhook_verify_token: Optional[str] = Field(
        None, description="Instagram webhook token"
    )

    # DNS Management
    godaddy_api_key: Optional[str] = Field(None, description="GoDaddy API key")
    godaddy_api_secret: Optional[str] = Field(None, description="GoDaddy API secret")
    godaddy_environment: Optional[str] = Field(None, description="GoDaddy environment")
    default_domain: Optional[str] = Field(None, description="Default domain")

    model_config = {
        "json_schema_extra": {
            "example": {
                "database_url": "postgresql://gator_user:password@localhost:5432/gator_dev",
                "openai_api_key": "sk-...",
                "secret_key": "your-secret-key-here",
                "environment": "production",
            }
        }
    }


class SetupStatusResponse(BaseModel):
    """Response with current setup status."""

    env_file_exists: bool
    env_file_path: str
    configured_sections: Dict[str, bool]
    current_config: Dict[str, Any]


class SetupConfigResponse(BaseModel):
    """Response after updating configuration."""

    success: bool
    message: str
    validation: Dict[str, Any]
    restart_required: bool = True


@router.get("/status")
async def get_setup_status(
    setup_service: SetupService = Depends(get_setup_service),
) -> SetupStatusResponse:
    """
    Get current setup status.

    Returns information about the current configuration state,
    including which sections are configured.
    """
    try:
        status = setup_service.get_setup_status()
        config = setup_service.get_current_config()

        return SetupStatusResponse(
            env_file_exists=status["env_file_exists"],
            env_file_path=status["env_file_path"],
            configured_sections=status["configured_sections"],
            current_config=config,
        )
    except Exception as e:
        logger.error(f"Failed to get setup status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get setup status: {str(e)}",
        )


@router.post("/config")
async def update_configuration(
    config_request: SetupConfigRequest,
    setup_service: SetupService = Depends(get_setup_service),
) -> SetupConfigResponse:
    """
    Update system configuration.

    Updates the .env file with provided configuration values.
    Only non-null values are updated. Application restart required
    for changes to take effect.

    Args:
        config_request: Configuration values to update

    Returns:
        Update result with validation information
    """
    try:
        # Convert request to dict, filtering out None values
        config_dict = {}
        for field_name, field_value in config_request.model_dump().items():
            if field_value is not None:
                # Convert field name to environment variable format
                env_key = field_name.upper()
                config_dict[env_key] = str(field_value)

        if not config_dict:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No configuration values provided",
            )

        # Validate configuration
        validation = setup_service.validate_config(config_dict)
        if not validation["valid"]:
            return SetupConfigResponse(
                success=False,
                message=f"Configuration validation failed: {', '.join(validation['errors'])}",
                validation=validation,
                restart_required=False,
            )

        # Update configuration
        success = setup_service.update_config(config_dict)

        if success:
            logger.info(
                f"Configuration updated successfully with {len(config_dict)} values"
            )

            message = "Configuration updated successfully. "
            if validation["warnings"]:
                message += f"Warnings: {', '.join(validation['warnings'])}. "
            message += "Restart the application for changes to take effect."

            return SetupConfigResponse(
                success=True,
                message=message,
                validation=validation,
                restart_required=True,
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update configuration",
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update configuration: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update configuration: {str(e)}",
        )


@router.get("/template")
async def get_configuration_template() -> Dict[str, Any]:
    """
    Get configuration template with all available settings.

    Returns a complete template showing all configuration options
    and their descriptions for the setup UI.
    """
    template = {
        "sections": {
            "database": {
                "title": "Database Configuration",
                "fields": {
                    "DATABASE_URL": {
                        "label": "Database URL",
                        "type": "text",
                        "placeholder": "postgresql://user:password@localhost:5432/gator_dev",
                        "required": True,
                    },
                    "DATABASE_TEST_URL": {
                        "label": "Test Database URL",
                        "type": "text",
                        "placeholder": "postgresql://user:password@localhost:5432/gator_test",
                    },
                    "REDIS_URL": {
                        "label": "Redis URL",
                        "type": "text",
                        "placeholder": "redis://localhost:6379/0",
                    },
                },
            },
            "ai_models": {
                "title": "AI Model Configuration",
                "fields": {
                    "AI_MODEL_PATH": {
                        "label": "Model Path",
                        "type": "text",
                        "placeholder": "/models",
                    },
                    "OPENAI_API_KEY": {
                        "label": "OpenAI API Key",
                        "type": "password",
                        "placeholder": "sk-...",
                    },
                    "ANTHROPIC_API_KEY": {
                        "label": "Anthropic API Key",
                        "type": "password",
                        "placeholder": "sk-ant-...",
                    },
                    "ELEVENLABS_API_KEY": {
                        "label": "ElevenLabs API Key",
                        "type": "password",
                    },
                    "HUGGING_FACE_TOKEN": {
                        "label": "Hugging Face Token",
                        "type": "password",
                    },
                },
            },
            "security": {
                "title": "Security Configuration",
                "fields": {
                    "SECRET_KEY": {
                        "label": "Secret Key",
                        "type": "password",
                        "required": True,
                    },
                    "JWT_SECRET": {
                        "label": "JWT Secret",
                        "type": "password",
                        "required": True,
                    },
                    "JWT_ALGORITHM": {
                        "label": "JWT Algorithm",
                        "type": "text",
                        "default": "HS256",
                    },
                    "ENCRYPTION_KEY": {
                        "label": "Encryption Key (Base64)",
                        "type": "password",
                    },
                },
            },
            "social_media": {
                "title": "Social Media APIs",
                "fields": {
                    "FACEBOOK_API_KEY": {
                        "label": "Facebook API Key",
                        "type": "password",
                    },
                    "FACEBOOK_API_SECRET": {
                        "label": "Facebook API Secret",
                        "type": "password",
                    },
                    "INSTAGRAM_API_KEY": {
                        "label": "Instagram API Key",
                        "type": "password",
                    },
                    "INSTAGRAM_API_SECRET": {
                        "label": "Instagram API Secret",
                        "type": "password",
                    },
                    "TWITTER_API_KEY": {"label": "Twitter API Key", "type": "password"},
                    "TWITTER_API_SECRET": {
                        "label": "Twitter API Secret",
                        "type": "password",
                    },
                },
            },
            "dns": {
                "title": "DNS Management",
                "fields": {
                    "GODADDY_API_KEY": {"label": "GoDaddy API Key", "type": "password"},
                    "GODADDY_API_SECRET": {
                        "label": "GoDaddy API Secret",
                        "type": "password",
                    },
                    "GODADDY_ENVIRONMENT": {
                        "label": "GoDaddy Environment",
                        "type": "text",
                        "default": "production",
                    },
                    "DEFAULT_DOMAIN": {"label": "Default Domain", "type": "text"},
                },
            },
            "application": {
                "title": "Application Settings",
                "fields": {
                    "DEBUG": {
                        "label": "Debug Mode",
                        "type": "boolean",
                        "default": "true",
                    },
                    "ENVIRONMENT": {
                        "label": "Environment",
                        "type": "text",
                        "default": "development",
                    },
                    "LOG_LEVEL": {
                        "label": "Log Level",
                        "type": "text",
                        "default": "DEBUG",
                    },
                },
            },
        }
    }

    return template
