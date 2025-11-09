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


@router.get("/ai-models/status")
async def get_ai_models_status() -> Dict[str, Any]:
    """
    Get AI model installation status and system capabilities.

    Returns information about installed models, available models,
    system hardware capabilities, and required dependency versions.
    """
    try:
        import sys
        import subprocess
        from pathlib import Path

        # Get system info
        system_info = {
            "python_version": sys.version,
            "platform": sys.platform,
        }

        # Try to use new ROCm utilities for enhanced GPU detection
        try:
            sys.path.insert(0, str(Path(__file__).parents[3]))
            from backend.utils.rocm_utils import (
                detect_rocm_version,
                check_pytorch_installation,
                get_pytorch_install_info,
                get_multi_gpu_config,
                generate_rocm_env_vars,
            )
            
            # Get ROCm version
            rocm_version = detect_rocm_version()
            if rocm_version:
                system_info["rocm_detected"] = True
                system_info["rocm_version_detected"] = str(rocm_version)
                system_info["rocm_6_5_plus"] = rocm_version.is_6_5_or_later
            
            # Get PyTorch installation info
            pytorch_info = check_pytorch_installation()
            system_info["gpu_available"] = pytorch_info["gpu_available"]
            system_info["torch_version"] = pytorch_info["version"]
            system_info["torch_installed"] = pytorch_info["installed"]
            system_info["gpu_count"] = pytorch_info["gpu_count"]
            
            if pytorch_info["installed"] and pytorch_info["is_rocm_build"]:
                system_info["is_rocm_build"] = True
                system_info["rocm_version"] = pytorch_info["rocm_build_version"]
            
            # Get detailed GPU architecture
            gpu_arch = pytorch_info.get("gpu_architecture", {})
            if gpu_arch.get("devices"):
                system_info["gpu_devices"] = gpu_arch["devices"]
                system_info["gpu_architectures"] = gpu_arch.get("architectures", [])
                system_info["total_gpu_memory_gb"] = gpu_arch.get("total_memory_gb", 0)
                system_info["multi_gpu"] = gpu_arch.get("multi_gpu", False)
                
                # Keep backward compatibility
                if gpu_arch["devices"]:
                    system_info["gpu_name"] = gpu_arch["devices"][0]["name"]
            
            # Get multi-GPU configuration if applicable
            if system_info.get("gpu_count", 0) > 1:
                multi_gpu_config = get_multi_gpu_config(system_info["gpu_count"])
                system_info["multi_gpu_config"] = multi_gpu_config
            
            # Get recommended environment variables
            env_vars = generate_rocm_env_vars(rocm_version, system_info.get("gpu_count"))
            system_info["recommended_env_vars"] = env_vars
            
        except ImportError:
            # Fallback to legacy GPU detection
            logger.warning("ROCm utilities not available, using legacy detection")
            try:
                import torch

                system_info["gpu_available"] = torch.cuda.is_available()
                system_info["torch_version"] = torch.__version__
                system_info["torch_installed"] = True
                
                if torch.cuda.is_available():
                    system_info["gpu_count"] = torch.cuda.device_count()
                    
                    # Get detailed information for all GPUs
                    gpu_devices = []
                    for i in range(torch.cuda.device_count()):
                        try:
                            props = torch.cuda.get_device_properties(i)
                            gpu_info = {
                                "device_id": i,
                                "name": torch.cuda.get_device_name(i),
                                "total_memory_gb": round(props.total_memory / (1024 ** 3), 2),
                                "compute_capability": f"{props.major}.{props.minor}",
                                "multi_processor_count": props.multi_processor_count,
                            }
                            gpu_devices.append(gpu_info)
                        except Exception as e:
                            logger.warning(f"Could not get properties for GPU {i}: {e}")
                            gpu_devices.append({
                                "device_id": i,
                                "name": "Unknown GPU",
                                "total_memory_gb": 0,
                                "error": str(e)
                            })
                    
                    system_info["gpu_devices"] = gpu_devices
                    system_info["gpu_name"] = torch.cuda.get_device_name(0) if gpu_devices else "Unknown"
                    
                    # Detect ROCm build version
                    if hasattr(torch.version, 'hip'):
                        rocm_version = getattr(torch.version, 'hip', None)
                        if rocm_version:
                            system_info["rocm_version"] = rocm_version
                            system_info["is_rocm_build"] = True
                        
            except ImportError:
                system_info["gpu_available"] = False
                system_info["torch_installed"] = False
                system_info["torch_version"] = "Not installed"

        # Get installed package versions for ML dependencies
        installed_versions = {}
        ml_packages = [
            "torch",
            "torchvision",
            "diffusers",
            "transformers",
            "accelerate",
            "huggingface_hub",
        ]
        for package in ml_packages:
            try:
                mod = __import__(package)
                installed_versions[package] = getattr(mod, "__version__", "Unknown")
            except ImportError:
                installed_versions[package] = "Not installed"

        # Get required versions from pyproject.toml
        # Path calculation: __file__ -> routes/ -> api/ -> backend/ -> src/ -> project_root
        project_root = Path(__file__).parents[4]
        pyproject_path = project_root / "pyproject.toml"

        required_versions = {
            "torch": "2.3.1+rocm5.7 (for AMD GPUs with MI-25)",
            "torchvision": "0.18.1+rocm5.7",
            "diffusers": ">=0.28.0",
            "transformers": ">=4.41.0",
            "accelerate": ">=0.29.0",
            "huggingface_hub": ">=0.23.0",
            "numpy": ">=1.24.0,<2.0",
        }

        # Try to parse actual requirements from pyproject.toml if it exists
        if pyproject_path.exists():
            try:
                import re

                content = pyproject_path.read_text()

                # Extract version constraints from dependencies section
                for package in [
                    "diffusers",
                    "transformers",
                    "accelerate",
                    "huggingface_hub",
                ]:
                    pattern = rf'"{package}>=([^"]+)"'
                    match = re.search(pattern, content)
                    if match:
                        required_versions[package] = f">={match.group(1)}"

                # Extract ROCm-specific versions from optional dependencies
                for package in ["torch", "torchvision"]:
                    pattern = rf'"{package}==([^"]+)"'
                    match = re.search(pattern, content)
                    if match:
                        required_versions[package] = match.group(1)

            except Exception as e:
                logger.warning(f"Could not parse pyproject.toml: {e}")

        # Check models directory
        models_dir = Path("./models")
        models_exist = models_dir.exists()

        installed_models = []
        if models_exist:
            # Check for installed models
            for category in ["text", "image", "voice"]:
                category_path = models_dir / category
                if category_path.exists():
                    for model_path in category_path.iterdir():
                        if model_path.is_dir():
                            installed_models.append(
                                {
                                    "name": model_path.name,
                                    "category": category,
                                    "path": str(model_path),
                                }
                            )

        # Available models for installation
        available_models = [
            {
                "name": "Stable Diffusion XL",
                "category": "image",
                "description": "High-quality image generation",
                "size": "6.9 GB",
                "requires_gpu": True,
            },
            {
                "name": "Llama 3.1 8B",
                "category": "text",
                "description": "Fast text generation model",
                "size": "16 GB",
                "requires_gpu": True,
            },
            {
                "name": "GPT-4 (API)",
                "category": "text",
                "description": "OpenAI GPT-4 via API",
                "size": "N/A",
                "requires_gpu": False,
                "requires_api_key": "OPENAI_API_KEY",
            },
            {
                "name": "DALL-E 3 (API)",
                "category": "image",
                "description": "OpenAI DALL-E 3 via API",
                "size": "N/A",
                "requires_gpu": False,
                "requires_api_key": "OPENAI_API_KEY",
            },
        ]

        # Check if setup script exists in project root
        # Path calculation: __file__ -> routes/ -> api/ -> backend/ -> src/ -> project_root
        project_root = Path(__file__).parents[4]
        setup_script = project_root / "setup_ai_models.py"

        return {
            "system": system_info,
            "installed_versions": installed_versions,
            "required_versions": required_versions,
            "compatibility_note": "PyTorch 2.3.1 is the latest version compatible with AMD MI-25 GPUs (ROCm 5.7)",
            "models_directory": str(models_dir.absolute()) if models_exist else None,
            "installed_models": installed_models,
            "available_models": available_models,
            "setup_script_available": setup_script.exists(),
        }

    except Exception as e:
        logger.error(f"Failed to get AI model status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get AI model status: {str(e)}",
        )


@router.post("/ai-models/analyze")
async def analyze_system_for_models() -> Dict[str, Any]:
    """
    Analyze system capabilities and recommend compatible AI models.

    Runs the setup_ai_models.py script with --analyze flag to determine
    which models can be installed on the current system.
    """
    try:
        import subprocess
        import sys
        import os
        from pathlib import Path

        # Get project root (setup script is in project root, not src)
        # Path calculation: __file__ -> routes/ -> api/ -> backend/ -> src/ -> project_root
        project_root = Path(__file__).parents[4]
        setup_script = project_root / "setup_ai_models.py"

        # Run the setup script with analyze flag
        result = subprocess.run(
            [sys.executable, setup_script, "--analyze"],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode == 0:
            return {
                "success": True,
                "output": result.stdout,
                "recommendations": "Check output for detailed analysis",
            }
        else:
            return {
                "success": False,
                "error": result.stderr,
                "output": result.stdout,
            }

    except subprocess.TimeoutExpired:
        raise HTTPException(
            status_code=status.HTTP_408_REQUEST_TIMEOUT,
            detail="Analysis timed out after 30 seconds",
        )
    except Exception as e:
        logger.error(f"Failed to analyze system: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to analyze system: {str(e)}",
        )


@router.get("/ai-models/recommendations")
async def get_model_recommendations() -> Dict[str, Any]:
    """
    Get structured model recommendations based on system capabilities.

    Returns a structured response with installable, upgradeable, and API-only models.
    """
    try:
        import sys
        from pathlib import Path

        # Import ModelSetupManager from setup script
        project_root = Path(__file__).parents[4]
        sys.path.insert(0, str(project_root))

        try:
            from setup_ai_models import ModelSetupManager

            manager = ModelSetupManager()
            sys_info = manager.get_system_info()
            recommendations = manager.analyze_system_requirements()

            return {
                "success": True,
                "system_info": sys_info,
                "recommendations": recommendations,
            }
        finally:
            # Clean up sys.path
            if str(project_root) in sys.path:
                sys.path.remove(str(project_root))

    except Exception as e:
        logger.error(f"Failed to get model recommendations: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get model recommendations: {str(e)}",
        )


class ModelInstallRequest(BaseModel):
    """Request to install AI models."""

    model_names: list[str] = Field(..., description="List of model names to install")
    model_type: str = Field("text", description="Model type (text, image, voice)")


@router.post("/ai-models/install")
async def install_models(request: ModelInstallRequest) -> Dict[str, Any]:
    """
    Install specified AI models.

    Initiates the installation of one or more models. This is a long-running
    operation that downloads and configures models.
    """
    try:
        import subprocess
        import sys
        from pathlib import Path

        project_root = Path(__file__).parents[4]
        setup_script = project_root / "setup_ai_models.py"

        # Build command with model names
        cmd = [sys.executable, str(setup_script), "--install"] + request.model_names

        logger.info(f"Starting model installation: {request.model_names}")

        # Run installation in background (non-blocking for async operation)
        # For now, we run it synchronously with a longer timeout
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minutes for model downloads
        )

        # Always return both stdout and stderr for transparency
        # Even successful installations may have warnings or partial failures
        response = {
            "success": result.returncode == 0,
            "message": (
                f"Installation completed for {len(request.model_names)} model(s)"
                if result.returncode == 0
                else "Installation failed or partially completed"
            ),
            "models": request.model_names,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "return_code": result.returncode,
        }

        # Log the installation result
        if result.returncode == 0:
            logger.info(f"Model installation completed: {request.model_names}")
        else:
            logger.warning(
                f"Model installation failed with code {result.returncode}: {request.model_names}"
            )

        return response

    except subprocess.TimeoutExpired:
        raise HTTPException(
            status_code=status.HTTP_408_REQUEST_TIMEOUT,
            detail="Model installation timed out (>5 minutes). Check logs for status.",
        )
    except Exception as e:
        logger.error(f"Failed to install models: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to install models: {str(e)}",
        )


class ModelEnableRequest(BaseModel):
    """Request to enable/disable an AI model."""

    model_name: str = Field(..., description="Model name")
    enabled: bool = Field(..., description="Whether to enable or disable the model")


@router.post("/ai-models/enable")
async def enable_model(request: ModelEnableRequest) -> Dict[str, Any]:
    """
    Enable or disable an installed AI model.

    Updates the model configuration to mark it as enabled or disabled for use.
    """
    try:
        from pathlib import Path
        import json

        models_dir = Path("./models")
        config_path = models_dir / "model_config.json"

        # Load or create configuration
        if config_path.exists():
            with open(config_path, "r") as f:
                config = json.load(f)
        else:
            config = {"enabled_models": {}}

        # Ensure enabled_models section exists
        if "enabled_models" not in config:
            config["enabled_models"] = {}

        # Update model status
        config["enabled_models"][request.model_name] = request.enabled

        # Save configuration
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        status_text = "enabled" if request.enabled else "disabled"
        logger.info(f"Model {request.model_name} {status_text}")

        return {
            "success": True,
            "message": f"Model {request.model_name} {status_text} successfully",
            "model_name": request.model_name,
            "enabled": request.enabled,
        }

    except Exception as e:
        logger.error(f"Failed to enable/disable model: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to enable/disable model: {str(e)}",
        )


@router.post("/ai-models/fix-dependencies")
async def fix_dependencies() -> Dict[str, Any]:
    """
    Install or update missing/outdated ML dependencies.

    Runs pip install to fix missing or outdated packages required for AI models.
    This excludes torch and torchvision to preserve ROCm-specific installations,
    and installs/upgrades diffusers, transformers, accelerate, huggingface_hub,
    numpy, and other dependencies.
    """
    try:
        import subprocess
        import sys
        from pathlib import Path
        import re

        logger.info(
            "Starting dependency fix installation (excluding torch/torchvision)"
        )

        # Get project root to access pyproject.toml
        project_root = Path(__file__).parents[4]
        pyproject_path = project_root / "pyproject.toml"

        # Read pyproject.toml to get dependency list
        if not pyproject_path.exists():
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="pyproject.toml not found",
            )

        content = pyproject_path.read_text()

        # Extract dependencies from pyproject.toml, excluding torch and torchvision
        # Match pattern: "package>=version" or "package[extra]>=version"
        dependency_pattern = r'"([^"]+)"'
        dependencies_section = False
        packages_to_install = []

        for line in content.split("\n"):
            line = line.strip()
            if line == "dependencies = [":
                dependencies_section = True
                continue
            elif dependencies_section and line == "]":
                break
            elif dependencies_section and line.startswith('"'):
                match = re.search(dependency_pattern, line)
                if match:
                    dep = match.group(1)
                    # Skip torch and torchvision to preserve ROCm installations
                    if not dep.startswith("torch==") and not dep.startswith(
                        "torchvision=="
                    ):
                        packages_to_install.append(dep)

        logger.info(
            f"Installing {len(packages_to_install)} packages (excluding torch/torchvision)"
        )

        # Install packages one by one to avoid dependency conflicts
        all_stdout = []
        all_stderr = []
        failed_packages = []

        for package in packages_to_install:
            logger.info(f"Installing/upgrading: {package}")
            cmd = [sys.executable, "-m", "pip", "install", "--upgrade", package]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minutes per package
            )

            all_stdout.append(f"=== Installing {package} ===\n{result.stdout}")
            all_stderr.append(result.stderr)

            if result.returncode != 0:
                failed_packages.append(package)
                logger.warning(f"Failed to install {package}: {result.stderr}")
            else:
                logger.info(f"Successfully installed/upgraded {package}")

        # Determine overall success
        success = len(failed_packages) == 0

        # Build response
        response = {
            "success": success,
            "message": (
                f"Successfully installed/upgraded all {len(packages_to_install)} packages (torch/torchvision preserved)"
                if success
                else f"Installed {len(packages_to_install) - len(failed_packages)}/{len(packages_to_install)} packages. Failed: {', '.join(failed_packages)}"
            ),
            "stdout": "\n".join(all_stdout),
            "stderr": "\n".join(all_stderr),
            "packages_installed": len(packages_to_install) - len(failed_packages),
            "packages_failed": len(failed_packages),
            "failed_packages": failed_packages,
        }

        # Log the installation result
        if success:
            logger.info(
                "Dependency fix completed successfully (torch/torchvision preserved)"
            )
        else:
            logger.warning(
                f"Dependency fix partially completed. Failed packages: {failed_packages}"
            )

        return response

    except subprocess.TimeoutExpired:
        raise HTTPException(
            status_code=status.HTTP_408_REQUEST_TIMEOUT,
            detail="Dependency installation timed out. Check logs for status.",
        )
    except Exception as e:
        logger.error(f"Failed to fix dependencies: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fix dependencies: {str(e)}",
        )
