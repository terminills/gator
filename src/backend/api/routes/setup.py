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
@router.get("/models/status")  # Alias for backward compatibility
async def get_ai_models_status() -> Dict[str, Any]:
    """
    Get AI model installation status and system capabilities.

    Returns information about installed models, available models,
    system hardware capabilities, and required dependency versions.
    
    This endpoint now integrates with the AIModelManager to properly
    detect and report loaded models.
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

        # Determine numpy requirement based on installed PyTorch version
        numpy_requirement = ">=1.24.0,<2.0"  # Default/fallback
        torch_version = system_info.get("torch_version", "")
        if torch_version and torch_version != "Not installed":
            try:
                # Extract major.minor version from torch version string
                # Examples: "2.9.0+cu128" -> "2.9", "2.3.1+rocm5.7" -> "2.3"
                torch_ver_parts = torch_version.split("+")[0].split(".")
                torch_major = int(torch_ver_parts[0])
                torch_minor = int(torch_ver_parts[1]) if len(torch_ver_parts) > 1 else 0
                
                # PyTorch 2.0+ generally requires numpy>=1.21.0
                # PyTorch 2.9+ is compatible with numpy 1.26.x and 2.x
                # PyTorch 2.4-2.8 works with numpy 1.24-1.26
                # PyTorch 2.0-2.3 works with numpy 1.21-1.26
                if torch_major == 2 and torch_minor >= 9:
                    numpy_requirement = ">=1.26.0"  # No upper bound for PyTorch 2.9+
                elif torch_major == 2 and torch_minor >= 4:
                    numpy_requirement = ">=1.24.0,<2.0"  # PyTorch 2.4-2.8
                elif torch_major == 2:
                    numpy_requirement = ">=1.21.0,<2.0"  # PyTorch 2.0-2.3
                elif torch_major == 1:
                    numpy_requirement = ">=1.21.0,<1.27"  # PyTorch 1.x
            except (ValueError, IndexError) as e:
                logger.warning(f"Could not parse PyTorch version '{torch_version}' for numpy requirement: {e}")
        
        required_versions = {
            "torch": "2.3.1+rocm5.7 (for AMD GPUs with MI-25)",
            "torchvision": "0.18.1+rocm5.7",
            "diffusers": ">=0.28.0",
            "transformers": ">=4.41.0",
            "accelerate": ">=0.29.0",
            "huggingface_hub": ">=0.23.0",
            "numpy": numpy_requirement,
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

        # Get models from AIModelManager (which was initialized at startup)
        installed_models = []
        loaded_models_count = 0
        total_models_count = 0
        
        try:
            from backend.services.ai_models import ai_models
            
            # Get all available models from AIModelManager
            for category in ["text", "image", "voice", "video", "audio"]:
                category_models = ai_models.available_models.get(category, [])
                total_models_count += len(category_models)
                
                for model in category_models:
                    is_loaded = model.get("loaded", False)
                    if is_loaded:
                        loaded_models_count += 1
                    
                    # Add to installed models list with detailed info
                    model_info = {
                        "name": model.get("name"),
                        "display_name": model.get("display_name", model.get("name")),
                        "category": category,
                        "provider": model.get("provider", "unknown"),
                        "source": model.get("source", "local"),
                        "type": model.get("type", "unknown"),
                        "model_type": model.get("model_type", ""),  # CivitAI model type
                        "loaded": is_loaded,
                        "can_load": model.get("can_load", False),
                        "size_gb": model.get("size_gb", 0),
                        "description": model.get("description", ""),
                        "path": model.get("path", ""),
                        "inference_engine": model.get("inference_engine", ""),
                        "device": model.get("device", "cpu"),
                    }
                    
                    # Add CivitAI-specific fields if present
                    if model.get("source") == "civitai":
                        model_info["base_model"] = model.get("base_model", "")
                        model_info["trained_words"] = model.get("trained_words", [])
                        model_info["nsfw"] = model.get("nsfw", False)
                        model_info["civitai_model_id"] = model.get("civitai_model_id")
                        model_info["civitai_version_id"] = model.get("civitai_version_id")
                    
                    installed_models.append(model_info)
            
            logger.info(f"Retrieved {loaded_models_count}/{total_models_count} loaded models from AIModelManager")
            
        except Exception as e:
            logger.warning(f"Could not get models from AIModelManager: {e}")
            # Fallback to filesystem detection
            models_dir = Path("./models")
            models_exist = models_dir.exists()

            if models_exist:
                # Check for installed models in all categories
                for category in ["text", "image", "voice", "video", "audio"]:
                    category_path = models_dir / category
                    if category_path.exists():
                        for model_path in category_path.iterdir():
                            if model_path.is_dir():
                                # Get model size
                                try:
                                    total_size = sum(f.stat().st_size for f in model_path.rglob('*') if f.is_file())
                                    size_gb = round(total_size / (1024 ** 3), 2)
                                except Exception:
                                    size_gb = 0
                                
                                # Check if model has required files
                                has_config = (model_path / "config.json").exists()
                                has_model_files = any(model_path.glob("*.safetensors")) or any(model_path.glob("*.bin")) or any(model_path.glob("*.pt"))
                                is_valid = has_config or has_model_files
                                
                                total_models_count += 1
                                if is_valid:
                                    loaded_models_count += 1
                                
                                installed_models.append(
                                    {
                                        "name": model_path.name,
                                        "category": category,
                                        "path": str(model_path),
                                        "size_gb": size_gb,
                                        "loaded": is_valid,
                                        "is_valid": is_valid,
                                        "has_config": has_config,
                                        "has_model_files": has_model_files,
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
        
        # Check models directory for backward compatibility
        models_dir = Path("./models")

        return {
            "system": system_info,
            "installed_versions": installed_versions,
            "required_versions": required_versions,
            "compatibility_note": "PyTorch 2.3.1 is the latest version compatible with AMD MI-25 GPUs (ROCm 5.7)",
            "models_directory": str(models_dir.absolute()) if models_dir.exists() else None,
            "installed_models": installed_models,
            "loaded_models_count": loaded_models_count,
            "total_models_count": total_models_count,
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


@router.get("/ai-models/civitai-browse")
async def browse_civitai_models(
    query: Optional[str] = None,
    model_type: Optional[str] = None,
    limit: int = 12,
) -> Dict[str, Any]:
    """
    Browse popular models from CivitAI for easy installation.
    
    Returns a curated list of models suitable for the AI models setup page.
    """
    try:
        from backend.utils.civitai_utils import CivitAIClient, CivitAIModelType
        from backend.config.settings import get_settings
        
        settings = get_settings()
        api_key = getattr(settings, "civitai_api_key", None)
        allow_nsfw = getattr(settings, "civitai_allow_nsfw", False)
        
        client = CivitAIClient(api_key=api_key)
        
        # Convert model type string to enum if provided
        type_filter = None
        if model_type:
            try:
                type_filter = [CivitAIModelType(model_type)]
            except ValueError:
                logger.warning(f"Invalid CivitAI model type: {model_type}")
        
        # Get models from CivitAI
        result = await client.list_models(
            limit=limit,
            query=query or "stable diffusion",
            model_types=type_filter,
            sort="Highest Rated",
            period="Month",
            nsfw=False,  # Always false for setup page
        )
        
        # Transform models into setup-friendly format
        models = []
        for item in result.get("items", []):
            # Get the latest version
            versions = item.get("modelVersions", [])
            if not versions:
                continue
            
            latest_version = versions[0]
            files = latest_version.get("files", [])
            if not files:
                continue
            
            primary_file = files[0]
            size_kb = primary_file.get("sizeKB", 0)
            KB_TO_GB = 1024 * 1024  # KB to GB conversion factor
            size_gb = round(size_kb / KB_TO_GB, 2)
            
            models.append({
                "id": item.get("id"),
                "name": item.get("name"),
                "type": item.get("type"),
                "description": (item.get("description", "")[:150] + "...") if item.get("description") else "No description",
                "version_id": latest_version.get("id"),
                "version_name": latest_version.get("name"),
                "size_gb": size_gb,
                "creator": item.get("creator", {}).get("username", "Unknown"),
                "stats": {
                    "downloads": item.get("stats", {}).get("downloadCount", 0),
                    "rating": item.get("stats", {}).get("rating", 0),
                    "favorites": item.get("stats", {}).get("favoriteCount", 0),
                },
                "base_model": latest_version.get("baseModel", "Unknown"),
                "download_url": f"/api/v1/civitai/download",
                "nsfw": item.get("nsfw", False),
            })
        
        return {
            "success": True,
            "models": models,
            "source": "civitai",
            "total": len(models),
        }
        
    except Exception as e:
        logger.error(f"Failed to browse CivitAI models: {str(e)}")
        return {
            "success": False,
            "models": [],
            "error": str(e),
        }


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


class ModelUninstallRequest(BaseModel):
    """Request to uninstall an AI model."""

    model_name: str = Field(..., description="Model name")
    model_category: str = Field("text", description="Model category (text, image, voice)")


@router.post("/ai-models/uninstall")
async def uninstall_model(request: ModelUninstallRequest) -> Dict[str, Any]:
    """
    Uninstall an AI model.

    Removes the model files from disk and updates the configuration.
    """
    try:
        import shutil
        from pathlib import Path

        models_dir = Path("./models")
        model_path = models_dir / request.model_category / request.model_name

        if not model_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Model {request.model_name} not found in {request.model_category}"
            )

        # Remove model directory
        shutil.rmtree(model_path)
        logger.info(f"Removed model directory: {model_path}")

        # Update model configuration
        config_path = models_dir / "model_config.json"
        if config_path.exists():
            import json
            with open(config_path, "r") as f:
                config = json.load(f)
            
            # Remove from enabled models if present
            if "enabled_models" in config and request.model_name in config["enabled_models"]:
                del config["enabled_models"][request.model_name]
            
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)

        return {
            "success": True,
            "message": f"Model {request.model_name} uninstalled successfully",
            "model_name": request.model_name,
            "model_category": request.model_category,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to uninstall model: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to uninstall model: {str(e)}",
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


@router.get("/inference-engines/status")
async def get_inference_engines_status() -> Dict[str, Any]:
    """
    Get status of all inference engines (vLLM, llama.cpp, ComfyUI, Automatic1111, etc.).
    
    Returns information about which inference engines are installed and their versions.
    This helps users verify that engines installed via setup scripts are properly detected.
    """
    try:
        import sys
        from pathlib import Path
        
        # Add backend utilities to path
        backend_path = Path(__file__).parent.parent.parent
        if str(backend_path) not in sys.path:
            sys.path.insert(0, str(backend_path))
        
        from utils.model_detection import get_inference_engines_status
        
        # Get project root for base_dir
        project_root = Path(__file__).parents[4]
        
        engines_status = get_inference_engines_status(base_dir=project_root)
        
        # Group engines by category
        by_category = {}
        for engine_id, engine_info in engines_status.items():
            category = engine_info.get("category", "other")
            if category not in by_category:
                by_category[category] = []
            by_category[category].append({
                "id": engine_id,
                **engine_info
            })
        
        # Count installed vs not installed
        installed_count = sum(1 for e in engines_status.values() if e.get("status") == "installed")
        total_count = len(engines_status)
        
        return {
            "success": True,
            "engines": engines_status,
            "by_category": by_category,
            "summary": {
                "installed": installed_count,
                "not_installed": total_count - installed_count,
                "total": total_count
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get inference engines status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get inference engines status: {str(e)}",
        )


class InferenceEngineInstallRequest(BaseModel):
    """Request to install an inference engine."""
    
    engine_name: str = Field(..., description="Engine name (vllm, comfyui, llama-cpp)")
    install_path: Optional[str] = Field(None, description="Optional custom installation path")


@router.post("/inference-engines/install")
async def install_inference_engine(request: InferenceEngineInstallRequest) -> Dict[str, Any]:
    """
    Install an inference engine using the appropriate installation script.
    
    Triggers installation of vLLM, ComfyUI, or other inference engines using
    the scripts in the scripts/ directory.
    """
    try:
        import subprocess
        import sys
        from pathlib import Path
        
        project_root = Path(__file__).parents[4]
        scripts_dir = project_root / "scripts"
        
        # Map engine names to installation scripts
        script_map = {
            "vllm": "install_vllm_rocm.sh",
            "vllm-rocm": "install_vllm_rocm.sh",
            "comfyui": "install_comfyui_rocm.sh",
            "comfyui-rocm": "install_comfyui_rocm.sh",
        }
        
        engine_name = request.engine_name.lower()
        
        if engine_name not in script_map:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported engine: {engine_name}. Supported engines: {', '.join(script_map.keys())}"
            )
        
        script_name = script_map[engine_name]
        script_path = scripts_dir / script_name
        
        if not script_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Installation script not found: {script_path}"
            )
        
        logger.info(f"Starting installation of {engine_name}...")
        
        # Build command
        cmd = ["bash", str(script_path)]
        if request.install_path:
            cmd.append(request.install_path)
        
        # Run installation script
        # This can take a long time (10-30 minutes for compilation)
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800,  # 30 minutes timeout
            cwd=str(project_root)
        )
        
        response = {
            "success": result.returncode == 0,
            "message": (
                f"Successfully installed {engine_name}"
                if result.returncode == 0
                else f"Installation of {engine_name} failed or partially completed"
            ),
            "engine": engine_name,
            "script": script_name,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "return_code": result.returncode,
        }
        
        if result.returncode == 0:
            logger.info(f"Inference engine {engine_name} installed successfully")
        else:
            logger.warning(f"Inference engine {engine_name} installation failed with code {result.returncode}")
        
        return response
        
    except subprocess.TimeoutExpired:
        raise HTTPException(
            status_code=status.HTTP_408_REQUEST_TIMEOUT,
            detail=f"Installation of {request.engine_name} timed out (>30 minutes). Check logs for status."
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to install inference engine: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to install inference engine: {str(e)}",
        )


@router.get("/dependencies/health")
async def check_dependencies_health() -> Dict[str, Any]:
    """
    Comprehensive health check for all AI model dependencies.
    
    Validates that all required dependencies are installed and functional.
    This helps identify missing or incompatible packages before attempting
    content generation.
    
    Returns:
        Dict with health status for each dependency category
    """
    try:
        import sys
        from pathlib import Path
        
        health_status = {
            "overall_status": "healthy",
            "dependencies": {},
            "inference_engines": {},
            "ai_models": {},
            "issues": [],
            "warnings": [],
        }
        
        # Check Python packages
        required_packages = {
            "core": [
                "fastapi",
                "sqlalchemy",
                "pydantic",
                "httpx",
            ],
            "ml": [
                "torch",
                "torchvision", 
                "diffusers",
                "transformers",
                "accelerate",
                "huggingface_hub",
            ],
            "optional": [
                "vllm",
                "llama_cpp",
                "opencv-python",
            ],
        }
        
        for category, packages in required_packages.items():
            for package in packages:
                try:
                    mod = __import__(package.replace("-", "_"))
                    version = getattr(mod, "__version__", "unknown")
                    health_status["dependencies"][package] = {
                        "status": "installed",
                        "version": version,
                        "category": category,
                    }
                except ImportError as e:
                    health_status["dependencies"][package] = {
                        "status": "missing",
                        "category": category,
                        "error": str(e),
                    }
                    if category in ["core", "ml"]:
                        health_status["issues"].append(
                            f"Required package '{package}' is not installed"
                        )
                        health_status["overall_status"] = "unhealthy"
                    else:
                        health_status["warnings"].append(
                            f"Optional package '{package}' is not installed"
                        )
        
        # Check inference engines
        from backend.utils.model_detection import get_inference_engines_status
        
        project_root = Path(__file__).parents[4]
        engines_status = get_inference_engines_status(base_dir=project_root)
        
        for engine_id, engine_info in engines_status.items():
            health_status["inference_engines"][engine_id] = {
                "name": engine_info.get("name"),
                "status": engine_info.get("status"),
                "category": engine_info.get("category"),
            }
            if engine_info.get("version"):
                health_status["inference_engines"][engine_id]["version"] = engine_info["version"]
            if engine_info.get("path"):
                health_status["inference_engines"][engine_id]["path"] = engine_info["path"]
        
        # Check AI models availability
        try:
            from backend.services.ai_models import ai_models
            
            if ai_models.models_loaded:
                for category in ["text", "image", "voice", "video"]:
                    category_models = ai_models.available_models.get(category, [])
                    loaded_count = len([m for m in category_models if m.get("loaded", False)])
                    total_count = len(category_models)
                    
                    health_status["ai_models"][category] = {
                        "loaded": loaded_count,
                        "total": total_count,
                        "status": "ready" if loaded_count > 0 else "no_models",
                    }
                    
                    if total_count > 0 and loaded_count == 0:
                        health_status["warnings"].append(
                            f"No {category} models are loaded despite being available"
                        )
            else:
                health_status["ai_models"]["status"] = "not_initialized"
                health_status["warnings"].append(
                    "AI models manager not initialized"
                )
                
        except Exception as e:
            health_status["ai_models"]["status"] = "error"
            health_status["ai_models"]["error"] = str(e)
            health_status["issues"].append(f"AI models check failed: {str(e)}")
            health_status["overall_status"] = "degraded"
        
        # Set overall status based on issues
        if health_status["issues"]:
            if health_status["overall_status"] == "healthy":
                health_status["overall_status"] = "degraded"
        
        logger.info(
            f"Dependencies health check completed: {health_status['overall_status']}"
        )
        
        return health_status
        
    except Exception as e:
        logger.error(f"Failed to check dependencies health: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to check dependencies health: {str(e)}",
        )


@router.post("/ai-models/warm-up")
async def warm_up_models() -> Dict[str, Any]:
    """
    Warm up AI models for faster first request.
    
    Loads and initializes AI models if not already loaded, reducing
    latency on the first content generation request. This is useful
    after application startup or when models have been unloaded.
    
    Returns:
        Dict with warm-up results and timing information
    """
    try:
        import time
        from backend.services.ai_models import ai_models
        
        start_time = time.time()
        
        # Check if already loaded
        if ai_models.models_loaded:
            return {
                "status": "already_warm",
                "message": "AI models are already initialized",
                "models_loaded": True,
                "elapsed_time_seconds": 0,
            }
        
        # Initialize models
        logger.info("Starting AI models warm-up...")
        await ai_models.initialize_models()
        
        elapsed_time = time.time() - start_time
        
        # Count loaded models
        loaded_counts = {}
        for category in ["text", "image", "voice", "video"]:
            category_models = ai_models.available_models.get(category, [])
            loaded_count = len([m for m in category_models if m.get("loaded", False)])
            loaded_counts[category] = loaded_count
        
        total_loaded = sum(loaded_counts.values())
        
        logger.info(
            f"AI models warm-up completed in {elapsed_time:.2f}s. "
            f"Loaded {total_loaded} models."
        )
        
        return {
            "status": "success",
            "message": f"AI models warmed up successfully in {elapsed_time:.2f}s",
            "models_loaded": True,
            "elapsed_time_seconds": round(elapsed_time, 2),
            "loaded_counts": loaded_counts,
            "total_loaded": total_loaded,
        }
        
    except Exception as e:
        logger.error(f"Failed to warm up AI models: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to warm up AI models: {str(e)}",
        )


@router.get("/ai-models/telemetry")
async def get_model_telemetry() -> Dict[str, Any]:
    """
    Get telemetry data for AI model usage.
    
    Tracks which models are actually being used in production,
    helping identify unused models and optimization opportunities.
    
    Returns:
        Dict with usage statistics for each model
    """
    try:
        from backend.services.ai_models import ai_models
        from sqlalchemy.ext.asyncio import AsyncSession
        from backend.database.connection import get_db_session
        from fastapi import Depends
        
        telemetry = {
            "models": {},
            "summary": {
                "total_models": 0,
                "loaded_models": 0,
                "used_models": 0,
                "unused_models": 0,
            },
            "recommendations": [],
        }
        
        # Get model availability from AIModelManager
        for category in ["text", "image", "voice", "video"]:
            category_models = ai_models.available_models.get(category, [])
            
            for model in category_models:
                model_name = model.get("name")
                is_loaded = model.get("loaded", False)
                
                telemetry["models"][model_name] = {
                    "category": category,
                    "provider": model.get("provider", "unknown"),
                    "loaded": is_loaded,
                    "can_load": model.get("can_load", False),
                    "size_gb": model.get("size_gb", 0),
                    # Note: Usage tracking would require database queries
                    # This is a placeholder for the telemetry structure
                    "usage_count": 0,  # Placeholder
                    "last_used": None,  # Placeholder
                }
                
                telemetry["summary"]["total_models"] += 1
                if is_loaded:
                    telemetry["summary"]["loaded_models"] += 1
        
        # Generate recommendations based on telemetry
        for model_name, model_data in telemetry["models"].items():
            if model_data["loaded"] and model_data["usage_count"] == 0:
                telemetry["recommendations"].append({
                    "model": model_name,
                    "recommendation": "Consider unloading this model to free up resources",
                    "reason": "Model is loaded but has not been used",
                })
            elif not model_data["loaded"] and model_data["usage_count"] > 0:
                telemetry["recommendations"].append({
                    "model": model_name,
                    "recommendation": "Consider loading this model for better performance",
                    "reason": "Model is being used but requires loading on each request",
                })
        
        telemetry["summary"]["used_models"] = len([
            m for m in telemetry["models"].values() 
            if m["usage_count"] > 0
        ])
        telemetry["summary"]["unused_models"] = (
            telemetry["summary"]["total_models"] - 
            telemetry["summary"]["used_models"]
        )
        
        logger.info("Model telemetry retrieved successfully")
        
        return telemetry
        
    except Exception as e:
        logger.error(f"Failed to get model telemetry: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get model telemetry: {str(e)}",
        )
