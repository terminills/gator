"""
Setup Service

Manages initial system configuration and environment file setup.
"""

import os
from typing import Dict, Optional, Any
from pathlib import Path
from backend.config.logging import get_logger

logger = get_logger(__name__)


class SetupService:
    """Service for managing system setup and configuration."""

    def __init__(self, env_file_path: Optional[str] = None):
        """
        Initialize setup service.

        Args:
            env_file_path: Path to .env file (defaults to .env in project root)
        """
        if env_file_path is None:
            # Default to .env in project root
            project_root = Path(__file__).parent.parent.parent.parent
            self.env_file_path = project_root / ".env"
        else:
            self.env_file_path = Path(env_file_path)

        logger.info(f"Setup service initialized with env file: {self.env_file_path}")

    def get_current_config(self) -> Dict[str, Any]:
        """
        Get current configuration from environment file.

        Returns:
            Dictionary of current configuration values (sensitive values masked)
        """
        if not self.env_file_path.exists():
            logger.warning("Environment file does not exist")
            return {}

        config = {}
        try:
            with open(self.env_file_path, "r") as f:
                for line in f:
                    line = line.strip()
                    # Skip comments and empty lines
                    if not line or line.startswith("#"):
                        continue

                    # Parse key=value pairs
                    if "=" in line:
                        key, value = line.split("=", 1)
                        key = key.strip()
                        value = value.strip()

                        # Mask sensitive values
                        if any(
                            sensitive in key.upper()
                            for sensitive in ["KEY", "SECRET", "PASSWORD", "TOKEN"]
                        ):
                            if value and value != "" and not value.startswith("your_"):
                                config[key] = "***CONFIGURED***"
                            else:
                                config[key] = value
                        else:
                            config[key] = value
        except Exception as e:
            logger.error(f"Failed to read environment file: {e}")
            raise

        return config

    def update_config(self, config: Dict[str, str]) -> bool:
        """
        Update environment configuration file.

        Args:
            config: Dictionary of configuration key-value pairs

        Returns:
            True if successful, False otherwise
        """
        try:
            # Read existing file or create template structure
            existing_lines = []
            existing_keys = set()

            if self.env_file_path.exists():
                with open(self.env_file_path, "r") as f:
                    existing_lines = f.readlines()
                    for line in existing_lines:
                        if "=" in line and not line.strip().startswith("#"):
                            key = line.split("=", 1)[0].strip()
                            existing_keys.add(key)
            else:
                # Create from template
                logger.info("Creating new environment file from template")
                template_path = self.env_file_path.parent / ".env.template"
                if template_path.exists():
                    with open(template_path, "r") as f:
                        existing_lines = f.readlines()

            # Update values
            updated_lines = []
            for line in existing_lines:
                line_stripped = line.strip()

                # Keep comments and empty lines
                if not line_stripped or line_stripped.startswith("#"):
                    updated_lines.append(line)
                    continue

                # Update existing keys
                if "=" in line:
                    key = line.split("=", 1)[0].strip()
                    if key in config:
                        # Update with new value
                        updated_lines.append(f"{key}={config[key]}\n")
                        existing_keys.add(key)
                    else:
                        # Keep existing line
                        updated_lines.append(line)

            # Add new keys that weren't in the file
            for key, value in config.items():
                if key not in existing_keys:
                    updated_lines.append(f"{key}={value}\n")

            # Write updated configuration
            with open(self.env_file_path, "w") as f:
                f.writelines(updated_lines)

            logger.info(f"Updated {len(config)} configuration values")
            return True

        except Exception as e:
            logger.error(f"Failed to update environment file: {e}")
            return False

    def get_setup_status(self) -> Dict[str, Any]:
        """
        Get current setup status.

        Returns:
            Dictionary with setup status information
        """
        status = {
            "env_file_exists": self.env_file_path.exists(),
            "env_file_path": str(self.env_file_path),
            "configured_sections": {},
        }

        if not self.env_file_path.exists():
            return status

        # Check which sections are configured
        config = self.get_current_config()

        # Database
        status["configured_sections"]["database"] = bool(
            config.get("DATABASE_URL")
            and not config.get("DATABASE_URL", "").startswith("sqlite")
        )

        # AI Models
        status["configured_sections"]["ai_models"] = bool(
            config.get("OPENAI_API_KEY") == "***CONFIGURED***"
            or config.get("ANTHROPIC_API_KEY") == "***CONFIGURED***"
        )

        # Social Media
        status["configured_sections"]["social_media"] = bool(
            config.get("FACEBOOK_API_KEY") == "***CONFIGURED***"
            or config.get("INSTAGRAM_API_KEY") == "***CONFIGURED***"
        )

        # DNS Management
        status["configured_sections"]["dns"] = bool(
            config.get("GODADDY_API_KEY") == "***CONFIGURED***"
        )

        # Security
        status["configured_sections"]["security"] = bool(
            config.get("SECRET_KEY")
            and config.get("SECRET_KEY") != "your_super_secret_key_change_in_production"
        )

        return status

    def validate_config(self, config: Dict[str, str]) -> Dict[str, Any]:
        """
        Validate configuration values.

        Args:
            config: Configuration to validate

        Returns:
            Dictionary with validation results
        """
        errors = []
        warnings = []

        # Check for placeholder values
        for key, value in config.items():
            if value.startswith("your_") or value == "change_in_production":
                warnings.append(f"{key} appears to have a placeholder value")

        # Validate database URL format
        if "DATABASE_URL" in config:
            db_url = config["DATABASE_URL"]
            if not db_url.startswith(("postgresql://", "sqlite:///")):
                errors.append(
                    "DATABASE_URL must start with postgresql:// or sqlite:///"
                )

        # Validate email configuration
        if "SMTP_PORT" in config:
            try:
                port = int(config["SMTP_PORT"])
                if port < 1 or port > 65535:
                    errors.append("SMTP_PORT must be between 1 and 65535")
            except ValueError:
                errors.append("SMTP_PORT must be a valid integer")

        return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}


def get_setup_service(env_file_path: Optional[str] = None) -> SetupService:
    """
    Get setup service instance.

    Args:
        env_file_path: Optional path to .env file

    Returns:
        SetupService instance
    """
    return SetupService(env_file_path)
