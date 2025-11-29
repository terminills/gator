"""
Configuration Package

Provides centralized configuration management for the Gator platform.
"""

from .logging import get_logger, setup_logging
from .settings import Settings, get_settings

__all__ = ["get_settings", "Settings", "setup_logging", "get_logger"]
