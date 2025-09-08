"""
Configuration Package

Provides centralized configuration management for the Gator platform.
"""

from .settings import get_settings, Settings
from .logging import setup_logging, get_logger

__all__ = ["get_settings", "Settings", "setup_logging", "get_logger"]