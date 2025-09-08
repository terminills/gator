"""
Logging Configuration

Structured logging setup using structlog for better observability.
"""

import logging
import sys
from typing import Any

import structlog
from structlog.stdlib import filter_by_level
from structlog.dev import ConsoleRenderer

from backend.config.settings import get_settings


def setup_logging() -> None:
    """
    Configure structured logging for the application.
    
    Sets up structlog with appropriate processors and formatters
    based on the environment settings.
    """
    settings = get_settings()
    
    # Configure timestamping
    timestamper = structlog.processors.TimeStamper(fmt="ISO")
    
    # Shared processors
    shared_processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        timestamper,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]
    
    if settings.debug:
        # Development: pretty console output
        processors = shared_processors + [
            ConsoleRenderer(colors=True)
        ]
        formatter = None
    else:
        # Production: JSON formatting
        processors = shared_processors + [
            structlog.processors.JSONRenderer()
        ]
        formatter = None
    
    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Configure standard library logging
    handler = logging.StreamHandler(sys.stdout)
    if formatter:
        handler.setFormatter(formatter)
    
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(getattr(logging, settings.log_level.upper()))
    
    # Set specific logger levels
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(
        logging.INFO if settings.debug else logging.WARNING
    )


def get_logger(name: str) -> structlog.BoundLogger:
    """
    Get a structured logger instance.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        structlog.BoundLogger: Configured logger instance
    """
    return structlog.get_logger(name)