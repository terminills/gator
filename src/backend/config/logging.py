"""
Logging Configuration

Simple logging setup for the application.
"""

import logging
import sys


def setup_logging() -> None:
    """
    Configure simple logging for the application.
    """
    # Simple console logging for now
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)
