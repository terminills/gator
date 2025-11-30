"""
Logging Configuration

Enhanced logging setup with correlation IDs and structured output
for production observability.
"""

import logging
import sys
from contextvars import ContextVar
from typing import Optional
import uuid

# Context variables for request tracking
request_id: ContextVar[str] = ContextVar("request_id", default="")
correlation_id: ContextVar[str] = ContextVar("correlation_id", default="")
user_id: ContextVar[str] = ContextVar("user_id", default="")


class CorrelationFilter(logging.Filter):
    """
    Logging filter that adds correlation context to log records.

    Adds request_id, correlation_id, and user_id to every log record
    for distributed tracing and debugging.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """Add context variables to log record."""
        record.request_id = request_id.get("")
        record.correlation_id = correlation_id.get("")
        record.user_id = user_id.get("")
        return True


class StructuredFormatter(logging.Formatter):
    """
    Structured log formatter with JSON-like output for production.

    Format: timestamp - name - level - [request_id] [correlation_id] - message
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with correlation context."""
        # Build context string
        context_parts = []
        if hasattr(record, "request_id") and record.request_id:
            context_parts.append(f"req={record.request_id[:8]}")
        if hasattr(record, "correlation_id") and record.correlation_id:
            context_parts.append(f"cor={record.correlation_id[:8]}")
        if hasattr(record, "user_id") and record.user_id:
            context_parts.append(f"usr={record.user_id[:8]}")

        context_str = f"[{' '.join(context_parts)}] " if context_parts else ""

        # Format the message
        record.context = context_str
        return super().format(record)


def setup_logging(log_level: Optional[str] = None) -> None:
    """
    Configure enhanced logging for the application.

    Args:
        log_level: Override log level (default: INFO)
    """
    level = getattr(logging, (log_level or "INFO").upper(), logging.INFO)

    # Create formatter with context
    formatter = StructuredFormatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(context)s%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Create handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    handler.addFilter(CorrelationFilter())

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    for existing_handler in root_logger.handlers[:]:
        root_logger.removeHandler(existing_handler)

    root_logger.addHandler(handler)

    # Reduce noise from third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def generate_request_id() -> str:
    """Generate a unique request ID."""
    return str(uuid.uuid4())


def generate_correlation_id() -> str:
    """Generate a unique correlation ID for tracing across services."""
    return str(uuid.uuid4())


def set_request_context(
    req_id: Optional[str] = None,
    corr_id: Optional[str] = None,
    usr_id: Optional[str] = None,
) -> None:
    """
    Set logging context for the current request.

    Args:
        req_id: Request ID (generated if not provided)
        corr_id: Correlation ID (generated if not provided)
        usr_id: User ID (optional)
    """
    request_id.set(req_id or generate_request_id())
    correlation_id.set(corr_id or generate_correlation_id())
    if usr_id:
        user_id.set(usr_id)


def clear_request_context() -> None:
    """Clear logging context at end of request."""
    request_id.set("")
    correlation_id.set("")
    user_id.set("")

