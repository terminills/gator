"""
Standardized Error Response Models

Provides consistent error response formats across all API endpoints
for improved developer experience and error handling.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from backend.config.logging import get_logger, correlation_id, request_id

logger = get_logger(__name__)


class ErrorDetail(BaseModel):
    """Individual error detail for validation errors."""

    field: str = Field(..., description="Field that caused the error")
    message: str = Field(..., description="Error message")
    code: Optional[str] = Field(None, description="Error code")


class ErrorResponse(BaseModel):
    """
    Standardized error response format.

    Provides consistent error responses across all API endpoints
    with support for tracing and debugging.
    """

    error: str = Field(..., description="Error type/category")
    code: str = Field(..., description="Machine-readable error code")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[List[ErrorDetail]] = Field(
        None, description="Detailed error information for validation errors"
    )
    trace_id: Optional[str] = Field(
        None, description="Request trace ID for debugging"
    )
    correlation_id: Optional[str] = Field(
        None, description="Correlation ID for distributed tracing"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Error timestamp",
    )
    path: Optional[str] = Field(None, description="Request path that caused the error")

    model_config = {
        "json_schema_extra": {
            "example": {
                "error": "ValidationError",
                "code": "VALIDATION_FAILED",
                "message": "Request validation failed",
                "details": [
                    {
                        "field": "name",
                        "message": "Field is required",
                        "code": "required",
                    }
                ],
                "trace_id": "abc123",
                "correlation_id": "def456",
                "timestamp": "2024-11-30T12:00:00Z",
                "path": "/api/v1/personas/",
            }
        }
    }


# Standard error codes
class ErrorCodes:
    """Standard error codes for consistent error handling."""

    # Authentication errors (AUTH_*)
    AUTH_REQUIRED = "AUTH_REQUIRED"
    AUTH_INVALID_TOKEN = "AUTH_INVALID_TOKEN"
    AUTH_EXPIRED_TOKEN = "AUTH_EXPIRED_TOKEN"
    AUTH_INSUFFICIENT_PERMISSIONS = "AUTH_INSUFFICIENT_PERMISSIONS"

    # Validation errors (VALIDATION_*)
    VALIDATION_FAILED = "VALIDATION_FAILED"
    VALIDATION_INVALID_FORMAT = "VALIDATION_INVALID_FORMAT"
    VALIDATION_MISSING_FIELD = "VALIDATION_MISSING_FIELD"

    # Resource errors (RESOURCE_*)
    RESOURCE_NOT_FOUND = "RESOURCE_NOT_FOUND"
    RESOURCE_ALREADY_EXISTS = "RESOURCE_ALREADY_EXISTS"
    RESOURCE_CONFLICT = "RESOURCE_CONFLICT"

    # External service errors (SERVICE_*)
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
    SERVICE_TIMEOUT = "SERVICE_TIMEOUT"
    SERVICE_ERROR = "SERVICE_ERROR"

    # Rate limiting errors (RATE_*)
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"

    # Internal errors (INTERNAL_*)
    INTERNAL_ERROR = "INTERNAL_ERROR"
    INTERNAL_DATABASE_ERROR = "INTERNAL_DATABASE_ERROR"

    # ACD errors (ACD_*)
    ACD_CONTEXT_NOT_FOUND = "ACD_CONTEXT_NOT_FOUND"
    ACD_INVALID_STATE = "ACD_INVALID_STATE"
    ACD_PROCESSING_FAILED = "ACD_PROCESSING_FAILED"


class GatorHTTPException(HTTPException):
    """
    Enhanced HTTP exception with standardized error response.

    Use this instead of FastAPI's HTTPException for consistent
    error responses across the API.
    """

    def __init__(
        self,
        status_code: int,
        error: str,
        code: str,
        message: str,
        details: Optional[List[Dict[str, Any]]] = None,
    ):
        self.error = error
        self.code = code
        self.message = message
        self.details = details
        super().__init__(status_code=status_code, detail=message)


def create_error_response(
    status_code: int,
    error: str,
    code: str,
    message: str,
    details: Optional[List[ErrorDetail]] = None,
    path: Optional[str] = None,
) -> JSONResponse:
    """
    Create a standardized error response.

    Args:
        status_code: HTTP status code
        error: Error type/category
        code: Machine-readable error code
        message: Human-readable message
        details: Optional detailed error information
        path: Request path

    Returns:
        JSONResponse with standardized error format
    """
    response = ErrorResponse(
        error=error,
        code=code,
        message=message,
        details=details,
        trace_id=request_id.get(""),
        correlation_id=correlation_id.get(""),
        path=path,
    )

    return JSONResponse(
        status_code=status_code,
        content=response.model_dump(mode="json"),
    )


# Exception handlers to register with FastAPI
async def gator_exception_handler(
    request: Request, exc: GatorHTTPException
) -> JSONResponse:
    """Handler for GatorHTTPException."""
    details = None
    if exc.details:
        details = [ErrorDetail(**d) for d in exc.details]

    return create_error_response(
        status_code=exc.status_code,
        error=exc.error,
        code=exc.code,
        message=exc.message,
        details=details,
        path=request.url.path,
    )


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Handler for standard HTTPException."""
    # Map status codes to error types
    error_map = {
        400: ("BadRequest", ErrorCodes.VALIDATION_FAILED),
        401: ("Unauthorized", ErrorCodes.AUTH_REQUIRED),
        403: ("Forbidden", ErrorCodes.AUTH_INSUFFICIENT_PERMISSIONS),
        404: ("NotFound", ErrorCodes.RESOURCE_NOT_FOUND),
        409: ("Conflict", ErrorCodes.RESOURCE_CONFLICT),
        422: ("ValidationError", ErrorCodes.VALIDATION_FAILED),
        429: ("TooManyRequests", ErrorCodes.RATE_LIMIT_EXCEEDED),
        500: ("InternalError", ErrorCodes.INTERNAL_ERROR),
        502: ("BadGateway", ErrorCodes.SERVICE_ERROR),
        503: ("ServiceUnavailable", ErrorCodes.SERVICE_UNAVAILABLE),
        504: ("GatewayTimeout", ErrorCodes.SERVICE_TIMEOUT),
    }

    error, code = error_map.get(
        exc.status_code, ("Error", f"HTTP_{exc.status_code}")
    )

    return create_error_response(
        status_code=exc.status_code,
        error=error,
        code=code,
        message=str(exc.detail),
        path=request.url.path,
    )


async def validation_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    """Handler for Pydantic validation errors."""
    from pydantic import ValidationError

    details = []
    if isinstance(exc, ValidationError):
        for error in exc.errors():
            field = ".".join(str(loc) for loc in error.get("loc", []))
            details.append(
                ErrorDetail(
                    field=field,
                    message=error.get("msg", "Validation failed"),
                    code=error.get("type", "validation_error"),
                )
            )

    return create_error_response(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        error="ValidationError",
        code=ErrorCodes.VALIDATION_FAILED,
        message="Request validation failed",
        details=details,
        path=request.url.path,
    )


async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handler for unexpected exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)

    return create_error_response(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        error="InternalError",
        code=ErrorCodes.INTERNAL_ERROR,
        message="An unexpected error occurred",
        path=request.url.path,
    )


# Convenience functions for raising common errors
def raise_not_found(resource: str, resource_id: Any) -> None:
    """Raise a not found error."""
    raise GatorHTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        error="NotFound",
        code=ErrorCodes.RESOURCE_NOT_FOUND,
        message=f"{resource} with ID {resource_id} not found",
    )


def raise_validation_error(message: str, details: Optional[List[Dict]] = None) -> None:
    """Raise a validation error."""
    raise GatorHTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        error="ValidationError",
        code=ErrorCodes.VALIDATION_FAILED,
        message=message,
        details=details,
    )


def raise_auth_error(message: str = "Authentication required") -> None:
    """Raise an authentication error."""
    raise GatorHTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        error="Unauthorized",
        code=ErrorCodes.AUTH_REQUIRED,
        message=message,
    )


def raise_permission_error(message: str = "Insufficient permissions") -> None:
    """Raise a permission error."""
    raise GatorHTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        error="Forbidden",
        code=ErrorCodes.AUTH_INSUFFICIENT_PERMISSIONS,
        message=message,
    )


def raise_conflict_error(message: str) -> None:
    """Raise a conflict error."""
    raise GatorHTTPException(
        status_code=status.HTTP_409_CONFLICT,
        error="Conflict",
        code=ErrorCodes.RESOURCE_CONFLICT,
        message=message,
    )


def raise_service_error(service: str, message: str) -> None:
    """Raise an external service error."""
    raise GatorHTTPException(
        status_code=status.HTTP_502_BAD_GATEWAY,
        error="ServiceError",
        code=ErrorCodes.SERVICE_ERROR,
        message=f"{service}: {message}",
    )
