"""
Common API Dependencies

Centralized dependency injection and utilities for API route handlers.
This module extracts common patterns from routes to reduce code duplication.

Usage:
    from backend.api.dependencies import (
        get_persona_service,
        get_acd_service,
        handle_service_error,
        PaginationParams,
    )

    @router.get("/items")
    async def list_items(
        pagination: PaginationParams = Depends(),
        service: MyService = Depends(get_my_service),
    ):
        return await service.list_items(pagination.skip, pagination.limit)
"""

from typing import Any, Callable, Optional, Type, TypeVar
from uuid import UUID

from fastapi import Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config.logging import get_logger
from backend.database.connection import get_db_session
from backend.exceptions import (
    ACDError,
    AuthError,
    ContentGenerationError,
    GatorError,
    PersonaError,
    RecordNotFoundError,
    ResourceNotFoundError,
    ValidationError,
)

logger = get_logger(__name__)

# Type variable for generic service creation
T = TypeVar("T")


# =============================================================================
# Pagination Dependencies
# =============================================================================


class PaginationParams(BaseModel):
    """
    Standard pagination parameters for list endpoints.

    Usage:
        @router.get("/items")
        async def list_items(pagination: PaginationParams = Depends()):
            return await service.list(pagination.skip, pagination.limit)
    """

    skip: int = Field(default=0, ge=0, description="Number of records to skip (offset)")
    limit: int = Field(
        default=10, ge=1, le=100, description="Maximum records to return"
    )

    @property
    def offset(self) -> int:
        """Alias for skip, for SQLAlchemy compatibility."""
        return self.skip


class CursorPaginationParams(BaseModel):
    """
    Cursor-based pagination for large datasets.

    More efficient than offset-based pagination for large tables.
    """

    cursor: Optional[str] = Field(
        default=None, description="Cursor for next page (base64 encoded)"
    )
    limit: int = Field(
        default=10, ge=1, le=100, description="Maximum records to return"
    )


class TimeWindowParams(BaseModel):
    """
    Time window parameters for analytics and historical queries.
    """

    hours: int = Field(
        default=24, ge=1, le=8760, description="Time window in hours (max 1 year)"
    )


# =============================================================================
# Service Factory Dependencies
# =============================================================================


def create_service_dependency(
    service_class: Type[T],
) -> Callable[[AsyncSession], T]:
    """
    Factory function to create service dependency injection functions.

    Args:
        service_class: The service class to instantiate

    Returns:
        A dependency function that creates service instances

    Usage:
        get_my_service = create_service_dependency(MyService)

        @router.get("/items")
        async def list_items(service: MyService = Depends(get_my_service)):
            pass
    """

    def get_service(db: AsyncSession = Depends(get_db_session)) -> T:
        return service_class(db)

    return get_service


# =============================================================================
# Pre-built Service Dependencies
# =============================================================================

# Import services here to avoid circular imports
# These will be lazily imported when first accessed


def get_persona_service():
    """Get PersonaService dependency."""
    from backend.services.persona_service import PersonaService

    def _get_service(db: AsyncSession = Depends(get_db_session)) -> PersonaService:
        return PersonaService(db)

    return _get_service


def get_acd_service():
    """Get ACDService dependency."""
    from backend.services.acd_service import ACDService

    def _get_service(db: AsyncSession = Depends(get_db_session)) -> ACDService:
        return ACDService(db)

    return _get_service


def get_settings_service():
    """Get SettingsService dependency."""
    from backend.services.settings_service import SettingsService

    def _get_service(db: AsyncSession = Depends(get_db_session)) -> SettingsService:
        return SettingsService(db)

    return _get_service


def get_user_service():
    """Get UserService dependency."""
    from backend.services.user_service import UserService

    def _get_service(db: AsyncSession = Depends(get_db_session)) -> UserService:
        return UserService(db)

    return _get_service


def get_scheduled_post_service():
    """Get ScheduledPostService dependency."""
    from backend.services.scheduled_post_service import ScheduledPostService

    def _get_service(
        db: AsyncSession = Depends(get_db_session),
    ) -> ScheduledPostService:
        return ScheduledPostService(db)

    return _get_service


def get_hil_rating_service():
    """Get HILRatingService dependency."""
    from backend.services.hil_rating_service import HILRatingService

    def _get_service(db: AsyncSession = Depends(get_db_session)) -> HILRatingService:
        return HILRatingService(db)

    return _get_service


def get_content_generation_service():
    """Get ContentGenerationService dependency."""
    from backend.services.content_generation_service import ContentGenerationService

    def _get_service(
        db: AsyncSession = Depends(get_db_session),
    ) -> ContentGenerationService:
        return ContentGenerationService(db)

    return _get_service


def get_rss_ingestion_service():
    """Get RSSIngestionService dependency."""
    from backend.services.rss_ingestion_service import RSSIngestionService

    def _get_service(
        db: AsyncSession = Depends(get_db_session),
    ) -> RSSIngestionService:
        return RSSIngestionService(db)

    return _get_service


# =============================================================================
# Error Handling Utilities
# =============================================================================


def handle_service_error(error: Exception) -> HTTPException:
    """
    Convert service-layer exceptions to appropriate HTTP exceptions.

    Maps GatorError subclasses to appropriate HTTP status codes.

    Args:
        error: The exception to convert

    Returns:
        HTTPException with appropriate status code and detail

    Usage:
        try:
            result = await service.do_something()
        except Exception as e:
            raise handle_service_error(e)
    """
    # Map exception types to HTTP status codes
    error_mapping = {
        # 400 Bad Request
        ValidationError: status.HTTP_400_BAD_REQUEST,
        PersonaError: status.HTTP_400_BAD_REQUEST,
        # 401 Unauthorized
        AuthError: status.HTTP_401_UNAUTHORIZED,
        # 404 Not Found
        RecordNotFoundError: status.HTTP_404_NOT_FOUND,
        ResourceNotFoundError: status.HTTP_404_NOT_FOUND,
        # 500 Internal Server Error (default for content generation issues)
        ContentGenerationError: status.HTTP_500_INTERNAL_SERVER_ERROR,
        ACDError: status.HTTP_500_INTERNAL_SERVER_ERROR,
        GatorError: status.HTTP_500_INTERNAL_SERVER_ERROR,
    }

    # Find the most specific exception type
    for exc_type, status_code in error_mapping.items():
        if isinstance(error, exc_type):
            if isinstance(error, GatorError):
                return HTTPException(
                    status_code=status_code,
                    detail=error.to_dict(),
                )
            return HTTPException(
                status_code=status_code,
                detail=str(error),
            )

    # Default to 500 for unknown exceptions
    logger.error(f"Unhandled exception: {type(error).__name__}: {error}")
    return HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Internal server error",
    )


def raise_not_found(
    resource_type: str,
    resource_id: Optional[str] = None,
) -> HTTPException:
    """
    Create a standardized 404 Not Found response.

    Args:
        resource_type: Type of resource (e.g., "Persona", "User")
        resource_id: Optional ID of the resource

    Returns:
        HTTPException with 404 status

    Usage:
        if not persona:
            raise raise_not_found("Persona", str(persona_id))
    """
    detail = f"{resource_type} not found"
    if resource_id:
        detail = f"{resource_type} with ID '{resource_id}' not found"

    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=detail,
    )


def raise_bad_request(message: str, details: Optional[dict] = None) -> HTTPException:
    """
    Create a standardized 400 Bad Request response.

    Args:
        message: Error message
        details: Optional additional details

    Returns:
        HTTPException with 400 status
    """
    detail: Any = message
    if details:
        detail = {"message": message, "details": details}

    return HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=detail,
    )


# =============================================================================
# UUID Validation
# =============================================================================


def validate_uuid(value: str, param_name: str = "id") -> UUID:
    """
    Validate and convert a string to UUID.

    Args:
        value: String to convert
        param_name: Parameter name for error messages

    Returns:
        Validated UUID

    Raises:
        HTTPException: If validation fails
    """
    try:
        return UUID(value)
    except (ValueError, TypeError):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid UUID format for {param_name}: '{value}'",
        )


# =============================================================================
# Common Query Parameters
# =============================================================================


def get_active_only_param(
    active_only: bool = Query(True, description="Return only active records"),
) -> bool:
    """Dependency for active_only filter."""
    return active_only


def get_domain_param(
    domain: Optional[str] = Query(None, description="Filter by domain"),
) -> Optional[str]:
    """Dependency for domain filter."""
    return domain


def get_time_window_param(
    hours: int = Query(24, ge=1, le=8760, description="Time window in hours"),
) -> int:
    """Dependency for time window filter."""
    return hours


# =============================================================================
# Response Models
# =============================================================================


class SuccessResponse(BaseModel):
    """Standard success response."""

    success: bool = True
    message: str


class DeleteResponse(BaseModel):
    """Standard delete operation response."""

    success: bool = True
    deleted_id: str
    message: str = "Resource deleted successfully"


class CountResponse(BaseModel):
    """Response with count information."""

    count: int
    total: Optional[int] = None


class PaginatedResponse(BaseModel):
    """Generic paginated response wrapper."""

    items: list
    total: int
    skip: int
    limit: int
    has_more: bool


# =============================================================================
# Dependency Instances (Pre-configured)
# =============================================================================

# These are ready-to-use dependency instances
PersonaServiceDep = Depends(get_persona_service())
ACDServiceDep = Depends(get_acd_service())
SettingsServiceDep = Depends(get_settings_service())
UserServiceDep = Depends(get_user_service())
ScheduledPostServiceDep = Depends(get_scheduled_post_service())
HILRatingServiceDep = Depends(get_hil_rating_service())
ContentGenerationServiceDep = Depends(get_content_generation_service())
RSSIngestionServiceDep = Depends(get_rss_ingestion_service())
