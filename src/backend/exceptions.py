"""
Gator Exception Hierarchy

Centralized exception handling for the Gator AI Platform.
Provides a structured exception hierarchy for different error scenarios.

Usage:
    from backend.exceptions import GatorError, PersonaNotFoundError

    try:
        persona = await get_persona(id)
    except PersonaNotFoundError:
        return {"error": "Persona not found"}
"""

from typing import Any, Dict, Optional


class GatorError(Exception):
    """
    Base exception for all Gator platform errors.

    All custom exceptions should inherit from this class to provide
    consistent error handling across the application.

    Attributes:
        message: Human-readable error message
        error_code: Machine-readable error code for API responses
        details: Additional error context (optional)
    """

    error_code: str = "GATOR_ERROR"

    def __init__(
        self,
        message: str = "An error occurred",
        details: Optional[Dict[str, Any]] = None,
    ):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for API responses."""
        result = {
            "error": self.error_code,
            "message": self.message,
        }
        if self.details:
            result["details"] = self.details
        return result


# =============================================================================
# Database Exceptions
# =============================================================================


class DatabaseError(GatorError):
    """Base exception for database-related errors."""

    error_code = "DATABASE_ERROR"


class ConnectionError(DatabaseError):
    """Database connection failed."""

    error_code = "DATABASE_CONNECTION_ERROR"

    def __init__(self, message: str = "Failed to connect to database", **kwargs):
        super().__init__(message, **kwargs)


class TransactionError(DatabaseError):
    """Database transaction failed."""

    error_code = "DATABASE_TRANSACTION_ERROR"


class RecordNotFoundError(DatabaseError):
    """Requested record was not found."""

    error_code = "RECORD_NOT_FOUND"


# =============================================================================
# Persona Exceptions
# =============================================================================


class PersonaError(GatorError):
    """Base exception for persona-related errors."""

    error_code = "PERSONA_ERROR"


class PersonaNotFoundError(PersonaError):
    """Requested persona was not found."""

    error_code = "PERSONA_NOT_FOUND"

    def __init__(
        self,
        persona_id: Optional[str] = None,
        message: Optional[str] = None,
        **kwargs,
    ):
        if message is None:
            message = (
                f"Persona with ID '{persona_id}' not found"
                if persona_id
                else "Persona not found"
            )
        super().__init__(message, details={"persona_id": persona_id}, **kwargs)


class PersonaValidationError(PersonaError):
    """Persona data validation failed."""

    error_code = "PERSONA_VALIDATION_ERROR"


class PersonaCreationError(PersonaError):
    """Failed to create persona."""

    error_code = "PERSONA_CREATION_ERROR"


class PersonaUpdateError(PersonaError):
    """Failed to update persona."""

    error_code = "PERSONA_UPDATE_ERROR"


# =============================================================================
# Content Generation Exceptions
# =============================================================================


class ContentGenerationError(GatorError):
    """Base exception for content generation errors."""

    error_code = "CONTENT_GENERATION_ERROR"


class ModelNotAvailableError(ContentGenerationError):
    """Required AI model is not available or loaded."""

    error_code = "MODEL_NOT_AVAILABLE"

    def __init__(
        self,
        model_name: Optional[str] = None,
        model_type: Optional[str] = None,
        message: Optional[str] = None,
        **kwargs,
    ):
        if message is None:
            if model_name:
                message = f"Model '{model_name}' is not available"
            elif model_type:
                message = f"No {model_type} model is available"
            else:
                message = "Required AI model is not available"
        super().__init__(
            message,
            details={"model_name": model_name, "model_type": model_type},
            **kwargs,
        )


class ModelLoadError(ContentGenerationError):
    """Failed to load AI model."""

    error_code = "MODEL_LOAD_ERROR"


class PromptGenerationError(ContentGenerationError):
    """Failed to generate prompt for content generation."""

    error_code = "PROMPT_GENERATION_ERROR"


class ImageGenerationError(ContentGenerationError):
    """Failed to generate image."""

    error_code = "IMAGE_GENERATION_ERROR"


class TextGenerationError(ContentGenerationError):
    """Failed to generate text."""

    error_code = "TEXT_GENERATION_ERROR"


class VideoGenerationError(ContentGenerationError):
    """Failed to generate video."""

    error_code = "VIDEO_GENERATION_ERROR"


class VoiceGenerationError(ContentGenerationError):
    """Failed to generate voice/audio."""

    error_code = "VOICE_GENERATION_ERROR"


class ContentModerationError(ContentGenerationError):
    """Content failed moderation checks."""

    error_code = "CONTENT_MODERATION_ERROR"


# =============================================================================
# ACD (Autonomous Continuous Development) Exceptions
# =============================================================================


class ACDError(GatorError):
    """Base exception for ACD system errors."""

    error_code = "ACD_ERROR"


class ACDContextError(ACDError):
    """Error creating or managing ACD context."""

    error_code = "ACD_CONTEXT_ERROR"


class ACDReasoningError(ACDError):
    """Error during ACD reasoning/decision making."""

    error_code = "ACD_REASONING_ERROR"


class ACDOrchestrationError(ACDError):
    """Error during ACD task orchestration."""

    error_code = "ACD_ORCHESTRATION_ERROR"


# =============================================================================
# Authentication/Authorization Exceptions
# =============================================================================


class AuthError(GatorError):
    """Base exception for authentication/authorization errors."""

    error_code = "AUTH_ERROR"


class AuthenticationError(AuthError):
    """Authentication failed (invalid credentials)."""

    error_code = "AUTHENTICATION_ERROR"


class AuthorizationError(AuthError):
    """Authorization failed (insufficient permissions)."""

    error_code = "AUTHORIZATION_ERROR"


class TokenExpiredError(AuthError):
    """Authentication token has expired."""

    error_code = "TOKEN_EXPIRED"


class InvalidTokenError(AuthError):
    """Authentication token is invalid."""

    error_code = "INVALID_TOKEN"


# =============================================================================
# External Service Exceptions
# =============================================================================


class ExternalServiceError(GatorError):
    """Base exception for external service errors."""

    error_code = "EXTERNAL_SERVICE_ERROR"


class APIConnectionError(ExternalServiceError):
    """Failed to connect to external API."""

    error_code = "API_CONNECTION_ERROR"


class APIRateLimitError(ExternalServiceError):
    """External API rate limit exceeded."""

    error_code = "API_RATE_LIMIT"


class CivitaiError(ExternalServiceError):
    """Error with Civitai API."""

    error_code = "CIVITAI_ERROR"


class HuggingFaceError(ExternalServiceError):
    """Error with HuggingFace API."""

    error_code = "HUGGINGFACE_ERROR"


class OllamaError(ExternalServiceError):
    """Error with Ollama service."""

    error_code = "OLLAMA_ERROR"


class ComfyUIError(ExternalServiceError):
    """Error with ComfyUI service."""

    error_code = "COMFYUI_ERROR"


class SocialMediaError(ExternalServiceError):
    """Error with social media integration."""

    error_code = "SOCIAL_MEDIA_ERROR"


# =============================================================================
# Configuration Exceptions
# =============================================================================


class ConfigurationError(GatorError):
    """Base exception for configuration errors."""

    error_code = "CONFIGURATION_ERROR"


class MissingConfigurationError(ConfigurationError):
    """Required configuration is missing."""

    error_code = "MISSING_CONFIGURATION"

    def __init__(
        self,
        config_key: Optional[str] = None,
        message: Optional[str] = None,
        **kwargs,
    ):
        if message is None:
            message = (
                f"Missing required configuration: {config_key}"
                if config_key
                else "Required configuration is missing"
            )
        super().__init__(message, details={"config_key": config_key}, **kwargs)


class InvalidConfigurationError(ConfigurationError):
    """Configuration value is invalid."""

    error_code = "INVALID_CONFIGURATION"


# =============================================================================
# Validation Exceptions
# =============================================================================


class ValidationError(GatorError):
    """Base exception for validation errors."""

    error_code = "VALIDATION_ERROR"


class InvalidInputError(ValidationError):
    """Input data is invalid."""

    error_code = "INVALID_INPUT"


class InvalidUUIDError(ValidationError):
    """Invalid UUID format."""

    error_code = "INVALID_UUID"

    def __init__(self, value: Optional[str] = None, **kwargs):
        message = f"Invalid UUID format: '{value}'" if value else "Invalid UUID format"
        super().__init__(message, details={"value": value}, **kwargs)


# =============================================================================
# Resource Exceptions
# =============================================================================


class ResourceError(GatorError):
    """Base exception for resource-related errors."""

    error_code = "RESOURCE_ERROR"


class ResourceNotFoundError(ResourceError):
    """Requested resource was not found."""

    error_code = "RESOURCE_NOT_FOUND"


class ResourceExistsError(ResourceError):
    """Resource already exists (duplicate)."""

    error_code = "RESOURCE_EXISTS"


class ResourceLimitError(ResourceError):
    """Resource limit exceeded."""

    error_code = "RESOURCE_LIMIT_EXCEEDED"


class StorageError(ResourceError):
    """File storage operation failed."""

    error_code = "STORAGE_ERROR"


class FileNotFoundError(ResourceError):
    """Requested file was not found."""

    error_code = "FILE_NOT_FOUND"


# =============================================================================
# GPU/Hardware Exceptions
# =============================================================================


class HardwareError(GatorError):
    """Base exception for hardware-related errors."""

    error_code = "HARDWARE_ERROR"


class GPUError(HardwareError):
    """GPU-related error."""

    error_code = "GPU_ERROR"


class GPUMemoryError(GPUError):
    """Insufficient GPU memory."""

    error_code = "GPU_MEMORY_ERROR"


class GPUNotAvailableError(GPUError):
    """No GPU available."""

    error_code = "GPU_NOT_AVAILABLE"


# =============================================================================
# Helper Functions
# =============================================================================


def handle_exception(e: Exception) -> Dict[str, Any]:
    """
    Convert any exception to a standardized error response.

    Args:
        e: The exception to handle

    Returns:
        Dictionary suitable for API error responses
    """
    if isinstance(e, GatorError):
        return e.to_dict()

    # Handle standard exceptions
    return {
        "error": "INTERNAL_ERROR",
        "message": str(e) or "An unexpected error occurred",
    }
