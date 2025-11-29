"""
Tests for Gator Exception Hierarchy

Tests the centralized exception handling system.
"""

import pytest

from backend.exceptions import (
    ACDContextError,
    ACDError,
    ACDOrchestrationError,
    ACDReasoningError,
    APIConnectionError,
    APIRateLimitError,
    AuthenticationError,
    AuthorizationError,
    AuthError,
    CivitaiError,
    ComfyUIError,
    ConfigurationError,
    ContentGenerationError,
    ContentModerationError,
    DatabaseError,
    ExternalServiceError,
    GPUError,
    GPUMemoryError,
    GPUNotAvailableError,
    GatorError,
    HardwareError,
    HuggingFaceError,
    ImageGenerationError,
    InvalidConfigurationError,
    InvalidInputError,
    InvalidTokenError,
    InvalidUUIDError,
    MissingConfigurationError,
    ModelLoadError,
    ModelNotAvailableError,
    OllamaError,
    PersonaCreationError,
    PersonaError,
    PersonaNotFoundError,
    PersonaUpdateError,
    PersonaValidationError,
    PromptGenerationError,
    RecordNotFoundError,
    ResourceError,
    ResourceExistsError,
    ResourceLimitError,
    ResourceNotFoundError,
    SocialMediaError,
    StorageError,
    TextGenerationError,
    TokenExpiredError,
    TransactionError,
    ValidationError,
    VideoGenerationError,
    VoiceGenerationError,
    handle_exception,
)


class TestGatorError:
    """Tests for base GatorError class."""

    def test_basic_error(self):
        """Test basic error creation."""
        error = GatorError("Something went wrong")
        assert str(error) == "Something went wrong"
        assert error.message == "Something went wrong"
        assert error.error_code == "GATOR_ERROR"
        assert error.details == {}

    def test_error_with_details(self):
        """Test error with additional details."""
        details = {"key": "value", "count": 42}
        error = GatorError("Error with details", details=details)
        assert error.details == details

    def test_to_dict(self):
        """Test error serialization."""
        error = GatorError("Test error", details={"foo": "bar"})
        result = error.to_dict()
        assert result["error"] == "GATOR_ERROR"
        assert result["message"] == "Test error"
        assert result["details"] == {"foo": "bar"}

    def test_to_dict_without_details(self):
        """Test error serialization without details."""
        error = GatorError("Test error")
        result = error.to_dict()
        assert "details" not in result


class TestPersonaExceptions:
    """Tests for persona-related exceptions."""

    def test_persona_not_found_with_id(self):
        """Test PersonaNotFoundError with persona ID."""
        error = PersonaNotFoundError(persona_id="abc-123")
        assert "abc-123" in str(error)
        assert error.error_code == "PERSONA_NOT_FOUND"
        assert error.details["persona_id"] == "abc-123"

    def test_persona_not_found_without_id(self):
        """Test PersonaNotFoundError without persona ID."""
        error = PersonaNotFoundError()
        assert "not found" in str(error).lower()

    def test_persona_not_found_custom_message(self):
        """Test PersonaNotFoundError with custom message."""
        error = PersonaNotFoundError(message="Custom message")
        assert str(error) == "Custom message"

    def test_persona_validation_error(self):
        """Test PersonaValidationError."""
        error = PersonaValidationError("Invalid persona data")
        assert error.error_code == "PERSONA_VALIDATION_ERROR"
        assert isinstance(error, PersonaError)
        assert isinstance(error, GatorError)

    def test_persona_hierarchy(self):
        """Test persona exception hierarchy."""
        error = PersonaCreationError("Failed to create")
        assert isinstance(error, PersonaError)
        assert isinstance(error, GatorError)


class TestContentGenerationExceptions:
    """Tests for content generation exceptions."""

    def test_model_not_available_with_name(self):
        """Test ModelNotAvailableError with model name."""
        error = ModelNotAvailableError(model_name="llama-3.1")
        assert "llama-3.1" in str(error)
        assert error.error_code == "MODEL_NOT_AVAILABLE"

    def test_model_not_available_with_type(self):
        """Test ModelNotAvailableError with model type."""
        error = ModelNotAvailableError(model_type="image")
        assert "image" in str(error)

    def test_model_not_available_custom_message(self):
        """Test ModelNotAvailableError with custom message."""
        error = ModelNotAvailableError(message="Custom model error")
        assert str(error) == "Custom model error"

    def test_content_generation_hierarchy(self):
        """Test content generation exception hierarchy."""
        errors = [
            ImageGenerationError("Image failed"),
            TextGenerationError("Text failed"),
            VideoGenerationError("Video failed"),
            VoiceGenerationError("Voice failed"),
            ModelLoadError("Load failed"),
            PromptGenerationError("Prompt failed"),
            ContentModerationError("Moderation failed"),
        ]
        for error in errors:
            assert isinstance(error, ContentGenerationError)
            assert isinstance(error, GatorError)


class TestACDExceptions:
    """Tests for ACD system exceptions."""

    def test_acd_context_error(self):
        """Test ACDContextError."""
        error = ACDContextError("Context creation failed")
        assert error.error_code == "ACD_CONTEXT_ERROR"
        assert isinstance(error, ACDError)

    def test_acd_reasoning_error(self):
        """Test ACDReasoningError."""
        error = ACDReasoningError("Reasoning failed")
        assert error.error_code == "ACD_REASONING_ERROR"

    def test_acd_orchestration_error(self):
        """Test ACDOrchestrationError."""
        error = ACDOrchestrationError("Orchestration failed")
        assert error.error_code == "ACD_ORCHESTRATION_ERROR"


class TestAuthExceptions:
    """Tests for authentication/authorization exceptions."""

    def test_authentication_error(self):
        """Test AuthenticationError."""
        error = AuthenticationError("Invalid credentials")
        assert error.error_code == "AUTHENTICATION_ERROR"
        assert isinstance(error, AuthError)

    def test_authorization_error(self):
        """Test AuthorizationError."""
        error = AuthorizationError("Insufficient permissions")
        assert error.error_code == "AUTHORIZATION_ERROR"

    def test_token_expired_error(self):
        """Test TokenExpiredError."""
        error = TokenExpiredError("Token has expired")
        assert error.error_code == "TOKEN_EXPIRED"

    def test_invalid_token_error(self):
        """Test InvalidTokenError."""
        error = InvalidTokenError("Token is malformed")
        assert error.error_code == "INVALID_TOKEN"


class TestExternalServiceExceptions:
    """Tests for external service exceptions."""

    def test_api_connection_error(self):
        """Test APIConnectionError."""
        error = APIConnectionError("Failed to connect")
        assert error.error_code == "API_CONNECTION_ERROR"
        assert isinstance(error, ExternalServiceError)

    def test_api_rate_limit_error(self):
        """Test APIRateLimitError."""
        error = APIRateLimitError("Rate limit exceeded")
        assert error.error_code == "API_RATE_LIMIT"

    def test_civitai_error(self):
        """Test CivitaiError."""
        error = CivitaiError("Civitai API error")
        assert error.error_code == "CIVITAI_ERROR"

    def test_huggingface_error(self):
        """Test HuggingFaceError."""
        error = HuggingFaceError("HuggingFace error")
        assert error.error_code == "HUGGINGFACE_ERROR"

    def test_ollama_error(self):
        """Test OllamaError."""
        error = OllamaError("Ollama service error")
        assert error.error_code == "OLLAMA_ERROR"

    def test_comfyui_error(self):
        """Test ComfyUIError."""
        error = ComfyUIError("ComfyUI error")
        assert error.error_code == "COMFYUI_ERROR"

    def test_social_media_error(self):
        """Test SocialMediaError."""
        error = SocialMediaError("Social media error")
        assert error.error_code == "SOCIAL_MEDIA_ERROR"


class TestConfigurationExceptions:
    """Tests for configuration exceptions."""

    def test_missing_configuration_with_key(self):
        """Test MissingConfigurationError with config key."""
        error = MissingConfigurationError(config_key="API_KEY")
        assert "API_KEY" in str(error)
        assert error.details["config_key"] == "API_KEY"

    def test_missing_configuration_custom_message(self):
        """Test MissingConfigurationError with custom message."""
        error = MissingConfigurationError(message="Custom config error")
        assert str(error) == "Custom config error"

    def test_invalid_configuration_error(self):
        """Test InvalidConfigurationError."""
        error = InvalidConfigurationError("Invalid config value")
        assert error.error_code == "INVALID_CONFIGURATION"
        assert isinstance(error, ConfigurationError)


class TestValidationExceptions:
    """Tests for validation exceptions."""

    def test_invalid_input_error(self):
        """Test InvalidInputError."""
        error = InvalidInputError("Input is invalid")
        assert error.error_code == "INVALID_INPUT"
        assert isinstance(error, ValidationError)

    def test_invalid_uuid_with_value(self):
        """Test InvalidUUIDError with value."""
        error = InvalidUUIDError(value="not-a-uuid")
        assert "not-a-uuid" in str(error)
        assert error.details["value"] == "not-a-uuid"

    def test_invalid_uuid_without_value(self):
        """Test InvalidUUIDError without value."""
        error = InvalidUUIDError()
        assert "invalid uuid" in str(error).lower()


class TestResourceExceptions:
    """Tests for resource exceptions."""

    def test_resource_not_found_error(self):
        """Test ResourceNotFoundError."""
        error = ResourceNotFoundError("Resource not found")
        assert error.error_code == "RESOURCE_NOT_FOUND"
        assert isinstance(error, ResourceError)

    def test_resource_exists_error(self):
        """Test ResourceExistsError."""
        error = ResourceExistsError("Resource already exists")
        assert error.error_code == "RESOURCE_EXISTS"

    def test_resource_limit_error(self):
        """Test ResourceLimitError."""
        error = ResourceLimitError("Limit exceeded")
        assert error.error_code == "RESOURCE_LIMIT_EXCEEDED"

    def test_storage_error(self):
        """Test StorageError."""
        error = StorageError("Storage operation failed")
        assert error.error_code == "STORAGE_ERROR"


class TestHardwareExceptions:
    """Tests for hardware-related exceptions."""

    def test_gpu_error(self):
        """Test GPUError."""
        error = GPUError("GPU error occurred")
        assert error.error_code == "GPU_ERROR"
        assert isinstance(error, HardwareError)

    def test_gpu_memory_error(self):
        """Test GPUMemoryError."""
        error = GPUMemoryError("Out of GPU memory")
        assert error.error_code == "GPU_MEMORY_ERROR"
        assert isinstance(error, GPUError)

    def test_gpu_not_available_error(self):
        """Test GPUNotAvailableError."""
        error = GPUNotAvailableError("No GPU found")
        assert error.error_code == "GPU_NOT_AVAILABLE"


class TestDatabaseExceptions:
    """Tests for database exceptions."""

    def test_database_error(self):
        """Test DatabaseError."""
        error = DatabaseError("Database error")
        assert error.error_code == "DATABASE_ERROR"

    def test_transaction_error(self):
        """Test TransactionError."""
        error = TransactionError("Transaction failed")
        assert error.error_code == "DATABASE_TRANSACTION_ERROR"
        assert isinstance(error, DatabaseError)

    def test_record_not_found_error(self):
        """Test RecordNotFoundError."""
        error = RecordNotFoundError("Record not found")
        assert error.error_code == "RECORD_NOT_FOUND"


class TestHandleException:
    """Tests for handle_exception helper function."""

    def test_handle_gator_error(self):
        """Test handling GatorError subclass."""
        error = PersonaNotFoundError(persona_id="test-id")
        result = handle_exception(error)
        assert result["error"] == "PERSONA_NOT_FOUND"
        assert "test-id" in result["message"]

    def test_handle_standard_exception(self):
        """Test handling standard Python exception."""
        error = ValueError("Invalid value")
        result = handle_exception(error)
        assert result["error"] == "INTERNAL_ERROR"
        assert result["message"] == "Invalid value"

    def test_handle_exception_empty_message(self):
        """Test handling exception with empty message."""
        error = Exception()
        result = handle_exception(error)
        assert result["error"] == "INTERNAL_ERROR"
        assert result["message"] == "An unexpected error occurred"


class TestExceptionHierarchy:
    """Tests for exception class hierarchy."""

    def test_all_exceptions_inherit_from_gator_error(self):
        """Test that all custom exceptions inherit from GatorError."""
        exceptions = [
            DatabaseError,
            PersonaError,
            ContentGenerationError,
            ACDError,
            AuthError,
            ExternalServiceError,
            ConfigurationError,
            ValidationError,
            ResourceError,
            HardwareError,
        ]
        for exc_class in exceptions:
            error = exc_class("Test")
            assert isinstance(error, GatorError)

    def test_exception_can_be_caught_by_parent(self):
        """Test that child exceptions can be caught by parent class."""
        try:
            raise PersonaNotFoundError(persona_id="123")
        except GatorError as e:
            assert e.error_code == "PERSONA_NOT_FOUND"

    def test_exception_inheritance_chain(self):
        """Test the full inheritance chain."""
        error = GPUMemoryError("OOM")
        assert isinstance(error, GPUMemoryError)
        assert isinstance(error, GPUError)
        assert isinstance(error, HardwareError)
        assert isinstance(error, GatorError)
        assert isinstance(error, Exception)
