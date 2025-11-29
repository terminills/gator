"""
Base classes for AI model handlers.

Provides abstract base classes and common interfaces for all AI model types.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional

from backend.config.logging import get_logger

logger = get_logger(__name__)


class ModelType(str, Enum):
    """Types of AI models supported by the platform."""

    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    VOICE = "voice"


class ModelSource(str, Enum):
    """Source of AI models."""

    LOCAL = "local"
    CLOUD = "cloud"
    CIVITAI = "civitai"
    HUGGINGFACE = "huggingface"


class BaseModelHandler(ABC):
    """
    Abstract base class for AI model handlers.

    All model handlers (text, image, video, voice) should inherit from this
    class and implement the required methods.
    """

    def __init__(self, model_type: ModelType):
        """
        Initialize the model handler.

        Args:
            model_type: Type of models this handler manages
        """
        self.model_type = model_type
        self.available_models: Dict[str, Dict[str, Any]] = {}
        self.loaded_models: Dict[str, Any] = {}
        self._initialized = False

    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize the model handler and discover available models.

        This method should be called before using the handler.
        """
        pass

    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> Any:
        """
        Generate content using the model.

        Args:
            prompt: Input prompt for generation
            **kwargs: Additional generation parameters

        Returns:
            Generated content (type depends on model type)
        """
        pass

    @abstractmethod
    async def get_available_models(self) -> List[Dict[str, Any]]:
        """
        Get list of available models for this handler.

        Returns:
            List of model information dictionaries
        """
        pass

    async def load_model(self, model_name: str) -> bool:
        """
        Load a specific model into memory.

        Args:
            model_name: Name of the model to load

        Returns:
            True if model was loaded successfully
        """
        if model_name in self.loaded_models:
            logger.debug(f"Model {model_name} already loaded")
            return True

        if model_name not in self.available_models:
            logger.error(f"Model {model_name} not found in available models")
            return False

        return await self._load_model_impl(model_name)

    async def _load_model_impl(self, model_name: str) -> bool:
        """
        Implementation-specific model loading.

        Override in subclasses for custom loading behavior.
        """
        return True

    async def unload_model(self, model_name: str) -> bool:
        """
        Unload a model from memory.

        Args:
            model_name: Name of the model to unload

        Returns:
            True if model was unloaded successfully
        """
        if model_name not in self.loaded_models:
            logger.debug(f"Model {model_name} not loaded")
            return True

        return await self._unload_model_impl(model_name)

    async def _unload_model_impl(self, model_name: str) -> bool:
        """
        Implementation-specific model unloading.

        Override in subclasses for custom unloading behavior.
        """
        if model_name in self.loaded_models:
            del self.loaded_models[model_name]
        return True

    def is_initialized(self) -> bool:
        """Check if the handler has been initialized."""
        return self._initialized

    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific model.

        Args:
            model_name: Name of the model

        Returns:
            Model information dict or None if not found
        """
        return self.available_models.get(model_name)

    async def close(self) -> None:
        """
        Clean up resources and unload all models.
        """
        for model_name in list(self.loaded_models.keys()):
            await self.unload_model(model_name)
        self._initialized = False


class ModelCapabilities:
    """
    Describes the capabilities of a model.

    Used for model selection and compatibility checking.
    """

    def __init__(
        self,
        supports_nsfw: bool = False,
        supports_streaming: bool = False,
        supports_batching: bool = False,
        max_tokens: Optional[int] = None,
        max_resolution: Optional[tuple] = None,
        supported_formats: Optional[List[str]] = None,
    ):
        self.supports_nsfw = supports_nsfw
        self.supports_streaming = supports_streaming
        self.supports_batching = supports_batching
        self.max_tokens = max_tokens
        self.max_resolution = max_resolution
        self.supported_formats = supported_formats or []

    def to_dict(self) -> Dict[str, Any]:
        """Convert capabilities to dictionary."""
        return {
            "supports_nsfw": self.supports_nsfw,
            "supports_streaming": self.supports_streaming,
            "supports_batching": self.supports_batching,
            "max_tokens": self.max_tokens,
            "max_resolution": self.max_resolution,
            "supported_formats": self.supported_formats,
        }
