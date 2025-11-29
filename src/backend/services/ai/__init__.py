"""
AI Services Module

Provides modular AI model handling for the Gator platform.

This module is organized into specialized handlers:
- text_models: LLM text generation (Llama, Ollama, OpenAI, Anthropic)
- image_models: Image generation (Stable Diffusion, SDXL, FLUX, DALL-E)
- voice_models: Voice synthesis (XTTS, Piper, ElevenLabs, OpenAI TTS)
- video_models: Video generation (SVD, AnimateDiff, Runway)

Supporting modules:
- base: Abstract base classes and interfaces
- gpu_manager: GPU detection and load balancing
- model_loader: Model downloading and verification
- model_cache: LRU caching for loaded models

Usage:
    from backend.services.ai import AIModelManager

    # Use the unified manager
    manager = get_ai_manager()
    await manager.initialize()

    # Generate text
    text = await manager.generate_text("Hello, world!")

    # Generate image
    image = await manager.generate_image("A beautiful sunset")

    # Or use individual handlers
    from backend.services.ai import get_text_handler, get_image_handler

    text_handler = get_text_handler()
    await text_handler.initialize()
    result = await text_handler.generate("Prompt here")
"""

from typing import Any, Dict, List, Optional

from backend.config.logging import get_logger

from .base import BaseModelHandler, ModelCapabilities, ModelSource, ModelType
from .gpu_manager import GPUInfo, GPUManager, GPUType, get_gpu_manager
from .image_models import ImageModelHandler, get_image_handler
from .model_cache import ModelCache, get_model_cache
from .model_loader import (
    disable_safety_checker,
    download_model_from_civitai,
    download_model_from_huggingface,
    filter_scheduler_config,
    get_model_size_estimate,
    is_flux_model,
    strip_ansi_codes,
    verify_model_files_exist,
)
from .text_models import TextModelHandler, get_text_handler
from .video_models import VideoModelHandler, get_video_handler
from .voice_models import VoiceModelHandler, get_voice_handler

logger = get_logger(__name__)


class AIModelManager:
    """
    Unified facade for all AI model operations.

    Provides a high-level interface for text, image, voice, and video generation
    while handling model selection, caching, and GPU management automatically.

    This is a refactored version of the monolithic ai_models.py, now split into
    specialized handlers for better maintainability and modularity.
    """

    _instance: Optional["AIModelManager"] = None

    def __new__(cls) -> "AIModelManager":
        """Singleton pattern for AI manager."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize the AI model manager."""
        if self._initialized:
            return

        self.text_handler = get_text_handler()
        self.image_handler = get_image_handler()
        self.voice_handler = get_voice_handler()
        self.video_handler = get_video_handler()
        self.gpu_manager = get_gpu_manager()
        self.model_cache = get_model_cache()

        self._initialized = False

    async def initialize(self) -> None:
        """Initialize all model handlers."""
        if self._initialized:
            return

        logger.info("Initializing AI Model Manager...")

        # Initialize all handlers in parallel
        import asyncio

        await asyncio.gather(
            self.text_handler.initialize(),
            self.image_handler.initialize(),
            self.voice_handler.initialize(),
            self.video_handler.initialize(),
            return_exceptions=True,
        )

        self._initialized = True
        logger.info("AI Model Manager initialized")

    async def generate_text(self, prompt: str, **kwargs) -> str:
        """
        Generate text using the best available model.

        Args:
            prompt: Input prompt
            **kwargs: Additional generation parameters

        Returns:
            Generated text
        """
        if not self._initialized:
            await self.initialize()
        return await self.text_handler.generate(prompt, **kwargs)

    async def generate_image(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Generate an image using the best available model.

        Args:
            prompt: Text prompt for image generation
            **kwargs: Additional generation parameters

        Returns:
            Dictionary with image path, base64 data, and metadata
        """
        if not self._initialized:
            await self.initialize()
        return await self.image_handler.generate(prompt, **kwargs)

    async def generate_voice(self, text: str, **kwargs) -> Dict[str, Any]:
        """
        Generate speech from text.

        Args:
            text: Text to synthesize
            **kwargs: Additional generation parameters

        Returns:
            Dictionary with audio path, base64 data, and metadata
        """
        if not self._initialized:
            await self.initialize()
        return await self.voice_handler.generate(text, **kwargs)

    async def generate_video(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Generate a video.

        Args:
            prompt: Text prompt for video generation
            **kwargs: Additional generation parameters

        Returns:
            Dictionary with video path and metadata
        """
        if not self._initialized:
            await self.initialize()
        return await self.video_handler.generate(prompt, **kwargs)

    async def get_available_models(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get all available models organized by type.

        Returns:
            Dictionary with model lists for each type
        """
        if not self._initialized:
            await self.initialize()

        return {
            "text": await self.text_handler.get_available_models(),
            "image": await self.image_handler.get_available_models(),
            "voice": await self.voice_handler.get_available_models(),
            "video": await self.video_handler.get_available_models(),
        }

    async def get_system_info(self) -> Dict[str, Any]:
        """
        Get system information including GPU status.

        Returns:
            Dictionary with system information
        """
        gpu_info = self.gpu_manager.get_system_info()
        cache_stats = self.model_cache.get_stats()

        return {
            "gpu": gpu_info,
            "cache": cache_stats,
            "handlers": {
                "text": self.text_handler.is_initialized(),
                "image": self.image_handler.is_initialized(),
                "voice": self.voice_handler.is_initialized(),
                "video": self.video_handler.is_initialized(),
            },
        }

    async def close(self) -> None:
        """Clean up all resources."""
        logger.info("Shutting down AI Model Manager...")

        await self.text_handler.close()
        await self.image_handler.close()
        await self.voice_handler.close()
        await self.video_handler.close()

        self.model_cache.clear()
        self._initialized = False

        logger.info("AI Model Manager shut down")


# Global AI manager instance
_ai_manager: Optional[AIModelManager] = None


def get_ai_manager() -> AIModelManager:
    """
    Get the global AI model manager instance.

    Returns:
        AIModelManager singleton instance
    """
    global _ai_manager
    if _ai_manager is None:
        _ai_manager = AIModelManager()
    return _ai_manager


# Export all public classes and functions
__all__ = [
    # Main manager
    "AIModelManager",
    "get_ai_manager",
    # Individual handlers
    "TextModelHandler",
    "ImageModelHandler",
    "VoiceModelHandler",
    "VideoModelHandler",
    "get_text_handler",
    "get_image_handler",
    "get_voice_handler",
    "get_video_handler",
    # Base classes
    "BaseModelHandler",
    "ModelType",
    "ModelSource",
    "ModelCapabilities",
    # GPU management
    "GPUManager",
    "GPUInfo",
    "GPUType",
    "get_gpu_manager",
    # Model utilities
    "ModelCache",
    "get_model_cache",
    "verify_model_files_exist",
    "download_model_from_huggingface",
    "download_model_from_civitai",
    "is_flux_model",
    "disable_safety_checker",
    "filter_scheduler_config",
    "strip_ansi_codes",
    "get_model_size_estimate",
]
