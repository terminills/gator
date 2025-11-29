"""
Unit tests for the AI services module.

Tests the modular AI model handling system.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from backend.services.ai import (
    AIModelManager,
    get_ai_manager,
    ModelType,
    ModelSource,
    ModelCapabilities,
)
from backend.services.ai.base import BaseModelHandler
from backend.services.ai.gpu_manager import GPUManager, GPUType, GPUInfo, get_gpu_manager
from backend.services.ai.model_cache import ModelCache, get_model_cache
from backend.services.ai.model_loader import (
    is_flux_model,
    strip_ansi_codes,
    get_model_size_estimate,
)


class TestModelCapabilities:
    """Tests for ModelCapabilities class."""

    def test_default_capabilities(self):
        """Test default capability values."""
        caps = ModelCapabilities()
        assert caps.supports_nsfw is False
        assert caps.supports_streaming is False
        assert caps.supports_batching is False
        assert caps.max_tokens is None
        assert caps.max_resolution is None
        assert caps.supported_formats == []

    def test_custom_capabilities(self):
        """Test custom capability values."""
        caps = ModelCapabilities(
            supports_nsfw=True,
            supports_streaming=True,
            max_tokens=4096,
            max_resolution=(1024, 1024),
            supported_formats=["mp3", "wav"],
        )
        assert caps.supports_nsfw is True
        assert caps.supports_streaming is True
        assert caps.max_tokens == 4096
        assert caps.max_resolution == (1024, 1024)
        assert caps.supported_formats == ["mp3", "wav"]

    def test_to_dict(self):
        """Test converting capabilities to dictionary."""
        caps = ModelCapabilities(
            supports_nsfw=True,
            max_tokens=2048,
        )
        result = caps.to_dict()
        assert isinstance(result, dict)
        assert result["supports_nsfw"] is True
        assert result["max_tokens"] == 2048


class TestModelType:
    """Tests for ModelType enum."""

    def test_model_types(self):
        """Test model type values."""
        assert ModelType.TEXT == "text"
        assert ModelType.IMAGE == "image"
        assert ModelType.VIDEO == "video"
        assert ModelType.VOICE == "voice"


class TestModelSource:
    """Tests for ModelSource enum."""

    def test_model_sources(self):
        """Test model source values."""
        assert ModelSource.LOCAL == "local"
        assert ModelSource.CLOUD == "cloud"
        assert ModelSource.CIVITAI == "civitai"
        assert ModelSource.HUGGINGFACE == "huggingface"


class TestGPUManager:
    """Tests for GPUManager class."""

    def test_singleton_pattern(self):
        """Test that GPUManager is a singleton."""
        manager1 = get_gpu_manager()
        manager2 = get_gpu_manager()
        assert manager1 is manager2

    def test_cpu_fallback(self):
        """Test CPU fallback when no GPU is available."""
        manager = get_gpu_manager()
        # In test environment without GPU, should fall back to CPU
        assert manager.gpu_type in [GPUType.CPU, GPUType.CUDA, GPUType.ROCM]

    def test_get_system_info(self):
        """Test system info retrieval."""
        manager = get_gpu_manager()
        info = manager.get_system_info()
        assert "gpu_type" in info
        assert "gpu_count" in info
        assert "total_memory_gb" in info
        assert "gpus" in info

    def test_select_device_for_model(self):
        """Test device selection for model loading."""
        manager = get_gpu_manager()
        device = manager.select_device_for_model("test_model", required_memory_gb=4.0)
        # Device should be either "cpu" or "cuda:N" format
        assert device == "cpu" or device.startswith("cuda:")


class TestModelCache:
    """Tests for ModelCache class."""

    def test_cache_initialization(self):
        """Test cache initialization."""
        cache = ModelCache(max_memory_gb=16.0, max_models=5)
        assert cache.max_memory_gb == 16.0
        assert cache.max_models == 5

    def test_put_and_get(self):
        """Test putting and getting models from cache."""
        cache = ModelCache()
        mock_model = MagicMock()
        
        cache.put("test_model", mock_model, "text", "cpu", memory_mb=1000)
        
        result = cache.get("test_model")
        assert result is mock_model

    def test_cache_miss(self):
        """Test cache miss returns None."""
        cache = ModelCache()
        result = cache.get("nonexistent_model")
        assert result is None

    def test_contains(self):
        """Test contains method."""
        cache = ModelCache()
        mock_model = MagicMock()
        
        cache.put("test_model", mock_model, "text", "cpu")
        
        assert cache.contains("test_model") is True
        assert cache.contains("other_model") is False

    def test_remove(self):
        """Test removing model from cache."""
        cache = ModelCache()
        mock_model = MagicMock()
        
        cache.put("test_model", mock_model, "text", "cpu")
        result = cache.remove("test_model")
        
        assert result is True
        assert cache.contains("test_model") is False

    def test_cache_stats(self):
        """Test getting cache statistics."""
        cache = ModelCache()
        mock_model = MagicMock()
        
        cache.put("test_model", mock_model, "text", "cpu", memory_mb=500)
        
        stats = cache.get_stats()
        assert stats["cached_models"] == 1
        assert stats["memory_used_mb"] == 500
        assert len(stats["models"]) == 1


class TestModelLoader:
    """Tests for model loader utilities."""

    def test_strip_ansi_codes(self):
        """Test ANSI code stripping."""
        text_with_ansi = "\x1b[32mGreen text\x1b[0m and \x1b[1mBold\x1b[0m"
        result = strip_ansi_codes(text_with_ansi)
        assert result == "Green text and Bold"

    def test_strip_ansi_codes_no_codes(self):
        """Test stripping when no ANSI codes present."""
        plain_text = "Just plain text"
        result = strip_ansi_codes(plain_text)
        assert result == plain_text

    def test_is_flux_model_positive(self):
        """Test FLUX model detection - positive cases."""
        flux_models = [
            {"name": "flux.1-dev"},
            {"name": "FLUX-schnell"},
            {"model_id": "black-forest-labs/FLUX.1-dev"},
            {"path": "/models/flux1-nsfw.safetensors"},
            {"name": "fluxed-up-model"},
        ]
        for model in flux_models:
            assert is_flux_model(model) is True, f"Should detect as FLUX: {model}"

    def test_is_flux_model_negative(self):
        """Test FLUX model detection - negative cases."""
        non_flux_models = [
            {"name": "stable-diffusion-xl"},
            {"name": "sdxl-turbo"},
            {"name": "dreamshaper"},
            {"path": "/models/realvis.safetensors"},
        ]
        for model in non_flux_models:
            assert is_flux_model(model) is False, f"Should NOT detect as FLUX: {model}"

    def test_get_model_size_estimate_text(self):
        """Test size estimation for text models."""
        assert get_model_size_estimate("text", "llama-70b") >= 30.0
        assert get_model_size_estimate("text", "llama-8b") >= 4.0
        assert get_model_size_estimate("text", "model-3b") >= 2.0

    def test_get_model_size_estimate_image(self):
        """Test size estimation for image models."""
        assert get_model_size_estimate("image", "flux-dev") >= 20.0
        assert get_model_size_estimate("image", "sdxl-base") >= 6.0
        assert get_model_size_estimate("image", "sd-1.5") >= 3.0

    def test_get_model_size_estimate_video(self):
        """Test size estimation for video models."""
        assert get_model_size_estimate("video", "svd") >= 10.0

    def test_get_model_size_estimate_voice(self):
        """Test size estimation for voice models."""
        assert get_model_size_estimate("voice", "xtts") >= 1.0


class TestAIModelManager:
    """Tests for AIModelManager class."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset singleton before each test to ensure test isolation."""
        # Store original instance
        original_instance = AIModelManager._instance
        AIModelManager._instance = None
        yield
        # Restore original instance after test
        AIModelManager._instance = original_instance

    def test_singleton_pattern(self):
        """Test that AIModelManager is a singleton."""
        manager1 = get_ai_manager()
        manager2 = get_ai_manager()
        assert manager1 is manager2

    def test_has_all_handlers(self):
        """Test that manager has all model handlers."""
        manager = get_ai_manager()
        assert hasattr(manager, "text_handler")
        assert hasattr(manager, "image_handler")
        assert hasattr(manager, "voice_handler")
        assert hasattr(manager, "video_handler")

    @pytest.mark.asyncio
    async def test_get_system_info(self):
        """Test getting system info."""
        manager = get_ai_manager()
        # Initialize first
        manager._initialized = True  # Skip full init for test
        
        info = await manager.get_system_info()
        assert "gpu" in info
        assert "cache" in info
        assert "handlers" in info
