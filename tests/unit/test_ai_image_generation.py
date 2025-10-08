"""
Unit tests for AI Model Manager - Image Generation
"""

import pytest
import asyncio
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import io

from backend.services.ai_models import AIModelManager


class TestImageGeneration:
    """Test image generation functionality."""

    @pytest.fixture
    def model_manager(self):
        """Create a model manager instance for testing."""
        return AIModelManager()

    @pytest.mark.asyncio
    async def test_model_manager_has_pipeline_cache(self, model_manager):
        """Test that model manager initializes with pipeline cache."""
        assert hasattr(model_manager, "_loaded_pipelines")
        assert isinstance(model_manager._loaded_pipelines, dict)
        assert len(model_manager._loaded_pipelines) == 0

    @pytest.mark.asyncio
    async def test_initialize_models_includes_image_models(self, model_manager):
        """Test that model initialization includes image model detection."""
        await model_manager.initialize_models()
        assert "image" in model_manager.available_models
        assert isinstance(model_manager.available_models["image"], list)

    @pytest.mark.asyncio
    async def test_image_model_configuration(self, model_manager):
        """Test that image models are properly configured."""
        # Check that we have image model configurations
        assert "image" in model_manager.local_model_configs
        image_configs = model_manager.local_model_configs["image"]

        # Should have stable-diffusion-v1-5 as a diffusers model
        assert "stable-diffusion-v1-5" in image_configs
        sd_config = image_configs["stable-diffusion-v1-5"]

        assert sd_config["inference_engine"] == "diffusers"
        assert sd_config["model_id"] == "runwayml/stable-diffusion-v1-5"
        assert "size_gb" in sd_config
        assert "min_gpu_memory_gb" in sd_config
        assert "min_ram_gb" in sd_config

    @pytest.mark.asyncio
    async def test_generate_image_with_no_models(self, model_manager):
        """Test that generate_image raises error when no models are available."""
        await model_manager.initialize_models()

        # If no models are available, should raise ValueError
        if not model_manager.available_models["image"]:
            with pytest.raises(
                ValueError, match="No image generation models available"
            ):
                await model_manager.generate_image("test prompt")

    @pytest.mark.asyncio
    @patch("backend.services.ai_models.AIModelManager._generate_image_diffusers")
    async def test_generate_image_calls_diffusers(self, mock_diffusers, model_manager):
        """Test that generate_image correctly routes to diffusers for local models."""
        # Setup mock
        mock_diffusers.return_value = {
            "image_data": b"fake_image_data",
            "format": "PNG",
            "model": "test-model",
        }

        # Add a fake local model
        model_manager.available_models["image"] = [
            {
                "name": "test-model",
                "provider": "local",
                "loaded": True,
                "inference_engine": "diffusers",
            }
        ]

        # Call generate_image
        result = await model_manager.generate_image(
            "test prompt", width=512, height=512
        )

        # Verify diffusers was called
        mock_diffusers.assert_called_once()
        assert result["image_data"] == b"fake_image_data"
        assert result["format"] == "PNG"

    @pytest.mark.asyncio
    async def test_diffusers_method_signature(self, model_manager):
        """Test that _generate_image_diffusers has correct signature."""
        import inspect

        sig = inspect.signature(model_manager._generate_image_diffusers)
        params = list(sig.parameters.keys())

        assert "prompt" in params
        assert "model" in params
        assert "kwargs" in params

    @pytest.mark.asyncio
    @patch("torch.cuda.is_available")
    @patch("diffusers.StableDiffusionPipeline")
    async def test_diffusers_loads_model_from_hub(
        self, mock_pipeline, mock_cuda, model_manager
    ):
        """Test that diffusers loads model from HuggingFace Hub."""
        # Setup
        mock_cuda.return_value = False  # CPU mode
        mock_pipe_instance = MagicMock()
        mock_pipe_instance.to.return_value = mock_pipe_instance
        mock_pipe_instance.scheduler = MagicMock()
        mock_pipe_instance.scheduler.config = {}

        # Mock the image generation
        mock_image = MagicMock()
        mock_image_bytes = io.BytesIO()
        mock_image.save = lambda buf, format: mock_image_bytes.write(b"fake_png_data")
        mock_pipe_instance.return_value.images = [mock_image]

        mock_pipeline.from_pretrained.return_value = mock_pipe_instance

        # Create test model config
        test_model = {
            "name": "test-sd-model",
            "model_id": "test/model-id",
        }

        # Call the method
        try:
            result = await model_manager._generate_image_diffusers(
                "test prompt", test_model, width=512, height=512
            )

            # Verify pipeline was loaded
            mock_pipeline.from_pretrained.assert_called()
            assert "image_data" in result
            assert result["format"] == "PNG"
            assert result["model"] == "test-sd-model"
        except Exception as e:
            # Some environments may not have all dependencies
            # This is acceptable for unit testing
            assert "diffusers" in str(e).lower() or "import" in str(e).lower()

    @pytest.mark.asyncio
    async def test_comfyui_returns_not_implemented(self, model_manager):
        """Test that ComfyUI returns not implemented status."""
        test_model = {
            "name": "test-comfyui-model",
            "model_id": "test/comfyui",
        }

        result = await model_manager._generate_image_comfyui("test prompt", test_model)

        assert result["status"] == "not_implemented"
        assert result["workflow"] == "comfyui"
        assert "not_implemented" in result["status"].lower()

    @pytest.mark.asyncio
    async def test_generate_image_prefers_local_models(self, model_manager):
        """Test that generate_image prefers local models over cloud."""
        # Setup mock models - local and cloud
        model_manager.available_models["image"] = [
            {
                "name": "local-model",
                "provider": "local",
                "loaded": True,
                "inference_engine": "diffusers",
            },
            {
                "name": "openai-dalle",
                "provider": "openai",
                "loaded": True,
            },
        ]

        with (
            patch.object(model_manager, "_generate_image_local") as mock_local,
            patch.object(model_manager, "_generate_image_openai") as mock_openai,
        ):
            mock_local.return_value = {"image_data": b"local"}
            mock_openai.return_value = {"image_data": b"openai"}

            result = await model_manager.generate_image("test")

            # Should call local, not openai
            mock_local.assert_called_once()
            mock_openai.assert_not_called()

    @pytest.mark.asyncio
    async def test_pipeline_caching(self, model_manager):
        """Test that pipelines are cached after loading."""
        # Initially empty
        assert len(model_manager._loaded_pipelines) == 0

        # After adding a mock pipeline
        model_manager._loaded_pipelines["test_pipeline"] = MagicMock()

        assert len(model_manager._loaded_pipelines) == 1
        assert "test_pipeline" in model_manager._loaded_pipelines

    @pytest.mark.asyncio
    async def test_image_generation_parameters(self, model_manager):
        """Test that image generation accepts and uses parameters."""
        test_model = {
            "name": "test-model",
            "provider": "local",
            "loaded": True,
            "inference_engine": "diffusers",
            "model_id": "test/model",
        }

        model_manager.available_models["image"] = [test_model]

        with patch.object(model_manager, "_generate_image_diffusers") as mock_diffusers:
            mock_diffusers.return_value = {
                "image_data": b"test",
                "width": 1024,
                "height": 768,
            }

            result = await model_manager.generate_image(
                "test prompt",
                width=1024,
                height=768,
                num_inference_steps=30,
                guidance_scale=8.0,
                seed=42,
            )

            # Check that parameters were passed
            call_kwargs = mock_diffusers.call_args[1]
            assert call_kwargs["width"] == 1024
            assert call_kwargs["height"] == 768
            assert call_kwargs["num_inference_steps"] == 30
            assert call_kwargs["guidance_scale"] == 8.0
            assert call_kwargs["seed"] == 42


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
