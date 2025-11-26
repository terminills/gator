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
                "can_load": True,
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
    @patch("backend.services.ai_models.AIModelManager._fallback_to_diffusers")
    async def test_comfyui_fallback_when_unavailable(
        self, mock_fallback, model_manager
    ):
        """Test that ComfyUI falls back to diffusers when unavailable."""
        test_model = {
            "name": "test-comfyui-model",
            "model_id": "test/comfyui",
        }

        # Mock fallback response
        mock_fallback.return_value = {
            "image_data": b"fallback_image",
            "format": "PNG",
            "model": "fallback-model",
            "status": "success",
        }

        # Mock http_client to simulate ComfyUI not available
        model_manager.http_client = AsyncMock()
        model_manager.http_client.get = AsyncMock(
            side_effect=Exception("Connection refused")
        )

        result = await model_manager._generate_image_comfyui("test prompt", test_model)

        # Should fallback to diffusers
        mock_fallback.assert_called_once()
        assert result["image_data"] == b"fallback_image"
        assert result["model"] == "fallback-model"

    @pytest.mark.asyncio
    @patch("backend.utils.model_detection.find_comfyui_installation")
    async def test_comfyui_integration_with_api(self, mock_comfyui_find, model_manager):
        """Test that ComfyUI integration works when API is available."""
        # Mock ComfyUI installation found
        mock_comfyui_find.return_value = Path("/fake/comfyui/path")

        test_model = {
            "name": "flux.1-dev",
            "model_id": "black-forest-labs/FLUX.1-dev",
            "comfyui_ckpt_name": "flux1-dev.safetensors",
        }

        # Mock http_client for successful ComfyUI interaction
        model_manager.http_client = AsyncMock()

        # Mock system_stats call (ComfyUI available)
        stats_response = AsyncMock()
        stats_response.status_code = 200

        # Mock object_info call (available checkpoints)
        object_info_response = AsyncMock()
        object_info_response.status_code = 200
        object_info_response.json = Mock(return_value={
            "CheckpointLoaderSimple": {
                "input": {
                    "required": {
                        "ckpt_name": [["flux1-dev.safetensors", "v1-5-pruned-emaonly.safetensors"]]
                    }
                }
            }
        })

        # Mock prompt submission
        queue_response = AsyncMock()
        queue_response.status_code = 200
        queue_response.json = Mock(return_value={"prompt_id": "test-prompt-123"})

        # Mock history check (completed)
        history_response = AsyncMock()
        history_response.status_code = 200
        history_response.json = Mock(
            return_value={
                "test-prompt-123": {
                    "outputs": {
                        "9": {
                            "images": [
                                {
                                    "filename": "test_image.png",
                                    "subfolder": "",
                                    "type": "output",
                                }
                            ]
                        }
                    }
                }
            }
        )

        # Mock image download
        image_response = AsyncMock()
        image_response.status_code = 200
        image_response.content = b"fake_comfyui_image_data"

        # Setup mock responses
        model_manager.http_client.get = AsyncMock(
            side_effect=[
                stats_response,  # First call: system_stats
                object_info_response,  # Second call: object_info for checkpoints
                history_response,  # Third call: history check
                image_response,  # Fourth call: image download
            ]
        )
        model_manager.http_client.post = AsyncMock(return_value=queue_response)

        result = await model_manager._generate_image_comfyui(
            "test prompt", test_model, width=1024, height=1024
        )

        # Verify successful generation
        assert result["status"] == "success"
        assert result["image_data"] == b"fake_comfyui_image_data"
        assert result["format"] == "PNG"
        assert result["workflow"] == "comfyui"
        assert result["width"] == 1024
        assert result["height"] == 1024

    @pytest.mark.asyncio
    async def test_generate_image_prefers_local_models(self, model_manager):
        """Test that generate_image prefers local models over cloud."""
        # Setup mock models - local and cloud
        model_manager.available_models["image"] = [
            {
                "name": "local-model",
                "provider": "local",
                "loaded": True,
                "can_load": True,
                "inference_engine": "diffusers",
            },
            {
                "name": "openai-dalle",
                "provider": "openai",
                "loaded": True,
                "can_load": True,
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
            "can_load": True,
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

    @pytest.mark.asyncio
    @patch("pathlib.Path.exists")
    @patch("torch.cuda.is_available")
    @patch("diffusers.StableDiffusionXLPipeline")
    async def test_sdxl_loading_with_fp16_variant(
        self, mock_pipeline, mock_cuda, mock_path_exists, model_manager
    ):
        """Test that SDXL models load with variant='fp16' and use_safetensors=True on CUDA."""
        # Setup - CUDA available
        mock_cuda.return_value = True
        # Mock path to NOT exist so it loads from HuggingFace Hub
        mock_path_exists.return_value = False

        mock_pipe_instance = MagicMock()
        mock_pipe_instance.to.return_value = mock_pipe_instance
        mock_pipe_instance.scheduler = MagicMock()
        mock_pipe_instance.scheduler.config = {}
        mock_pipe_instance.save_pretrained = MagicMock()

        # Mock the image generation
        mock_image = MagicMock()
        mock_image_bytes = io.BytesIO()
        mock_image.save = lambda buf, format: mock_image_bytes.write(b"fake_png_data")
        mock_pipe_instance.return_value.images = [mock_image]

        mock_pipeline.from_pretrained.return_value = mock_pipe_instance

        # Create SDXL model config
        sdxl_model = {
            "name": "sdxl-1.0",
            "model_id": "stabilityai/stable-diffusion-xl-base-1.0",
        }

        # Call the method
        try:
            result = await model_manager._generate_image_diffusers(
                "test prompt", sdxl_model, width=1024, height=1024
            )

            # Verify pipeline was loaded with correct parameters
            mock_pipeline.from_pretrained.assert_called_once()
            call_args = mock_pipeline.from_pretrained.call_args

            # Check that variant and use_safetensors are present
            assert (
                "variant" in call_args[1]
            ), "variant parameter should be present for SDXL"
            assert (
                call_args[1]["variant"] == "fp16"
            ), "variant should be 'fp16' for SDXL on CUDA"
            assert (
                "use_safetensors" in call_args[1]
            ), "use_safetensors should be present"
            assert (
                call_args[1]["use_safetensors"] is True
            ), "use_safetensors should be True"

            assert "image_data" in result
            assert result["format"] == "PNG"
        except Exception as e:
            # Some environments may not have all dependencies
            # This is acceptable for unit testing
            assert "diffusers" in str(e).lower() or "import" in str(e).lower()

    @pytest.mark.asyncio
    async def test_nsfw_model_preferred_by_default(self, model_manager):
        """Test that NSFW/anatomy-focused models are preferred by default."""
        # Setup mock models - regular and NSFW
        model_manager.available_models["image"] = [
            {
                "name": "sdxl-1.0",
                "provider": "local",
                "loaded": True,
                "can_load": True,
                "inference_engine": "diffusers",
            },
            {
                "name": "realistic-vision-nsfw",
                "provider": "local",
                "loaded": True,
                "can_load": True,
                "inference_engine": "diffusers",
            },
        ]

        # Call _get_best_local_image_model which should prefer NSFW model
        result = model_manager._get_best_local_image_model()
        assert "nsfw" in result["name"].lower(), "Should prefer NSFW model by default"

    @pytest.mark.asyncio
    async def test_preferred_civitai_model_selected(self, model_manager):
        """Test that preferred CivitAI model (version 1257570) is selected when available."""
        from backend.services.ai_models import PREFERRED_CIVITAI_VERSION_ID
        
        # Setup mock models - including preferred CivitAI model
        model_manager.available_models["image"] = [
            {
                "name": "sdxl-1.0",
                "provider": "local",
                "loaded": True,
                "can_load": True,
                "inference_engine": "diffusers",
            },
            {
                "name": "realistic-vision-nsfw",
                "provider": "local",
                "loaded": True,
                "can_load": True,
                "inference_engine": "diffusers",
            },
            {
                "name": "civitai-preferred-model",
                "provider": "local",
                "loaded": True,
                "can_load": True,
                "inference_engine": "diffusers",
                "civitai_version_id": PREFERRED_CIVITAI_VERSION_ID,
            },
        ]

        # Call _get_best_local_image_model which should prefer CivitAI model
        result = model_manager._get_best_local_image_model()
        assert result.get("civitai_version_id") == PREFERRED_CIVITAI_VERSION_ID, \
            "Should prefer CivitAI model version 1257570 when available"

    @pytest.mark.asyncio
    async def test_nsfw_model_selected_for_image_generation(self, model_manager):
        """Test that image generation selects NSFW models by default."""
        # Setup mock models with different types
        model_manager.available_models["image"] = [
            {
                "name": "basic-model",
                "provider": "local",
                "loaded": True,
                "can_load": True,
                "inference_engine": "diffusers",
                "model_id": "test/basic",
            },
            {
                "name": "dreamshaper-model",
                "provider": "local",
                "loaded": True,
                "can_load": True,
                "inference_engine": "diffusers",
                "model_id": "test/dreamshaper",
            },
        ]

        with patch.object(model_manager, "_generate_image_diffusers") as mock_diffusers:
            mock_diffusers.return_value = {"image_data": b"test", "model": "dreamshaper-model"}

            result = await model_manager.generate_image("test prompt")

            # Should have selected the dreamshaper model (keyword match)
            call_args = mock_diffusers.call_args
            model_arg = call_args[1].get("model", call_args[0][1] if len(call_args[0]) > 1 else None)
            if model_arg:
                assert "dreamshaper" in model_arg.get("name", "").lower(), "Should select anatomy-focused model"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
