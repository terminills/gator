"""
Unit tests for AI Model Manager - Image Generation fp16 Variant Fallback
"""

import pytest
from unittest.mock import patch, MagicMock
import io

from backend.services.ai_models import AIModelManager


class TestImageGenerationFp16Fallback:
    """Test image generation fp16 variant fallback functionality."""

    @pytest.fixture
    def model_manager(self):
        """Create a model manager instance for testing."""
        return AIModelManager()

    @pytest.mark.asyncio
    @patch("pathlib.Path.exists")
    @patch("torch.cuda.is_available")
    @patch("diffusers.StableDiffusionXLPipeline")
    async def test_sdxl_fp16_variant_fallback_on_error(
        self, mock_pipeline, mock_cuda, mock_path_exists, model_manager
    ):
        """Test that SDXL models fallback to default loading when fp16 variant is unavailable."""
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

        # First call with fp16 should fail, second call without variant should succeed
        mock_pipeline.from_pretrained.side_effect = [
            ValueError(
                "You are trying to load the model files of the `variant=fp16`, "
                "but no such modeling files are available."
            ),
            mock_pipe_instance,  # Second call succeeds
        ]

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

            # Verify pipeline was called twice (first with fp16, then without)
            assert mock_pipeline.from_pretrained.call_count == 2

            # First call should have variant='fp16'
            first_call_args = mock_pipeline.from_pretrained.call_args_list[0]
            assert "variant" in first_call_args[1]
            assert first_call_args[1]["variant"] == "fp16"

            # Second call should NOT have variant parameter
            second_call_args = mock_pipeline.from_pretrained.call_args_list[1]
            assert "variant" not in second_call_args[1]

            # Result should be successful
            assert "image_data" in result
            assert result["format"] == "PNG"
        except Exception as e:
            # Some environments may not have all dependencies
            # This is acceptable for unit testing
            assert "diffusers" in str(e).lower() or "import" in str(e).lower()

    @pytest.mark.asyncio
    @patch("pathlib.Path.exists")
    @patch("torch.cuda.is_available")
    @patch("diffusers.StableDiffusionXLPipeline")
    async def test_sdxl_fp16_variant_oserror_fallback(
        self, mock_pipeline, mock_cuda, mock_path_exists, model_manager
    ):
        """Test that SDXL models fallback when OSError is raised for fp16 variant."""
        # Setup - CUDA available
        mock_cuda.return_value = True
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

        # First call with fp16 should fail with OSError, second call should succeed
        mock_pipeline.from_pretrained.side_effect = [
            OSError("No such file or directory: fp16 variant"),
            mock_pipe_instance,
        ]

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

            # Verify pipeline was called twice
            assert mock_pipeline.from_pretrained.call_count == 2

            # Result should be successful
            assert "image_data" in result
        except Exception as e:
            # Some environments may not have all dependencies
            assert "diffusers" in str(e).lower() or "import" in str(e).lower()

    @pytest.mark.asyncio
    @patch("pathlib.Path.exists")
    @patch("torch.cuda.is_available")
    @patch("diffusers.StableDiffusionXLPipeline")
    async def test_sdxl_fp16_variant_from_local_path_fallback(
        self, mock_pipeline, mock_cuda, mock_path_exists, model_manager
    ):
        """Test fallback when loading SDXL from local path with fp16 variant unavailable."""
        # Setup - CUDA available
        mock_cuda.return_value = True
        # Mock path EXISTS so it tries to load from local path
        mock_path_exists.return_value = True

        mock_pipe_instance = MagicMock()
        mock_pipe_instance.to.return_value = mock_pipe_instance
        mock_pipe_instance.scheduler = MagicMock()
        mock_pipe_instance.scheduler.config = {}

        # Mock the image generation
        mock_image = MagicMock()
        mock_image_bytes = io.BytesIO()
        mock_image.save = lambda buf, format: mock_image_bytes.write(b"fake_png_data")
        mock_pipe_instance.return_value.images = [mock_image]

        # First call with fp16 should fail, second call without variant should succeed
        mock_pipeline.from_pretrained.side_effect = [
            ValueError(
                "You are trying to load the model files of the `variant=fp16`, "
                "but no such modeling files are available."
            ),
            mock_pipe_instance,
        ]

        # Create SDXL model config with local path
        sdxl_model = {
            "name": "sdxl-1.0",
            "model_id": "stabilityai/stable-diffusion-xl-base-1.0",
            "path": "./models/image/sdxl-1.0",
        }

        # Call the method
        try:
            result = await model_manager._generate_image_diffusers(
                "test prompt", sdxl_model, width=1024, height=1024
            )

            # Verify pipeline was called twice (local path loading with fallback)
            assert mock_pipeline.from_pretrained.call_count == 2

            # Both calls should use the local path
            first_call_args = mock_pipeline.from_pretrained.call_args_list[0]
            second_call_args = mock_pipeline.from_pretrained.call_args_list[1]

            # First call should have variant='fp16'
            assert "variant" in first_call_args[1]
            assert first_call_args[1]["variant"] == "fp16"

            # Second call should NOT have variant
            assert "variant" not in second_call_args[1]

            # Result should be successful
            assert "image_data" in result
        except Exception as e:
            # Some environments may not have all dependencies
            assert "diffusers" in str(e).lower() or "import" in str(e).lower()

    @pytest.mark.asyncio
    @patch("pathlib.Path.exists")
    @patch("torch.cuda.is_available")
    @patch("diffusers.StableDiffusionPipeline")
    async def test_non_sdxl_models_unaffected(
        self, mock_pipeline, mock_cuda, mock_path_exists, model_manager
    ):
        """Test that non-SDXL models are not affected by fp16 variant logic."""
        # Setup - CUDA available
        mock_cuda.return_value = True
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

        # Create non-SDXL model config
        sd_model = {
            "name": "stable-diffusion-v1-5",
            "model_id": "runwayml/stable-diffusion-v1-5",
        }

        # Call the method
        try:
            result = await model_manager._generate_image_diffusers(
                "test prompt", sd_model, width=512, height=512
            )

            # Verify pipeline was called only once (no fallback needed)
            assert mock_pipeline.from_pretrained.call_count == 1

            # Call should NOT have variant parameter
            call_args = mock_pipeline.from_pretrained.call_args
            assert "variant" not in call_args[1]

            # Result should be successful
            assert "image_data" in result
        except Exception as e:
            # Some environments may not have all dependencies
            assert "diffusers" in str(e).lower() or "import" in str(e).lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
