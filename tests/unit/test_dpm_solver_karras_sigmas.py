"""
Unit tests to verify DPMSolverMultistepScheduler uses Karras sigmas.

This test verifies the fix for the issue:
"IndexError: index 81 is out of bounds for dimension 0 with size 81"

The issue was caused by DPMSolverMultistepScheduler attempting to access
an out-of-bounds index during generation. The fix enables use_karras_sigmas
to stabilize the step index calculation.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from backend.services.ai_models import AIModelManager


class TestDPMSolverKarrasSigmas:
    """Test that DPMSolverMultistepScheduler is configured with Karras sigmas."""

    @pytest.fixture
    def model_manager(self):
        """Create a model manager instance for testing."""
        return AIModelManager()

    @pytest.mark.asyncio
    @patch("diffusers.DPMSolverMultistepScheduler")
    @patch("diffusers.StableDiffusionXLPipeline")
    @patch("backend.services.ai_models.torch")
    async def test_dpm_solver_uses_karras_sigmas_sdxl(
        self, mock_torch, mock_pipeline_class, mock_scheduler_class, model_manager
    ):
        """Test that DPMSolverMultistepScheduler is configured with use_karras_sigmas for SDXL."""
        # Setup mocks
        mock_torch.cuda.is_available.return_value = True
        mock_pipeline_instance = MagicMock()
        mock_pipeline_instance.to.return_value = mock_pipeline_instance
        mock_pipeline_instance.scheduler = MagicMock()
        mock_pipeline_instance.scheduler.config = {}
        mock_pipeline_instance.enable_attention_slicing = MagicMock()
        mock_pipeline_instance.enable_xformers_memory_efficient_attention = MagicMock()
        mock_pipeline_class.from_pretrained.return_value = mock_pipeline_instance

        mock_scheduler_instance = MagicMock()
        mock_scheduler_class.from_config.return_value = mock_scheduler_instance

        # Setup model info for SDXL
        model = {
            "name": "sdxl-1.0",
            "model_id": "stabilityai/stable-diffusion-xl-base-1.0",
            "path": str(Path("/fake/path/sdxl-1.0")),
        }

        # Mock path.exists to return True (model already downloaded)
        with patch("pathlib.Path.exists", return_value=True):
            try:
                await model_manager._generate_image_diffusers(
                    "test prompt", model, width=512, height=512
                )
            except Exception:
                # We expect some errors due to mocking, but we're checking the scheduler config
                pass

        # Verify DPMSolverMultistepScheduler.from_config was called
        assert mock_scheduler_class.from_config.called

        # Get the kwargs passed to from_config
        call_kwargs = mock_scheduler_class.from_config.call_args[1]

        # Verify use_karras_sigmas is set to True
        assert "use_karras_sigmas" in call_kwargs
        assert call_kwargs["use_karras_sigmas"] is True

    @pytest.mark.asyncio
    @patch("diffusers.DPMSolverMultistepScheduler")
    @patch("diffusers.StableDiffusionPipeline")
    @patch("backend.services.ai_models.torch")
    async def test_dpm_solver_uses_karras_sigmas_sd15(
        self, mock_torch, mock_pipeline_class, mock_scheduler_class, model_manager
    ):
        """Test that DPMSolverMultistepScheduler is configured with use_karras_sigmas for SD 1.5."""
        # Setup mocks
        mock_torch.cuda.is_available.return_value = True
        mock_pipeline_instance = MagicMock()
        mock_pipeline_instance.to.return_value = mock_pipeline_instance
        mock_pipeline_instance.scheduler = MagicMock()
        mock_pipeline_instance.scheduler.config = {}
        mock_pipeline_instance.enable_attention_slicing = MagicMock()
        mock_pipeline_instance.enable_xformers_memory_efficient_attention = MagicMock()
        mock_pipeline_class.from_pretrained.return_value = mock_pipeline_instance

        mock_scheduler_instance = MagicMock()
        mock_scheduler_class.from_config.return_value = mock_scheduler_instance

        # Setup model info for SD 1.5
        model = {
            "name": "stable-diffusion-v1-5",
            "model_id": "runwayml/stable-diffusion-v1-5",
            "path": str(Path("/fake/path/stable-diffusion-v1-5")),
        }

        # Mock path.exists to return True (model already downloaded)
        with patch("pathlib.Path.exists", return_value=True):
            try:
                await model_manager._generate_image_diffusers(
                    "test prompt", model, width=512, height=512
                )
            except Exception:
                # We expect some errors due to mocking, but we're checking the scheduler config
                pass

        # Verify DPMSolverMultistepScheduler.from_config was called
        assert mock_scheduler_class.from_config.called

        # Get the kwargs passed to from_config
        call_kwargs = mock_scheduler_class.from_config.call_args[1]

        # Verify use_karras_sigmas is set to True
        assert "use_karras_sigmas" in call_kwargs
        assert call_kwargs["use_karras_sigmas"] is True
