"""
Unit tests to verify SDXL models don't receive safety_checker parameters.

This test verifies the fix for the issue:
"Image generation failed: 'NoneType' object has no attribute 'tokenize'"

The issue was caused by passing safety_checker and requires_safety_checker
parameters to StableDiffusionXLPipeline, which doesn't support them.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from backend.services.ai_models import AIModelManager


class TestSDXLSafetyCheckerFix:
    """Test that SDXL models don't receive safety_checker parameters."""

    @pytest.fixture
    def model_manager(self):
        """Create a model manager instance for testing."""
        return AIModelManager()

    @pytest.mark.asyncio
    @patch("diffusers.StableDiffusionXLPipeline")
    @patch("backend.services.ai_models.torch")
    async def test_sdxl_no_safety_checker_params_local_path(
        self, mock_torch, mock_pipeline_class, model_manager
    ):
        """Test that SDXL models loaded from local path don't get safety_checker params."""
        # Setup mocks
        mock_torch.cuda.is_available.return_value = True
        mock_pipeline_instance = MagicMock()
        mock_pipeline_instance.to.return_value = mock_pipeline_instance
        mock_pipeline_instance.scheduler = MagicMock()
        mock_pipeline_instance.scheduler.config = {}
        mock_pipeline_class.from_pretrained.return_value = mock_pipeline_instance

        # Setup model info for SDXL
        model = {
            "name": "sdxl-1.0",
            "model_id": "stabilityai/stable-diffusion-xl-base-1.0",
            "path": str(Path("/fake/path/sdxl-1.0")),
        }

        # Mock path.exists to return True
        with patch("pathlib.Path.exists", return_value=True):
            try:
                await model_manager._generate_image_diffusers(
                    "test prompt", model, width=512, height=512
                )
            except Exception:
                # We expect some errors due to mocking, but we're only checking the call
                pass

        # Verify from_pretrained was called
        assert mock_pipeline_class.from_pretrained.called
        
        # Get the kwargs passed to from_pretrained
        call_kwargs = mock_pipeline_class.from_pretrained.call_args[1]
        
        # Verify safety_checker params are NOT in the kwargs for SDXL
        assert "safety_checker" not in call_kwargs
        assert "requires_safety_checker" not in call_kwargs
        
        # Verify torch_dtype is still present
        assert "torch_dtype" in call_kwargs

    @pytest.mark.asyncio
    @patch("diffusers.StableDiffusionXLPipeline")
    @patch("backend.services.ai_models.torch")
    async def test_sdxl_no_safety_checker_params_hub(
        self, mock_torch, mock_pipeline_class, model_manager
    ):
        """Test that SDXL models loaded from HuggingFace Hub don't get safety_checker params."""
        # Setup mocks
        mock_torch.cuda.is_available.return_value = True
        mock_pipeline_instance = MagicMock()
        mock_pipeline_instance.to.return_value = mock_pipeline_instance
        mock_pipeline_instance.scheduler = MagicMock()
        mock_pipeline_instance.scheduler.config = {}
        mock_pipeline_instance.save_pretrained = MagicMock()
        mock_pipeline_class.from_pretrained.return_value = mock_pipeline_instance

        # Setup model info for SDXL
        model = {
            "name": "sdxl-1.0",
            "model_id": "stabilityai/stable-diffusion-xl-base-1.0",
            "path": str(Path("/fake/path/sdxl-1.0")),
        }

        # Mock path.exists to return False (model not downloaded yet)
        with patch("pathlib.Path.exists", return_value=False), \
             patch("pathlib.Path.mkdir"):
            try:
                await model_manager._generate_image_diffusers(
                    "test prompt", model, width=512, height=512
                )
            except Exception:
                # We expect some errors due to mocking, but we're only checking the call
                pass

        # Verify from_pretrained was called
        assert mock_pipeline_class.from_pretrained.called
        
        # Get the kwargs passed to from_pretrained (may be called twice due to fp16 fallback)
        # Check the last call (fallback without variant)
        call_kwargs = mock_pipeline_class.from_pretrained.call_args[1]
        
        # Verify safety_checker params are NOT in the kwargs for SDXL
        assert "safety_checker" not in call_kwargs
        assert "requires_safety_checker" not in call_kwargs
        
        # Verify torch_dtype is still present
        assert "torch_dtype" in call_kwargs

    @pytest.mark.asyncio
    @patch("diffusers.StableDiffusionPipeline")
    @patch("backend.services.ai_models.torch")
    async def test_sd15_has_safety_checker_params(
        self, mock_torch, mock_pipeline_class, model_manager
    ):
        """Test that SD 1.5 models DO get safety_checker params."""
        # Setup mocks
        mock_torch.cuda.is_available.return_value = True
        mock_pipeline_instance = MagicMock()
        mock_pipeline_instance.to.return_value = mock_pipeline_instance
        mock_pipeline_instance.scheduler = MagicMock()
        mock_pipeline_instance.scheduler.config = {}
        mock_pipeline_class.from_pretrained.return_value = mock_pipeline_instance

        # Setup model info for SD 1.5
        model = {
            "name": "stable-diffusion-v1-5",
            "model_id": "runwayml/stable-diffusion-v1-5",
            "path": str(Path("/fake/path/stable-diffusion-v1-5")),
        }

        # Mock path.exists to return True
        with patch("pathlib.Path.exists", return_value=True):
            try:
                await model_manager._generate_image_diffusers(
                    "test prompt", model, width=512, height=512
                )
            except Exception:
                # We expect some errors due to mocking, but we're only checking the call
                pass

        # Verify from_pretrained was called
        assert mock_pipeline_class.from_pretrained.called
        
        # Get the kwargs passed to from_pretrained
        call_kwargs = mock_pipeline_class.from_pretrained.call_args[1]
        
        # Verify safety_checker params ARE in the kwargs for SD 1.5
        assert "safety_checker" in call_kwargs
        assert call_kwargs["safety_checker"] is None
        assert "requires_safety_checker" in call_kwargs
        assert call_kwargs["requires_safety_checker"] is False
        
        # Verify torch_dtype is still present
        assert "torch_dtype" in call_kwargs
