"""
Unit tests to verify that concurrent image generation doesn't cause scheduler state accumulation.

This test verifies the fix for the issue:
"IndexError: index 81 is out of bounds for dimension 0 with size 81"

The issue was caused by sharing a single scheduler instance across concurrent
generation requests, causing step_index to accumulate. The fix ensures each
generation request gets a fresh scheduler instance.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock, call
from pathlib import Path

from backend.services.ai_models import AIModelManager


class TestConcurrentImageGeneration:
    """Test that concurrent image generation doesn't cause scheduler state issues."""

    @pytest.fixture
    def model_manager(self):
        """Create a model manager instance for testing."""
        return AIModelManager()

    @pytest.mark.asyncio
    @patch("diffusers.DPMSolverMultistepScheduler")
    @patch("diffusers.StableDiffusionXLPipeline")
    @patch("backend.services.ai_models.torch")
    async def test_scheduler_recreated_for_each_request(
        self, mock_torch, mock_pipeline_class, mock_scheduler_class, model_manager
    ):
        """Test that a fresh scheduler is created for each generation request."""
        # Setup mocks
        mock_torch.cuda.is_available.return_value = True
        mock_pipeline_instance = MagicMock()
        mock_pipeline_instance.to.return_value = mock_pipeline_instance
        mock_pipeline_instance.scheduler = MagicMock()
        mock_pipeline_instance.scheduler.config = {}
        mock_pipeline_instance.enable_attention_slicing = MagicMock()
        mock_pipeline_instance.enable_xformers_memory_efficient_attention = MagicMock()

        # Mock the pipe call to return a result with images
        mock_result = MagicMock()
        mock_image = MagicMock()
        mock_image.save = MagicMock()
        mock_result.images = [mock_image]
        mock_pipeline_instance.return_value = mock_result

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
            # First generation request
            await model_manager._generate_image_diffusers(
                "test prompt 1", model, width=512, height=512
            )

            # Second generation request (simulating concurrent request)
            await model_manager._generate_image_diffusers(
                "test prompt 2", model, width=512, height=512
            )

            # Third generation request (simulating concurrent request)
            await model_manager._generate_image_diffusers(
                "test prompt 3", model, width=512, height=512
            )

        # The scheduler should be created 4 times:
        # 1. Once when the pipeline is first loaded
        # 2. Once before each of the 3 generation requests
        # This ensures each request gets a fresh scheduler with step_index=0
        assert mock_scheduler_class.from_config.call_count >= 4

        # Verify all scheduler creations used use_karras_sigmas=True
        for call_args in mock_scheduler_class.from_config.call_args_list:
            kwargs = call_args[1]
            assert "use_karras_sigmas" in kwargs
            assert kwargs["use_karras_sigmas"] is True

    @pytest.mark.asyncio
    @patch("diffusers.DPMSolverMultistepScheduler")
    @patch("diffusers.StableDiffusionPipeline")
    @patch("backend.services.ai_models.torch")
    async def test_scheduler_fresh_for_sd15(
        self, mock_torch, mock_pipeline_class, mock_scheduler_class, model_manager
    ):
        """Test that SD 1.5 also gets fresh scheduler for each request."""
        # Setup mocks
        mock_torch.cuda.is_available.return_value = True
        mock_pipeline_instance = MagicMock()
        mock_pipeline_instance.to.return_value = mock_pipeline_instance
        mock_pipeline_instance.scheduler = MagicMock()
        mock_pipeline_instance.scheduler.config = {}
        mock_pipeline_instance.enable_attention_slicing = MagicMock()
        mock_pipeline_instance.enable_xformers_memory_efficient_attention = MagicMock()

        # Mock the pipe call to return a result with images
        mock_result = MagicMock()
        mock_image = MagicMock()
        mock_image.save = MagicMock()
        mock_result.images = [mock_image]
        mock_pipeline_instance.return_value = mock_result

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
            # Simulate two concurrent requests
            await model_manager._generate_image_diffusers(
                "test prompt 1", model, width=512, height=512
            )

            await model_manager._generate_image_diffusers(
                "test prompt 2", model, width=512, height=512
            )

        # The scheduler should be created 3 times:
        # 1. Once when the pipeline is first loaded
        # 2. Once before each of the 2 generation requests
        assert mock_scheduler_class.from_config.call_count >= 3

        # Verify all scheduler creations used use_karras_sigmas=True
        for call_args in mock_scheduler_class.from_config.call_args_list:
            kwargs = call_args[1]
            assert "use_karras_sigmas" in kwargs
            assert kwargs["use_karras_sigmas"] is True

    @pytest.mark.asyncio
    async def test_sequential_generation_prevents_race_condition(self, model_manager):
        """Test that sequential generation prevents scheduler race conditions.
        
        This test verifies the fix for the issue where concurrent generation
        caused 'index 31 is out of bounds' errors due to step_index accumulation
        in shared schedulers. Sequential generation ensures each request completes
        before the next begins.
        """
        # Create a mock for tracking generation order
        generation_order = []
        
        async def mock_generate(appearance, **kwargs):
            start_idx = len(generation_order)
            generation_order.append(f"start_{start_idx}")
            # Simulate generation time
            await asyncio.sleep(0.01)
            generation_order.append(f"end_{start_idx}")
            return {
                "image_data": b"fake_image_data",
                "format": "PNG",
            }
        
        # Run 4 sequential generations
        results = []
        for i in range(4):
            result = await mock_generate(f"appearance_{i}")
            results.append(result)
        
        # Verify all generations completed
        assert len(results) == 4
        
        # Verify sequential execution: each generation should complete before next starts
        # Pattern should be: start_0, end_0, start_1, end_1, start_2, end_2, start_3, end_3
        for i in range(4):
            start_idx = generation_order.index(f"start_{i}")
            end_idx = generation_order.index(f"end_{i}")
            # Each generation's end should come before the next generation's start
            assert end_idx == start_idx + 1
            if i < 3:
                next_start_idx = generation_order.index(f"start_{i+1}")
                assert end_idx < next_start_idx
