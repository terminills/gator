"""
Tests for Multi-GPU Image Generation

Validates the batch image generation functionality with multi-GPU support.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch

from backend.services.ai_models import AIModelManager


@pytest.fixture
def ai_model_manager():
    """Create an AIModelManager for testing."""
    return AIModelManager()


class TestMultiGPUImageGeneration:
    """Test suite for multi-GPU batch image generation."""

    @pytest.mark.asyncio
    async def test_generate_images_batch_empty_prompts(self, ai_model_manager):
        """Test that batch generation handles empty prompt list."""
        result = await ai_model_manager.generate_images_batch([])

        assert result == []

    @pytest.mark.asyncio
    async def test_generate_images_batch_single_prompt(self, ai_model_manager):
        """Test batch generation with single prompt."""
        # Mock the generate_image method
        with patch.object(
            ai_model_manager, "generate_image", new_callable=AsyncMock
        ) as mock_generate:
            mock_generate.return_value = {
                "image_data": b"test_image_data",
                "format": "PNG",
                "status": "success",
            }

            # Mock no local models available to trigger fallback
            ai_model_manager.available_models["image"] = []

            result = await ai_model_manager.generate_images_batch(["Test prompt"])

            # Should call generate_image once
            assert mock_generate.call_count == 1
            assert len(result) == 1

    @pytest.mark.asyncio
    async def test_generate_images_batch_multiple_prompts_no_gpu(
        self, ai_model_manager
    ):
        """Test batch generation with multiple prompts and no GPU."""
        prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]

        with patch("torch.cuda.is_available", return_value=False):
            with patch.object(
                ai_model_manager, "_generate_image_local", new_callable=AsyncMock
            ) as mock_generate_local:
                mock_generate_local.return_value = {
                    "image_data": b"test",
                    "format": "PNG",
                }

                # Set up a fake local model
                ai_model_manager.available_models["image"] = [
                    {"provider": "local", "loaded": True, "name": "test-model"}
                ]

                result = await ai_model_manager.generate_images_batch(prompts)

                # Should process sequentially
                assert len(result) == 3
                assert mock_generate_local.call_count == 3

    @pytest.mark.asyncio
    async def test_generate_images_batch_with_multi_gpu(self, ai_model_manager):
        """Test batch generation distributes across multiple GPUs."""
        prompts = ["Prompt 1", "Prompt 2", "Prompt 3", "Prompt 4"]

        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.device_count", return_value=2):
                with patch.object(
                    ai_model_manager,
                    "_generate_image_on_device",
                    new_callable=AsyncMock,
                ) as mock_generate_device:
                    mock_generate_device.return_value = {
                        "image_data": b"test",
                        "format": "PNG",
                        "status": "success",
                    }

                    # Set up a fake local model
                    ai_model_manager.available_models["image"] = [
                        {"provider": "local", "loaded": True, "name": "test-model"}
                    ]

                    result = await ai_model_manager.generate_images_batch(prompts)

                    # Should call _generate_image_on_device for each prompt
                    assert mock_generate_device.call_count == 4
                    assert len(result) == 4

                    # Verify device distribution (should alternate: 0,1,0,1)
                    calls = mock_generate_device.call_args_list
                    device_ids = [call[0][2] for call in calls]  # device_id is 3rd arg
                    assert device_ids == [0, 1, 0, 1]

    @pytest.mark.asyncio
    async def test_generate_image_on_device_adds_device_id(self, ai_model_manager):
        """Test that device-specific generation adds device_id to kwargs."""
        model = {"name": "test-model"}

        with patch.object(
            ai_model_manager, "_generate_image_local", new_callable=AsyncMock
        ) as mock_generate_local:
            mock_generate_local.return_value = {"image_data": b"test", "format": "PNG"}

            result = await ai_model_manager._generate_image_on_device(
                "Test prompt", model, device_id=1, width=512, height=512
            )

            # Should call _generate_image_local with device_id in kwargs
            mock_generate_local.assert_called_once()
            call_kwargs = mock_generate_local.call_args[1]
            assert "device_id" in call_kwargs
            assert call_kwargs["device_id"] == 1

            # Result should include device_id
            assert "device_id" in result
            assert result["device_id"] == 1

    @pytest.mark.asyncio
    async def test_generate_images_batch_handles_exceptions(self, ai_model_manager):
        """Test that batch generation handles individual failures gracefully."""
        prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]

        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.device_count", return_value=2):
                with patch.object(
                    ai_model_manager,
                    "_generate_image_on_device",
                    new_callable=AsyncMock,
                ) as mock_generate_device:
                    # First call succeeds, second fails, third succeeds
                    mock_generate_device.side_effect = [
                        {"image_data": b"test1", "format": "PNG"},
                        Exception("Generation failed"),
                        {"image_data": b"test3", "format": "PNG"},
                    ]

                    ai_model_manager.available_models["image"] = [
                        {"provider": "local", "loaded": True, "name": "test-model"}
                    ]

                    result = await ai_model_manager.generate_images_batch(prompts)

                    # Should return all results (including error)
                    assert len(result) == 3
                    assert result[0]["format"] == "PNG"
                    assert "error" in result[1]
                    assert result[2]["format"] == "PNG"

    @pytest.mark.asyncio
    async def test_generate_images_batch_no_local_models_uses_cloud(
        self, ai_model_manager
    ):
        """Test batch generation falls back to sequential when no local models."""
        prompts = ["Prompt 1", "Prompt 2"]

        with patch.object(
            ai_model_manager, "generate_image", new_callable=AsyncMock
        ) as mock_generate:
            mock_generate.return_value = {"image_data": b"test", "format": "PNG"}

            # No local models available
            ai_model_manager.available_models["image"] = []

            result = await ai_model_manager.generate_images_batch(prompts)

            # Should fall back to sequential generation
            assert len(result) == 2
            assert mock_generate.call_count == 2
