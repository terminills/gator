"""
Tests for AI-powered video frame generation integration.

Verifies that video processing service can use AI models to generate
actual image frames instead of placeholder frames.
"""

import pytest
import numpy as np
from unittest.mock import AsyncMock, patch, MagicMock
from pathlib import Path

from backend.services.video_processing_service import (
    VideoProcessingService,
    VideoQuality,
    TransitionType,
)
from backend.services.ai_models import AIModelManager


@pytest.fixture
def video_service():
    """Create a VideoProcessingService for testing."""
    return VideoProcessingService(output_dir="/tmp/test_videos")


@pytest.fixture
def mock_ai_manager():
    """Create a mock AIModelManager."""
    manager = MagicMock(spec=AIModelManager)

    # Mock successful image generation
    async def mock_generate_image(*args, **kwargs):
        # Return a mock image result
        # Create a simple test image (100x100 PNG)
        from PIL import Image
        import io

        width = kwargs.get("width", 512)
        height = kwargs.get("height", 512)

        # Create a simple colored image
        img = Image.new("RGB", (width, height), color=(73, 109, 137))

        # Convert to bytes
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format="PNG")
        image_data = img_byte_arr.getvalue()

        return {
            "image_data": image_data,
            "format": "PNG",
            "width": width,
            "height": height,
            "model": "test-model",
        }

    manager.generate_image = AsyncMock(side_effect=mock_generate_image)
    return manager


class TestAIVideoFrameGeneration:
    """Test suite for AI-powered video frame generation."""

    @pytest.mark.asyncio
    async def test_generate_single_frame_with_ai(self, video_service, mock_ai_manager):
        """Test that single frame generation uses AI when provided."""
        prompt = "A beautiful mountain landscape"
        quality = VideoQuality.STANDARD

        frame = await video_service._generate_single_frame(
            prompt=prompt,
            quality=quality,
            frame_index=0,
            use_ai_generation=True,
            ai_model_manager=mock_ai_manager,
        )

        # Verify frame is generated
        assert isinstance(frame, np.ndarray)
        assert frame.shape[2] == 3  # RGB/BGR channels

        # Verify AI manager was called
        mock_ai_manager.generate_image.assert_called_once()
        call_kwargs = mock_ai_manager.generate_image.call_args[1]
        assert call_kwargs["prompt"] == prompt
        assert call_kwargs["width"] == 1280  # STANDARD quality
        assert call_kwargs["height"] == 720

    @pytest.mark.asyncio
    async def test_generate_single_frame_without_ai(self, video_service):
        """Test that single frame generation works without AI (placeholder)."""
        prompt = "Test prompt"
        quality = VideoQuality.STANDARD

        frame = await video_service._generate_single_frame(
            prompt=prompt,
            quality=quality,
            frame_index=0,
            use_ai_generation=False,
        )

        # Verify placeholder frame is generated
        assert isinstance(frame, np.ndarray)
        assert frame.shape[2] == 3
        assert frame.shape[0] == 720  # STANDARD height
        assert frame.shape[1] == 1280  # STANDARD width

    @pytest.mark.asyncio
    async def test_generate_single_frame_ai_fallback(self, video_service):
        """Test that frame generation falls back to placeholder if AI fails."""
        # Create a mock manager that raises an exception
        failing_manager = MagicMock(spec=AIModelManager)
        failing_manager.generate_image = AsyncMock(
            side_effect=Exception("AI generation failed")
        )

        prompt = "Test prompt"
        quality = VideoQuality.STANDARD

        frame = await video_service._generate_single_frame(
            prompt=prompt,
            quality=quality,
            frame_index=0,
            use_ai_generation=True,
            ai_model_manager=failing_manager,
        )

        # Should still return a frame (placeholder)
        assert isinstance(frame, np.ndarray)
        assert frame.shape[2] == 3

    @pytest.mark.asyncio
    async def test_frame_by_frame_video_with_ai(self, video_service, mock_ai_manager):
        """Test full video generation with AI frame generation."""
        prompts = ["Scene 1", "Scene 2"]

        result = await video_service.generate_frame_by_frame_video(
            prompts=prompts,
            duration_per_frame=1.0,
            quality=VideoQuality.DRAFT,  # Use DRAFT for faster testing
            transition=TransitionType.FADE,
            use_ai_generation=True,
            ai_model_manager=mock_ai_manager,
        )

        # Verify video was generated
        assert "file_path" in result
        assert result["num_scenes"] == 2
        assert result["duration"] == 2.0

        # Verify AI was called for each frame
        assert mock_ai_manager.generate_image.call_count == 2

    @pytest.mark.asyncio
    async def test_different_quality_settings_with_ai(
        self, video_service, mock_ai_manager
    ):
        """Test that AI generation uses correct resolution for different qualities."""
        qualities_and_resolutions = [
            (VideoQuality.DRAFT, (854, 480)),
            (VideoQuality.STANDARD, (1280, 720)),
            (VideoQuality.HIGH, (1920, 1080)),
        ]

        for quality, expected_resolution in qualities_and_resolutions:
            mock_ai_manager.generate_image.reset_mock()

            frame = await video_service._generate_single_frame(
                prompt="Test",
                quality=quality,
                frame_index=0,
                use_ai_generation=True,
                ai_model_manager=mock_ai_manager,
            )

            # Verify correct resolution was requested
            call_kwargs = mock_ai_manager.generate_image.call_args[1]
            assert call_kwargs["width"] == expected_resolution[0]
            assert call_kwargs["height"] == expected_resolution[1]

            # Verify frame has correct shape
            assert frame.shape[1] == expected_resolution[0]
            assert frame.shape[0] == expected_resolution[1]


class TestAIModelsVideoIntegration:
    """Test integration between AIModelManager and video generation."""

    @pytest.mark.asyncio
    async def test_ai_manager_passes_itself_to_video_service(self):
        """Test that AIModelManager passes itself for frame generation."""
        from backend.services.ai_models import AIModelManager

        manager = AIModelManager()
        await manager.initialize_models()

        # Mock the video service to verify parameters
        with patch(
            "backend.services.video_processing_service.VideoProcessingService.generate_frame_by_frame_video"
        ) as mock_video_gen:
            mock_video_gen.return_value = {
                "file_path": "/tmp/test.mp4",
                "duration": 1.0,
                "resolution": "1920x1080",
                "fps": 30,
                "format": "MP4",
                "num_scenes": 1,
                "transition_type": "fade",
            }

            await manager._generate_video_frame_by_frame(
                prompt="Test prompt",
                quality="high",
                use_ai_generation=True,
            )

            # Verify video service was called with AI manager
            mock_video_gen.assert_called_once()
            call_kwargs = mock_video_gen.call_args[1]
            assert "ai_model_manager" in call_kwargs
            assert call_kwargs["ai_model_manager"] is manager
            assert call_kwargs["use_ai_generation"] is True

    @pytest.mark.asyncio
    async def test_ai_manager_respects_use_ai_generation_flag(self):
        """Test that use_ai_generation flag is respected."""
        from backend.services.ai_models import AIModelManager

        manager = AIModelManager()
        await manager.initialize_models()

        with patch(
            "backend.services.video_processing_service.VideoProcessingService.generate_frame_by_frame_video"
        ) as mock_video_gen:
            mock_video_gen.return_value = {
                "file_path": "/tmp/test.mp4",
                "duration": 1.0,
                "resolution": "1920x1080",
                "fps": 30,
                "format": "MP4",
                "num_scenes": 1,
                "transition_type": "fade",
            }

            # Test with AI generation disabled
            await manager._generate_video_frame_by_frame(
                prompt="Test prompt",
                quality="high",
                use_ai_generation=False,
            )

            call_kwargs = mock_video_gen.call_args[1]
            assert call_kwargs["use_ai_generation"] is False
