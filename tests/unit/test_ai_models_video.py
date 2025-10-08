"""
Tests for AI Models Video Generation

Tests the Q2-Q3 2025 advanced video generation features in AI models manager.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from pathlib import Path

from backend.services.ai_models import AIModelManager


class TestAIModelsVideoGeneration:
    """Test suite for AI models video generation."""

    @pytest.fixture
    def ai_manager(self):
        """Create AI model manager instance."""
        manager = AIModelManager()
        manager.models_loaded = True
        return manager

    @pytest.fixture
    def mock_video_models(self, ai_manager):
        """Setup mock video models."""
        ai_manager.available_models["video"] = [
            {
                "name": "frame-by-frame-generator",
                "type": "multi-frame-video",
                "provider": "local",
                "loaded": True,
                "features": [
                    "multi_scene",
                    "transitions",
                    "audio_sync",
                    "storyboarding",
                ],
            },
            {
                "name": "stable-video-diffusion",
                "type": "image-to-video",
                "provider": "local",
                "loaded": False,
                "min_vram_gb": 24,
            },
            {
                "name": "runway-gen2",
                "type": "text-to-video",
                "provider": "runway",
                "loaded": False,
                "api_required": True,
            },
        ]
        return ai_manager

    def test_video_models_initialized(self, mock_video_models):
        """Test video models are properly initialized."""
        video_models = mock_video_models.available_models["video"]

        assert len(video_models) == 3
        assert any(m["name"] == "frame-by-frame-generator" for m in video_models)
        assert any(m["name"] == "stable-video-diffusion" for m in video_models)
        assert any(m["name"] == "runway-gen2" for m in video_models)

    def test_frame_by_frame_model_features(self, mock_video_models):
        """Test frame-by-frame generator has correct features."""
        fbf_model = next(
            m
            for m in mock_video_models.available_models["video"]
            if m["name"] == "frame-by-frame-generator"
        )

        assert "multi_scene" in fbf_model["features"]
        assert "transitions" in fbf_model["features"]
        assert "audio_sync" in fbf_model["features"]
        assert "storyboarding" in fbf_model["features"]
        assert fbf_model["loaded"] is True

    @pytest.mark.asyncio
    async def test_generate_video_with_frame_by_frame(self, mock_video_models):
        """Test video generation using frame-by-frame generator."""
        with patch(
            "backend.services.ai_models.AIModelManager._generate_video_frame_by_frame"
        ) as mock_generate:
            mock_generate.return_value = {
                "file_path": "/tmp/test_video.mp4",
                "duration": 6.0,
                "resolution": "1920x1080",
                "format": "MP4",
                "num_scenes": 2,
            }

            result = await mock_video_models.generate_video(
                prompt="Test video", video_type="multi_frame", quality="high"
            )

            assert result["file_path"] == "/tmp/test_video.mp4"
            assert result["duration"] == 6.0
            assert mock_generate.called

    @pytest.mark.asyncio
    async def test_generate_video_with_single_prompt(self, mock_video_models):
        """Test video generation with single text prompt."""
        with patch(
            "backend.services.ai_models.AIModelManager._generate_video_frame_by_frame"
        ) as mock_generate:
            mock_generate.return_value = {
                "file_path": "/tmp/single_frame.mp4",
                "duration": 4.0,
                "resolution": "1920x1080",
                "format": "MP4",
            }

            result = await mock_video_models.generate_video(
                prompt="Single scene video", video_type="single_frame"
            )

            assert result["duration"] == 4.0
            assert mock_generate.called

    @pytest.mark.asyncio
    async def test_generate_video_with_multiple_prompts(self, mock_video_models):
        """Test video generation with multiple prompts."""
        with patch(
            "backend.services.ai_models.AIModelManager._generate_video_frame_by_frame"
        ) as mock_generate:
            mock_generate.return_value = {
                "file_path": "/tmp/multi_scene.mp4",
                "duration": 9.0,
                "num_scenes": 3,
            }

            prompts = ["Scene 1", "Scene 2", "Scene 3"]

            result = await mock_video_models.generate_video(
                prompt=prompts, video_type="multi_frame", duration_per_frame=3.0
            )

            assert result["num_scenes"] == 3
            assert mock_generate.called

    @pytest.mark.asyncio
    async def test_synchronize_audio_to_video(self, mock_video_models):
        """Test audio synchronization with video."""
        with patch(
            "backend.services.video_processing_service.VideoProcessingService.synchronize_audio_with_video"
        ) as mock_sync:
            mock_sync.return_value = {
                "file_path": "/tmp/video_with_audio.mp4",
                "duration": 10.0,
                "has_audio": True,
                "audio_source": "/tmp/audio.mp3",
            }

            result = await mock_video_models.synchronize_audio_to_video(
                video_path="/tmp/video.mp4", audio_path="/tmp/audio.mp3"
            )

            assert result["has_audio"] is True
            assert result["audio_source"] == "/tmp/audio.mp3"
            assert mock_sync.called

    @pytest.mark.asyncio
    async def test_create_video_storyboard(self, mock_video_models):
        """Test storyboard creation."""
        with patch(
            "backend.services.video_processing_service.VideoProcessingService.create_storyboard"
        ) as mock_storyboard:
            mock_storyboard.return_value = {
                "file_path": "/tmp/storyboard.mp4",
                "duration": 12.0,
                "num_scenes": 4,
                "scene_markers": [
                    {"scene": 1, "timestamp": 0.0},
                    {"scene": 2, "timestamp": 3.0},
                    {"scene": 3, "timestamp": 6.0},
                    {"scene": 4, "timestamp": 9.0},
                ],
            }

            scenes = [
                {"prompt": "Scene 1", "duration": 3.0, "transition": "fade"},
                {"prompt": "Scene 2", "duration": 3.0, "transition": "crossfade"},
                {"prompt": "Scene 3", "duration": 3.0, "transition": "wipe"},
                {"prompt": "Scene 4", "duration": 3.0, "transition": "dissolve"},
            ]

            result = await mock_video_models.create_video_storyboard(
                scenes=scenes, quality="high"
            )

            assert result["num_scenes"] == 4
            assert len(result["scene_markers"]) == 4
            assert mock_storyboard.called

    @pytest.mark.asyncio
    async def test_video_generation_error_handling(self, mock_video_models):
        """Test error handling in video generation."""
        # Empty available models
        mock_video_models.available_models["video"] = []

        with pytest.raises(ValueError, match="No video generation models available"):
            await mock_video_models.generate_video(
                prompt="Test", video_type="single_frame"
            )

    @pytest.mark.asyncio
    async def test_stable_video_diffusion_placeholder(self, mock_video_models):
        """Test Stable Video Diffusion returns placeholder."""
        result = await mock_video_models._generate_video_svd(
            prompt="Test SVD generation"
        )

        assert result["status"] == "placeholder"
        assert result["model"] == "stable-video-diffusion"
        assert "SVD requires model download" in result["note"]

    @pytest.mark.asyncio
    async def test_runway_gen2_requires_api_key(self, mock_video_models):
        """Test Runway Gen-2 requires API key."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="RUNWAY_API_KEY not configured"):
                await mock_video_models._generate_video_runway(
                    prompt="Test Runway generation"
                )

    @pytest.mark.asyncio
    async def test_runway_gen2_with_api_key(self, mock_video_models):
        """Test Runway Gen-2 with API key configured."""
        with patch.dict("os.environ", {"RUNWAY_API_KEY": "test_key"}):
            result = await mock_video_models._generate_video_runway(
                prompt="Test Runway generation"
            )

            assert result["model"] == "runway-gen2"
            assert result["status"] == "placeholder"

    @pytest.mark.asyncio
    async def test_frame_by_frame_with_quality_settings(self, mock_video_models):
        """Test frame-by-frame generation with quality settings."""
        with patch(
            "backend.services.video_processing_service.VideoProcessingService.generate_frame_by_frame_video"
        ) as mock_gen:
            mock_gen.return_value = {
                "file_path": "/tmp/video.mp4",
                "duration": 6.0,
                "quality": "premium",
                "resolution": "3840x2160",
            }

            result = await mock_video_models._generate_video_frame_by_frame(
                prompt="Test", quality="premium", duration_per_frame=2.0
            )

            assert result["quality"] == "premium"
            assert "2160" in result["resolution"]
            # Verify the mock was called with correct quality parameter
            call_kwargs = mock_gen.call_args[1]
            assert call_kwargs["quality"].value == "premium"

    @pytest.mark.asyncio
    async def test_frame_by_frame_with_transitions(self, mock_video_models):
        """Test frame-by-frame generation with different transitions."""
        with patch(
            "backend.services.video_processing_service.VideoProcessingService.generate_frame_by_frame_video"
        ) as mock_gen:
            mock_gen.return_value = {
                "file_path": "/tmp/video.mp4",
                "transition_type": "crossfade",
            }

            result = await mock_video_models._generate_video_frame_by_frame(
                prompt="Test", transition="crossfade"
            )

            assert result["transition_type"] == "crossfade"
            # Verify the mock was called with correct transition parameter
            call_kwargs = mock_gen.call_args[1]
            assert call_kwargs["transition"].value == "crossfade"

    @pytest.mark.asyncio
    async def test_video_generation_with_list_prompts(self, mock_video_models):
        """Test handling of list prompts in frame-by-frame generation."""
        with patch(
            "backend.services.video_processing_service.VideoProcessingService.generate_frame_by_frame_video"
        ) as mock_gen:
            mock_gen.return_value = {"file_path": "/tmp/video.mp4", "num_scenes": 3}

            prompts = ["Scene 1", "Scene 2", "Scene 3"]

            result = await mock_video_models._generate_video_frame_by_frame(
                prompt=prompts
            )

            # Should pass the list to video service
            call_args = mock_gen.call_args
            assert call_args[1]["prompts"] == prompts

    @pytest.mark.asyncio
    async def test_video_generation_with_single_string_prompt(self, mock_video_models):
        """Test handling of single string prompt."""
        with patch(
            "backend.services.video_processing_service.VideoProcessingService.generate_frame_by_frame_video"
        ) as mock_gen:
            mock_gen.return_value = {"file_path": "/tmp/video.mp4"}

            result = await mock_video_models._generate_video_frame_by_frame(
                prompt="Single scene"
            )

            # Should convert to list
            call_args = mock_gen.call_args
            assert call_args[1]["prompts"] == ["Single scene"]

    @pytest.mark.asyncio
    async def test_audio_sync_without_output_path(self, mock_video_models):
        """Test audio sync generates default output path."""
        with patch(
            "backend.services.video_processing_service.VideoProcessingService.synchronize_audio_with_video"
        ) as mock_sync:
            mock_sync.return_value = {"file_path": "/tmp/video_with_audio.mp4"}

            result = await mock_video_models.synchronize_audio_to_video(
                video_path="/tmp/video.mp4", audio_path="/tmp/audio.mp3"
            )

            # Should pass None for output_path
            call_args = mock_sync.call_args
            assert call_args[1]["output_path"] is None

    @pytest.mark.asyncio
    async def test_audio_sync_with_custom_output_path(self, mock_video_models):
        """Test audio sync with custom output path."""
        with patch(
            "backend.services.video_processing_service.VideoProcessingService.synchronize_audio_with_video"
        ) as mock_sync:
            mock_sync.return_value = {"file_path": "/custom/output.mp4"}

            result = await mock_video_models.synchronize_audio_to_video(
                video_path="/tmp/video.mp4",
                audio_path="/tmp/audio.mp3",
                output_path="/custom/output.mp4",
            )

            assert result["file_path"] == "/custom/output.mp4"

    def test_video_model_configurations(self, mock_video_models):
        """Test video model configurations are complete."""
        for model in mock_video_models.available_models["video"]:
            assert "name" in model
            assert "type" in model
            assert "provider" in model
            assert "loaded" in model

            # Frame-by-frame should have features
            if model["name"] == "frame-by-frame-generator":
                assert "features" in model
                assert len(model["features"]) > 0


class TestVideoModelInitialization:
    """Test video model initialization in AI manager."""

    @pytest.mark.asyncio
    async def test_video_models_loaded_on_init(self):
        """Test video models are loaded during initialization."""
        manager = AIModelManager()
        await manager._initialize_video_models()

        assert len(manager.available_models["video"]) > 0

    @pytest.mark.asyncio
    async def test_all_video_model_types_present(self):
        """Test all expected video model types are initialized."""
        manager = AIModelManager()
        await manager._initialize_video_models()

        model_names = [m["name"] for m in manager.available_models["video"]]

        assert "frame-by-frame-generator" in model_names
        assert "stable-video-diffusion" in model_names
        assert "runway-gen2" in model_names

    @pytest.mark.asyncio
    async def test_frame_by_frame_always_available(self):
        """Test frame-by-frame generator is always available."""
        manager = AIModelManager()
        await manager._initialize_video_models()

        fbf_model = next(
            m
            for m in manager.available_models["video"]
            if m["name"] == "frame-by-frame-generator"
        )

        assert fbf_model["loaded"] is True
        assert fbf_model["provider"] == "local"

    @pytest.mark.asyncio
    async def test_svd_requires_vram(self):
        """Test SVD has VRAM requirements."""
        manager = AIModelManager()
        await manager._initialize_video_models()

        svd_model = next(
            m
            for m in manager.available_models["video"]
            if m["name"] == "stable-video-diffusion"
        )

        assert "min_vram_gb" in svd_model
        assert svd_model["min_vram_gb"] == 24

    @pytest.mark.asyncio
    async def test_runway_has_api_flag(self):
        """Test Runway model has API requirement flag."""
        manager = AIModelManager()
        await manager._initialize_video_models()

        runway_model = next(
            m for m in manager.available_models["video"] if m["name"] == "runway-gen2"
        )

        assert runway_model["api_required"] is True
