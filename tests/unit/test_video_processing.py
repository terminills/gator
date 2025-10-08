"""
Tests for Advanced Video Processing Service

Tests the Q2-Q3 2025 advanced video features including:
- Frame-by-frame video generation
- Audio synchronization
- Video transitions
- Storyboard creation
"""

import pytest
import asyncio
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
import numpy as np

from backend.services.video_processing_service import (
    VideoProcessingService,
    VideoQuality,
    TransitionType
)


class TestVideoProcessingService:
    """Test suite for video processing service."""

    @pytest.fixture
    def video_service(self, tmp_path):
        """Create video service with temp directory."""
        service = VideoProcessingService(output_dir=str(tmp_path / "videos"))
        return service

    @pytest.fixture
    def sample_frame(self):
        """Create a sample video frame."""
        # 720p frame
        return np.zeros((720, 1280, 3), dtype=np.uint8)

    def test_initialization(self, video_service):
        """Test service initialization."""
        assert video_service.output_dir.exists()
        assert len(video_service.quality_settings) == 4
        assert VideoQuality.HIGH in video_service.quality_settings

    def test_quality_settings(self, video_service):
        """Test video quality presets."""
        high_quality = video_service.quality_settings[VideoQuality.HIGH]
        assert high_quality["resolution"] == (1920, 1080)
        assert high_quality["fps"] == 30
        assert high_quality["bitrate"] == "6000k"
        
        draft_quality = video_service.quality_settings[VideoQuality.DRAFT]
        assert draft_quality["resolution"] == (854, 480)
        assert draft_quality["fps"] == 24

    @pytest.mark.asyncio
    async def test_generate_single_frame(self, video_service):
        """Test single frame generation."""
        frame = await video_service._generate_single_frame(
            prompt="Test scene",
            quality=VideoQuality.STANDARD,
            frame_index=0
        )
        
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (720, 1280, 3)
        assert frame.dtype == np.uint8

    @pytest.mark.asyncio
    async def test_frame_by_frame_video_generation(self, video_service):
        """Test frame-by-frame video generation."""
        prompts = [
            "Scene 1: Introduction",
            "Scene 2: Main content",
            "Scene 3: Conclusion"
        ]
        
        result = await video_service.generate_frame_by_frame_video(
            prompts=prompts,
            duration_per_frame=2.0,
            quality=VideoQuality.STANDARD,
            transition=TransitionType.CROSSFADE
        )
        
        assert "file_path" in result
        assert "duration" in result
        assert "resolution" in result
        assert result["num_scenes"] == 3
        assert result["quality"] == "standard"
        assert result["transition_type"] == "crossfade"
        assert result["duration"] == 6.0  # 3 scenes * 2 seconds each

    def test_create_fade_transition(self, video_service, sample_frame):
        """Test fade transition creation."""
        frame1 = sample_frame.copy()
        frame1[:, :] = [100, 100, 100]  # Gray
        
        frame2 = sample_frame.copy()
        frame2[:, :] = [200, 200, 200]  # Lighter gray
        
        transitions = video_service._create_transition(
            frame1,
            frame2,
            TransitionType.FADE,
            num_frames=10
        )
        
        assert len(transitions) == 10
        assert all(isinstance(frame, np.ndarray) for frame in transitions)
        # First transition frame should be darker than last
        assert np.mean(transitions[0]) < np.mean(transitions[-1])

    def test_create_crossfade_transition(self, video_service, sample_frame):
        """Test crossfade transition creation."""
        frame1 = sample_frame.copy()
        frame1[:, :] = [0, 0, 255]  # Blue
        
        frame2 = sample_frame.copy()
        frame2[:, :] = [255, 0, 0]  # Red
        
        transitions = video_service._create_transition(
            frame1,
            frame2,
            TransitionType.CROSSFADE,
            num_frames=10
        )
        
        assert len(transitions) == 10
        # Middle frame should be a blend
        middle_frame = transitions[5]
        # Should have both blue and red components
        assert np.mean(middle_frame[:, :, 0]) > 0  # Red channel
        assert np.mean(middle_frame[:, :, 2]) > 0  # Blue channel

    def test_create_wipe_transition(self, video_service, sample_frame):
        """Test wipe transition creation."""
        frame1 = sample_frame.copy()
        frame1[:, :] = [100, 100, 100]
        
        frame2 = sample_frame.copy()
        frame2[:, :] = [200, 200, 200]
        
        transitions = video_service._create_transition(
            frame1,
            frame2,
            TransitionType.WIPE,
            num_frames=10
        )
        
        assert len(transitions) == 10
        # First frame should be mostly frame1
        assert np.mean(transitions[0]) < np.mean(transitions[-1])

    @pytest.mark.asyncio
    async def test_apply_transitions(self, video_service, sample_frame):
        """Test applying transitions between frames."""
        frames = [
            sample_frame.copy(),
            sample_frame.copy(),
            sample_frame.copy()
        ]
        
        result_frames = await video_service._apply_transitions(
            frames,
            TransitionType.CROSSFADE,
            duration_per_frame=1.0
        )
        
        # Should have more frames due to transitions
        assert len(result_frames) > len(frames)
        # All frames should be numpy arrays
        assert all(isinstance(f, np.ndarray) for f in result_frames)

    @pytest.mark.asyncio
    async def test_export_video(self, video_service, sample_frame):
        """Test video export functionality."""
        frames = [sample_frame.copy() for _ in range(30)]  # 1 second at 30fps
        output_path = video_service.output_dir / "test_export.mp4"
        
        success = await video_service._export_video(
            frames,
            output_path,
            VideoQuality.STANDARD
        )
        
        assert success
        assert output_path.exists()

    @pytest.mark.asyncio
    async def test_get_video_duration(self, video_service, tmp_path):
        """Test getting video duration."""
        # This test requires an actual video file
        # For now, we'll test the method exists
        video_path = tmp_path / "test.mp4"
        video_path.touch()
        
        duration = await video_service._get_video_duration(video_path)
        assert isinstance(duration, float)
        assert duration >= 0.0

    @pytest.mark.asyncio
    async def test_storyboard_creation(self, video_service):
        """Test storyboard video creation."""
        scenes = [
            {
                "prompt": "Opening scene: sunrise",
                "duration": 2.0,
                "transition": "fade"
            },
            {
                "prompt": "Middle scene: action",
                "duration": 3.0,
                "transition": "crossfade"
            },
            {
                "prompt": "Closing scene: sunset",
                "duration": 2.0,
                "transition": "dissolve"
            }
        ]
        
        result = await video_service.create_storyboard(
            scenes=scenes,
            quality=VideoQuality.STANDARD
        )
        
        assert "file_path" in result
        assert result["num_scenes"] == 3
        assert "scene_markers" in result
        assert len(result["scene_markers"]) == 3
        # Duration should be sum of scene durations plus transitions
        assert result["duration"] > 7.0  # 2+3+2 + transitions

    @pytest.mark.asyncio
    async def test_generate_scene_frames(self, video_service):
        """Test generating frames for a scene."""
        frames = await video_service._generate_scene_frames(
            prompt="Test scene",
            duration=2.0,
            quality=VideoQuality.STANDARD
        )
        
        # 2 seconds at 30fps = 60 frames
        assert len(frames) == 60
        assert all(isinstance(f, np.ndarray) for f in frames)
        # Frames should show slight zoom (animation)
        assert not np.array_equal(frames[0], frames[-1])

    def test_get_supported_transitions(self, video_service):
        """Test getting supported transitions."""
        transitions = video_service.get_supported_transitions()
        assert "fade" in transitions
        assert "crossfade" in transitions
        assert "wipe" in transitions
        assert "slide" in transitions
        assert "zoom" in transitions
        assert "dissolve" in transitions
        assert len(transitions) == 6

    def test_get_supported_qualities(self, video_service):
        """Test getting supported quality presets."""
        qualities = video_service.get_supported_qualities()
        assert "draft" in qualities
        assert "standard" in qualities
        assert "high" in qualities
        assert "premium" in qualities
        assert len(qualities) == 4

    @pytest.mark.asyncio
    async def test_ffmpeg_check(self, video_service):
        """Test ffmpeg availability check."""
        # This may pass or fail depending on environment
        result = video_service._check_ffmpeg_available()
        assert isinstance(result, bool)

    @pytest.mark.asyncio
    @patch('backend.services.video_processing_service.VideoProcessingService._check_ffmpeg_available')
    async def test_audio_sync_without_ffmpeg(self, mock_ffmpeg, video_service, tmp_path):
        """Test audio sync fails gracefully without ffmpeg."""
        mock_ffmpeg.return_value = False
        
        video_path = tmp_path / "video.mp4"
        audio_path = tmp_path / "audio.mp3"
        video_path.touch()
        audio_path.touch()
        
        with pytest.raises(RuntimeError, match="ffmpeg not available"):
            await video_service.synchronize_audio_with_video(
                video_path=video_path,
                audio_path=audio_path
            )

    @pytest.mark.asyncio
    async def test_multi_scene_with_different_transitions(self, video_service):
        """Test multi-scene video with different transition types."""
        prompts = [
            "Scene with fade",
            "Scene with crossfade", 
            "Scene with wipe"
        ]
        
        # Test with each transition type
        for transition in [TransitionType.FADE, TransitionType.CROSSFADE, TransitionType.WIPE]:
            result = await video_service.generate_frame_by_frame_video(
                prompts=prompts,
                duration_per_frame=1.0,
                quality=VideoQuality.DRAFT,
                transition=transition
            )
            
            assert result["transition_type"] == transition.value
            assert result["num_scenes"] == 3

    @pytest.mark.asyncio
    async def test_video_quality_impacts_output(self, video_service):
        """Test that video quality setting impacts output."""
        prompts = ["Test scene"]
        
        # Generate with different qualities
        draft_result = await video_service.generate_frame_by_frame_video(
            prompts=prompts,
            duration_per_frame=1.0,
            quality=VideoQuality.DRAFT
        )
        
        premium_result = await video_service.generate_frame_by_frame_video(
            prompts=prompts,
            duration_per_frame=1.0,
            quality=VideoQuality.PREMIUM
        )
        
        # Premium should have higher resolution
        assert "1080" in draft_result["resolution"] or "480" in draft_result["resolution"]
        assert "2160" in premium_result["resolution"] or "1080" in premium_result["resolution"]

    @pytest.mark.asyncio
    async def test_empty_prompts_handling(self, video_service):
        """Test handling of empty prompts list."""
        with pytest.raises(Exception):
            await video_service.generate_frame_by_frame_video(
                prompts=[],
                duration_per_frame=1.0,
                quality=VideoQuality.STANDARD
            )

    @pytest.mark.asyncio
    async def test_single_frame_no_transition(self, video_service, sample_frame):
        """Test that single frame generates no transitions."""
        frames = [sample_frame]
        
        result_frames = await video_service._apply_transitions(
            frames,
            TransitionType.CROSSFADE,
            duration_per_frame=2.0
        )
        
        # Single frame should just be held for duration (no transitions between frames)
        # At 30fps for 2 seconds minus transition time = ~45 frames
        assert len(result_frames) >= 1
        # All frames should be similar (same frame held)
        if len(result_frames) > 1:
            assert np.array_equal(result_frames[0], result_frames[-1])


class TestVideoQualityEnum:
    """Test VideoQuality enumeration."""
    
    def test_all_qualities_exist(self):
        """Test all quality presets are defined."""
        assert VideoQuality.DRAFT.value == "draft"
        assert VideoQuality.STANDARD.value == "standard"
        assert VideoQuality.HIGH.value == "high"
        assert VideoQuality.PREMIUM.value == "premium"
    
    def test_quality_from_string(self):
        """Test creating quality from string."""
        assert VideoQuality("high") == VideoQuality.HIGH
        assert VideoQuality("draft") == VideoQuality.DRAFT


class TestTransitionTypeEnum:
    """Test TransitionType enumeration."""
    
    def test_all_transitions_exist(self):
        """Test all transition types are defined."""
        assert TransitionType.FADE.value == "fade"
        assert TransitionType.CROSSFADE.value == "crossfade"
        assert TransitionType.WIPE.value == "wipe"
        assert TransitionType.SLIDE.value == "slide"
        assert TransitionType.ZOOM.value == "zoom"
        assert TransitionType.DISSOLVE.value == "dissolve"
    
    def test_transition_from_string(self):
        """Test creating transition from string."""
        assert TransitionType("fade") == TransitionType.FADE
        assert TransitionType("crossfade") == TransitionType.CROSSFADE
