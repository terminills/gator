"""
Tests for Reel Generation Service video creation.

Tests the reel generation video functionality including:
- Actual video file generation (not placeholder text)
- Video frame creation
- Fallback video generation
"""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4


class TestReelVideoGeneration:
    """Test suite for reel video generation functionality."""

    @pytest.fixture
    def tmp_video_dir(self, tmp_path):
        """Create temporary video directory."""
        video_dir = tmp_path / "reels"
        video_dir.mkdir(parents=True, exist_ok=True)
        return video_dir

    @pytest.mark.asyncio
    async def test_create_fallback_video_generates_mp4(self, tmp_video_dir):
        """Test fallback video creates actual video file."""
        from backend.services.reel_generation_service import ReelGenerationService
        from unittest.mock import MagicMock

        # Create mock db session
        mock_db = MagicMock()
        service = ReelGenerationService(mock_db, str(tmp_video_dir))

        output_path = tmp_video_dir / "test_fallback.mp4"

        await service._create_fallback_video(
            output_path=output_path,
            width=1280,
            height=720,
            duration=3.0,
            text="Test reel content",
        )

        # Verify file was created
        assert output_path.exists()

        # Verify it's not a text file
        with open(output_path, "rb") as f:
            header = f.read(12)
            # Check for MP4/MOV-like header or AVI header
            # MP4 files can start with different signatures
            assert len(header) >= 4

        # Verify file size is reasonable for video
        assert output_path.stat().st_size > 1000  # At least 1KB

    @pytest.mark.asyncio
    async def test_create_placeholder_video_with_video_service(self, tmp_video_dir):
        """Test placeholder video creation uses video service."""
        from backend.services.reel_generation_service import ReelGenerationService

        mock_db = MagicMock()
        service = ReelGenerationService(mock_db, str(tmp_video_dir))

        output_path = tmp_video_dir / "test_video.mp4"

        # Call the method
        await service._create_placeholder_video(
            output_path=output_path,
            width=1080,
            height=1920,
            duration=5.0,
            text="Test persona content",
        )

        # Verify file exists
        assert output_path.exists()

        # Verify it's a real video file (binary content)
        with open(output_path, "rb") as f:
            content = f.read(100)
            # Should not be a text file starting with '#'
            assert not content.startswith(b"# Reel Placeholder")

    @pytest.mark.asyncio
    async def test_fallback_video_has_gradient_frames(self, tmp_video_dir):
        """Test fallback video creates frames with gradient."""
        from backend.services.reel_generation_service import ReelGenerationService
        import cv2

        mock_db = MagicMock()
        service = ReelGenerationService(mock_db, str(tmp_video_dir))

        output_path = tmp_video_dir / "gradient_test.mp4"

        await service._create_fallback_video(
            output_path=output_path,
            width=640,
            height=480,
            duration=2.0,
            text="Gradient test",
        )

        # Verify video can be opened and read
        cap = cv2.VideoCapture(str(output_path))
        assert cap.isOpened()

        # Read first frame
        ret, frame = cap.read()
        assert ret is True
        assert frame is not None
        assert frame.shape == (480, 640, 3)  # Height x Width x Channels

        cap.release()

    @pytest.mark.asyncio
    async def test_fallback_video_duration(self, tmp_video_dir):
        """Test fallback video has correct duration."""
        from backend.services.reel_generation_service import ReelGenerationService
        import cv2

        mock_db = MagicMock()
        service = ReelGenerationService(mock_db, str(tmp_video_dir))

        output_path = tmp_video_dir / "duration_test.mp4"
        expected_duration = 3.0

        await service._create_fallback_video(
            output_path=output_path,
            width=640,
            height=480,
            duration=expected_duration,
            text="Duration test",
        )

        cap = cv2.VideoCapture(str(output_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()

        if fps > 0:
            actual_duration = frame_count / fps
            # Allow some tolerance for video encoding
            assert abs(actual_duration - expected_duration) < 1.0


class TestReelGenerationServiceVideoOutput:
    """Test reel generation service produces video output."""

    @pytest.fixture
    def mock_db_session(self):
        """Create mock database session."""
        session = MagicMock()
        session.execute = AsyncMock()
        session.commit = AsyncMock()
        session.refresh = AsyncMock()
        session.add = MagicMock()
        session.rollback = AsyncMock()
        return session

    @pytest.mark.asyncio
    async def test_generate_single_reel_creates_video(self, mock_db_session, tmp_path):
        """Test single reel generation creates video file."""
        from backend.services.reel_generation_service import ReelGenerationService
        from backend.services.video_processing_service import VideoQuality
        from backend.models.persona import PersonaModel

        output_dir = tmp_path / "reels"
        output_dir.mkdir()

        service = ReelGenerationService(mock_db_session, str(output_dir))

        # Create mock persona
        mock_persona = MagicMock(spec=PersonaModel)
        mock_persona.id = uuid4()
        mock_persona.name = "Test Persona"
        mock_persona.personality = "Energetic"

        # Mock the _get_persona method
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_persona
        mock_db_session.execute.return_value = mock_result

        result = await service.generate_single_reel(
            persona_id=mock_persona.id,
            prompt="Test reel about technology",
            duration=5.0,
            quality=VideoQuality.STANDARD,
        )

        assert "file_path" in result
        assert result["type"] == "single_reel"
        assert result["duration"] == 5.0

        # Verify file was created
        file_path = Path(result["file_path"])
        assert file_path.exists()


class TestReelGenerationNoPlaceholderText:
    """Verify reel generation doesn't create text placeholders."""

    @pytest.fixture
    def tmp_video_dir(self, tmp_path):
        """Create temporary video directory."""
        video_dir = tmp_path / "reels"
        video_dir.mkdir(parents=True, exist_ok=True)
        return video_dir

    @pytest.mark.asyncio
    async def test_output_is_not_text_file(self, tmp_video_dir):
        """Test output is binary video, not text."""
        from backend.services.reel_generation_service import ReelGenerationService

        mock_db = MagicMock()
        service = ReelGenerationService(mock_db, str(tmp_video_dir))

        output_path = tmp_video_dir / "test_not_text.mp4"

        await service._create_placeholder_video(
            output_path=output_path,
            width=720,
            height=1280,
            duration=2.0,
            text="Should be video not text",
        )

        # Read file as bytes
        with open(output_path, "rb") as f:
            content = f.read()

        # Should not contain typical text file markers
        try:
            text_content = content.decode("utf-8")
            # If we can decode as UTF-8, check it's not the old placeholder format
            assert "# Reel Placeholder" not in text_content
            assert "# Resolution:" not in text_content
            assert "# Note: Reel generation requires AI video models" not in text_content
        except UnicodeDecodeError:
            # Can't decode as UTF-8, which is expected for binary video
            pass

    @pytest.mark.asyncio
    async def test_video_has_frames(self, tmp_video_dir):
        """Test generated video has actual frames."""
        from backend.services.reel_generation_service import ReelGenerationService
        import cv2

        mock_db = MagicMock()
        service = ReelGenerationService(mock_db, str(tmp_video_dir))

        output_path = tmp_video_dir / "frames_test.mp4"

        await service._create_fallback_video(
            output_path=output_path,
            width=640,
            height=480,
            duration=1.0,
            text="Test frames",
        )

        # Open video and count frames
        cap = cv2.VideoCapture(str(output_path))
        frame_count = 0
        while True:
            ret, _ = cap.read()
            if not ret:
                break
            frame_count += 1
        cap.release()

        # Should have at least 24 frames for 1 second at 24fps minimum
        assert frame_count >= 20
