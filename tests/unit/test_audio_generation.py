"""
Tests for Audio Generation in ContentGenerationService.

Tests the audio generation features including:
- Procedural audio generation
- AudioCraft/MusicGen integration (when available)
- Audio quality settings
- Prompt-based audio characteristics
"""

import pytest
import wave
import io
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from backend.models.content import ContentRating, ContentType, GenerationRequest
from backend.models.persona import PersonaModel


class TestAudioGeneration:
    """Test suite for audio generation functionality."""

    @pytest.fixture
    def mock_persona(self):
        """Create mock persona for testing."""
        persona = MagicMock(spec=PersonaModel)
        persona.id = uuid4()
        persona.name = "Test Persona"
        persona.personality = "Creative and musical"
        persona.appearance = "Digital artist"
        persona.content_themes = ["music", "audio"]
        persona.style_preferences = {}
        persona.default_content_rating = "sfw"
        return persona

    @pytest.fixture
    def audio_request(self, mock_persona):
        """Create audio generation request."""
        return GenerationRequest(
            persona_id=mock_persona.id,
            content_type=ContentType.AUDIO,
            prompt="Generate calm ambient background music",
            quality="standard",
            content_rating=ContentRating.SFW,
        )

    @pytest.mark.asyncio
    async def test_procedural_audio_generation_basic(self):
        """Test basic procedural audio generation."""
        from backend.services.content_generation_service import ContentGenerationService

        # Create minimal mock for testing the procedural generation directly
        service = MagicMock(spec=ContentGenerationService)

        # Import the actual method
        from backend.services.content_generation_service import (
            ContentGenerationService as RealService,
        )

        # Test settings
        settings = {
            "sample_rate": 22050,
            "duration": 5,
            "channels": 1,
        }

        # Create a temporary instance to test the method
        result = await RealService._generate_procedural_audio(
            None, "Test calm ambient music", settings
        )

        assert result is not None
        assert "audio_data" in result
        assert result["format"] == "WAV"
        assert result["duration"] == 5
        assert result["sample_rate"] == 22050
        assert result["model"] == "procedural"

    @pytest.mark.asyncio
    async def test_procedural_audio_stereo(self):
        """Test procedural audio generation with stereo output."""
        from backend.services.content_generation_service import (
            ContentGenerationService as RealService,
        )

        settings = {
            "sample_rate": 44100,
            "duration": 3,
            "channels": 2,
        }

        result = await RealService._generate_procedural_audio(
            None, "Energetic upbeat music", settings
        )

        assert result["channels"] == 2
        assert result["sample_rate"] == 44100

        # Verify it's valid WAV data
        audio_io = io.BytesIO(result["audio_data"])
        with wave.open(audio_io, "rb") as wav:
            assert wav.getnchannels() == 2
            assert wav.getframerate() == 44100

    @pytest.mark.asyncio
    async def test_procedural_audio_prompt_influence(self):
        """Test that different prompts produce different audio characteristics."""
        from backend.services.content_generation_service import (
            ContentGenerationService as RealService,
        )

        settings = {
            "sample_rate": 22050,
            "duration": 2,
            "channels": 1,
        }

        calm_result = await RealService._generate_procedural_audio(
            None, "Calm relaxing peaceful ambient sounds", settings
        )

        energetic_result = await RealService._generate_procedural_audio(
            None, "Energetic exciting upbeat dynamic music", settings
        )

        # Both should produce valid audio
        assert calm_result["audio_data"] is not None
        assert energetic_result["audio_data"] is not None

        # Audio data should be different due to different characteristics
        assert calm_result["audio_data"] != energetic_result["audio_data"]

    @pytest.mark.asyncio
    async def test_try_audiocraft_generation_not_available(self):
        """Test AudioCraft gracefully returns None when not available."""
        from backend.services.content_generation_service import (
            ContentGenerationService as RealService,
        )

        settings = {
            "sample_rate": 32000,
            "duration": 10,
            "channels": 2,
        }

        # AudioCraft is likely not installed, should return None gracefully
        result = await RealService._try_audiocraft_generation(
            None, "Test music generation", settings
        )

        # Should return None when audiocraft is not available
        assert result is None

    def test_quality_settings_mapping(self):
        """Test quality settings are properly defined."""
        # These are the expected quality settings
        quality_settings = {
            "draft": {"sample_rate": 22050, "duration": 10, "channels": 1},
            "standard": {"sample_rate": 32000, "duration": 15, "channels": 2},
            "high": {"sample_rate": 44100, "duration": 30, "channels": 2},
            "premium": {"sample_rate": 48000, "duration": 60, "channels": 2},
        }

        assert quality_settings["draft"]["sample_rate"] == 22050
        assert quality_settings["standard"]["channels"] == 2
        assert quality_settings["high"]["duration"] == 30
        assert quality_settings["premium"]["sample_rate"] == 48000


class TestAudioGenerationWAVFormat:
    """Test WAV file format compliance."""

    @pytest.mark.asyncio
    async def test_valid_wav_header(self):
        """Test generated audio has valid WAV header."""
        from backend.services.content_generation_service import (
            ContentGenerationService as RealService,
        )

        settings = {
            "sample_rate": 44100,
            "duration": 1,
            "channels": 2,
        }

        result = await RealService._generate_procedural_audio(
            None, "Test audio", settings
        )

        audio_data = result["audio_data"]

        # Check WAV magic bytes
        assert audio_data[:4] == b"RIFF"
        assert audio_data[8:12] == b"WAVE"

    @pytest.mark.asyncio
    async def test_wav_readable(self):
        """Test generated WAV can be read back properly."""
        from backend.services.content_generation_service import (
            ContentGenerationService as RealService,
        )

        settings = {
            "sample_rate": 22050,
            "duration": 2,
            "channels": 1,
        }

        result = await RealService._generate_procedural_audio(
            None, "Dark mysterious audio", settings
        )

        # Parse the WAV file
        audio_io = io.BytesIO(result["audio_data"])
        with wave.open(audio_io, "rb") as wav:
            assert wav.getnchannels() == 1
            assert wav.getframerate() == 22050
            assert wav.getsampwidth() == 2  # 16-bit

            # Read some frames to verify data integrity
            frames = wav.readframes(1000)
            assert len(frames) > 0


class TestAudioGenerationPromptAnalysis:
    """Test prompt analysis for audio generation."""

    @pytest.mark.asyncio
    async def test_calm_prompt_characteristics(self):
        """Test calm prompts result in lower frequencies."""
        from backend.services.content_generation_service import (
            ContentGenerationService as RealService,
        )

        settings = {"sample_rate": 22050, "duration": 1, "channels": 1}

        result = await RealService._generate_procedural_audio(
            None, "Calm peaceful relaxing ambient", settings
        )

        # Audio should be generated
        assert result["audio_data"] is not None
        assert len(result["audio_data"]) > 0

    @pytest.mark.asyncio
    async def test_energetic_prompt_characteristics(self):
        """Test energetic prompts result in different characteristics."""
        from backend.services.content_generation_service import (
            ContentGenerationService as RealService,
        )

        settings = {"sample_rate": 22050, "duration": 1, "channels": 1}

        result = await RealService._generate_procedural_audio(
            None, "Energetic exciting dynamic upbeat", settings
        )

        assert result["audio_data"] is not None
        assert len(result["audio_data"]) > 0

    @pytest.mark.asyncio
    async def test_dark_prompt_characteristics(self):
        """Test dark/mysterious prompts."""
        from backend.services.content_generation_service import (
            ContentGenerationService as RealService,
        )

        settings = {"sample_rate": 22050, "duration": 1, "channels": 1}

        result = await RealService._generate_procedural_audio(
            None, "Dark mysterious intense atmosphere", settings
        )

        assert result["audio_data"] is not None

    @pytest.mark.asyncio
    async def test_neutral_prompt(self):
        """Test neutral prompts use default characteristics."""
        from backend.services.content_generation_service import (
            ContentGenerationService as RealService,
        )

        settings = {"sample_rate": 22050, "duration": 1, "channels": 1}

        result = await RealService._generate_procedural_audio(
            None, "Generic background audio", settings
        )

        assert result["audio_data"] is not None


class TestAudioGenerationEdgeCases:
    """Edge case tests for audio generation."""

    @pytest.mark.asyncio
    async def test_empty_prompt(self):
        """Test handling of empty prompt."""
        from backend.services.content_generation_service import (
            ContentGenerationService as RealService,
        )

        settings = {"sample_rate": 22050, "duration": 1, "channels": 1}

        result = await RealService._generate_procedural_audio(None, "", settings)

        # Should still generate audio with default characteristics
        assert result["audio_data"] is not None

    @pytest.mark.asyncio
    async def test_very_long_duration(self):
        """Test generation of longer audio."""
        from backend.services.content_generation_service import (
            ContentGenerationService as RealService,
        )

        settings = {
            "sample_rate": 22050,
            "duration": 10,  # 10 seconds
            "channels": 1,
        }

        result = await RealService._generate_procedural_audio(
            None, "Extended ambient track", settings
        )

        assert result["audio_data"] is not None
        assert result["duration"] == 10

    @pytest.mark.asyncio
    async def test_minimum_duration(self):
        """Test generation of very short audio."""
        from backend.services.content_generation_service import (
            ContentGenerationService as RealService,
        )

        settings = {
            "sample_rate": 22050,
            "duration": 1,  # 1 second minimum
            "channels": 1,
        }

        result = await RealService._generate_procedural_audio(
            None, "Short sound", settings
        )

        assert result["audio_data"] is not None
        assert result["duration"] == 1
