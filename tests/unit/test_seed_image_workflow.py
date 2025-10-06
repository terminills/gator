"""
Test Seed Image Generation Workflow

Tests the new seed image upload, generation, and approval functionality.
"""

import pytest
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from pathlib import Path
import tempfile
import os

from backend.services.persona_service import PersonaService
from backend.models.persona import PersonaCreate, PersonaUpdate, BaseImageStatus
from backend.services.ai_models import AIModelManager


class TestSeedImageWorkflow:
    """Test suite for seed image generation workflow."""

    @pytest.fixture
    def sample_persona_data(self):
        """Create sample persona data for testing."""
        return PersonaCreate(
            name="Test AI Persona",
            appearance="Young woman with dark hair, professional style",
            personality="Friendly, intelligent, tech-savvy",
            content_themes=["technology", "AI"],
            style_preferences={"style": "professional"},
            base_appearance_description="Detailed appearance: professional woman in her late 20s with shoulder-length dark hair, modern business casual attire, warm smile",
        )

    async def test_create_persona_with_base_image_status(
        self, db_session, sample_persona_data
    ):
        """Test persona creation includes base_image_status field."""
        service = PersonaService(db_session)

        result = await service.create_persona(sample_persona_data)

        assert result.base_image_status == "pending_upload"
        assert result.base_image_path is None
        assert result.appearance_locked is False

    async def test_approve_baseline_without_image_fails(
        self, db_session, sample_persona_data
    ):
        """Test that approving baseline without an image raises error."""
        service = PersonaService(db_session)

        # Create persona without base image
        persona = await service.create_persona(sample_persona_data)

        # Try to approve - should fail
        with pytest.raises(ValueError, match="does not have a base image"):
            await service.approve_baseline(persona.id)

    async def test_approve_baseline_with_image_success(
        self, db_session, sample_persona_data
    ):
        """Test successful baseline approval with existing image."""
        service = PersonaService(db_session)

        # Create persona
        persona = await service.create_persona(sample_persona_data)

        # Update with base image path
        updates = PersonaUpdate(
            base_image_path="/opt/gator/data/models/base_images/test.png",
            base_image_status=BaseImageStatus.DRAFT,
        )
        persona = await service.update_persona(persona.id, updates)

        # Approve baseline
        approved = await service.approve_baseline(persona.id)

        assert approved is not None
        assert approved.base_image_status == "approved"
        assert approved.appearance_locked is True

    async def test_save_image_to_disk(self, db_session, sample_persona_data):
        """Test saving image data to disk."""
        service = PersonaService(db_session)

        # Create persona
        persona = await service.create_persona(sample_persona_data)

        # Create temporary directory for test
        with tempfile.TemporaryDirectory() as tmpdir:
            # Mock the base_images_dir to use temp directory
            with patch("backend.services.persona_service.Path") as mock_path:
                mock_path.return_value = Path(tmpdir)

                # Test image data
                image_data = b"fake_image_data_for_testing"

                # Save image
                result_path = await service._save_image_to_disk(
                    persona_id=persona.id, image_data=image_data
                )

                # Verify path was returned
                assert result_path is not None
                assert isinstance(result_path, str)

    async def test_update_persona_with_base_image_status(
        self, db_session, sample_persona_data
    ):
        """Test updating persona with base_image_status."""
        service = PersonaService(db_session)

        # Create persona
        persona = await service.create_persona(sample_persona_data)
        assert persona.base_image_status == "pending_upload"

        # Update to DRAFT status
        updates = PersonaUpdate(base_image_status=BaseImageStatus.DRAFT)
        updated = await service.update_persona(persona.id, updates)

        assert updated.base_image_status == "draft"

        # Update to APPROVED status
        updates = PersonaUpdate(base_image_status=BaseImageStatus.APPROVED)
        updated = await service.update_persona(persona.id, updates)

        assert updated.base_image_status == "approved"

    async def test_update_persona_with_base_image_path(
        self, db_session, sample_persona_data
    ):
        """Test updating persona with base_image_path."""
        service = PersonaService(db_session)

        # Create persona
        persona = await service.create_persona(sample_persona_data)
        assert persona.base_image_path is None

        # Update with image path
        test_path = "/opt/gator/data/models/base_images/test_persona.png"
        updates = PersonaUpdate(base_image_path=test_path)
        updated = await service.update_persona(persona.id, updates)

        assert updated.base_image_path == test_path


class TestAIModelManagerReferenceGeneration:
    """Test suite for AI model manager reference image generation."""

    @pytest.fixture
    def ai_manager(self):
        """Create AI model manager instance."""
        manager = AIModelManager()
        return manager

    @pytest.mark.asyncio
    async def test_generate_reference_image_openai_structure(self, ai_manager):
        """Test that _generate_reference_image_openai has correct structure."""
        # Verify method exists
        assert hasattr(ai_manager, "_generate_reference_image_openai")

        # Mock the HTTP client
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [{"url": "https://example.com/image.png"}]
        }

        mock_image_response = Mock()
        mock_image_response.content = b"fake_image_data"

        ai_manager.http_client = AsyncMock()
        ai_manager.http_client.post.return_value = mock_response
        ai_manager.http_client.get.return_value = mock_image_response

        # Mock the settings to have OpenAI API key
        ai_manager.settings.openai_api_key = "test_key"

        # Test generation
        result = await ai_manager._generate_reference_image_openai(
            appearance_prompt="Professional woman with dark hair",
            personality_context="friendly and professional",
        )

        # Verify result structure
        assert "image_data" in result
        assert "format" in result
        assert "model" in result
        assert result["model"] == "dall-e-3"
        assert result["provider"] == "openai"

    @pytest.mark.asyncio
    async def test_generate_reference_image_local_structure(self, ai_manager):
        """Test that _generate_reference_image_local has correct structure."""
        # Verify method exists
        assert hasattr(ai_manager, "_generate_reference_image_local")

        # Mock the _generate_image_diffusers method
        mock_result = {
            "image_data": b"fake_image_data",
            "format": "PNG",
            "width": 1024,
            "height": 1024,
            "model": "stable-diffusion-xl",
            "provider": "local",
        }

        ai_manager._generate_image_diffusers = AsyncMock(return_value=mock_result)
        ai_manager._get_best_local_image_model = Mock(
            return_value={"name": "stable-diffusion-xl", "provider": "local"}
        )

        # Test generation
        result = await ai_manager._generate_reference_image_local(
            appearance_prompt="Professional woman with dark hair",
            personality_context="friendly and professional",
        )

        # Verify result structure
        assert "image_data" in result
        assert "format" in result

    def test_get_best_local_image_model(self, ai_manager):
        """Test _get_best_local_image_model selection logic."""
        # Setup mock models
        ai_manager.available_models = {
            "image": [
                {"name": "stable-diffusion-xl", "provider": "local", "loaded": True},
                {"name": "stable-diffusion-v1.5", "provider": "local", "loaded": True},
            ]
        }

        # Test it prefers SDXL models
        result = ai_manager._get_best_local_image_model()
        assert "xl" in result["name"].lower()

    def test_get_best_local_image_model_no_models_raises(self, ai_manager):
        """Test _get_best_local_image_model raises when no models available."""
        # Setup empty models
        ai_manager.available_models = {"image": []}

        # Should raise exception
        with pytest.raises(Exception, match="No local image models available"):
            ai_manager._get_best_local_image_model()


class TestBaseImageStatusEnum:
    """Test suite for BaseImageStatus enum."""

    def test_base_image_status_values(self):
        """Test BaseImageStatus enum has correct values."""
        assert BaseImageStatus.PENDING_UPLOAD.value == "pending_upload"
        assert BaseImageStatus.DRAFT.value == "draft"
        assert BaseImageStatus.APPROVED.value == "approved"
        assert BaseImageStatus.REJECTED.value == "rejected"

    def test_base_image_status_string_conversion(self):
        """Test BaseImageStatus enum can be compared as strings."""
        status = BaseImageStatus.DRAFT
        assert status.value == "draft"
        assert str(status.value) == "draft"
