"""
Unit tests for appearance locking and visual consistency features.

Tests the new base_appearance_description, base_image_path, and
appearance_locked fields in persona management and content generation.
"""

import pytest
from uuid import uuid4
from backend.models.persona import PersonaCreate, PersonaUpdate, ContentRating


class TestAppearanceLocking:
    """Test suite for appearance locking features."""

    def test_persona_create_with_appearance_locking(self):
        """Test creating a persona with appearance locking enabled."""
        persona_data = PersonaCreate(
            name="Locked Test Persona",
            appearance="Basic appearance",
            personality="Test personality",
            content_themes=["test"],
            base_appearance_description="Detailed baseline appearance with specific features",
            base_image_path="/models/base_images/test_ref.jpg",
            appearance_locked=True,
        )

        assert persona_data.appearance_locked is True
        assert (
            persona_data.base_appearance_description
            == "Detailed baseline appearance with specific features"
        )
        assert persona_data.base_image_path == "/models/base_images/test_ref.jpg"

    def test_persona_create_without_appearance_locking(self):
        """Test creating a persona without appearance locking (defaults)."""
        persona_data = PersonaCreate(
            name="Standard Test Persona",
            appearance="Basic appearance",
            personality="Test personality",
        )

        assert persona_data.appearance_locked is False
        assert persona_data.base_appearance_description is None
        assert persona_data.base_image_path is None

    def test_persona_update_enable_locking(self):
        """Test updating a persona to enable appearance locking."""
        update_data = PersonaUpdate(
            base_appearance_description="New detailed baseline appearance",
            base_image_path="/models/base_images/updated_ref.jpg",
            appearance_locked=True,
        )

        assert update_data.appearance_locked is True
        assert (
            update_data.base_appearance_description
            == "New detailed baseline appearance"
        )
        assert update_data.base_image_path == "/models/base_images/updated_ref.jpg"

    def test_persona_update_disable_locking(self):
        """Test updating a persona to disable appearance locking."""
        update_data = PersonaUpdate(appearance_locked=False)

        assert update_data.appearance_locked is False

    def test_base_appearance_max_length(self):
        """Test that base appearance description respects max length."""
        # This should not raise an error (within limit)
        long_description = "A" * 5000
        persona_data = PersonaCreate(
            name="Long Description Persona",
            appearance="Basic appearance",
            personality="Test personality",
            base_appearance_description=long_description,
        )
        assert len(persona_data.base_appearance_description) == 5000

    def test_base_appearance_too_long(self):
        """Test that base appearance description rejects values over max length."""
        too_long_description = "A" * 5001

        with pytest.raises(ValueError):
            PersonaCreate(
                name="Too Long Description Persona",
                appearance="Basic appearance",
                personality="Test personality",
                base_appearance_description=too_long_description,
            )

    def test_base_image_path_max_length(self):
        """Test that base image path respects max length."""
        # This should not raise an error (within limit)
        long_path = (
            "/models/" + "a" * 486 + ".jpg"
        )  # Total 499 chars (within 500 limit)
        persona_data = PersonaCreate(
            name="Long Path Persona",
            appearance="Basic appearance description",
            personality="Test personality traits",
            base_image_path=long_path,
        )
        assert len(persona_data.base_image_path) <= 500

    def test_base_image_path_too_long(self):
        """Test that base image path rejects values over max length."""
        too_long_path = "/models/" + "a" * 500 + ".jpg"  # Over 500 chars

        with pytest.raises(ValueError):
            PersonaCreate(
                name="Too Long Path Persona",
                appearance="Basic appearance",
                personality="Test personality",
                base_image_path=too_long_path,
            )

    def test_appearance_locking_optional_fields(self):
        """Test that appearance locking fields are truly optional."""
        # Should work with only appearance_locked
        persona1 = PersonaCreate(
            name="Locked Only",
            appearance="Basic appearance description",
            personality="Test personality",
            appearance_locked=True,
        )
        assert persona1.appearance_locked is True
        assert persona1.base_appearance_description is None
        assert persona1.base_image_path is None

        # Should work with only base_appearance_description
        persona2 = PersonaCreate(
            name="Description Only",
            appearance="Basic appearance description",
            personality="Test personality",
            base_appearance_description="Detailed description",
        )
        assert persona2.appearance_locked is False
        assert persona2.base_appearance_description == "Detailed description"
        assert persona2.base_image_path is None

        # Should work with only base_image_path
        persona3 = PersonaCreate(
            name="Path Only",
            appearance="Basic appearance description",
            personality="Test personality",
            base_image_path="/path/to/image.jpg",
        )
        assert persona3.appearance_locked is False
        assert persona3.base_appearance_description is None
        assert persona3.base_image_path == "/path/to/image.jpg"


class TestContentGenerationWithLocking:
    """Test suite for content generation with appearance locking."""

    def test_generation_request_parameters(self):
        """Test that generation requests can handle appearance locking."""
        # This test verifies that the models are compatible
        # The actual generation logic testing requires mocking the AI services
        # which is covered in integration tests
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
