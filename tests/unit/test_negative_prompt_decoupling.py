"""
Tests for the negative prompt decoupling feature.

This module tests the persona-specific negative prompt functionality,
which allows personas to override style-based negative prompt defaults.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock

from backend.models.persona import PersonaCreate, PersonaUpdate, PersonaResponse
from backend.services.ai_models import AIModelManager


class TestNegativePromptPersonaModels:
    """Tests for the default_negative_prompt field in Pydantic models."""

    def test_persona_create_default_negative_prompt(self):
        """Test that PersonaCreate includes default_negative_prompt with correct default."""
        persona = PersonaCreate(
            name="Test Persona",
            appearance="A young woman with dark hair",
            personality="Friendly and outgoing",
        )
        
        assert persona.default_negative_prompt == (
            "ugly, blurry, low quality, distorted, deformed, bad anatomy"
        )

    def test_persona_create_custom_negative_prompt(self):
        """Test that PersonaCreate accepts custom negative prompt."""
        custom_negative = "bright colors, happy, sunshine, realistic"
        persona = PersonaCreate(
            name="Goth Girl",
            appearance="A pale woman with black hair and dark makeup",
            personality="Moody and mysterious",
            default_negative_prompt=custom_negative,
        )
        
        assert persona.default_negative_prompt == custom_negative

    def test_persona_update_negative_prompt(self):
        """Test that PersonaUpdate can update negative prompt."""
        updates = PersonaUpdate(
            default_negative_prompt="cartoon, anime, 3d render"
        )
        
        assert updates.default_negative_prompt == "cartoon, anime, 3d render"

    def test_persona_update_negative_prompt_optional(self):
        """Test that PersonaUpdate doesn't require negative prompt."""
        updates = PersonaUpdate(name="New Name")
        
        assert updates.default_negative_prompt is None

    def test_persona_response_includes_negative_prompt(self):
        """Test that PersonaResponse includes default_negative_prompt."""
        from uuid import uuid4
        from datetime import datetime
        
        response = PersonaResponse(
            id=uuid4(),
            name="Test",
            appearance="test appearance description",
            personality="test personality description",
            content_themes=[],
            style_preferences={},
            default_content_rating="sfw",
            allowed_content_ratings=["sfw"],
            platform_restrictions={},
            is_active=True,
            generation_count=0,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            default_negative_prompt="custom negative prompt",
        )
        
        assert response.default_negative_prompt == "custom negative prompt"


class TestBuildStyleSpecificPrompt:
    """Tests for the _build_style_specific_prompt method with persona negative prompts."""

    @pytest.fixture
    def ai_manager(self):
        """Create an AIModelManager instance for testing."""
        manager = AIModelManager()
        return manager

    def test_style_default_negative_prompt(self, ai_manager):
        """Test that style-specific negative prompt is used when no persona prompt provided."""
        enhanced, negative = ai_manager._build_style_specific_prompt(
            base_prompt="A beautiful woman",
            image_style="photorealistic",
            use_long_prompt=True,
            persona_negative_prompt=None,
        )
        
        # Should use the photorealistic style's default negative prompt
        assert "cartoon" in negative
        assert "anime" in negative
        assert "3d render" in negative

    def test_anime_style_default_negative_prompt(self, ai_manager):
        """Test that anime style uses correct default negative prompt."""
        enhanced, negative = ai_manager._build_style_specific_prompt(
            base_prompt="An anime character",
            image_style="anime",
            use_long_prompt=True,
            persona_negative_prompt=None,
        )
        
        # Anime style's default negative should include "realistic" and "photorealistic"
        # but NOT "anime"
        assert "realistic" in negative
        assert "photorealistic" in negative
        # Note: The current implementation DOES have style-based defaults
        # which includes things that might conflict with the style itself

    def test_persona_negative_prompt_override(self, ai_manager):
        """Test that persona negative prompt completely overrides style default."""
        custom_negative = "bright colors, happy, daytime, sunshine"
        
        enhanced, negative = ai_manager._build_style_specific_prompt(
            base_prompt="A goth woman",
            image_style="photorealistic",
            use_long_prompt=True,
            persona_negative_prompt=custom_negative,
        )
        
        # Should use the persona's custom negative prompt exactly
        assert negative == custom_negative
        # Should NOT contain the style default elements
        assert "cartoon" not in negative
        assert "anime" not in negative

    def test_anime_persona_custom_negative(self, ai_manager):
        """Test that anime persona can use custom negative without 'anime' in it."""
        # This is the key use case: anime persona shouldn't have "anime" in negative
        anime_negative = "ugly, blurry, low quality, distorted, deformed, bad anatomy"
        
        enhanced, negative = ai_manager._build_style_specific_prompt(
            base_prompt="A cute anime girl",
            image_style="anime",
            use_long_prompt=True,
            persona_negative_prompt=anime_negative,
        )
        
        # Should use persona's negative prompt
        assert negative == anime_negative
        # Should NOT contain the photorealistic defaults that would harm anime
        assert "realistic" not in negative or negative == anime_negative
        
    def test_enhanced_prompt_not_affected_by_persona_negative(self, ai_manager):
        """Test that persona negative prompt doesn't affect the enhanced positive prompt."""
        enhanced1, _ = ai_manager._build_style_specific_prompt(
            base_prompt="A woman",
            image_style="photorealistic",
            use_long_prompt=True,
            persona_negative_prompt=None,
        )
        
        enhanced2, _ = ai_manager._build_style_specific_prompt(
            base_prompt="A woman",
            image_style="photorealistic",
            use_long_prompt=True,
            persona_negative_prompt="custom negative prompt",
        )
        
        # Enhanced prompts should be identical regardless of negative prompt
        assert enhanced1 == enhanced2

    def test_truncation_with_persona_negative(self, ai_manager):
        """Test that persona negative prompt is properly truncated for SD 1.5."""
        # Create a very long negative prompt
        long_negative = "ugly, " * 100  # Very long prompt
        
        enhanced, negative = ai_manager._build_style_specific_prompt(
            base_prompt="A woman",
            image_style="photorealistic",
            use_long_prompt=False,  # SD 1.5 mode, should truncate
            persona_negative_prompt=long_negative,
        )
        
        # Negative prompt should be truncated
        # Word count should be limited (roughly 75 tokens / 1.3 ≈ 57 words max)
        word_count = len(negative.split())
        assert word_count < 60  # Should be truncated


class TestNegativePromptIntegration:
    """Integration tests for negative prompt in image generation."""

    @pytest.fixture
    def ai_manager(self):
        """Create an AIModelManager instance for testing."""
        manager = AIModelManager()
        return manager

    def test_persona_negative_prompt_passed_through_kwargs(self, ai_manager):
        """Test that persona_negative_prompt is correctly extracted from kwargs."""
        # This tests the _generate_reference_image_local method's kwargs handling
        # We can't easily test the full generation flow, but we can verify
        # the method signature accepts the parameter
        
        # The method should accept persona_negative_prompt in kwargs
        # This is verified by checking that the kwargs handling extracts it
        import inspect
        
        # Get the source of _generate_reference_image_local
        source = inspect.getsource(ai_manager._generate_reference_image_local)
        
        # Verify the method extracts persona_negative_prompt from kwargs
        assert 'persona_negative_prompt' in source
        assert 'kwargs.get("persona_negative_prompt")' in source


class TestPersonaWithNegativePromptFlow:
    """End-to-end flow tests for persona negative prompt usage."""

    def test_anime_persona_workflow(self):
        """Test the workflow for creating an anime persona with custom negative."""
        # Create an anime persona with a negative prompt that doesn't include "anime"
        anime_persona = PersonaCreate(
            name="Sakura Chan",
            appearance="A cute anime girl with pink hair and big eyes",
            personality="Cheerful and energetic",
            image_style="anime",
            default_negative_prompt=(
                "ugly, blurry, low quality, distorted, deformed, "
                "bad anatomy, western cartoon, 3d render"
            ),
        )
        
        # Verify the negative prompt doesn't contain "anime"
        assert "anime" not in anime_persona.default_negative_prompt.lower()
        # But it should contain quality-related negatives
        assert "ugly" in anime_persona.default_negative_prompt.lower()
        assert "blurry" in anime_persona.default_negative_prompt.lower()

    def test_photorealistic_persona_workflow(self):
        """Test the workflow for creating a photorealistic persona."""
        photo_persona = PersonaCreate(
            name="Emma Professional",
            appearance="A professional woman in business attire",
            personality="Confident and professional",
            image_style="photorealistic",
            default_negative_prompt=(
                "cartoon, anime, 3d render, illustration, painting, "
                "drawing, art, sketched, ugly, blurry, low quality"
            ),
        )
        
        # Verify the negative prompt contains style-appropriate negatives
        assert "cartoon" in photo_persona.default_negative_prompt.lower()
        assert "anime" in photo_persona.default_negative_prompt.lower()

    def test_goth_persona_custom_negative(self):
        """Test a goth persona with style-specific negative prompt."""
        goth_persona = PersonaCreate(
            name="Raven Blackwood",
            appearance="A pale woman with black hair, dark makeup, gothic clothing",
            personality="Mysterious, introspective, poetic",
            image_style="cinematic",
            default_negative_prompt=(
                "bright colors, happy expression, sunshine, colorful, "
                "cheerful, pastel colors, ugly, blurry, low quality"
            ),
        )
        
        # Goth persona should avoid bright/happy elements
        assert "bright colors" in goth_persona.default_negative_prompt.lower()
        assert "happy" in goth_persona.default_negative_prompt.lower()
        assert "sunshine" in goth_persona.default_negative_prompt.lower()
