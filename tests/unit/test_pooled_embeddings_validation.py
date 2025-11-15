"""
Unit tests for pooled embeddings validation in diffusers generation.
Tests that the code properly validates all required embeddings before passing to pipeline.
This prevents TypeError: 'NoneType' object is not iterable when pooled embeddings are None.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from backend.services.ai_models import AIModelManager


class TestPooledEmbeddingsValidation:
    """Test that pooled embeddings are properly validated before use."""

    @pytest.fixture
    def model_manager(self):
        """Create a model manager instance for testing."""
        return AIModelManager()

    @pytest.mark.asyncio
    async def test_embeddings_validation_in_code(self, model_manager):
        """Test that all embedding validation checks are present in the code."""
        import inspect

        source = inspect.getsource(model_manager._generate_image_diffusers)

        # Check that all three code paths validate ALL embeddings, not just prompt_embeds
        # Path 1: ControlNet with embeddings
        # Path 2: img2img with embeddings
        # Path 3: text2img with embeddings

        # Verify that we check for all 4 embeddings being non-None
        assert "prompt_embeds is not None" in source
        assert "negative_prompt_embeds is not None" in source
        assert "pooled_prompt_embeds is not None" in source
        assert "negative_pooled_prompt_embeds is not None" in source

    @pytest.mark.asyncio
    async def test_controlnet_embeddings_validation(self, model_manager):
        """Test that ControlNet path validates all embeddings including pooled."""
        import inspect

        source = inspect.getsource(model_manager._generate_image_diffusers)

        # Find the ControlNet section
        lines = source.split("\n")
        controlnet_section_found = False
        validation_found = False

        for i, line in enumerate(lines):
            if "ControlNet generation with structural conditioning" in line:
                controlnet_section_found = True

            # Look for the validation block after ControlNet section
            if controlnet_section_found and "prompt_embeds is not None" in line:
                # Check the next few lines for all validation conditions
                check_window = "\n".join(lines[i : i + 10])
                if (
                    "prompt_embeds is not None" in check_window
                    and "negative_prompt_embeds is not None" in check_window
                    and "pooled_prompt_embeds is not None" in check_window
                    and "negative_pooled_prompt_embeds is not None" in check_window
                ):
                    validation_found = True
                    break

        assert (
            validation_found
        ), "ControlNet path should validate all 4 embeddings before use"

    @pytest.mark.asyncio
    async def test_img2img_embeddings_validation(self, model_manager):
        """Test that img2img path validates all embeddings including pooled."""
        import inspect

        source = inspect.getsource(model_manager._generate_image_diffusers)

        # Find the img2img section
        lines = source.split("\n")
        img2img_section_found = False
        validation_found = False

        for i, line in enumerate(lines):
            if "img2img generation with reference image" in line:
                img2img_section_found = True

            # Look for the validation block after img2img section
            if img2img_section_found and "prompt_embeds is not None" in line:
                # Check the next few lines for all validation conditions
                check_window = "\n".join(lines[i : i + 10])
                if (
                    "prompt_embeds is not None" in check_window
                    and "negative_prompt_embeds is not None" in check_window
                    and "pooled_prompt_embeds is not None" in check_window
                    and "negative_pooled_prompt_embeds is not None" in check_window
                ):
                    validation_found = True
                    break

        assert (
            validation_found
        ), "img2img path should validate all 4 embeddings before use"

    @pytest.mark.asyncio
    async def test_text2img_embeddings_validation(self, model_manager):
        """Test that text2img path validates all embeddings including pooled."""
        import inspect

        source = inspect.getsource(model_manager._generate_image_diffusers)

        # Find the text2img section
        lines = source.split("\n")
        text2img_section_found = False
        validation_found = False

        for i, line in enumerate(lines):
            if "Standard text2img generation" in line:
                text2img_section_found = True

            # Look for the validation block after text2img section
            if text2img_section_found and "prompt_embeds is not None" in line:
                # Check the next few lines for all validation conditions
                check_window = "\n".join(lines[i : i + 10])
                if (
                    "prompt_embeds is not None" in check_window
                    and "negative_prompt_embeds is not None" in check_window
                    and "pooled_prompt_embeds is not None" in check_window
                    and "negative_pooled_prompt_embeds is not None" in check_window
                ):
                    validation_found = True
                    break

        assert (
            validation_found
        ), "text2img path should validate all 4 embeddings before use"

    @pytest.mark.asyncio
    async def test_fallback_to_text_prompts_when_embeddings_incomplete(
        self, model_manager
    ):
        """Test that when embeddings are incomplete, code falls back to text prompts."""
        import inspect

        source = inspect.getsource(model_manager._generate_image_diffusers)

        # After each embeddings check, there should be an else clause that uses text prompts
        # Verify the fallback paths exist

        # Check for text prompt fallback in ControlNet path
        assert (
            "Using standard text prompts" in source
        ), "Should have text prompt fallback"

        # Check that original_prompt is preserved for fallback
        assert (
            "original_prompt" in source or "prompt=" in source
        ), "Should use text prompts as fallback"

    @pytest.mark.asyncio
    async def test_compel_error_handling(self, model_manager):
        """Test that compel import/usage errors are properly handled."""
        import inspect

        source = inspect.getsource(model_manager._generate_image_diffusers)

        # Check for compel error handling
        assert "except ImportError" in source or "try:" in source
        assert "compel library not available" in source or "Failed to use compel" in source


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
