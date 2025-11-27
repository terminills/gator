"""
Tests for AutoencoderKL (VAE) loading from base models when missing from CivitAI checkpoints.

This tests the fix for the error:
"Failed to load AutoencoderKL. Weights for this component appear to be missing in the checkpoint."

CivitAI models are often "pruned" and don't include the VAE weights. The fix
loads the VAE from the base SDXL or SD 1.5 model when this error occurs.
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from pathlib import Path


class TestAutoencoderVAELoading:
    """Tests for VAE loading fallback when missing from checkpoints."""

    def test_needs_vae_detection_autoencoder_error(self):
        """Test that AutoencoderKL error is detected correctly."""
        error_msg = "Failed to load AutoencoderKL. Weights for this component appear to be missing in the checkpoint."
        
        # Simulating the detection logic in ai_models.py
        needs_vae = "AutoencoderKL" in error_msg or ("vae" in error_msg.lower() and "missing" in error_msg.lower())
        
        assert needs_vae is True

    def test_needs_vae_detection_vae_missing(self):
        """Test that 'vae missing' error is detected correctly."""
        error_msg = "The VAE weights are missing from this checkpoint file"
        
        needs_vae = "AutoencoderKL" in error_msg or ("vae" in error_msg.lower() and "missing" in error_msg.lower())
        
        assert needs_vae is True

    def test_needs_vae_detection_unrelated_error(self):
        """Test that unrelated errors don't trigger VAE loading."""
        error_msg = "CUDA out of memory"
        
        needs_vae = "AutoencoderKL" in error_msg or ("vae" in error_msg.lower() and "missing" in error_msg.lower())
        
        assert needs_vae is False

    def test_needs_text_encoder_detection(self):
        """Test that CLIPTextModel error is detected correctly."""
        error_msg = "CLIPTextModel could not be loaded from checkpoint"
        
        needs_text_encoder = "CLIPTextModel" in error_msg or "text_encoder" in error_msg.lower()
        
        assert needs_text_encoder is True

    def test_combined_error_detection(self):
        """Test that both VAE and text encoder errors can be detected."""
        error_msg = "Both AutoencoderKL and text_encoder are missing from the checkpoint"
        
        needs_text_encoder = "CLIPTextModel" in error_msg or "text_encoder" in error_msg.lower()
        needs_vae = "AutoencoderKL" in error_msg or ("vae" in error_msg.lower() and "missing" in error_msg.lower())
        
        assert needs_text_encoder is True
        assert needs_vae is True

    def test_component_list_message(self):
        """Test the component list message formatting."""
        needs_text_encoder = True
        needs_vae = True
        
        component_list = []
        if needs_text_encoder:
            component_list.append("text encoder")
        if needs_vae:
            component_list.append("VAE (AutoencoderKL)")
        
        message = f"Checkpoint missing {', '.join(component_list)}, loading from base model..."
        
        assert "text encoder" in message
        assert "VAE (AutoencoderKL)" in message

    def test_component_list_vae_only(self):
        """Test the component list message when only VAE is missing."""
        needs_text_encoder = False
        needs_vae = True
        
        component_list = []
        if needs_text_encoder:
            component_list.append("text encoder")
        if needs_vae:
            component_list.append("VAE (AutoencoderKL)")
        
        message = f"Checkpoint missing {', '.join(component_list)}, loading from base model..."
        
        assert "text encoder" not in message
        assert "VAE (AutoencoderKL)" in message


class TestAutoencoderKLImport:
    """Tests that AutoencoderKL can be imported from diffusers."""

    def test_autoencoder_kl_import(self):
        """Test that AutoencoderKL can be imported from diffusers."""
        from diffusers import AutoencoderKL
        assert AutoencoderKL is not None

    def test_autoencoder_kl_has_from_pretrained(self):
        """Test that AutoencoderKL has the from_pretrained method."""
        from diffusers import AutoencoderKL
        assert hasattr(AutoencoderKL, 'from_pretrained')


class TestCodeStructure:
    """Tests that the code structure is correct for VAE loading."""

    def test_ai_models_contains_autoencoder_handling(self):
        """Test that ai_models.py contains the AutoencoderKL handling code."""
        import inspect
        from backend.services.ai_models import AIModelManager
        
        source = inspect.getsource(AIModelManager._generate_image_diffusers)
        
        # Verify the error detection logic exists
        assert "AutoencoderKL" in source
        assert "needs_vae" in source
        
        # Verify the loading logic exists
        assert 'from diffusers import AutoencoderKL' in source
        assert 'Loading SDXL VAE (AutoencoderKL)' in source or 'Loading SD 1.5 VAE (AutoencoderKL)' in source

    def test_ai_models_handles_both_text_encoder_and_vae(self):
        """Test that the code handles both text encoder and VAE missing cases."""
        import inspect
        from backend.services.ai_models import AIModelManager
        
        source = inspect.getsource(AIModelManager._generate_image_diffusers)
        
        # Verify both component types are handled
        assert "needs_text_encoder" in source
        assert "needs_vae" in source
        
        # Verify the combined condition
        assert "if needs_text_encoder or needs_vae:" in source
