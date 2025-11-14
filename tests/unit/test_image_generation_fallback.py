"""
Unit tests for image generation model fallback mechanism.

Tests that when a model fails due to incomplete components or other errors,
the system tries alternative available models instead of failing immediately.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from backend.services.ai_models import AIModelManager


class TestImageGenerationFallback:
    """Tests for image generation fallback mechanism."""

    @pytest.mark.asyncio
    async def test_fallback_to_alternative_model_on_sdxl_failure(self):
        """Test that generation falls back to SD 1.5 when SDXL has incomplete components."""
        # Setup
        manager = AIModelManager()
        manager.available_models = {
            "text": [],
            "image": [
                {
                    "name": "sdxl-1.0",
                    "type": "text-to-image",
                    "model_id": "stabilityai/stable-diffusion-xl-base-1.0",
                    "provider": "local",
                    "inference_engine": "diffusers",
                    "loaded": True,
                    "can_load": True,
                    "size_gb": 7,
                },
                {
                    "name": "stable-diffusion-v1-5",
                    "type": "text-to-image",
                    "model_id": "runwayml/stable-diffusion-v1-5",
                    "provider": "local",
                    "inference_engine": "diffusers",
                    "loaded": True,
                    "can_load": True,
                    "size_gb": 4,
                },
            ],
            "voice": [],
            "video": [],
            "audio": [],
        }
        manager.models_loaded = True

        # Mock _generate_image_local to simulate SDXL failure and SD 1.5 success
        async def mock_generate_local(prompt, model, **kwargs):
            if model["name"] == "sdxl-1.0":
                raise ValueError(
                    "SDXL pipeline has None text encoders: text_encoder=True, text_encoder_2=False"
                )
            elif model["name"] == "stable-diffusion-v1-5":
                return {
                    "image_data": b"fake_image_data",
                    "format": "PNG",
                    "model": model["name"],
                    "width": 512,
                    "height": 512,
                }
            else:
                raise ValueError(f"Unknown model: {model['name']}")

        # Mock ComfyUI check to return not available
        with patch.object(
            manager.http_client, "get", new_callable=AsyncMock
        ) as mock_get:
            mock_get.side_effect = Exception("ComfyUI not available")

            with patch.object(manager, "_generate_image_local", new=mock_generate_local):
                # Execute
                result = await manager.generate_image(
                    "A test image",
                    width=512,
                    height=512,
                )

                # Verify
                assert result is not None
                assert result.get("model") == "stable-diffusion-v1-5"
                benchmark_data = result.get("benchmark_data", {})
                assert "sdxl-1.0" in benchmark_data.get("failed_models", [])
                assert benchmark_data.get("model_selected") == "stable-diffusion-v1-5"

    @pytest.mark.asyncio
    async def test_fallback_records_failed_models(self):
        """Test that failed models are properly tracked in benchmark data."""
        # Setup
        manager = AIModelManager()
        manager.available_models = {
            "text": [],
            "image": [
                {
                    "name": "model-1",
                    "provider": "local",
                    "inference_engine": "diffusers",
                    "loaded": True,
                    "can_load": True,
                    "size_gb": 5,
                },
                {
                    "name": "model-2",
                    "provider": "local",
                    "inference_engine": "diffusers",
                    "loaded": True,
                    "can_load": True,
                    "size_gb": 4,
                },
                {
                    "name": "model-3",
                    "provider": "local",
                    "inference_engine": "diffusers",
                    "loaded": True,
                    "can_load": True,
                    "size_gb": 3,
                },
            ],
            "voice": [],
            "video": [],
            "audio": [],
        }
        manager.models_loaded = True

        call_count = {"count": 0}

        # Mock _generate_image_local to fail first two, succeed on third
        async def mock_generate_local(prompt, model, **kwargs):
            call_count["count"] += 1
            if model["name"] in ["model-1", "model-2"]:
                raise ValueError(f"Model {model['name']} failed")
            return {
                "image_data": b"fake_image_data",
                "format": "PNG",
                "model": model["name"],
                "width": 512,
                "height": 512,
            }

        with patch.object(
            manager.http_client, "get", new_callable=AsyncMock
        ) as mock_get:
            mock_get.side_effect = Exception("ComfyUI not available")

            with patch.object(manager, "_generate_image_local", new=mock_generate_local):
                # Execute
                result = await manager.generate_image("test prompt")

                # Verify
                assert result is not None
                assert result.get("model") == "model-3"
                assert call_count["count"] == 3
                benchmark_data = result.get("benchmark_data", {})
                failed_models = benchmark_data.get("failed_models", [])
                assert "model-1" in failed_models
                assert "model-2" in failed_models
                assert len(failed_models) == 2

    @pytest.mark.asyncio
    async def test_all_models_fail_raises_last_error(self):
        """Test that when all models fail, the last error is raised."""
        # Setup
        manager = AIModelManager()
        manager.available_models = {
            "text": [],
            "image": [
                {
                    "name": "model-1",
                    "provider": "local",
                    "inference_engine": "diffusers",
                    "loaded": True,
                    "can_load": True,
                    "size_gb": 5,
                },
                {
                    "name": "model-2",
                    "provider": "local",
                    "inference_engine": "diffusers",
                    "loaded": True,
                    "can_load": True,
                    "size_gb": 4,
                },
            ],
            "voice": [],
            "video": [],
            "audio": [],
        }
        manager.models_loaded = True

        # Mock _generate_image_local to always fail
        async def mock_generate_local(prompt, model, **kwargs):
            raise ValueError(f"Model {model['name']} failed")

        with patch.object(
            manager.http_client, "get", new_callable=AsyncMock
        ) as mock_get:
            mock_get.side_effect = Exception("ComfyUI not available")

            with patch.object(manager, "_generate_image_local", new=mock_generate_local):
                # Execute and verify exception
                with pytest.raises(ValueError) as exc_info:
                    await manager.generate_image("test prompt")

                assert "Model model-2 failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_incomplete_model_error_logging(self):
        """Test that incomplete model errors provide helpful context."""
        # Setup
        manager = AIModelManager()
        manager.available_models = {
            "text": [],
            "image": [
                {
                    "name": "sdxl-incomplete",
                    "provider": "local",
                    "inference_engine": "diffusers",
                    "loaded": True,
                    "can_load": True,
                    "size_gb": 7,
                },
                {
                    "name": "sd-working",
                    "provider": "local",
                    "inference_engine": "diffusers",
                    "loaded": True,
                    "can_load": True,
                    "size_gb": 4,
                },
            ],
            "voice": [],
            "video": [],
            "audio": [],
        }
        manager.models_loaded = True

        # Mock _generate_image_local
        async def mock_generate_local(prompt, model, **kwargs):
            if model["name"] == "sdxl-incomplete":
                raise ValueError("SDXL pipeline has None text encoders")
            return {
                "image_data": b"fake",
                "format": "PNG",
                "model": model["name"],
                "width": 512,
                "height": 512,
            }

        with patch.object(
            manager.http_client, "get", new_callable=AsyncMock
        ) as mock_get:
            mock_get.side_effect = Exception("ComfyUI not available")

            with patch.object(manager, "_generate_image_local", new=mock_generate_local):
                # Execute
                result = await manager.generate_image("test")

                # Verify fallback worked
                assert result.get("model") == "sd-working"
                benchmark_data = result.get("benchmark_data", {})
                assert "sdxl-incomplete" in benchmark_data.get("failed_models", [])

    @pytest.mark.asyncio
    async def test_single_model_success_no_fallback(self):
        """Test that when first model succeeds, no fallback is attempted."""
        # Setup
        manager = AIModelManager()
        manager.available_models = {
            "text": [],
            "image": [
                {
                    "name": "model-1",
                    "provider": "local",
                    "inference_engine": "diffusers",
                    "loaded": True,
                    "can_load": True,
                    "size_gb": 5,
                },
                {
                    "name": "model-2",
                    "provider": "local",
                    "inference_engine": "diffusers",
                    "loaded": True,
                    "can_load": True,
                    "size_gb": 4,
                },
            ],
            "voice": [],
            "video": [],
            "audio": [],
        }
        manager.models_loaded = True

        call_count = {"count": 0}

        # Mock _generate_image_local to succeed on first call
        async def mock_generate_local(prompt, model, **kwargs):
            call_count["count"] += 1
            return {
                "image_data": b"fake_image_data",
                "format": "PNG",
                "model": model["name"],
                "width": 512,
                "height": 512,
            }

        with patch.object(
            manager.http_client, "get", new_callable=AsyncMock
        ) as mock_get:
            mock_get.side_effect = Exception("ComfyUI not available")

            with patch.object(manager, "_generate_image_local", new=mock_generate_local):
                # Execute
                result = await manager.generate_image("test prompt")

                # Verify only one call was made
                assert call_count["count"] == 1
                assert result.get("model") == "model-1"
                benchmark_data = result.get("benchmark_data", {})
                assert len(benchmark_data.get("failed_models", [])) == 0
