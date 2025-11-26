"""
Tests for CivitAI Model Detection in AIModelManager

Tests the automatic detection and registration of CivitAI models
that have been downloaded to the models/civitai/ directory.
"""

import asyncio
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, patch

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


class TestCivitAIModelDetection:
    """Test CivitAI model detection in AIModelManager."""

    @pytest.mark.asyncio
    async def test_civitai_model_detection_with_metadata(self):
        """Test that CivitAI models with metadata are properly detected."""
        from backend.services.ai_models import AIModelManager

        with TemporaryDirectory() as tmpdir:
            # Setup mock CivitAI model directory
            models_dir = Path(tmpdir) / "models"
            civitai_dir = models_dir / "civitai"
            civitai_dir.mkdir(parents=True)

            # Create mock model file
            model_file = civitai_dir / "test_model.safetensors"
            model_file.touch()

            # Create mock metadata file
            metadata = {
                "source": "civitai",
                "model_id": 144203,
                "version_id": 160240,
                "version_name": "v1.0",
                "model_name": "NSFW POV All In One SDXL",
                "base_model": "SDXL 1.0",
                "file_name": "test_model.safetensors",
                "file_size_kb": 76800,
                "trained_words": ["pov", "nsfw"],
                "license": "open",
                "nsfw": True,
                "type": "LORA",
            }

            metadata_file = civitai_dir / "test_model_metadata.json"
            with open(metadata_file, "w") as f:
                json.dump(metadata, f)

            # Create AIModelManager with custom models dir
            manager = AIModelManager()
            manager.models_dir = models_dir

            # Initialize models
            await manager.initialize_models()

            # Check that the CivitAI model was detected
            image_models = manager.available_models.get("image", [])
            civitai_models = [m for m in image_models if m.get("source") == "civitai"]

            assert len(civitai_models) == 1, "Should detect 1 CivitAI model"

            model = civitai_models[0]
            assert model["source"] == "civitai"
            assert model["base_model"] == "SDXL 1.0"
            assert model["trained_words"] == ["pov", "nsfw"]
            assert model["nsfw"] is True
            assert model["model_type"] == "LORA"
            assert model["civitai_model_id"] == 144203
            assert model["civitai_version_id"] == 160240

    @pytest.mark.asyncio
    async def test_civitai_model_detection_without_civitai_dir(self):
        """Test that missing civitai directory is handled gracefully."""
        from backend.services.ai_models import AIModelManager

        with TemporaryDirectory() as tmpdir:
            models_dir = Path(tmpdir) / "models"
            models_dir.mkdir(parents=True)
            # Note: civitai subdirectory is not created

            manager = AIModelManager()
            manager.models_dir = models_dir

            # Should not raise an error
            await manager.initialize_models()

            # Check that no CivitAI models were added
            image_models = manager.available_models.get("image", [])
            civitai_models = [m for m in image_models if m.get("source") == "civitai"]
            assert len(civitai_models) == 0

    @pytest.mark.asyncio
    async def test_civitai_model_detection_with_missing_model_file(self):
        """Test that missing model files are handled gracefully."""
        from backend.services.ai_models import AIModelManager

        with TemporaryDirectory() as tmpdir:
            models_dir = Path(tmpdir) / "models"
            civitai_dir = models_dir / "civitai"
            civitai_dir.mkdir(parents=True)

            # Create metadata file without corresponding model file
            metadata = {
                "model_name": "Missing Model",
                "base_model": "SDXL 1.0",
                "file_name": "nonexistent.safetensors",
            }

            metadata_file = civitai_dir / "missing_model_metadata.json"
            with open(metadata_file, "w") as f:
                json.dump(metadata, f)

            manager = AIModelManager()
            manager.models_dir = models_dir

            # Should not raise an error
            await manager.initialize_models()

            # Check that no CivitAI models were added
            image_models = manager.available_models.get("image", [])
            civitai_models = [m for m in image_models if m.get("source") == "civitai"]
            assert len(civitai_models) == 0

    @pytest.mark.asyncio
    async def test_civitai_model_detection_with_checkpoint(self):
        """Test detection of CivitAI Checkpoint models."""
        from backend.services.ai_models import AIModelManager

        with TemporaryDirectory() as tmpdir:
            models_dir = Path(tmpdir) / "models"
            civitai_dir = models_dir / "civitai"
            civitai_dir.mkdir(parents=True)

            # Create mock checkpoint model
            model_file = civitai_dir / "realistic_vision.safetensors"
            model_file.touch()

            metadata = {
                "model_name": "Realistic Vision V5.1",
                "base_model": "SD 1.5",
                "file_name": "realistic_vision.safetensors",
                "type": "Checkpoint",
                "model_id": 12345,
                "version_id": 67890,
            }

            metadata_file = civitai_dir / "realistic_vision_metadata.json"
            with open(metadata_file, "w") as f:
                json.dump(metadata, f)

            manager = AIModelManager()
            manager.models_dir = models_dir

            await manager.initialize_models()

            image_models = manager.available_models.get("image", [])
            civitai_models = [m for m in image_models if m.get("source") == "civitai"]

            assert len(civitai_models) == 1
            model = civitai_models[0]
            assert model["model_type"] == "Checkpoint"
            assert model["type"] == "text-to-image"

    @pytest.mark.asyncio
    async def test_civitai_model_detection_with_lora(self):
        """Test detection of CivitAI LORA models."""
        from backend.services.ai_models import AIModelManager

        with TemporaryDirectory() as tmpdir:
            models_dir = Path(tmpdir) / "models"
            civitai_dir = models_dir / "civitai"
            civitai_dir.mkdir(parents=True)

            # Create mock LORA model
            model_file = civitai_dir / "detail_enhancer.safetensors"
            model_file.touch()

            metadata = {
                "model_name": "Detail Enhancer",
                "base_model": "SDXL 1.0",
                "file_name": "detail_enhancer.safetensors",
                "type": "LORA",
                "trained_words": ["detailed", "hires"],
            }

            metadata_file = civitai_dir / "detail_enhancer_metadata.json"
            with open(metadata_file, "w") as f:
                json.dump(metadata, f)

            manager = AIModelManager()
            manager.models_dir = models_dir

            await manager.initialize_models()

            image_models = manager.available_models.get("image", [])
            civitai_models = [m for m in image_models if m.get("source") == "civitai"]

            assert len(civitai_models) == 1
            model = civitai_models[0]
            assert model["model_type"] == "LORA"
            assert model["type"] == "lora"
            assert model["trained_words"] == ["detailed", "hires"]


class TestCivitAIModelSelection:
    """Test CivitAI model selection in optimal model selection."""

    @pytest.mark.asyncio
    async def test_select_model_with_trigger_word(self):
        """Test that models with matching trigger words are selected."""
        from backend.services.ai_models import AIModelManager

        with TemporaryDirectory() as tmpdir:
            models_dir = Path(tmpdir) / "models"
            civitai_dir = models_dir / "civitai"
            civitai_dir.mkdir(parents=True)

            # Create mock model with trigger word
            model_file = civitai_dir / "pov_model.safetensors"
            model_file.touch()

            metadata = {
                "model_name": "POV Style SDXL",
                "base_model": "SDXL 1.0",
                "file_name": "pov_model.safetensors",
                "type": "LORA",
                "trained_words": ["pov", "perspective"],
            }

            metadata_file = civitai_dir / "pov_model_metadata.json"
            with open(metadata_file, "w") as f:
                json.dump(metadata, f)

            manager = AIModelManager()
            manager.models_dir = models_dir

            await manager.initialize_models()

            # Test model selection with prompt containing trigger word
            prompt = "A beautiful sunset from POV angle"
            available_models = manager.available_models.get("image", [])
            loaded_models = [m for m in available_models if m.get("loaded")]

            if loaded_models:
                selected = await manager._select_optimal_model(
                    prompt=prompt,
                    content_type="image",
                    available_models=loaded_models,
                )

                # Should select the CivitAI model because prompt contains "pov"
                assert selected.get("source") == "civitai"
                assert "pov" in [w.lower() for w in selected.get("trained_words", [])]

    @pytest.mark.asyncio
    async def test_select_model_by_display_name(self):
        """Test that models can be selected by display name preference."""
        from backend.services.ai_models import AIModelManager

        with TemporaryDirectory() as tmpdir:
            models_dir = Path(tmpdir) / "models"
            civitai_dir = models_dir / "civitai"
            civitai_dir.mkdir(parents=True)

            # Create mock model
            model_file = civitai_dir / "nsfw_model.safetensors"
            model_file.touch()

            metadata = {
                "model_name": "NSFW POV All In One",
                "base_model": "SDXL 1.0",
                "file_name": "nsfw_model.safetensors",
                "type": "LORA",
                "nsfw": True,
            }

            metadata_file = civitai_dir / "nsfw_model_metadata.json"
            with open(metadata_file, "w") as f:
                json.dump(metadata, f)

            manager = AIModelManager()
            manager.models_dir = models_dir

            await manager.initialize_models()

            # Test model selection with nsfw_model preference
            prompt = "A photo"
            available_models = manager.available_models.get("image", [])
            loaded_models = [m for m in available_models if m.get("loaded")]

            if loaded_models:
                selected = await manager._select_optimal_model(
                    prompt=prompt,
                    content_type="image",
                    available_models=loaded_models,
                    nsfw_model="NSFW POV",
                )

                # Should select the model that matches the nsfw_model preference
                assert selected.get("source") == "civitai"
                assert "NSFW POV" in selected.get("display_name", "")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
