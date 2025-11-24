"""
Tests for CivitAI Integration

Tests the CivitAI API client and download functionality.
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from backend.utils.civitai_utils import (
    CivitAIClient,
    CivitAIModelType,
    CivitAIBaseModel,
)


class TestCivitAIClient:
    """Test CivitAI API client."""
    
    @pytest.mark.asyncio
    async def test_client_initialization(self):
        """Test that client can be initialized."""
        client = CivitAIClient()
        assert client is not None
        assert client.base_url == "https://civitai.com/api/v1"
    
    @pytest.mark.asyncio
    async def test_client_with_api_key(self):
        """Test client initialization with API key."""
        api_key = "test_key_123"
        client = CivitAIClient(api_key=api_key)
        assert client.api_key == api_key
        
        headers = client._get_headers()
        assert "Authorization" in headers
        assert headers["Authorization"] == f"Bearer {api_key}"
    
    @pytest.mark.asyncio
    async def test_list_models_parameters(self):
        """Test that list_models accepts correct parameters."""
        client = CivitAIClient()
        
        # This will make a real API call - only run if CIVITAI_TESTS_ENABLED
        import os
        if os.environ.get("CIVITAI_TESTS_ENABLED") != "true":
            pytest.skip("CivitAI API tests disabled")
        
        result = await client.list_models(
            limit=5,
            page=1,
            query="stable diffusion",
            model_types=[CivitAIModelType.CHECKPOINT],
        )
        
        assert "items" in result
        assert isinstance(result["items"], list)
    
    def test_model_type_enum(self):
        """Test that model type enum has expected values."""
        assert CivitAIModelType.CHECKPOINT.value == "Checkpoint"
        assert CivitAIModelType.LORA.value == "LORA"
        assert CivitAIModelType.TEXTUAL_INVERSION.value == "TextualInversion"
    
    def test_base_model_enum(self):
        """Test that base model enum has expected values."""
        assert CivitAIBaseModel.SD_1_5.value == "SD 1.5"
        assert CivitAIBaseModel.SDXL_1_0.value == "SDXL 1.0"
        assert CivitAIBaseModel.FLUX_1.value == "Flux.1"


class TestCivitAIUtilityFunctions:
    """Test utility functions for CivitAI."""
    
    @pytest.mark.asyncio
    async def test_list_civitai_models_basic(self):
        """Test basic model listing."""
        from backend.utils.civitai_utils import list_civitai_models
        
        import os
        if os.environ.get("CIVITAI_TESTS_ENABLED") != "true":
            pytest.skip("CivitAI API tests disabled")
        
        models = await list_civitai_models(
            query="anime",
            limit=3,
        )
        
        assert isinstance(models, list)
        assert len(models) <= 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
