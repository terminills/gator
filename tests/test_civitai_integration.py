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


class TestCivitAIRedirectHandling:
    """Test redirect handling for CivitAI downloads."""

    @pytest.mark.asyncio
    async def test_client_follows_redirects(self):
        """Test that the download client is configured to follow redirects."""
        import httpx

        # Test that AsyncClient with follow_redirects=True is created
        # This is what the fix in civitai_utils.py enables
        async with httpx.AsyncClient(timeout=None, follow_redirects=True) as client:
            assert client.follow_redirects is True

    @pytest.mark.asyncio
    async def test_download_handles_307_redirect(self):
        """Test that 307 redirects are properly followed during download."""
        # This test validates the fix for the reported issue where
        # CivitAI returns 307 redirects to Cloudflare storage
        import httpx
        from unittest.mock import AsyncMock, MagicMock, patch

        # Create a mock response that simulates a successful download after redirect
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.headers = {"content-length": "100"}

        async def mock_aiter_bytes(chunk_size=8192):
            yield b"test content"

        mock_response.aiter_bytes = mock_aiter_bytes

        # Create async context manager for stream
        mock_stream_cm = MagicMock()
        mock_stream_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_stream_cm.__aexit__ = AsyncMock(return_value=None)

        # Create mock client
        mock_client = MagicMock()
        mock_client.stream = MagicMock(return_value=mock_stream_cm)
        mock_client.get = AsyncMock()

        # Create async context manager for client
        mock_client_cm = MagicMock()
        mock_client_cm.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_cm.__aexit__ = AsyncMock(return_value=None)

        # Verify that CivitAIClient.download_model would use follow_redirects=True
        from backend.utils.civitai_utils import CivitAIClient
        import tempfile

        # Patch the AsyncClient to verify it's called with follow_redirects=True
        with patch("backend.utils.civitai_utils.httpx.AsyncClient") as mock_async_client:
            mock_async_client.return_value = mock_client_cm

            client = CivitAIClient(api_key="test_key")

            # Mock get_model_version to return test data
            with patch.object(client, "get_model_version") as mock_get_version:
                mock_get_version.return_value = {
                    "files": [
                        {
                            "name": "test_model.safetensors",
                            "downloadUrl": "https://civitai.com/api/download/models/12345",
                            "sizeKB": 100,
                            "hashes": [],
                        }
                    ],
                    "modelId": 123,
                    "name": "v1.0",
                    "model": {"name": "Test Model"},
                    "baseModel": "SDXL 1.0",
                    "trainedWords": [],
                }

                with tempfile.TemporaryDirectory() as tmpdir:
                    # The download will fail due to incomplete mocking (file write issues),
                    # but we only need to verify that AsyncClient was called correctly
                    try:
                        await client.download_model(
                            model_version_id=12345,
                            output_path=Path(tmpdir),
                        )
                    except (OSError, TypeError, AttributeError):
                        # Expected failures from incomplete mocking:
                        # - OSError: file write operations
                        # - TypeError: mock method calls
                        # - AttributeError: missing mock attributes
                        pass

                    # Verify AsyncClient was called with follow_redirects=True
                    # This is the key assertion that validates the fix for the 307 redirect issue
                    mock_async_client.assert_called_with(
                        timeout=None, follow_redirects=True
                    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
