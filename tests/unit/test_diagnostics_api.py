"""
Tests for diagnostics API endpoints.
"""

import pytest
from httpx import AsyncClient, ASGITransport
from fastapi import status

from backend.api.main import app


@pytest.mark.asyncio
async def test_ai_activity_endpoint():
    """Test AI activity diagnostics endpoint."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/v1/diagnostics/ai-activity")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # Verify response structure
        assert "total_generations" in data
        assert "successful_generations" in data
        assert "failed_generations" in data
        assert "fallback_generations" in data
        assert "success_rate" in data
        assert "by_content_type" in data
        assert "by_persona" in data
        assert "active_contexts" in data
        assert "completed_contexts" in data
        assert "failed_contexts" in data
        assert "recent_errors" in data
        
        # Verify data types
        assert isinstance(data["total_generations"], int)
        assert isinstance(data["success_rate"], (int, float))
        assert isinstance(data["by_content_type"], dict)
        assert isinstance(data["recent_errors"], list)


@pytest.mark.asyncio
async def test_generation_attempts_endpoint():
    """Test generation attempts endpoint."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/v1/diagnostics/generation-attempts?limit=10")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # Should return a list
        assert isinstance(data, list)
        
        # Each item should have expected fields
        for attempt in data:
            assert "persona_id" in attempt
            assert "persona_name" in attempt
            assert "content_type" in attempt
            assert "status" in attempt
            assert "created_at" in attempt
            assert "has_acd_context" in attempt


@pytest.mark.asyncio
async def test_ai_models_endpoint():
    """Test AI models activity endpoint."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/v1/diagnostics/ai-models")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # Should return a list
        assert isinstance(data, list)
        
        # Each model should have expected fields
        for model in data:
            assert "model_name" in model
            assert "provider" in model
            assert "total_calls" in model
            assert "successful_calls" in model
            assert "failed_calls" in model


@pytest.mark.asyncio
async def test_acd_contexts_endpoint():
    """Test ACD contexts endpoint."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/v1/diagnostics/acd-contexts?limit=5")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # Should return a list
        assert isinstance(data, list)
        
        # Each context should have expected fields
        for context in data:
            assert "id" in context
            assert "ai_phase" in context
            assert "ai_status" in context
            assert "ai_state" in context
            assert "created_at" in context


@pytest.mark.asyncio
async def test_diagnostics_with_filters():
    """Test diagnostics endpoints with filters."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Test with hours filter
        response = await client.get("/api/v1/diagnostics/ai-activity?hours=48")
        assert response.status_code == status.HTTP_200_OK
        
        # Test generation attempts with content type filter
        response = await client.get("/api/v1/diagnostics/generation-attempts?content_type=image")
        assert response.status_code == status.HTTP_200_OK
        
        # Test ACD contexts with phase filter
        response = await client.get("/api/v1/diagnostics/acd-contexts?phase=IMAGE_GENERATION")
        assert response.status_code == status.HTTP_200_OK
