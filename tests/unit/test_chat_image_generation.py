"""
Test Chat Image Generation Feature

Tests the chat-based image generation functionality that uses Ollama
to create image prompts from persona and chat context.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from backend.api.routes.persona import (
    ChatImageRequest,
    ChatImageResponse,
    _generate_image_prompt_with_ollama,
    _generate_fallback_image_prompt,
)


class TestChatImageRequest:
    """Test ChatImageRequest model validation."""
    
    def test_default_values(self):
        """Test that request has sensible defaults."""
        request = ChatImageRequest()
        assert request.message == ""
        assert request.custom_prompt == ""
        assert request.include_nsfw is True
    
    def test_with_message(self):
        """Test request with a chat message."""
        request = ChatImageRequest(
            message="Let's take a romantic photo together",
            include_nsfw=True
        )
        assert request.message == "Let's take a romantic photo together"
        assert request.include_nsfw is True
    
    def test_with_custom_prompt(self):
        """Test request with custom prompt additions."""
        request = ChatImageRequest(
            message="Beach vacation",
            custom_prompt="sunset lighting, golden hour",
            include_nsfw=False
        )
        assert request.custom_prompt == "sunset lighting, golden hour"
        assert request.include_nsfw is False


class TestChatImageResponse:
    """Test ChatImageResponse model."""
    
    def test_response_structure(self):
        """Test response contains all required fields."""
        response = ChatImageResponse(
            image_data="data:image/png;base64,abc123",
            image_prompt="A beautiful portrait of...",
            persona_name="Test Persona",
            model_used="sdxl-1.0",
            timestamp="2024-01-01T00:00:00",
            width=1024,
            height=1024,
        )
        
        assert response.image_data.startswith("data:image/png;base64,")
        assert response.image_prompt == "A beautiful portrait of..."
        assert response.persona_name == "Test Persona"
        assert response.model_used == "sdxl-1.0"
        assert response.width == 1024
        assert response.height == 1024


class TestFallbackImagePrompt:
    """Test fallback prompt generation without Ollama."""
    
    def test_basic_prompt_from_appearance(self):
        """Test prompt generation with just appearance."""
        mock_persona = MagicMock()
        mock_persona.name = "Luna"
        mock_persona.appearance = "Young woman with long black hair and blue eyes"
        mock_persona.personality = None
        mock_persona.default_content_rating = "sfw"
        
        prompt = _generate_fallback_image_prompt(
            persona=mock_persona,
            chat_message="",
            custom_prompt="",
            include_nsfw=False,
        )
        
        assert "Young woman with long black hair and blue eyes" in prompt
        assert "highly detailed" in prompt
    
    def test_prompt_with_personality(self):
        """Test prompt includes personality-based elements."""
        mock_persona = MagicMock()
        mock_persona.name = "Luna"
        mock_persona.appearance = "Young woman with long black hair"
        mock_persona.personality = "Confident and energetic"
        mock_persona.default_content_rating = "sfw"
        
        prompt = _generate_fallback_image_prompt(
            persona=mock_persona,
            chat_message="",
            custom_prompt="",
            include_nsfw=False,
        )
        
        assert "confident" in prompt.lower()
    
    def test_prompt_with_chat_context(self):
        """Test prompt incorporates chat message."""
        mock_persona = MagicMock()
        mock_persona.name = "Luna"
        mock_persona.appearance = "Young woman"
        mock_persona.personality = None
        mock_persona.default_content_rating = "sfw"
        
        prompt = _generate_fallback_image_prompt(
            persona=mock_persona,
            chat_message="romantic beach sunset",
            custom_prompt="",
            include_nsfw=False,
        )
        
        assert "romantic beach sunset" in prompt
    
    def test_prompt_with_custom_additions(self):
        """Test prompt includes custom prompt text."""
        mock_persona = MagicMock()
        mock_persona.name = "Luna"
        mock_persona.appearance = "Young woman"
        mock_persona.personality = None
        mock_persona.default_content_rating = "sfw"
        
        prompt = _generate_fallback_image_prompt(
            persona=mock_persona,
            chat_message="",
            custom_prompt="cinematic lighting, shallow depth of field",
            include_nsfw=False,
        )
        
        assert "cinematic lighting" in prompt
    
    def test_nsfw_elements_when_enabled(self):
        """Test NSFW elements added when enabled and allowed."""
        mock_persona = MagicMock()
        mock_persona.name = "Luna"
        mock_persona.appearance = "Young woman"
        mock_persona.personality = "seductive and alluring"
        mock_persona.default_content_rating = "nsfw"
        
        prompt = _generate_fallback_image_prompt(
            persona=mock_persona,
            chat_message="",
            custom_prompt="",
            include_nsfw=True,
        )
        
        assert "sensual" in prompt.lower() or "artistic nude" in prompt.lower()
    
    def test_no_nsfw_when_disabled(self):
        """Test no NSFW elements when disabled."""
        mock_persona = MagicMock()
        mock_persona.name = "Luna"
        mock_persona.appearance = "Young woman"
        mock_persona.personality = None
        mock_persona.default_content_rating = "nsfw"
        
        prompt = _generate_fallback_image_prompt(
            persona=mock_persona,
            chat_message="",
            custom_prompt="",
            include_nsfw=False,
        )
        
        assert "nude" not in prompt.lower()
        assert "explicit" not in prompt.lower()


class TestOllamaPromptGeneration:
    """Test Ollama-based prompt generation."""
    
    @pytest.mark.asyncio
    async def test_fallback_when_ollama_unavailable(self):
        """Test fallback is used when Ollama is not available."""
        mock_persona = MagicMock()
        mock_persona.name = "Luna"
        mock_persona.appearance = "Young woman with blonde hair"
        mock_persona.personality = "Friendly and outgoing"
        mock_persona.default_content_rating = "sfw"
        
        # Mock httpx to simulate Ollama being unavailable
        with patch("backend.api.routes.persona.httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get = AsyncMock(side_effect=Exception("Connection refused"))
            mock_client.return_value.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_client.return_value.__aexit__ = AsyncMock()
            
            prompt = await _generate_image_prompt_with_ollama(
                persona=mock_persona,
                chat_message="Hello",
                custom_prompt="",
                include_nsfw=False,
            )
        
        # Should get a fallback prompt
        assert prompt is not None
        assert len(prompt) > 0
        assert "Young woman with blonde hair" in prompt
    
    @pytest.mark.asyncio
    async def test_prompt_generation_with_ollama(self):
        """Test successful prompt generation with Ollama."""
        mock_persona = MagicMock()
        mock_persona.name = "Luna"
        mock_persona.appearance = "Young woman with silver hair and violet eyes"
        mock_persona.personality = "Mysterious and enigmatic"
        mock_persona.default_content_rating = "sfw"
        
        # Mock successful Ollama response
        with patch("backend.api.routes.persona.httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            
            # Mock /api/tags response
            mock_tags_response = AsyncMock()
            mock_tags_response.status_code = 200
            mock_tags_response.json.return_value = {
                "models": [{"name": "dolphin-mixtral:8x7b"}]
            }
            
            # Mock /api/generate response
            mock_generate_response = AsyncMock()
            mock_generate_response.status_code = 200
            mock_generate_response.json.return_value = {
                "response": "Ethereal portrait of a mystical young woman with flowing silver hair and captivating violet eyes, mysterious expression, dramatic chiaroscuro lighting, fantasy art style, highly detailed, digital painting"
            }
            
            async def mock_get(*args, **kwargs):
                return mock_tags_response
            
            async def mock_post(*args, **kwargs):
                return mock_generate_response
            
            mock_instance.get = mock_get
            mock_instance.post = mock_post
            mock_client.return_value.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_client.return_value.__aexit__ = AsyncMock()
            
            prompt = await _generate_image_prompt_with_ollama(
                persona=mock_persona,
                chat_message="Tell me about your mysterious past",
                custom_prompt="",
                include_nsfw=False,
            )
        
        # Should get the mocked Ollama response
        assert "silver hair" in prompt.lower() or "violet eyes" in prompt.lower() or "mystical" in prompt.lower()


class TestChatImageEndpoint:
    """Test the chat image generation endpoint (requires database connection)."""
    
    @pytest.mark.asyncio
    async def test_endpoint_route_exists(self):
        """Test that the chat/generate-image route is registered."""
        from backend.api.main import app
        
        # Check that the route exists in the app
        routes = [r.path for r in app.routes]
        
        # The route pattern should exist (with path parameter)
        matching_routes = [r for r in routes if "chat/generate-image" in r]
        assert len(matching_routes) > 0, "chat/generate-image route should be registered"
    
    @pytest.mark.asyncio
    async def test_request_validation(self):
        """Test that request model validates correctly."""
        # Valid request
        valid_request = ChatImageRequest(
            message="Hello",
            custom_prompt="studio lighting",
            include_nsfw=False
        )
        assert valid_request.message == "Hello"
        
        # Empty request (should use defaults)
        empty_request = ChatImageRequest()
        assert empty_request.message == ""
        assert empty_request.include_nsfw is True
