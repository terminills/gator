"""
Tests for Plugin System GatorAPI Implementation

Tests the GatorAPI client methods for plugin interactions with the platform.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4
import httpx

from backend.plugins import GatorAPI, PluginMetadata, PluginType, GatorPlugin


class TestGatorAPI:
    """Test suite for GatorAPI client."""

    @pytest.fixture
    def api_client(self):
        """Create a test API client."""
        return GatorAPI(api_key="test-api-key", base_url="http://localhost:8000")

    @pytest.fixture
    def mock_http_client(self, api_client):
        """Mock the HTTP client."""
        mock_client = AsyncMock()
        api_client._http_client = mock_client
        return mock_client

    @pytest.mark.asyncio
    async def test_generate_content_success(self, api_client, mock_http_client):
        """Test successful content generation via API."""
        # Arrange
        persona_id = str(uuid4())
        prompt = "Create amazing artwork"
        expected_response = {
            "id": str(uuid4()),
            "persona_id": persona_id,
            "content_type": "image",
            "prompt": prompt,
            "file_path": "/path/to/content.png",
            "metadata": {"format": "PNG", "size": 1024},
        }

        mock_response = MagicMock()
        mock_response.json.return_value = expected_response
        mock_response.raise_for_status = MagicMock()
        mock_http_client.post.return_value = mock_response

        # Act
        result = await api_client.generate_content(
            persona_id=persona_id, prompt=prompt, content_type="image"
        )

        # Assert
        assert result == expected_response
        mock_http_client.post.assert_called_once()
        call_args = mock_http_client.post.call_args
        assert call_args[0][0] == "http://localhost:8000/api/v1/content/generate"
        assert call_args[1]["json"]["persona_id"] == persona_id
        assert call_args[1]["json"]["prompt"] == prompt
        assert call_args[1]["json"]["content_type"] == "image"

    @pytest.mark.asyncio
    async def test_generate_content_with_optional_params(
        self, api_client, mock_http_client
    ):
        """Test content generation with optional parameters."""
        # Arrange
        persona_id = str(uuid4())
        mock_response = MagicMock()
        mock_response.json.return_value = {"id": str(uuid4())}
        mock_response.raise_for_status = MagicMock()
        mock_http_client.post.return_value = mock_response

        # Act
        await api_client.generate_content(
            persona_id=persona_id,
            prompt="Test prompt",
            content_type="video",
            content_rating="nsfw",
            target_platforms=["instagram", "twitter"],
            style_override={"aesthetic": "dark"},
        )

        # Assert
        call_args = mock_http_client.post.call_args
        payload = call_args[1]["json"]
        assert payload["content_rating"] == "nsfw"
        assert payload["target_platforms"] == ["instagram", "twitter"]
        assert payload["style_override"] == {"aesthetic": "dark"}

    @pytest.mark.asyncio
    async def test_generate_content_api_error(self, api_client, mock_http_client):
        """Test content generation with API error."""
        # Arrange
        mock_http_client.post.side_effect = httpx.HTTPError("Connection failed")

        # Act & Assert
        with pytest.raises(httpx.HTTPError, match="Failed to generate content"):
            await api_client.generate_content(
                persona_id=str(uuid4()), prompt="Test", content_type="text"
            )

    @pytest.mark.asyncio
    async def test_get_persona_success(self, api_client, mock_http_client):
        """Test successful persona retrieval."""
        # Arrange
        persona_id = str(uuid4())
        expected_response = {
            "id": persona_id,
            "name": "Test Persona",
            "appearance": "Modern style",
            "personality": "Friendly",
            "content_themes": ["art", "tech"],
            "style_preferences": {"aesthetic": "futuristic"},
            "stats": {"generation_count": 42},
        }

        mock_response = MagicMock()
        mock_response.json.return_value = expected_response
        mock_response.raise_for_status = MagicMock()
        mock_http_client.get.return_value = mock_response

        # Act
        result = await api_client.get_persona(persona_id)

        # Assert
        assert result == expected_response
        mock_http_client.get.assert_called_once_with(
            f"http://localhost:8000/api/v1/personas/{persona_id}"
        )

    @pytest.mark.asyncio
    async def test_get_persona_not_found(self, api_client, mock_http_client):
        """Test persona retrieval when persona doesn't exist."""
        # Arrange
        persona_id = str(uuid4())
        error_response = MagicMock()
        error_response.status_code = 404
        mock_http_client.get.side_effect = httpx.HTTPStatusError(
            "Not Found", request=MagicMock(), response=error_response
        )

        # Act & Assert
        with pytest.raises(ValueError, match="Persona not found"):
            await api_client.get_persona(persona_id)

    @pytest.mark.asyncio
    async def test_publish_content_success(self, api_client, mock_http_client):
        """Test successful content publishing."""
        # Arrange
        content_id = str(uuid4())
        platforms = ["instagram", "twitter"]
        expected_response = {
            "content_id": content_id,
            "results": [
                {
                    "platform": "instagram",
                    "status": "success",
                    "post_id": "ig_12345",
                    "url": "https://instagram.com/p/12345",
                },
                {
                    "platform": "twitter",
                    "status": "success",
                    "post_id": "tw_67890",
                    "url": "https://twitter.com/post/67890",
                },
            ],
        }

        mock_response = MagicMock()
        mock_response.json.return_value = expected_response
        mock_response.raise_for_status = MagicMock()
        mock_http_client.post.return_value = mock_response

        # Act
        result = await api_client.publish_content(
            content_id=content_id,
            platforms=platforms,
            caption="Check out this amazing content!",
            hashtags=["ai", "art"],
        )

        # Assert
        assert result == expected_response
        mock_http_client.post.assert_called_once()
        call_args = mock_http_client.post.call_args
        assert call_args[0][0] == "http://localhost:8000/api/v1/social/publish"
        payload = call_args[1]["json"]
        assert payload["content_id"] == content_id
        assert payload["platforms"] == platforms
        assert payload["caption"] == "Check out this amazing content!"
        assert payload["hashtags"] == ["ai", "art"]

    @pytest.mark.asyncio
    async def test_publish_content_with_schedule(self, api_client, mock_http_client):
        """Test content publishing with scheduled time."""
        # Arrange
        mock_response = MagicMock()
        mock_response.json.return_value = {"content_id": str(uuid4()), "results": []}
        mock_response.raise_for_status = MagicMock()
        mock_http_client.post.return_value = mock_response

        # Act
        await api_client.publish_content(
            content_id=str(uuid4()),
            platforms=["instagram"],
            schedule_time="2025-12-25T10:00:00",
            platform_specific={"instagram": {"location": "New York"}},
        )

        # Assert
        call_args = mock_http_client.post.call_args
        payload = call_args[1]["json"]
        assert payload["schedule_time"] == "2025-12-25T10:00:00"
        assert payload["platform_specific"]["instagram"]["location"] == "New York"

    @pytest.mark.asyncio
    async def test_publish_content_api_error(self, api_client, mock_http_client):
        """Test content publishing with API error."""
        # Arrange
        mock_http_client.post.side_effect = httpx.HTTPError("Publishing failed")

        # Act & Assert
        with pytest.raises(httpx.HTTPError, match="Failed to publish content"):
            await api_client.publish_content(
                content_id=str(uuid4()), platforms=["instagram"]
            )

    @pytest.mark.asyncio
    async def test_api_client_close(self, api_client, mock_http_client):
        """Test proper cleanup of HTTP client."""
        # Act
        await api_client.close()

        # Assert
        mock_http_client.aclose.assert_called_once()

    def test_api_client_initialization(self):
        """Test API client initialization with default and custom settings."""
        # Test with defaults
        client1 = GatorAPI(api_key="key1")
        assert client1.api_key == "key1"
        assert client1._base_url == "http://localhost:8000"

        # Test with custom base URL
        client2 = GatorAPI(api_key="key2", base_url="https://api.example.com")
        assert client2.api_key == "key2"
        assert client2._base_url == "https://api.example.com"


class TestPluginBase:
    """Test suite for plugin base classes."""

    def test_plugin_metadata_creation(self):
        """Test plugin metadata model."""
        metadata = PluginMetadata(
            name="Test Plugin",
            version="1.0.0",
            author="Test Author",
            description="A test plugin",
            plugin_type=PluginType.CONTENT_GENERATOR,
            homepage="https://example.com",
            repository="https://github.com/test/plugin",
            tags=["test", "demo"],
            permissions=["content.read", "content.write"],
            dependencies={"requests": "^2.31.0"},
        )

        assert metadata.name == "Test Plugin"
        assert metadata.version == "1.0.0"
        assert metadata.plugin_type == PluginType.CONTENT_GENERATOR
        assert "test" in metadata.tags
        assert "content.read" in metadata.permissions

    def test_plugin_type_enum(self):
        """Test plugin type enumeration."""
        assert PluginType.CONTENT_GENERATOR == "content_generator"
        assert PluginType.SOCIAL_INTEGRATION == "social_integration"
        assert PluginType.ANALYTICS == "analytics"
        assert PluginType.AI_MODEL == "ai_model"
        assert PluginType.WORKFLOW == "workflow"
        assert PluginType.STORAGE_CDN == "storage_cdn"
        assert PluginType.OTHER == "other"


class MockPlugin(GatorPlugin):
    """Mock plugin for testing."""

    async def initialize(self):
        """Initialize mock plugin."""
        self.initialized = True

    async def shutdown(self):
        """Shutdown mock plugin."""
        self.initialized = False


class TestGatorPlugin:
    """Test suite for GatorPlugin base class."""

    def test_plugin_initialization(self):
        """Test plugin initialization with config."""
        config = {"api_key": "test-key", "custom_setting": "value"}
        plugin = MockPlugin(config)

        assert plugin.config == config
        assert plugin.enabled is True
        assert plugin.api.api_key == "test-key"

    @pytest.mark.asyncio
    async def test_plugin_lifecycle(self):
        """Test plugin lifecycle hooks."""
        plugin = MockPlugin({"api_key": "test"})

        # Initialize
        await plugin.initialize()
        assert plugin.initialized is True

        # Shutdown
        await plugin.shutdown()
        assert plugin.initialized is False

    @pytest.mark.asyncio
    async def test_plugin_optional_methods_not_implemented(self):
        """Test that optional plugin methods raise NotImplementedError."""
        plugin = MockPlugin({"api_key": "test"})

        with pytest.raises(NotImplementedError, match="Content generation not supported"):
            await plugin.generate_content({}, "test prompt")

        with pytest.raises(NotImplementedError, match="Analytics not supported"):
            await plugin.get_analytics({})

        with pytest.raises(NotImplementedError, match="Webhook processing not supported"):
            await plugin.process_webhook("event", {})

    @pytest.mark.asyncio
    async def test_plugin_lifecycle_hooks_default_behavior(self):
        """Test that lifecycle hooks have default implementations."""
        plugin = MockPlugin({"api_key": "test"})

        # These should not raise exceptions
        result = await plugin.on_content_generated({"content": "test"})
        assert result == {"content": "test"}  # Should return unchanged

        await plugin.on_persona_created({"persona": "test"})
        await plugin.on_persona_updated({"persona": "test"})
        await plugin.on_post_published({"post": "test"})
        await plugin.on_schedule_created({"schedule": "test"})
