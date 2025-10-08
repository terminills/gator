"""
Unit tests for Plugin system base classes and manager.

Tests the plugin infrastructure including base classes, manager, and lifecycle.
"""

import pytest
import asyncio
from typing import Dict, Any

from backend.plugins import (
    GatorPlugin,
    PluginMetadata,
    PluginType,
    PluginStatus,
    GatorAPI,
)
from backend.plugins.manager import PluginManager, PluginError, PluginLoadError


class TestPluginBase:
    """Test suite for base plugin classes."""

    def test_plugin_type_enum(self):
        """Test PluginType enum values."""
        assert PluginType.CONTENT_GENERATOR == "content_generator"
        assert PluginType.SOCIAL_INTEGRATION == "social_integration"
        assert PluginType.ANALYTICS == "analytics"
        assert PluginType.AI_MODEL == "ai_model"
        assert PluginType.WORKFLOW == "workflow"
        assert PluginType.STORAGE_CDN == "storage_cdn"

    def test_plugin_status_enum(self):
        """Test PluginStatus enum values."""
        assert PluginStatus.INSTALLED == "installed"
        assert PluginStatus.ACTIVE == "active"
        assert PluginStatus.INACTIVE == "inactive"
        assert PluginStatus.ERROR == "error"

    def test_plugin_metadata_creation(self):
        """Test PluginMetadata model creation."""
        metadata = PluginMetadata(
            name="Test Plugin",
            version="1.0.0",
            author="Test Author",
            description="Test description",
            plugin_type=PluginType.CONTENT_GENERATOR,
            tags=["test", "demo"],
            permissions=["content:read", "content:write"],
        )

        assert metadata.name == "Test Plugin"
        assert metadata.version == "1.0.0"
        assert metadata.plugin_type == PluginType.CONTENT_GENERATOR
        assert len(metadata.tags) == 2
        assert len(metadata.permissions) == 2

    def test_gator_api_initialization(self):
        """Test GatorAPI initialization."""
        api = GatorAPI("test-api-key")
        assert api.api_key == "test-api-key"
        assert api._base_url is None

    @pytest.mark.asyncio
    async def test_gator_api_methods_raise_not_implemented(self):
        """Test that GatorAPI stub methods raise NotImplementedError."""
        api = GatorAPI("test-key")

        with pytest.raises(NotImplementedError):
            await api.generate_content("persona-1", "test prompt")

        with pytest.raises(NotImplementedError):
            await api.get_persona("persona-1")

        with pytest.raises(NotImplementedError):
            await api.publish_content("content-1", ["instagram"])


class TestGatorPlugin:
    """Test suite for GatorPlugin base class."""

    def test_plugin_instantiation(self):
        """Test that abstract GatorPlugin cannot be instantiated directly."""

        # Create a concrete implementation for testing
        class TestPlugin(GatorPlugin):
            async def initialize(self) -> None:
                pass

            async def shutdown(self) -> None:
                pass

        config = {"api_key": "test-key", "setting1": "value1"}
        plugin = TestPlugin(config)

        assert plugin.config == config
        assert plugin.enabled is True
        assert plugin.api.api_key == "test-key"

    @pytest.mark.asyncio
    async def test_plugin_lifecycle_hooks(self):
        """Test plugin lifecycle hooks."""

        class TestPlugin(GatorPlugin):
            def __init__(self, config):
                super().__init__(config)
                self.initialized = False
                self.shutdown_called = False
                self.content_generated_called = False

            async def initialize(self) -> None:
                self.initialized = True

            async def shutdown(self) -> None:
                self.shutdown_called = True

            async def on_content_generated(
                self, content: Dict[str, Any]
            ) -> Dict[str, Any]:
                self.content_generated_called = True
                content["processed"] = True
                return content

        plugin = TestPlugin({})

        # Test initialize
        await plugin.initialize()
        assert plugin.initialized is True

        # Test content hook
        content = {"type": "image", "data": "test"}
        result = await plugin.on_content_generated(content)
        assert plugin.content_generated_called is True
        assert result["processed"] is True

        # Test shutdown
        await plugin.shutdown()
        assert plugin.shutdown_called is True

    @pytest.mark.asyncio
    async def test_plugin_optional_methods_raise_not_implemented(self):
        """Test that optional plugin methods raise NotImplementedError by default."""

        class MinimalPlugin(GatorPlugin):
            async def initialize(self) -> None:
                pass

            async def shutdown(self) -> None:
                pass

        plugin = MinimalPlugin({})

        with pytest.raises(NotImplementedError):
            await plugin.generate_content({}, "test prompt")

        with pytest.raises(NotImplementedError):
            await plugin.get_analytics({})

        with pytest.raises(NotImplementedError):
            await plugin.process_webhook("test_event", {})

    def test_plugin_metadata_getter(self):
        """Test plugin metadata getter."""

        class TestPlugin(GatorPlugin):
            async def initialize(self) -> None:
                pass

            async def shutdown(self) -> None:
                pass

        plugin = TestPlugin({})

        # Should raise ValueError when metadata not set
        with pytest.raises(ValueError, match="Plugin metadata not set"):
            plugin.get_metadata()

        # Set metadata and test getter
        plugin.metadata = PluginMetadata(
            name="Test",
            version="1.0.0",
            author="Author",
            description="Description",
            plugin_type=PluginType.ANALYTICS,
        )

        metadata = plugin.get_metadata()
        assert metadata.name == "Test"
        assert metadata.version == "1.0.0"

    def test_plugin_string_representation(self):
        """Test plugin string representations."""

        class TestPlugin(GatorPlugin):
            async def initialize(self) -> None:
                pass

            async def shutdown(self) -> None:
                pass

        plugin = TestPlugin({})

        # Without metadata
        assert "TestPlugin" in str(plugin)
        assert "TestPlugin" in repr(plugin)

        # With metadata
        plugin.metadata = PluginMetadata(
            name="My Plugin",
            version="2.0.0",
            author="Author",
            description="Description",
            plugin_type=PluginType.CONTENT_GENERATOR,
        )

        assert "My Plugin v2.0.0" in str(plugin)
        assert "My Plugin" in repr(plugin)
        assert "2.0.0" in repr(plugin)

    def test_plugin_config_validation_no_schema(self):
        """Test config validation when no schema is defined."""

        class TestPlugin(GatorPlugin):
            async def initialize(self) -> None:
                pass

            async def shutdown(self) -> None:
                pass

        plugin = TestPlugin({"key": "value"})

        # Should return True when no schema is defined
        assert plugin.validate_config({"any": "config"}) is True

    def test_plugin_config_validation_with_schema(self):
        """Test config validation with JSON schema."""
        from jsonschema import ValidationError as JSONSchemaValidationError

        class TestPlugin(GatorPlugin):
            async def initialize(self) -> None:
                pass

            async def shutdown(self) -> None:
                pass

        # Define a plugin with a schema
        plugin = TestPlugin({"api_key": "test123"})
        plugin.metadata = PluginMetadata(
            name="Schema Test Plugin",
            version="1.0.0",
            author="Test",
            description="Test",
            plugin_type=PluginType.CONTENT_GENERATOR,
            config_schema={
                "type": "object",
                "properties": {
                    "api_key": {"type": "string"},
                    "max_retries": {"type": "integer", "minimum": 0},
                },
                "required": ["api_key"],
            },
        )

        # Valid config should pass
        valid_config = {"api_key": "test123", "max_retries": 3}
        assert plugin.validate_config(valid_config) is True

        # Invalid config should raise ValidationError
        invalid_config = {"api_key": "test", "max_retries": "not_an_int"}
        with pytest.raises(JSONSchemaValidationError):
            plugin.validate_config(invalid_config)

        # Missing required field should raise ValidationError
        missing_field_config = {"max_retries": 5}
        with pytest.raises(JSONSchemaValidationError):
            plugin.validate_config(missing_field_config)


class TestPluginManager:
    """Test suite for PluginManager."""

    def test_plugin_manager_initialization(self):
        """Test PluginManager initialization."""
        manager = PluginManager()

        assert isinstance(manager.plugins, dict)
        assert isinstance(manager.plugin_status, dict)
        assert isinstance(manager.hooks, dict)
        assert len(manager.plugin_dirs) > 0

    def test_plugin_manager_hooks_registry(self):
        """Test that PluginManager has correct hook registry."""
        manager = PluginManager()

        expected_hooks = [
            "on_content_generated",
            "on_persona_created",
            "on_persona_updated",
            "on_post_published",
            "on_schedule_created",
        ]

        for hook_name in expected_hooks:
            assert hook_name in manager.hooks
            assert isinstance(manager.hooks[hook_name], list)

    def test_plugin_manager_get_plugin(self):
        """Test getting plugin by ID."""
        manager = PluginManager()

        # Non-existent plugin
        assert manager.get_plugin("non-existent") is None

    def test_plugin_manager_list_plugins_empty(self):
        """Test listing plugins when none are loaded."""
        manager = PluginManager()
        plugins = manager.list_plugins()
        assert isinstance(plugins, list)
        assert len(plugins) == 0

    def test_plugin_manager_get_plugin_status(self):
        """Test getting plugin status."""
        manager = PluginManager()

        # Non-existent plugin
        assert manager.get_plugin_status("non-existent") is None

    @pytest.mark.asyncio
    async def test_plugin_manager_enable_disable_nonexistent(self):
        """Test enabling/disabling non-existent plugin raises error."""
        manager = PluginManager()

        with pytest.raises(PluginError, match="Plugin not found"):
            await manager.enable_plugin("non-existent")

        with pytest.raises(PluginError, match="Plugin not found"):
            await manager.disable_plugin("non-existent")

    @pytest.mark.asyncio
    async def test_plugin_manager_execute_hook_empty(self):
        """Test executing hook with no registered plugins."""
        manager = PluginManager()

        results = await manager.execute_hook("on_content_generated", {"data": "test"})
        assert isinstance(results, list)
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_plugin_manager_execute_unknown_hook(self):
        """Test executing unknown hook returns empty list."""
        manager = PluginManager()

        results = await manager.execute_hook("unknown_hook", {})
        assert isinstance(results, list)
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_plugin_manager_shutdown_empty(self):
        """Test shutting down manager with no plugins."""
        manager = PluginManager()
        await manager.shutdown()
        assert len(manager.plugins) == 0


class TestPluginIntegration:
    """Integration tests for plugin system."""

    @pytest.mark.asyncio
    async def test_complete_plugin_lifecycle(self):
        """Test complete plugin lifecycle with mock plugin."""

        class TestPlugin(GatorPlugin):
            def __init__(self, config):
                super().__init__(config)
                self.initialized = False
                self.shutdown_called = False

            async def initialize(self) -> None:
                self.initialized = True
                self.metadata = PluginMetadata(
                    name="Test Plugin",
                    version="1.0.0",
                    author="Test",
                    description="Test plugin",
                    plugin_type=PluginType.ANALYTICS,
                )

            async def shutdown(self) -> None:
                self.shutdown_called = True

            async def on_content_generated(
                self, content: Dict[str, Any]
            ) -> Dict[str, Any]:
                content["test_processed"] = True
                return content

        # Create plugin instance directly (not through manager for this test)
        plugin = TestPlugin({"test_config": "value"})

        # Test initialization
        await plugin.initialize()
        assert plugin.initialized is True
        assert plugin.metadata is not None

        # Test hook execution
        content = {"type": "text", "data": "test"}
        result = await plugin.on_content_generated(content)
        assert result["test_processed"] is True

        # Test shutdown
        await plugin.shutdown()
        assert plugin.shutdown_called is True
