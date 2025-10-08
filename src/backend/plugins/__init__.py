"""
Plugin System Base Classes

Base classes and interfaces for the Gator Plugin/Marketplace system.
Enables developers to create custom plugins that extend platform functionality.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from enum import Enum
from pydantic import BaseModel, Field
import jsonschema
from jsonschema import ValidationError as JSONSchemaValidationError


class PluginType(str, Enum):
    """Plugin type classification."""

    CONTENT_GENERATOR = "content_generator"
    SOCIAL_INTEGRATION = "social_integration"
    ANALYTICS = "analytics"
    AI_MODEL = "ai_model"
    WORKFLOW = "workflow"
    STORAGE_CDN = "storage_cdn"
    OTHER = "other"


class PluginStatus(str, Enum):
    """Plugin lifecycle status."""

    INSTALLED = "installed"
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    UPDATING = "updating"


class PluginMetadata(BaseModel):
    """Plugin metadata and configuration."""

    name: str = Field(..., description="Plugin name")
    version: str = Field(..., description="Plugin version (semver)")
    author: str = Field(..., description="Plugin author name")
    description: str = Field(..., description="Plugin description")
    plugin_type: PluginType = Field(..., description="Plugin type classification")
    homepage: Optional[str] = Field(None, description="Plugin homepage URL")
    repository: Optional[str] = Field(None, description="Plugin repository URL")
    license: str = Field(default="MIT", description="Plugin license")
    tags: List[str] = Field(default_factory=list, description="Plugin tags")
    permissions: List[str] = Field(
        default_factory=list, description="Required permissions"
    )
    dependencies: Dict[str, str] = Field(
        default_factory=dict, description="Plugin dependencies"
    )
    config_schema: Optional[Dict[str, Any]] = Field(
        None, description="Plugin configuration schema"
    )


class GatorAPI:
    """
    API client for plugins to interact with Gator platform.

    Provides authenticated access to platform features and data.
    """

    def __init__(self, api_key: str):
        """
        Initialize API client.

        Args:
            api_key: Plugin API key for authentication
        """
        self.api_key = api_key
        self._base_url = None  # Will be set by plugin manager

    async def generate_content(
        self,
        persona_id: str,
        prompt: str,
        content_type: str = "text",
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate content using platform's content generation service.

        Args:
            persona_id: Persona UUID
            prompt: Content generation prompt
            content_type: Type of content to generate
            **kwargs: Additional generation parameters

        Returns:
            Generated content data

        Raises:
            NotImplementedError: This is a stub for plugin development
        """
        raise NotImplementedError(
            "API client methods will be implemented by plugin manager"
        )

    async def get_persona(self, persona_id: str) -> Dict[str, Any]:
        """
        Fetch persona data.

        Args:
            persona_id: Persona UUID

        Returns:
            Persona data dictionary

        Raises:
            NotImplementedError: This is a stub for plugin development
        """
        raise NotImplementedError(
            "API client methods will be implemented by plugin manager"
        )

    async def publish_content(
        self,
        content_id: str,
        platforms: List[str],
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Publish content to social media platforms.

        Args:
            content_id: Content UUID
            platforms: List of platform names
            **kwargs: Platform-specific parameters

        Returns:
            Publishing results

        Raises:
            NotImplementedError: This is a stub for plugin development
        """
        raise NotImplementedError(
            "API client methods will be implemented by plugin manager"
        )


class GatorPlugin(ABC):
    """
    Base class for all Gator plugins.

    Plugins extend platform functionality by implementing this interface
    and defining lifecycle hooks and custom methods.

    Example:
        ```python
        class CustomImageStylePlugin(GatorPlugin):
            async def initialize(self) -> None:
                self.style_model = await load_model()

            async def on_content_generated(self, content: Dict) -> Dict:
                if content['type'] == 'image':
                    content['data'] = self.apply_style(content['data'])
                return content
        ```
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize plugin with configuration.

        Args:
            config: Plugin configuration dictionary
        """
        self.config = config
        self.api = GatorAPI(config.get("api_key", ""))
        self.enabled = True
        self.metadata: Optional[PluginMetadata] = None

    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize plugin resources.

        Called when plugin is loaded. Use this to:
        - Load models or external resources
        - Establish connections
        - Validate configuration
        - Prepare plugin state

        Raises:
            Exception: If initialization fails
        """
        pass

    @abstractmethod
    async def shutdown(self) -> None:
        """
        Cleanup plugin resources.

        Called when plugin is unloaded or system shuts down. Use this to:
        - Release resources
        - Close connections
        - Save state
        - Cleanup temporary files
        """
        pass

    # Optional lifecycle hooks - override as needed

    async def on_content_generated(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """
        Hook called after content is generated.

        Use this to:
        - Post-process generated content
        - Apply filters or transformations
        - Add metadata
        - Validate output

        Args:
            content: Generated content data

        Returns:
            Modified content data (or unchanged if no modifications)
        """
        return content

    async def on_persona_created(self, persona: Dict[str, Any]) -> None:
        """
        Hook called when new persona is created.

        Args:
            persona: Persona data
        """
        pass

    async def on_persona_updated(self, persona: Dict[str, Any]) -> None:
        """
        Hook called when persona is updated.

        Args:
            persona: Updated persona data
        """
        pass

    async def on_post_published(self, post: Dict[str, Any]) -> None:
        """
        Hook called when content is published to social media.

        Args:
            post: Published post data including platform and post ID
        """
        pass

    async def on_schedule_created(self, schedule: Dict[str, Any]) -> None:
        """
        Hook called when new content schedule is created.

        Args:
            schedule: Schedule data
        """
        pass

    # Optional custom methods - implement if plugin provides specific features

    async def generate_content(
        self,
        persona: Dict[str, Any],
        prompt: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate content using plugin's custom logic.

        Override this if plugin provides custom content generation.

        Args:
            persona: Persona data
            prompt: Generation prompt
            **kwargs: Additional parameters

        Returns:
            Generated content data

        Raises:
            NotImplementedError: If plugin doesn't support content generation
        """
        raise NotImplementedError("Content generation not supported by this plugin")

    async def get_analytics(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get custom analytics data.

        Override this if plugin provides custom analytics.

        Args:
            params: Analytics parameters

        Returns:
            Analytics data

        Raises:
            NotImplementedError: If plugin doesn't support analytics
        """
        raise NotImplementedError("Analytics not supported by this plugin")

    async def process_webhook(
        self,
        event: str,
        data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Process webhook events from external services.

        Override this if plugin needs to handle webhooks.

        Args:
            event: Event type
            data: Event data

        Returns:
            Processing result

        Raises:
            NotImplementedError: If plugin doesn't support webhooks
        """
        raise NotImplementedError("Webhook processing not supported by this plugin")

    def get_metadata(self) -> PluginMetadata:
        """
        Get plugin metadata.

        Returns:
            Plugin metadata

        Raises:
            ValueError: If metadata not set
        """
        if self.metadata is None:
            raise ValueError("Plugin metadata not set")
        return self.metadata

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate plugin configuration against JSON schema.

        Override this to add custom validation logic.

        Args:
            config: Configuration to validate

        Returns:
            True if valid, False otherwise

        Raises:
            JSONSchemaValidationError: If configuration is invalid
        """
        # Basic validation - check required fields
        if self.metadata and self.metadata.config_schema:
            try:
                # Validate configuration against JSON schema
                jsonschema.validate(instance=config, schema=self.metadata.config_schema)
                return True
            except JSONSchemaValidationError as e:
                # Log validation error and re-raise for caller to handle
                raise JSONSchemaValidationError(
                    f"Plugin configuration validation failed: {e.message}"
                )
        return True

    def __str__(self) -> str:
        """String representation of plugin."""
        if self.metadata:
            return f"{self.metadata.name} v{self.metadata.version}"
        return f"GatorPlugin({self.__class__.__name__})"

    def __repr__(self) -> str:
        """Detailed representation of plugin."""
        if self.metadata:
            return (
                f"GatorPlugin(name='{self.metadata.name}', "
                f"version='{self.metadata.version}', "
                f"type='{self.metadata.plugin_type}', "
                f"enabled={self.enabled})"
            )
        return f"GatorPlugin(class='{self.__class__.__name__}', enabled={self.enabled})"
