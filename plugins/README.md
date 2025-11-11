# Gator Plugin System

The Gator Plugin System allows developers to extend platform functionality through custom plugins. Plugins can interact with the platform via the GatorAPI client and respond to lifecycle events.

## Features

- **GatorAPI Client**: HTTP-based API client for interacting with platform services
- **Lifecycle Hooks**: Respond to events like content generation, persona creation, and publishing
- **Plugin Manager**: Automatic plugin discovery, loading, and lifecycle management
- **Type Safety**: Built with Pydantic models and type hints
- **Authentication**: API key-based authentication for secure access

## GatorAPI Client

The `GatorAPI` class provides methods to interact with the Gator platform:

### Content Generation

```python
content = await api.generate_content(
    persona_id="uuid-here",
    prompt="Create amazing artwork",
    content_type="image",  # image, video, text, audio, voice
    content_rating="sfw",  # sfw, moderate, nsfw
    target_platforms=["instagram", "twitter"],
    style_override={"aesthetic": "futuristic"}
)
```

### Persona Management

```python
persona = await api.get_persona("persona-uuid")
# Returns: {
#   "id": "uuid",
#   "name": "Persona Name",
#   "appearance": "Description",
#   "personality": "Traits",
#   "content_themes": ["art", "tech"],
#   "style_preferences": {...},
#   "stats": {"generation_count": 42}
# }
```

### Content Publishing

```python
result = await api.publish_content(
    content_id="content-uuid",
    platforms=["instagram", "twitter"],
    caption="Check out my latest creation!",
    hashtags=["ai", "art", "gator"],
    schedule_time="2025-12-25T10:00:00",  # Optional
    platform_specific={
        "instagram": {"location": "New York"}
    }
)
```

## Creating a Plugin

### 1. Create Plugin Directory

```bash
plugins/
  my-plugin/
    main.py          # Plugin implementation
    manifest.json    # Plugin metadata
```

### 2. Implement Plugin Class

```python
# main.py
from typing import Dict, Any
from backend.plugins import GatorPlugin, PluginMetadata, PluginType


class MyPlugin(GatorPlugin):
    """My custom plugin."""

    async def initialize(self) -> None:
        """Initialize plugin resources."""
        self.metadata = PluginMetadata(
            name="My Plugin",
            version="1.0.0",
            author="Your Name",
            description="What your plugin does",
            plugin_type=PluginType.CONTENT_GENERATOR,
            tags=["custom", "example"],
            permissions=["content:read", "content:write"],
        )
        
        # Access configuration
        self.my_setting = self.config.get("my_setting", "default")

    async def shutdown(self) -> None:
        """Cleanup resources."""
        await self.api.close()

    async def on_content_generated(
        self, content: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process generated content."""
        # Modify content here
        content["processed_by"] = self.metadata.name
        return content


# Export plugin class
__all__ = ["MyPlugin"]
```

### 3. Create Manifest

```json
{
  "name": "My Plugin",
  "version": "1.0.0",
  "author": "Your Name",
  "description": "Plugin description",
  "plugin_type": "content_generator",
  "homepage": "https://example.com",
  "license": "MIT",
  "tags": ["custom", "example"],
  "permissions": ["content:read", "content:write"],
  "dependencies": {},
  "config_schema": {
    "type": "object",
    "properties": {
      "api_key": {
        "type": "string",
        "description": "API authentication key"
      },
      "my_setting": {
        "type": "string",
        "description": "Custom setting",
        "default": "default_value"
      }
    },
    "required": ["api_key"]
  }
}
```

## Lifecycle Hooks

Plugins can override these methods to respond to platform events:

### Content Hooks

```python
async def on_content_generated(self, content: Dict[str, Any]) -> Dict[str, Any]:
    """Called after content is generated. Can modify content."""
    return content
```

### Persona Hooks

```python
async def on_persona_created(self, persona: Dict[str, Any]) -> None:
    """Called when new persona is created."""
    pass

async def on_persona_updated(self, persona: Dict[str, Any]) -> None:
    """Called when persona is updated."""
    pass
```

### Publishing Hooks

```python
async def on_post_published(self, post: Dict[str, Any]) -> None:
    """Called when content is published to social media."""
    pass

async def on_schedule_created(self, schedule: Dict[str, Any]) -> None:
    """Called when content schedule is created."""
    pass
```

## Custom Methods

Plugins can implement custom methods for specific functionality:

### Content Generation

```python
async def generate_content(
    self, persona: Dict[str, Any], prompt: str, **kwargs
) -> Dict[str, Any]:
    """Custom content generation logic."""
    return await self.api.generate_content(
        persona_id=persona["id"],
        prompt=prompt,
        content_type=kwargs.get("content_type", "image")
    )
```

### Analytics

```python
async def get_analytics(self, params: Dict[str, Any]) -> Dict[str, Any]:
    """Provide custom analytics."""
    return {
        "metric1": 100,
        "metric2": 200
    }
```

### Webhooks

```python
async def process_webhook(
    self, event: str, data: Dict[str, Any]
) -> Dict[str, Any]:
    """Handle webhook events from external services."""
    return {"status": "processed"}
```

## Plugin Types

- `CONTENT_GENERATOR`: Content generation and modification
- `SOCIAL_INTEGRATION`: Social media platform integration
- `ANALYTICS`: Analytics and reporting
- `AI_MODEL`: Custom AI model integration
- `WORKFLOW`: Workflow automation
- `STORAGE_CDN`: Storage and CDN integration
- `OTHER`: Other functionality

## Example Plugins

### 1. Example Content Filter Plugin

Location: `plugins/example-plugin/`

Demonstrates:
- Content filtering and keyword replacement
- Configuration handling
- Lifecycle hook implementation

### 2. Example API Integration Plugin

Location: `plugins/example-api-plugin/`

Demonstrates:
- Using GatorAPI to generate content
- Fetching persona information
- Publishing content to social media
- Error handling and analytics

## Loading Plugins

```python
from backend.plugins.manager import plugin_manager
from pathlib import Path

# Discover plugins
plugins = await plugin_manager.discover_plugins()

# Load a plugin
await plugin_manager.load_plugin(
    plugin_id="my-plugin",
    plugin_path=Path("plugins/my-plugin"),
    config={
        "api_key": "your-api-key",
        "my_setting": "custom_value"
    }
)

# Execute hooks
results = await plugin_manager.execute_hook(
    "on_content_generated",
    content_data
)
```

## Testing

Run plugin tests:

```bash
python -m pytest tests/unit/test_plugin_api.py -v
```

## Best Practices

1. **Error Handling**: Always wrap API calls in try-except blocks
2. **Resource Cleanup**: Call `await self.api.close()` in `shutdown()`
3. **Configuration Validation**: Validate config in `initialize()`
4. **Type Safety**: Use type hints for all methods
5. **Documentation**: Document your plugin's functionality
6. **Testing**: Write tests for your plugin logic

## API Reference

Full API documentation available at:
- GatorAPI: `src/backend/plugins/__init__.py`
- Plugin Manager: `src/backend/plugins/manager.py`
- Plugin Models: `src/backend/models/plugin.py`

## Support

For questions or issues:
- GitHub Issues: https://github.com/terminills/gator/issues
- Documentation: See repository README.md
