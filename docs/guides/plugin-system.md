# Gator Plugin System

## Overview

The Gator Plugin System enables developers to extend the platform's functionality through a well-defined API. Plugins can add new content generation capabilities, integrate with external services, provide analytics, and more.

## Architecture

```
┌────────────────────────────────────────────┐
│         Gator Core Platform                │
├────────────────────────────────────────────┤
│                                            │
│  ┌──────────────────────────────────────┐ │
│  │      Plugin Manager                   │ │
│  │  - Discovery & Loading                │ │
│  │  - Lifecycle Management               │ │
│  │  - Hook Execution                     │ │
│  └──────────────────────────────────────┘ │
│                                            │
│  ┌──────────────────────────────────────┐ │
│  │      Plugin API Gateway               │ │
│  │  - Authentication                     │ │
│  │  - Rate Limiting                      │ │
│  │  - Resource Access                    │ │
│  └──────────────────────────────────────┘ │
│                                            │
└────────────────┬───────────────────────────┘
                 │
    ┌────────────┼────────────┐
    │            │            │
    ↓            ↓            ↓
┌─────────┐  ┌─────────┐  ┌─────────┐
│Plugin A │  │Plugin B │  │Plugin C │
└─────────┘  └─────────┘  └─────────┘
```

## Plugin Types

### 1. Content Generators
Extend content generation with new styles, formats, or models.

**Examples:**
- Custom image styles (cyberpunk, watercolor, etc.)
- Video generation providers (Runway, Pika)
- Voice cloning services
- Text-to-3D model generation

### 2. Social Media Integrations
Connect to additional platforms or enhance existing integrations.

**Examples:**
- TikTok publisher
- Pinterest automation
- LinkedIn scheduler
- Discord bot integration

### 3. Analytics Extensions
Provide advanced analytics and insights.

**Examples:**
- Predictive engagement AI
- Competitor analysis
- ROI calculators
- Custom dashboards

### 4. AI Model Integrations
Integrate new AI models and services.

**Examples:**
- Custom LLM providers
- Specialized image models
- Translation services
- Content moderation models

### 5. Workflow Automation
Automate repetitive tasks and create complex workflows.

**Examples:**
- Auto-responders
- Content repurposing pipelines
- Multi-platform publishing rules
- A/B testing automation

## Creating a Plugin

### 1. Plugin Structure

```
my-plugin/
├── manifest.json       # Plugin metadata
├── main.py            # Plugin implementation
├── requirements.txt   # Python dependencies (optional)
└── README.md          # Plugin documentation
```

### 2. Manifest File (manifest.json)

```json
{
  "name": "My Awesome Plugin",
  "slug": "my-awesome-plugin",
  "version": "1.0.0",
  "author": "Your Name",
  "author_email": "you@example.com",
  "description": "Plugin description",
  "plugin_type": "content_generator",
  "homepage": "https://github.com/yourusername/my-plugin",
  "repository": "https://github.com/yourusername/my-plugin",
  "license": "MIT",
  "tags": ["content", "generator", "custom"],
  "permissions": [
    "content:read",
    "content:write",
    "persona:read"
  ],
  "dependencies": {
    "python": ">=3.9",
    "packages": [
      "requests>=2.28.0",
      "pillow>=9.0.0"
    ]
  },
  "config_schema": {
    "type": "object",
    "properties": {
      "api_key": {
        "type": "string",
        "description": "API key for external service"
      },
      "model": {
        "type": "string",
        "default": "default",
        "description": "Model to use"
      }
    },
    "required": ["api_key"]
  }
}
```

### 3. Plugin Implementation (main.py)

```python
from typing import Dict, Any
from backend.plugins import GatorPlugin, PluginMetadata, PluginType

class MyAwesomePlugin(GatorPlugin):
    """My awesome plugin implementation."""
    
    async def initialize(self) -> None:
        """Initialize plugin resources."""
        # Set metadata
        self.metadata = PluginMetadata(
            name="My Awesome Plugin",
            version="1.0.0",
            author="Your Name",
            description="Plugin description",
            plugin_type=PluginType.CONTENT_GENERATOR,
            tags=["content", "generator"],
            permissions=["content:read", "content:write"],
        )
        
        # Load configuration
        self.api_key = self.config.get("api_key")
        self.model = self.config.get("model", "default")
        
        # Initialize resources
        print(f"✅ {self.metadata.name} initialized")
    
    async def shutdown(self) -> None:
        """Cleanup resources."""
        print(f"🔴 {self.metadata.name} shutting down")
    
    # Optional: Implement lifecycle hooks
    
    async def on_content_generated(
        self, content: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process generated content.
        
        Args:
            content: Generated content data
            
        Returns:
            Modified content (or unchanged)
        """
        # Add your content processing logic
        if content.get("type") == "image":
            # Process image content
            content["processed_by"] = self.metadata.name
        
        return content
    
    async def on_persona_created(self, persona: Dict[str, Any]) -> None:
        """Called when new persona is created."""
        pass
    
    async def on_post_published(self, post: Dict[str, Any]) -> None:
        """Called when content is published."""
        pass
    
    # Optional: Implement custom methods
    
    async def generate_content(
        self,
        persona: Dict[str, Any],
        prompt: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Custom content generation.
        
        Args:
            persona: Persona data
            prompt: Generation prompt
            **kwargs: Additional parameters
            
        Returns:
            Generated content data
        """
        # Implement your custom generation logic
        return {
            "type": "text",
            "text": f"Generated content for: {prompt}",
            "generator": self.metadata.name,
        }
```

## Lifecycle Hooks

Plugins can implement the following hooks to react to platform events:

### on_content_generated(content)
Called after content is generated. Use this to:
- Post-process content
- Apply filters or transformations
- Add metadata
- Validate output

### on_persona_created(persona)
Called when a new persona is created.

### on_persona_updated(persona)
Called when a persona is updated.

### on_post_published(post)
Called when content is published to social media.

### on_schedule_created(schedule)
Called when a new content schedule is created.

## API Access

Plugins have access to the Gator platform through the `GatorAPI` client:

```python
# In your plugin
async def some_method(self):
    # Generate content
    content = await self.api.generate_content(
        persona_id="persona-uuid",
        prompt="Generate an image",
        content_type="image"
    )
    
    # Get persona data
    persona = await self.api.get_persona("persona-uuid")
    
    # Publish content
    result = await self.api.publish_content(
        content_id="content-uuid",
        platforms=["instagram", "twitter"]
    )
```

## API Endpoints

### Marketplace

```http
GET /api/v1/plugins/marketplace
  ?plugin_type=content_generator
  &featured=true
  &search=keyword
  &skip=0
  &limit=50
```

Get available plugins from marketplace.

```http
GET /api/v1/plugins/marketplace/{plugin_slug}
```

Get details of a specific plugin.

### Installation Management

```http
GET /api/v1/plugins/installed
  ?enabled=true
```

List installed plugins.

```http
POST /api/v1/plugins/install
Content-Type: application/json

{
  "plugin_slug": "my-plugin",
  "config": {
    "api_key": "your-key",
    "setting": "value"
  }
}
```

Install a plugin.

```http
PUT /api/v1/plugins/installed/{installation_id}
Content-Type: application/json

{
  "config": {
    "api_key": "updated-key"
  },
  "enabled": true
}
```

Update plugin configuration.

```http
DELETE /api/v1/plugins/installed/{installation_id}
```

Uninstall a plugin.

### Reviews

```http
GET /api/v1/plugins/marketplace/{plugin_slug}/reviews
  ?skip=0
  &limit=50
```

List plugin reviews.

```http
POST /api/v1/plugins/marketplace/{plugin_slug}/reviews
Content-Type: application/json

{
  "rating": 5,
  "title": "Great plugin!",
  "review_text": "This plugin is amazing..."
}
```

Create a review.

## Testing Your Plugin

Create a test file to validate your plugin:

```python
import pytest
from my_plugin.main import MyAwesomePlugin

@pytest.mark.asyncio
async def test_plugin_initialization():
    """Test plugin initializes correctly."""
    config = {"api_key": "test-key", "model": "test-model"}
    plugin = MyAwesomePlugin(config)
    
    await plugin.initialize()
    
    assert plugin.metadata is not None
    assert plugin.api_key == "test-key"
    assert plugin.model == "test-model"
    
    await plugin.shutdown()

@pytest.mark.asyncio
async def test_content_hook():
    """Test content generation hook."""
    plugin = MyAwesomePlugin({"api_key": "test"})
    await plugin.initialize()
    
    content = {"type": "image", "data": "test"}
    result = await plugin.on_content_generated(content)
    
    assert "processed_by" in result
    
    await plugin.shutdown()
```

## Best Practices

### 1. Resource Management
- Always cleanup resources in `shutdown()`
- Use async/await for I/O operations
- Implement proper error handling

### 2. Configuration
- Validate configuration in `validate_config()`
- Provide sensible defaults
- Document all configuration options

### 3. Error Handling
- Never crash the platform
- Log errors appropriately
- Return meaningful error messages

### 4. Performance
- Cache expensive operations
- Implement rate limiting for API calls
- Be mindful of memory usage

### 5. Security
- Validate all inputs
- Use secure API credentials
- Request only necessary permissions

## Example: Content Filter Plugin

See `plugins/example-plugin/` for a complete working example that demonstrates:
- Plugin initialization and configuration
- Content filtering and transformation
- Lifecycle hook implementation
- Configuration validation

## Support

For questions or issues with plugin development:
1. Review the plugin specification
2. Check example plugins
3. Open an issue on GitHub
4. Contact the development team

## License

Plugins can use any license, but we recommend:
- MIT for maximum compatibility
- Apache 2.0 for patent protection
- GPL for copyleft requirements

---

**Last Updated**: January 2025
**Version**: 1.0.0
