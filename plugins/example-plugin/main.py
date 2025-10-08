"""
Example Content Filter Plugin

Demonstrates the Gator plugin system by implementing a simple content filter
that can block or replace specific keywords in generated content.
"""

from typing import Dict, Any
from backend.plugins import GatorPlugin, PluginMetadata, PluginType


class ExampleContentFilterPlugin(GatorPlugin):
    """
    Example plugin that filters content based on keywords.

    This plugin demonstrates:
    - Plugin initialization and configuration
    - Lifecycle hooks (on_content_generated)
    - Configuration handling
    - Content transformation
    """

    async def initialize(self) -> None:
        """
        Initialize the plugin.

        Loads configuration and sets up plugin metadata.
        """
        # Set plugin metadata
        self.metadata = PluginMetadata(
            name="Example Content Filter Plugin",
            version="1.0.0",
            author="Gator Team",
            description="Example plugin that demonstrates content filtering",
            plugin_type=PluginType.CONTENT_GENERATOR,
            tags=["example", "content", "filter"],
            permissions=["content:read", "content:write"],
        )

        # Load configuration
        self.filter_keywords = self.config.get("filter_keywords", [])
        self.replacement_text = self.config.get("replacement_text", "[filtered]")

        print(f"✅ {self.metadata.name} initialized")
        print(f"   Filter keywords: {self.filter_keywords}")
        print(f"   Replacement text: {self.replacement_text}")

    async def shutdown(self) -> None:
        """Cleanup plugin resources."""
        print(f"🔴 {self.metadata.name} shutting down")

    async def on_content_generated(
        self, content: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Hook called after content is generated.

        Filters content by replacing configured keywords.

        Args:
            content: Generated content data

        Returns:
            Modified content with filtered text
        """
        # Only process text content
        if content.get("type") not in ["text", "caption"]:
            return content

        # Get the text to filter
        text = content.get("text", "")

        # Apply filters
        filtered_text = text
        filtered_count = 0

        for keyword in self.filter_keywords:
            if keyword.lower() in filtered_text.lower():
                # Case-insensitive replacement
                import re

                filtered_text = re.sub(
                    keyword, self.replacement_text, filtered_text, flags=re.IGNORECASE
                )
                filtered_count += 1

        # Update content if anything was filtered
        if filtered_count > 0:
            content["text"] = filtered_text
            content["filtered"] = True
            content["filter_count"] = filtered_count

            print(
                f"🔍 Filtered {filtered_count} keyword(s) from content"
            )

        return content

    async def on_persona_created(self, persona: Dict[str, Any]) -> None:
        """Hook called when new persona is created."""
        print(
            f"👤 New persona created: {persona.get('name', 'Unknown')}"
        )

    async def on_post_published(self, post: Dict[str, Any]) -> None:
        """Hook called when content is published."""
        print(
            f"📤 Content published to {post.get('platform', 'unknown')}"
        )

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate plugin configuration.

        Args:
            config: Configuration to validate

        Returns:
            True if valid, False otherwise
        """
        # Check that filter_keywords is a list if provided
        if "filter_keywords" in config:
            if not isinstance(config["filter_keywords"], list):
                print("❌ filter_keywords must be a list")
                return False

        # Check that replacement_text is a string if provided
        if "replacement_text" in config:
            if not isinstance(config["replacement_text"], str):
                print("❌ replacement_text must be a string")
                return False

        return True


# Export the plugin class
__all__ = ["ExampleContentFilterPlugin"]
