"""
Example API Integration Plugin

Demonstrates how to use the GatorAPI client to interact with the platform.
Shows content generation, persona retrieval, and content publishing.
"""

from typing import Dict, Any
from backend.plugins import GatorPlugin, PluginMetadata, PluginType


class ExampleAPIPlugin(GatorPlugin):
    """
    Example plugin demonstrating GatorAPI usage.

    This plugin shows how to:
    - Use GatorAPI to generate content
    - Fetch persona information
    - Publish content to social media
    - Handle API errors gracefully
    """

    async def initialize(self) -> None:
        """Initialize the plugin and set up API client."""
        # Set plugin metadata
        self.metadata = PluginMetadata(
            name="Example API Integration Plugin",
            version="1.0.0",
            author="Gator Team",
            description="Example plugin demonstrating GatorAPI usage",
            plugin_type=PluginType.CONTENT_GENERATOR,
            tags=["example", "api", "integration"],
            permissions=["content:read", "content:write", "persona:read", "social:publish"],
        )

        # Load configuration
        self.auto_publish = self.config.get("auto_publish", False)
        self.target_platforms = self.config.get("target_platforms", ["twitter"])
        
        print(f"✅ {self.metadata.name} initialized")
        print(f"   Auto-publish: {self.auto_publish}")
        print(f"   Target platforms: {self.target_platforms}")

    async def shutdown(self) -> None:
        """Cleanup plugin resources."""
        # Close the API client connection
        await self.api.close()
        print(f"🔴 {self.metadata.name} shutting down")

    async def generate_content(
        self, persona: Dict[str, Any], prompt: str, **kwargs
    ) -> Dict[str, Any]:
        """
        Generate content using the platform's API.

        This demonstrates using GatorAPI to generate content programmatically.

        Args:
            persona: Persona data dictionary
            prompt: Content generation prompt
            **kwargs: Additional generation parameters

        Returns:
            Generated content data
        """
        try:
            print(f"🎨 Generating content for persona: {persona.get('name', 'Unknown')}")
            
            # Use GatorAPI to generate content
            content = await self.api.generate_content(
                persona_id=persona["id"],
                prompt=prompt,
                content_type=kwargs.get("content_type", "image"),
                content_rating=kwargs.get("content_rating", "sfw"),
                target_platforms=kwargs.get("target_platforms", self.target_platforms),
            )
            
            print(f"✅ Content generated: {content.get('id')}")
            
            # Auto-publish if enabled
            if self.auto_publish and content.get("id"):
                await self._publish_content(content)
            
            return content
            
        except Exception as e:
            print(f"❌ Content generation failed: {str(e)}")
            raise

    async def on_persona_created(self, persona: Dict[str, Any]) -> None:
        """
        Hook called when new persona is created.
        
        Demonstrates fetching persona details via API.
        """
        try:
            print(f"👤 New persona created: {persona.get('name', 'Unknown')}")
            
            # Fetch full persona details using GatorAPI
            if persona.get("id"):
                full_persona = await self.api.get_persona(persona["id"])
                print(f"   Themes: {full_persona.get('content_themes', [])}")
                print(f"   Style: {full_persona.get('style_preferences', {}).get('aesthetic', 'default')}")
                
        except Exception as e:
            print(f"⚠️ Could not fetch persona details: {str(e)}")

    async def on_content_generated(
        self, content: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Hook called after content is generated.
        
        Can modify content or trigger additional actions.
        """
        print(f"📝 Content generated: {content.get('id')} (type: {content.get('type')})")
        
        # Add custom metadata
        content["plugin_processed"] = True
        content["plugin_name"] = self.metadata.name
        
        return content

    async def on_post_published(self, post: Dict[str, Any]) -> None:
        """Hook called when content is published to social media."""
        platform = post.get("platform", "unknown")
        post_id = post.get("post_id", "N/A")
        print(f"📤 Content published to {platform} (post ID: {post_id})")

    async def _publish_content(self, content: Dict[str, Any]) -> None:
        """
        Internal method to publish content to social media.
        
        Demonstrates using GatorAPI to publish content.
        """
        try:
            content_id = content.get("id")
            if not content_id:
                print("⚠️ Cannot publish: No content ID")
                return
            
            print(f"📤 Publishing content {content_id} to platforms: {self.target_platforms}")
            
            # Use GatorAPI to publish content
            result = await self.api.publish_content(
                content_id=content_id,
                platforms=self.target_platforms,
                caption=content.get("description", "Check out my latest creation!"),
                hashtags=["ai", "gator", "content"],
            )
            
            # Check results
            for platform_result in result.get("results", []):
                platform = platform_result.get("platform")
                status = platform_result.get("status")
                
                if status == "success":
                    post_url = platform_result.get("url", "N/A")
                    print(f"✅ Published to {platform}: {post_url}")
                else:
                    error = platform_result.get("error", "Unknown error")
                    print(f"❌ Failed to publish to {platform}: {error}")
                    
        except Exception as e:
            print(f"❌ Publishing failed: {str(e)}")

    async def get_analytics(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get custom analytics data.
        
        This example shows how a plugin could provide analytics.
        """
        persona_id = params.get("persona_id")
        
        if not persona_id:
            return {"error": "persona_id required"}
        
        try:
            # Fetch persona using API
            persona = await self.api.get_persona(persona_id)
            
            # Return analytics data
            return {
                "persona_name": persona.get("name"),
                "total_content": persona.get("stats", {}).get("generation_count", 0),
                "active_platforms": self.target_platforms,
                "auto_publish_enabled": self.auto_publish,
            }
            
        except Exception as e:
            return {"error": str(e)}


# Export the plugin class
__all__ = ["ExampleAPIPlugin"]
