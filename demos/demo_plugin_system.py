#!/usr/bin/env python
"""
Plugin System Demo

Demonstrates the Gator Plugin System functionality including:
- Plugin discovery
- Plugin loading
- Hook execution
- Plugin lifecycle management
"""

import asyncio
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from backend.plugins.manager import PluginManager
from backend.config.logging import get_logger

logger = get_logger(__name__)


async def main():
    """Run plugin system demo."""
    print("🎭 Gator Plugin System - Demo")
    print("=" * 50)

    # Initialize plugin manager
    print("\n📦 Initializing Plugin Manager...")
    plugin_dirs = [Path("plugins")]
    manager = PluginManager(plugin_dirs=plugin_dirs)

    # Discover plugins
    print("\n🔍 Discovering Plugins...")
    discovered = await manager.discover_plugins()

    if not discovered:
        print("   ℹ️  No plugins found in plugin directories")
        print(f"   Plugin directories: {[str(d) for d in plugin_dirs]}")
    else:
        print(f"   ✅ Found {len(discovered)} plugin(s):")
        for plugin_info in discovered:
            manifest = plugin_info.get("manifest", {})
            print(f"      • {manifest.get('name', plugin_info['plugin_id'])}")
            print(f"        Version: {manifest.get('version', 'unknown')}")
            print(f"        Type: {manifest.get('plugin_type', 'unknown')}")
            print(f"        Path: {plugin_info['path']}")

    # Try to load an example plugin if available
    if discovered:
        print("\n📥 Loading First Plugin...")
        first_plugin = discovered[0]
        plugin_id = first_plugin["plugin_id"]
        plugin_path = Path(first_plugin["path"])

        # Example configuration
        config = {
            "api_key": "demo-api-key",
            "filter_keywords": ["bad", "spam", "test"],
            "replacement_text": "[filtered]",
        }

        try:
            plugin = await manager.load_plugin(plugin_id, plugin_path, config)
            print(f"   ✅ Plugin loaded: {plugin}")

            # List loaded plugins
            print("\n📋 Loaded Plugins:")
            for plugin_info in manager.list_plugins():
                print(f"   • {plugin_info['metadata']['name']}")
                print(f"     Status: {plugin_info['status']}")
                print(f"     Enabled: {plugin_info['enabled']}")

            # Test hook execution
            print("\n🎣 Testing Hook Execution...")
            print("   Simulating content generation event...")

            test_content = {
                "type": "text",
                "text": "This is a test message with spam content that should be filtered",
                "persona_id": "test-persona",
            }

            print(f"   Original content: {test_content['text']}")

            results = await manager.execute_hook("on_content_generated", test_content)

            if results:
                for result in results:
                    print(f"   ✅ Hook executed by: {result['plugin_id']}")
                    processed_content = result["result"]
                    if processed_content.get("filtered"):
                        print(f"   Filtered content: {processed_content['text']}")
                        print(
                            f"   Filter count: {processed_content.get('filter_count', 0)}"
                        )

            # Test other hooks
            print("\n🎣 Testing Other Hooks...")

            print("   Simulating persona creation event...")
            await manager.execute_hook(
                "on_persona_created", {"name": "Test Persona", "id": "test-123"}
            )

            print("   Simulating post publication event...")
            await manager.execute_hook(
                "on_post_published",
                {"platform": "instagram", "post_id": "abc123", "content_id": "xyz789"},
            )

            # Test enable/disable
            print("\n🔌 Testing Plugin Enable/Disable...")
            print(f"   Current status: {manager.get_plugin_status(plugin_id)}")

            await manager.disable_plugin(plugin_id)
            print(f"   After disable: {manager.get_plugin_status(plugin_id)}")

            await manager.enable_plugin(plugin_id)
            print(f"   After enable: {manager.get_plugin_status(plugin_id)}")

            # Shutdown
            print("\n🔴 Shutting Down Plugin Manager...")
            await manager.shutdown()

            print("\n✅ Demo completed successfully!")

        except Exception as e:
            print(f"\n❌ Error during demo: {str(e)}")
            import traceback

            traceback.print_exc()
            await manager.shutdown()

    else:
        print("\n⚠️  No plugins to load. Create a plugin to see the system in action!")
        print(
            "\nTo create a plugin, follow the guide in: docs/PLUGIN_SYSTEM.md"
        )

    print("\n" + "=" * 50)
    print("🎉 Plugin System Demo Complete!")


if __name__ == "__main__":
    asyncio.run(main())
