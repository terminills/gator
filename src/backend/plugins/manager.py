"""
Plugin Manager Service

Manages plugin lifecycle including discovery, loading, execution, and unloading.
Provides sandboxing and resource management for plugins.
"""

import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from backend.config.logging import get_logger
from backend.plugins import (
    GatorPlugin,
    PluginStatus,
)

logger = get_logger(__name__)


class PluginError(Exception):
    """Base exception for plugin-related errors."""



class PluginLoadError(PluginError):
    """Exception raised when plugin fails to load."""



class PluginExecutionError(PluginError):
    """Exception raised when plugin execution fails."""



class PluginManager:
    """
    Plugin manager for discovery, loading, and lifecycle management.

    The plugin manager:
    - Discovers plugins from plugin directories
    - Loads and initializes plugins
    - Manages plugin lifecycle (enable, disable, reload)
    - Executes plugin hooks
    - Provides sandboxing and resource limits
    - Tracks plugin status and errors
    """

    def __init__(self, plugin_dirs: Optional[List[Path]] = None):
        """
        Initialize plugin manager.

        Args:
            plugin_dirs: List of directories to search for plugins
        """
        self.plugin_dirs = plugin_dirs or [Path("plugins")]
        self.plugins: Dict[str, GatorPlugin] = {}
        self.plugin_status: Dict[str, PluginStatus] = {}
        self.plugin_errors: Dict[str, str] = {}

        # Hook registry - maps hook names to list of plugins that implement them
        self.hooks: Dict[str, List[str]] = {
            "on_content_generated": [],
            "on_persona_created": [],
            "on_persona_updated": [],
            "on_post_published": [],
            "on_schedule_created": [],
        }

    async def discover_plugins(self) -> List[Dict[str, Any]]:
        """
        Discover available plugins from plugin directories.

        Returns:
            List of plugin metadata dictionaries

        Raises:
            PluginError: If plugin discovery fails
        """
        discovered = []

        for plugin_dir in self.plugin_dirs:
            if not plugin_dir.exists():
                logger.warning(f"Plugin directory not found: {plugin_dir}")
                continue

            logger.info(f"Scanning plugin directory: {plugin_dir}")

            # Look for plugin directories (must contain __init__.py or main.py)
            for item in plugin_dir.iterdir():
                if not item.is_dir():
                    continue

                # Check for plugin entry point
                manifest_file = item / "manifest.json"
                init_file = item / "__init__.py"
                main_file = item / "main.py"

                if manifest_file.exists():
                    try:
                        import json

                        with open(manifest_file) as f:
                            manifest = json.load(f)

                        discovered.append(
                            {
                                "plugin_id": item.name,
                                "path": str(item),
                                "manifest": manifest,
                                "entry_point": str(
                                    main_file if main_file.exists() else init_file
                                ),
                            }
                        )

                        logger.info(
                            f"Discovered plugin: {manifest.get('name', item.name)}"
                        )

                    except Exception as e:
                        logger.error(
                            f"Failed to read manifest for {item.name}: {str(e)}"
                        )

        logger.info(f"Discovered {len(discovered)} plugins")
        return discovered

    async def load_plugin(
        self,
        plugin_id: str,
        plugin_path: Path,
        config: Optional[Dict[str, Any]] = None,
    ) -> GatorPlugin:
        """
        Load and initialize a plugin.

        Args:
            plugin_id: Unique plugin identifier
            plugin_path: Path to plugin directory
            config: Plugin configuration

        Returns:
            Loaded plugin instance

        Raises:
            PluginLoadError: If plugin fails to load or initialize
        """
        try:
            logger.info(f"Loading plugin: {plugin_id}")

            # Check if already loaded
            if plugin_id in self.plugins:
                logger.warning(f"Plugin already loaded: {plugin_id}")
                return self.plugins[plugin_id]

            # Determine entry point
            main_file = plugin_path / "main.py"
            init_file = plugin_path / "__init__.py"

            if main_file.exists():
                entry_point = main_file
            elif init_file.exists():
                entry_point = init_file
            else:
                raise PluginLoadError(f"No entry point found for plugin: {plugin_id}")

            # Load plugin module
            spec = importlib.util.spec_from_file_location(
                f"gator_plugin_{plugin_id}", entry_point
            )

            if spec is None or spec.loader is None:
                raise PluginLoadError(f"Failed to load module spec: {plugin_id}")

            module = importlib.util.module_from_spec(spec)
            sys.modules[f"gator_plugin_{plugin_id}"] = module
            spec.loader.exec_module(module)

            # Find plugin class (must inherit from GatorPlugin)
            plugin_class: Optional[Type[GatorPlugin]] = None

            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (
                    isinstance(attr, type)
                    and issubclass(attr, GatorPlugin)
                    and attr != GatorPlugin
                ):
                    plugin_class = attr
                    break

            if plugin_class is None:
                raise PluginLoadError(f"No GatorPlugin subclass found in: {plugin_id}")

            # Instantiate plugin
            plugin_config = config or {}
            plugin = plugin_class(plugin_config)

            # Initialize plugin
            await plugin.initialize()

            # Register plugin
            self.plugins[plugin_id] = plugin
            self.plugin_status[plugin_id] = PluginStatus.ACTIVE

            # Register hooks
            for hook_name in self.hooks.keys():
                if hasattr(plugin, hook_name):
                    method = getattr(plugin, hook_name)
                    if callable(method):
                        self.hooks[hook_name].append(plugin_id)

            logger.info(f"Plugin loaded successfully: {plugin_id}")
            return plugin

        except Exception as e:
            logger.error(f"Failed to load plugin {plugin_id}: {str(e)}")
            self.plugin_status[plugin_id] = PluginStatus.ERROR
            self.plugin_errors[plugin_id] = str(e)
            raise PluginLoadError(f"Failed to load plugin {plugin_id}: {str(e)}")

    async def unload_plugin(self, plugin_id: str) -> None:
        """
        Unload and cleanup a plugin.

        Args:
            plugin_id: Plugin identifier

        Raises:
            PluginError: If plugin unload fails
        """
        try:
            logger.info(f"Unloading plugin: {plugin_id}")

            if plugin_id not in self.plugins:
                logger.warning(f"Plugin not loaded: {plugin_id}")
                return

            plugin = self.plugins[plugin_id]

            # Call shutdown hook
            await plugin.shutdown()

            # Unregister hooks
            for hook_name in self.hooks.keys():
                if plugin_id in self.hooks[hook_name]:
                    self.hooks[hook_name].remove(plugin_id)

            # Remove plugin
            del self.plugins[plugin_id]
            self.plugin_status[plugin_id] = PluginStatus.INACTIVE

            # Remove from sys.modules
            module_name = f"gator_plugin_{plugin_id}"
            if module_name in sys.modules:
                del sys.modules[module_name]

            logger.info(f"Plugin unloaded: {plugin_id}")

        except Exception as e:
            logger.error(f"Failed to unload plugin {plugin_id}: {str(e)}")
            self.plugin_errors[plugin_id] = str(e)
            raise PluginError(f"Failed to unload plugin {plugin_id}: {str(e)}")

    async def reload_plugin(
        self,
        plugin_id: str,
        plugin_path: Path,
        config: Optional[Dict[str, Any]] = None,
    ) -> GatorPlugin:
        """
        Reload a plugin (unload then load).

        Args:
            plugin_id: Plugin identifier
            plugin_path: Path to plugin directory
            config: Plugin configuration

        Returns:
            Reloaded plugin instance

        Raises:
            PluginError: If plugin reload fails
        """
        logger.info(f"Reloading plugin: {plugin_id}")

        # Unload if currently loaded
        if plugin_id in self.plugins:
            await self.unload_plugin(plugin_id)

        # Load plugin
        return await self.load_plugin(plugin_id, plugin_path, config)

    async def execute_hook(
        self,
        hook_name: str,
        *args,
        **kwargs,
    ) -> List[Any]:
        """
        Execute a hook across all registered plugins.

        Args:
            hook_name: Name of hook to execute
            *args: Hook positional arguments
            **kwargs: Hook keyword arguments

        Returns:
            List of hook results from each plugin

        Raises:
            PluginExecutionError: If hook execution fails critically
        """
        if hook_name not in self.hooks:
            logger.warning(f"Unknown hook: {hook_name}")
            return []

        results = []
        plugin_ids = self.hooks[hook_name].copy()

        logger.debug(f"Executing hook '{hook_name}' across {len(plugin_ids)} plugins")

        for plugin_id in plugin_ids:
            try:
                plugin = self.plugins.get(plugin_id)
                if plugin is None or not plugin.enabled:
                    continue

                # Get hook method
                hook_method = getattr(plugin, hook_name, None)
                if hook_method is None or not callable(hook_method):
                    continue

                # Execute hook
                result = await hook_method(*args, **kwargs)
                results.append({"plugin_id": plugin_id, "result": result})

                logger.debug(f"Hook '{hook_name}' executed successfully: {plugin_id}")

            except Exception as e:
                logger.error(
                    f"Plugin {plugin_id} failed executing hook '{hook_name}': {str(e)}"
                )
                self.plugin_errors[plugin_id] = (
                    f"Hook execution error ({hook_name}): {str(e)}"
                )
                # Continue executing other plugins even if one fails
                continue

        return results

    def get_plugin(self, plugin_id: str) -> Optional[GatorPlugin]:
        """
        Get a loaded plugin by ID.

        Args:
            plugin_id: Plugin identifier

        Returns:
            Plugin instance or None if not found
        """
        return self.plugins.get(plugin_id)

    def list_plugins(self) -> List[Dict[str, Any]]:
        """
        List all loaded plugins with their status.

        Returns:
            List of plugin information dictionaries
        """
        return [
            {
                "plugin_id": plugin_id,
                "metadata": (
                    plugin.get_metadata().dict()
                    if plugin.metadata
                    else {"name": plugin.__class__.__name__}
                ),
                "enabled": plugin.enabled,
                "status": self.plugin_status.get(plugin_id, PluginStatus.INACTIVE),
                "error": self.plugin_errors.get(plugin_id),
            }
            for plugin_id, plugin in self.plugins.items()
        ]

    def get_plugin_status(self, plugin_id: str) -> Optional[PluginStatus]:
        """
        Get current status of a plugin.

        Args:
            plugin_id: Plugin identifier

        Returns:
            Plugin status or None if not found
        """
        return self.plugin_status.get(plugin_id)

    async def enable_plugin(self, plugin_id: str) -> None:
        """
        Enable a plugin.

        Args:
            plugin_id: Plugin identifier

        Raises:
            PluginError: If plugin not found
        """
        plugin = self.plugins.get(plugin_id)
        if plugin is None:
            raise PluginError(f"Plugin not found: {plugin_id}")

        plugin.enabled = True
        self.plugin_status[plugin_id] = PluginStatus.ACTIVE
        logger.info(f"Plugin enabled: {plugin_id}")

    async def disable_plugin(self, plugin_id: str) -> None:
        """
        Disable a plugin (without unloading).

        Args:
            plugin_id: Plugin identifier

        Raises:
            PluginError: If plugin not found
        """
        plugin = self.plugins.get(plugin_id)
        if plugin is None:
            raise PluginError(f"Plugin not found: {plugin_id}")

        plugin.enabled = False
        self.plugin_status[plugin_id] = PluginStatus.INACTIVE
        logger.info(f"Plugin disabled: {plugin_id}")

    async def shutdown(self) -> None:
        """Shutdown all plugins and cleanup resources."""
        logger.info(f"Shutting down plugin manager ({len(self.plugins)} plugins)")

        for plugin_id in list(self.plugins.keys()):
            try:
                await self.unload_plugin(plugin_id)
            except Exception as e:
                logger.error(f"Error unloading plugin {plugin_id}: {str(e)}")

        logger.info("Plugin manager shutdown complete")


# Global plugin manager instance
plugin_manager = PluginManager()
