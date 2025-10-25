"""Plugin manager for discovering and loading model plugins."""

import importlib
import importlib.metadata
import importlib.util
from typing import Dict, List, Type, Optional, Any
from pathlib import Path
import inspect

from lucidity.base_model import BaseModel, ModelMetadata


class PluginManager:
    """
    Manager for discovering and loading model plugins.

    Plugins can be registered via:
    1. Entry points (installed packages)
    2. Direct registration (for development)
    3. Directory scanning (local plugins)
    """

    def __init__(self):
        """Initialize the plugin manager."""
        self._plugins: Dict[str, Type[BaseModel]] = {}
        self._instances: Dict[str, BaseModel] = {}

    def discover_plugins(self) -> None:
        """
        Discover all available plugins via entry points.

        Plugins should register an entry point in the 'lucidity.models' group.
        """
        try:
            entry_points = importlib.metadata.entry_points()

            # Handle different versions of importlib.metadata
            if hasattr(entry_points, 'select'):
                # Python 3.10+
                lucidity_eps = entry_points.select(group='lucidity.models')
            else:
                # Python 3.9
                lucidity_eps = entry_points.get('lucidity.models', [])

            for ep in lucidity_eps:
                try:
                    plugin_class = ep.load()
                    if self._validate_plugin(plugin_class):
                        self._plugins[ep.name] = plugin_class
                except Exception as e:
                    print(f"Warning: Failed to load plugin '{ep.name}': {e}")

        except Exception as e:
            print(f"Warning: Failed to discover plugins: {e}")

    def discover_from_directory(self, directory: Path) -> None:
        """
        Discover plugins from a directory.

        Args:
            directory: Directory containing plugin modules
        """
        if not directory.exists() or not directory.is_dir():
            return

        for python_file in directory.glob("*.py"):
            if python_file.name.startswith("_"):
                continue

            try:
                # Import the module
                spec = importlib.util.spec_from_file_location(
                    python_file.stem, python_file
                )
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)

                    # Find all BaseModel subclasses in the module
                    for name, obj in inspect.getmembers(module, inspect.isclass):
                        if (issubclass(obj, BaseModel) and
                            obj is not BaseModel and
                            self._validate_plugin(obj)):
                            # Use the model's metadata name as the plugin name
                            temp_instance = obj()
                            metadata = temp_instance.get_metadata()
                            self._plugins[metadata.name] = obj

            except Exception as e:
                print(f"Warning: Failed to load plugin from {python_file}: {e}")

    def register_plugin(self, name: str, plugin_class: Type[BaseModel]) -> None:
        """
        Manually register a plugin.

        Args:
            name: Name to register the plugin under
            plugin_class: Plugin class (must inherit from BaseModel)
        """
        if not self._validate_plugin(plugin_class):
            raise ValueError(f"Invalid plugin class: {plugin_class}")

        self._plugins[name] = plugin_class

    def _validate_plugin(self, plugin_class: Type) -> bool:
        """
        Validate that a plugin class is properly implemented.

        Args:
            plugin_class: Class to validate

        Returns:
            True if valid, False otherwise
        """
        if not inspect.isclass(plugin_class):
            return False

        if not issubclass(plugin_class, BaseModel):
            return False

        if plugin_class is BaseModel:
            return False

        # Check that required methods are implemented
        required_methods = ['get_metadata', 'initialize', 'process_frame']
        for method_name in required_methods:
            if not hasattr(plugin_class, method_name):
                return False

        return True

    def get_plugin(self, name: str, config: Optional[Dict[str, Any]] = None) -> BaseModel:
        """
        Get an instance of a plugin.

        Args:
            name: Name of the plugin
            config: Optional configuration dictionary

        Returns:
            Instance of the plugin

        Raises:
            KeyError: If plugin not found
        """
        if name not in self._plugins:
            raise KeyError(f"Plugin '{name}' not found. Available: {list(self._plugins.keys())}")

        # Check if we already have an instance with this config
        instance_key = f"{name}:{id(config)}"
        if instance_key in self._instances:
            return self._instances[instance_key]

        # Create new instance
        plugin_class = self._plugins[name]
        instance = plugin_class(config=config)
        self._instances[instance_key] = instance

        return instance

    def list_plugins(self) -> List[str]:
        """
        Get list of available plugin names.

        Returns:
            List of plugin names
        """
        return list(self._plugins.keys())

    def get_plugin_metadata(self, name: str) -> ModelMetadata:
        """
        Get metadata for a plugin without instantiating it.

        Args:
            name: Name of the plugin

        Returns:
            ModelMetadata for the plugin

        Raises:
            KeyError: If plugin not found
        """
        if name not in self._plugins:
            raise KeyError(f"Plugin '{name}' not found")

        # Create temporary instance to get metadata
        plugin_class = self._plugins[name]
        temp_instance = plugin_class()
        return temp_instance.get_metadata()

    def get_all_metadata(self) -> Dict[str, ModelMetadata]:
        """
        Get metadata for all available plugins.

        Returns:
            Dictionary mapping plugin names to their metadata
        """
        metadata_dict = {}
        for name in self._plugins.keys():
            try:
                metadata_dict[name] = self.get_plugin_metadata(name)
            except Exception as e:
                print(f"Warning: Failed to get metadata for '{name}': {e}")

        return metadata_dict

    def clear_instances(self) -> None:
        """Clear all cached plugin instances."""
        # Call cleanup on all instances
        for instance in self._instances.values():
            try:
                instance.cleanup()
            except Exception as e:
                print(f"Warning: Cleanup failed for instance: {e}")

        self._instances.clear()


# Global plugin manager instance
_global_plugin_manager: Optional[PluginManager] = None


def get_plugin_manager() -> PluginManager:
    """
    Get the global plugin manager instance.

    Returns:
        PluginManager instance
    """
    global _global_plugin_manager
    if _global_plugin_manager is None:
        _global_plugin_manager = PluginManager()
        _global_plugin_manager.discover_plugins()

    return _global_plugin_manager
