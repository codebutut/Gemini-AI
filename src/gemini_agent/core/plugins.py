import importlib.util
import inspect
import json
import logging
import os
import subprocess
from typing import Any

from google.genai import types


class Plugin:
    """Base class for all plugins."""

    name: str = "Base Plugin"
    description: str = "Description of the plugin."
    version: str = "0.1.0"
    author: str = "Unknown"

    def __init__(self):
        self.enabled = True
        self.config = {}
        self.filepath: str | None = None

    def get_tools(self) -> list[types.FunctionDeclaration]:
        """Return a list of tool definitions provided by this plugin."""
        return []

    def execute_tool(self, tool_name: str, args: dict[str, Any]) -> Any:
        """Execute a tool provided by this plugin."""
        raise NotImplementedError(
            "Plugins must implement execute_tool if they provide tools."
        )

    def load_config(self, config_path: str):
        """Load configuration from a JSON file."""
        try:
            with open(config_path, "r") as f:
                self.config = json.load(f)
        except FileNotFoundError:
            self.config = {}

    def save_config(self, config_path: str):
        """Save configuration to a JSON file."""
        with open(config_path, "w") as f:
            json.dump(self.config, f, indent=4)


class PluginManager:
    """Manages discovery and loading of plugins."""

    def __init__(self, plugins_dir: str = "plugins", config_dir: str = "config"):
        self.plugins_dir = plugins_dir
        self.config_dir = config_dir
        self.plugins: dict[str, Plugin] = {}
        self.logger = logging.getLogger(__name__)

        if not os.path.exists(self.plugins_dir):
            os.makedirs(self.plugins_dir)
        if not os.path.exists(self.config_dir):
            os.makedirs(self.config_dir)

    def discover_plugins(self):
        """Discover and load plugins from the plugins directory."""
        self.plugins = {}
        for filename in os.listdir(self.plugins_dir):
            if filename.endswith(".py") and not filename.startswith("__"):
                filepath = os.path.join(self.plugins_dir, filename)
                self.load_plugin(filepath)

    def load_plugin(self, filepath: str):
        """Load a plugin from a file."""
        try:
            module_name = os.path.splitext(os.path.basename(filepath))[0]
            spec = importlib.util.spec_from_file_location(module_name, filepath)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                for _name, obj in inspect.getmembers(module):
                    if (
                        inspect.isclass(obj)
                        and issubclass(obj, Plugin)
                        and obj is not Plugin
                    ):
                        plugin_instance = obj()
                        plugin_instance.filepath = filepath
                        config_path = os.path.join(
                            self.config_dir, f"{plugin_instance.name}.json"
                        )
                        plugin_instance.load_config(config_path)
                        self.plugins[plugin_instance.name] = plugin_instance
                        self.logger.info(f"Loaded plugin: {plugin_instance.name}")
        except Exception as e:
            self.logger.error(f"Failed to load plugin from {filepath}: {e}")

    def install_plugin(self, package_name: str):
        """Install a plugin from PyPI."""
        try:
            subprocess.check_call(
                ["pip", "install", "-t", self.plugins_dir, package_name]
            )
            self.logger.info(f"Installed plugin: {package_name}")
            self.discover_plugins()
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to install plugin {package_name}: {e}")

    def uninstall_plugin(self, plugin_name: str):
        """Uninstall a plugin."""
        if plugin_name in self.plugins:
            plugin = self.plugins[plugin_name]
            if plugin.filepath and os.path.exists(plugin.filepath):
                os.remove(plugin.filepath)
                self.logger.info(f"Uninstalled plugin: {plugin_name}")
                # Also remove the config file
                config_path = os.path.join(self.config_dir, f"{plugin.name}.json")
                if os.path.exists(config_path):
                    os.remove(config_path)
                self.discover_plugins()
            else:
                self.logger.error(f"Plugin file not found for {plugin_name}")
        else:
            self.logger.error(f"Plugin not found: {plugin_name}")

    def configure_plugin(self, plugin_name: str, key: str, value: Any):
        """Configure a plugin."""
        if plugin_name in self.plugins:
            plugin = self.plugins[plugin_name]
            plugin.config[key] = value
            config_path = os.path.join(self.config_dir, f"{plugin.name}.json")
            plugin.save_config(config_path)
            self.logger.info(f"Configured plugin: {plugin_name}")
        else:
            self.logger.error(f"Plugin not found: {plugin_name}")

    def get_enabled_plugins(self) -> list[Plugin]:
        """Return a list of currently enabled plugins."""
        return [p for p in self.plugins.values() if p.enabled]

    def get_all_tools(self) -> list[types.FunctionDeclaration]:
        """Collect all tools from enabled plugins."""
        all_tools = []
        for plugin in self.get_enabled_plugins():
            all_tools.extend(plugin.get_tools())
        return all_tools

    def execute_plugin_tool(self, tool_name: str, args: dict[str, Any]) -> Any:
        """Find the plugin that provides the tool and execute it."""
        for plugin in self.get_enabled_plugins():
            for tool_decl in plugin.get_tools():
                if tool_decl.name == tool_name:
                    return plugin.execute_tool(tool_name, args)
        raise ValueError(f"Tool '{tool_name}' not found in any enabled plugin.")
