import importlib.util
import inspect
import json
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from google.genai import types

from gemini_agent.core.plugins import Plugin


class ExtensionManager:
    """
    Manages both internal plugins and external MCP servers.
    Provides an automated mechanism to Install, Uninstall, or Configure extensions.
    """

    def __init__(
        self,
        plugins_dir: str = "plugins",
        config_dir: str = "config",
        mcp_config_path: str = "mcp_config.json",
    ):
        self.plugins_dir = Path(plugins_dir)
        self.config_dir = Path(config_dir)
        self.mcp_config_path = Path(mcp_config_path)
        self.plugins: dict[str, Plugin] = {}
        self.logger = logging.getLogger(__name__)

        if not self.plugins_dir.exists():
            self.plugins_dir.mkdir(parents=True)
        if not self.config_dir.exists():
            self.config_dir.mkdir(parents=True)

    # --- Plugin Management ---

    def discover_plugins(self):
        """Discover and load plugins from the plugins directory."""
        self.plugins = {}
        for item in self.plugins_dir.iterdir():
            if (
                item.is_file()
                and item.suffix == ".py"
                and not item.name.startswith("__")
            ):
                self.load_plugin(str(item))
            elif item.is_dir() and not item.name.startswith("__"):
                # Check for __init__.py in directory
                init_file = item / "__init__.py"
                if init_file.exists():
                    self.load_plugin(str(init_file))

    def load_plugin(self, filepath: str):
        """Load a plugin from a file."""
        try:
            module_name = Path(filepath).stem
            if module_name == "__init__":
                module_name = Path(filepath).parent.name

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
                        config_path = self.config_dir / f"{plugin_instance.name}.json"
                        plugin_instance.load_config(str(config_path))
                        self.plugins[plugin_instance.name] = plugin_instance
                        self.logger.info(f"Loaded plugin: {plugin_instance.name}")
        except Exception as e:
            self.logger.error(f"Failed to load plugin from {filepath}: {e}")

    def install_plugin(self, package_name: str) -> str:
        """Install a plugin from PyPI."""
        try:
            # Install to plugins directory
            subprocess.check_call(
                [
                    os.sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "-t",
                    str(self.plugins_dir),
                    package_name,
                ]
            )
            self.discover_plugins()
            return f"Successfully installed plugin: {package_name}"
        except subprocess.CalledProcessError as e:
            err_msg = f"Failed to install plugin {package_name}: {e}"
            self.logger.error(err_msg)
            return err_msg

    def uninstall_plugin(self, plugin_name: str) -> str:
        """Uninstall a plugin."""
        if plugin_name in self.plugins:
            plugin = self.plugins[plugin_name]
            if plugin.filepath and os.path.exists(plugin.filepath):
                path_to_remove = Path(plugin.filepath)
                if path_to_remove.name == "__init__.py":
                    path_to_remove = path_to_remove.parent

                if path_to_remove.is_dir():
                    shutil.rmtree(path_to_remove)
                else:
                    path_to_remove.unlink()

                self.logger.info(f"Uninstalled plugin: {plugin_name}")
                # Also remove the config file
                config_path = self.config_dir / f"{plugin.name}.json"
                if config_path.exists():
                    config_path.unlink()

                del self.plugins[plugin_name]
                self.discover_plugins()
                return f"Successfully uninstalled plugin: {plugin_name}"
            else:
                return f"Plugin file not found for {plugin_name}"
        else:
            return f"Plugin not found: {plugin_name}"

    def configure_plugin(self, plugin_name: str, key: str, value: Any) -> str:
        """Configure a plugin."""
        if plugin_name in self.plugins:
            plugin = self.plugins[plugin_name]
            plugin.config[key] = value
            config_path = self.config_dir / f"{plugin.name}.json"
            plugin.save_config(str(config_path))
            self.logger.info(f"Configured plugin: {plugin_name}")
            return f"Successfully configured plugin {plugin_name}: {key}={value}"
        else:
            return f"Plugin not found: {plugin_name}"

    # --- MCP Management ---

    def _load_mcp_config(self) -> Dict[str, Any]:
        if self.mcp_config_path.exists():
            try:
                with open(self.mcp_config_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Failed to load MCP config: {e}")
        return {"mcpServers": {}}

    def _save_mcp_config(self, config: Dict[str, Any]):
        try:
            with open(self.mcp_config_path, "w") as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save MCP config: {e}")

    def add_mcp_server(
        self,
        name: str,
        command: str,
        args: List[str],
        env: Optional[Dict[str, str]] = None,
    ) -> str:
        """Add or update an MCP server configuration."""
        config = self._load_mcp_config()
        config["mcpServers"][name] = {
            "command": command,
            "args": args,
            "env": env or {},
        }
        self._save_mcp_config(config)
        return f"Successfully added/updated MCP server: {name}"

    def remove_mcp_server(self, name: str) -> str:
        """Remove an MCP server configuration."""
        config = self._load_mcp_config()
        if name in config["mcpServers"]:
            del config["mcpServers"][name]
            self._save_mcp_config(config)
            return f"Successfully removed MCP server: {name}"
        else:
            return f"MCP server {name} not found"

    def configure_mcp_server(self, name: str, key: str, value: Any) -> str:
        """Configure an existing MCP server (e.g., update env or args)."""
        config = self._load_mcp_config()
        if name in config["mcpServers"]:
            if key in ["command", "args", "env"]:
                config["mcpServers"][name][key] = value
                self._save_mcp_config(config)
                return f"Successfully configured MCP server {name}: {key}={value}"
            else:
                return f"Invalid configuration key for MCP server: {key}. Use 'command', 'args', or 'env'."
        else:
            return f"MCP server {name} not found"

    # --- Unified API ---

    def list_extensions(self) -> Dict[str, Any]:
        """List all installed plugins and MCP servers."""
        mcp_config = self._load_mcp_config()
        return {
            "plugins": {
                name: {"version": p.version, "description": p.description}
                for name, p in self.plugins.items()
            },
            "mcp_servers": mcp_config.get("mcpServers", {}),
        }

    def get_all_tools(self) -> list[types.FunctionDeclaration]:
        """Collect all tools from enabled plugins."""
        all_tools = []
        for plugin in self.plugins.values():
            if plugin.enabled:
                all_tools.extend(plugin.get_tools())
        return all_tools

    def execute_plugin_tool(self, tool_name: str, args: dict[str, Any]) -> Any:
        """Find the plugin that provides the tool and execute it."""
        for plugin in self.plugins.values():
            if plugin.enabled:
                for tool_decl in plugin.get_tools():
                    if tool_decl.name == tool_name:
                        return plugin.execute_tool(tool_name, args)
        raise ValueError(f"Tool '{tool_name}' not found in any enabled plugin.")
