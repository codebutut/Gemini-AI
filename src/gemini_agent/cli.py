import argparse
import json
import sys
from gemini_agent.core.extension_manager import ExtensionManager


def handle_cli():
    """Main entry point for the Gemini CLI."""
    parser = argparse.ArgumentParser(description="Gemini Agent CLI")
    subparsers = parser.add_subparsers(dest="command")

    # Extension CLI
    extension_parser = subparsers.add_parser(
        "extension", help="Manage extensions (plugins and MCP servers)"
    )
    ext_subparsers = extension_parser.add_subparsers(dest="ext_command")

    # List
    ext_subparsers.add_parser("list", help="List all extensions")

    # Install Plugin
    install_plugin_parser = ext_subparsers.add_parser(
        "install-plugin", help="Install a plugin from PyPI"
    )
    install_plugin_parser.add_argument(
        "package_name", help="The name of the plugin package"
    )

    # Uninstall Plugin
    uninstall_plugin_parser = ext_subparsers.add_parser(
        "uninstall-plugin", help="Uninstall a plugin"
    )
    uninstall_plugin_parser.add_argument("plugin_name", help="The name of the plugin")

    # Add MCP
    add_mcp_parser = ext_subparsers.add_parser("add-mcp", help="Add an MCP server")
    add_mcp_parser.add_argument("name", help="Name of the MCP server")
    add_mcp_parser.add_argument("command", help="Command to run")
    add_mcp_parser.add_argument("--args", nargs="*", help="Arguments for the command")

    # Remove MCP
    remove_mcp_parser = ext_subparsers.add_parser(
        "remove-mcp", help="Remove an MCP server"
    )
    remove_mcp_parser.add_argument("name", help="Name of the MCP server")

    args = parser.parse_args()

    if not args.command:
        return False  # No CLI command, proceed to GUI

    extension_mgr = ExtensionManager()
    extension_mgr.discover_plugins()

    if args.command == "extension":
        if args.ext_command == "list":
            print(json.dumps(extension_mgr.list_extensions(), indent=2))
        elif args.ext_command == "install-plugin":
            print(extension_mgr.install_plugin(args.package_name))
        elif args.ext_command == "uninstall-plugin":
            print(extension_mgr.uninstall_plugin(args.plugin_name))
        elif args.ext_command == "add-mcp":
            print(
                extension_mgr.add_mcp_server(args.name, args.command, args.args or [])
            )
        elif args.ext_command == "remove-mcp":
            print(extension_mgr.remove_mcp_server(args.name))
        sys.exit(0)

    return True
