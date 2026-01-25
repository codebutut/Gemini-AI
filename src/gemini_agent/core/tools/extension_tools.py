from typing import Any
from pydantic import BaseModel, Field
from . import tool, validate_args


class ManageExtensionArgs(BaseModel):
    operation: str = Field(
        ...,
        description="Operation to perform: 'install', 'uninstall', 'configure', 'list', 'add_mcp', 'remove_mcp'.",
    )
    extension_type: str = Field(
        ..., description="Type of extension: 'plugin' or 'mcp'."
    )
    name: str = Field(..., description="Name of the extension or package.")
    key: str | None = Field(
        None, description="Configuration key (for 'configure' operation)."
    )
    value: Any | None = Field(
        None, description="Configuration value (for 'configure' operation)."
    )
    command: str | None = Field(None, description="Command for MCP server.")
    args: list[str] | None = Field(None, description="Arguments for MCP server.")
    env: dict[str, str] | None = Field(
        None, description="Environment variables for MCP server."
    )


@tool
@validate_args(ManageExtensionArgs)
def manage_extension(
    operation: str,
    extension_type: str,
    name: str,
    key: str | None = None,
    value: Any | None = None,
    command: str | None = None,
    args: list[str] | None = None,
    env: dict[str, str] | None = None,
) -> str:
    """
    Automated mechanism to Install, Uninstall, or Configure extensions (plugins or MCP servers).

    Args:
        operation: 'install', 'uninstall', 'configure', 'list', 'add_mcp', 'remove_mcp'.
        extension_type: 'plugin' or 'mcp'.
        name: Name of the extension or package.
        key: Configuration key.
        value: Configuration value.
        command: Command for MCP server.
        args: Arguments for MCP server.
        env: Environment variables for MCP server.
    """
    return "Extension operation requested."
