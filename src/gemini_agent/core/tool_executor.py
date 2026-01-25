import json
import logging
import threading
from collections.abc import Callable
from typing import Any

from gemini_agent.config.app_config import AppConfig
from gemini_agent.core import tools
from gemini_agent.core.extension_manager import ExtensionManager


class ToolExecutor:
    """
    Handles execution of tools, including special session-specific tools.
    Thread-safe for parallel execution.
    """

    def __init__(
        self,
        status_callback: Callable[[str], None],
        terminal_callback: Callable[[str, str], None],
        confirmation_callback: Callable[
            [str, dict[str, Any]], tuple[bool, dict[str, Any] | None]
        ],
        extension_manager: ExtensionManager | None = None,
    ):
        self.status_callback = status_callback
        self.terminal_callback = terminal_callback
        self.confirmation_callback = confirmation_callback
        self.extension_manager = extension_manager
        self._lock = threading.Lock()

        # Session state managed by the worker but updated via tools
        self.current_plan = ""
        self.current_specs = ""

        # Handlers for special tools
        self.special_handlers = {
            "update_plan": self._handle_update_plan,
            "update_specs": self._handle_update_specs,
            "read_file": self._handle_read_file,
            "write_file": self._handle_write_file,
            "manage_extension": self._handle_manage_extension,
        }

    def execute(self, fn_name: str, fn_args: dict[str, Any]) -> str:
        """Execute a tool and return the result."""
        sanitized_args = self._sanitize_args(fn_name, fn_args)

        # Check for special handlers
        if fn_name in self.special_handlers:
            return self.special_handlers[fn_name](fn_args)

        # Check for dangerous tools
        if fn_name in AppConfig.DANGEROUS_TOOLS:
            # Confirmation must be serialized to avoid multiple dialogs at once
            with self._lock:
                allowed, modified_args = self.confirmation_callback(fn_name, fn_args)

            if not allowed:
                self.status_callback(f"🚫 Denied: {fn_name}")
                self.terminal_callback(f"🚫 Denied: {fn_name}\n", "error")
                return f"Error: User denied execution of '{fn_name}'."

            if modified_args:
                fn_args = modified_args
                sanitized_args = self._sanitize_args(fn_name, fn_args)  # Re-sanitize
                self.terminal_callback("✏️ Args modified by user.\n", "info")

        self.status_callback(f"⚙️ Executing: {fn_name}...")
        self.terminal_callback(f"⚙️ Executing: {fn_name}({sanitized_args})\n", "info")

        try:
            if fn_name in tools.TOOL_FUNCTIONS:
                result = tools.TOOL_FUNCTIONS[fn_name](**fn_args)
                self.status_callback(f"✅ Completed: {fn_name}")
                self.terminal_callback(f"✅ {fn_name} completed.\n", "success")
                return str(result)
            elif self.extension_manager:
                try:
                    result = self.extension_manager.execute_plugin_tool(
                        fn_name, fn_args
                    )
                    self.status_callback(f"✅ Completed (Plugin): {fn_name}")
                    self.terminal_callback(
                        f"✅ {fn_name} (Plugin) completed.\n", "success"
                    )
                    return str(result)
                except ValueError:
                    return f"Error: Tool '{fn_name}' not found."
            else:
                return f"Error: Tool '{fn_name}' not found."
        except Exception as e:
            error_msg = f"Error in {fn_name}: {str(e)}"
            logging.error(f"Execution failure for {fn_name}: {e}", exc_info=True)
            self.status_callback(f"❌ Failed: {fn_name}")
            self.terminal_callback(f"❌ {error_msg}\n", "error")
            return error_msg

    def _sanitize_args(self, fn_name: str, fn_args: dict[str, Any]) -> dict[str, Any]:
        """Sanitizes sensitive information in tool arguments."""
        sensitive_keys = {"api_key", "password", "token", "secret", "content"}
        sanitized = {}

        for key, value in fn_args.items():
            key_lower = key.lower()
            if any(sk in key_lower for sk in sensitive_keys):
                if isinstance(value, str):
                    if len(value) > 20:
                        sanitized[key] = f"{value[:5]}...[REDACTED]...{value[-5:]}"
                    else:
                        sanitized[key] = "[REDACTED]"
                else:
                    sanitized[key] = value
            else:
                sanitized[key] = value
        return sanitized

    def _handle_update_plan(self, fn_args: dict[str, Any]) -> str:
        with self._lock:
            self.current_plan = fn_args.get("content", "")
        self.status_callback("✅ Plan updated")
        self.terminal_callback("📝 Plan updated\n", "success")
        return "Plan updated successfully in session context."

    def _handle_update_specs(self, fn_args: dict[str, Any]) -> str:
        with self._lock:
            self.current_specs = fn_args.get("content", "")
        self.status_callback("✅ Specifications updated")
        self.terminal_callback("📝 Specifications updated\n", "success")
        return "Specifications updated successfully in session context."

    def _handle_read_file(self, fn_args: dict[str, Any]) -> str:
        filepath = fn_args.get("filepath", "")
        if filepath == "plan.md":
            self.status_callback("📖 Reading virtual plan.md")
            with self._lock:
                return (
                    self.current_plan
                    if self.current_plan
                    else "plan.md is currently empty."
                )
        if filepath == "specs.md":
            self.status_callback("📖 Reading virtual specs.md")
            with self._lock:
                return (
                    self.current_specs
                    if self.current_specs
                    else "specs.md is currently empty."
                )

        # Fallback to real file read
        return str(tools.read_file(**fn_args))

    def _handle_write_file(self, fn_args: dict[str, Any]) -> str:
        filepath = fn_args.get("filepath", "")
        content = fn_args.get("content", "")
        if filepath == "plan.md":
            with self._lock:
                self.current_plan = content
            self.status_callback("✅ Plan updated (via write_file)")
            self.terminal_callback("📝 Plan updated (via write_file)\n", "success")
            return "Successfully updated plan.md in session context."
        if filepath == "specs.md":
            with self._lock:
                self.current_specs = content
            self.status_callback("✅ Specifications updated (via write_file)")
            self.terminal_callback(
                "📝 Specifications updated (via write_file)\n", "success"
            )
            return "Successfully updated specs.md in session context."

        # Fallback to real file write
        return str(tools.write_file(**fn_args))

    def _handle_manage_extension(self, fn_args: dict[str, Any]) -> str:
        if not self.extension_manager:
            return "Error: Extension manager not available."

        operation = fn_args.get("operation")
        ext_type = fn_args.get("extension_type")
        name = fn_args.get("name")

        self.status_callback(f"⚙️ Managing Extension: {operation} {name}...")
        self.terminal_callback(
            f"⚙️ Extension Op: {operation} {ext_type} {name}\n", "info"
        )

        if operation == "list":
            return json.dumps(self.extension_manager.list_extensions(), indent=2)

        if ext_type == "plugin":
            if operation == "install":
                return self.extension_manager.install_plugin(name)
            elif operation == "uninstall":
                return self.extension_manager.uninstall_plugin(name)
            elif operation == "configure":
                return self.extension_manager.configure_plugin(
                    name, fn_args.get("key"), fn_args.get("value")
                )
        elif ext_type == "mcp":
            if operation == "add_mcp":
                return self.extension_manager.add_mcp_server(
                    name,
                    fn_args.get("command"),
                    fn_args.get("args"),
                    fn_args.get("env"),
                )
            elif operation == "remove_mcp":
                return self.extension_manager.remove_mcp_server(name)
            elif operation == "configure":
                return self.extension_manager.configure_mcp_server(
                    name, fn_args.get("key"), fn_args.get("value")
                )

        return f"Error: Unsupported operation '{operation}' for extension type '{ext_type}'."
