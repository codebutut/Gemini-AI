import json
import logging
from pathlib import Path

from fastmcp import FastMCP

# Configure logging
logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP("GeminiAgent")

# --- MCP Resources ---


@mcp.resource("agent://logs/{session_id}")
def get_session_logs(session_id: str) -> str:
    """Exposes the chat history for a specific session."""
    history_file = Path("history.json")
    if not history_file.exists():
        return "No history found."

    try:
        with open(history_file, encoding="utf-8") as f:
            history = json.load(f)
            session = history.get(session_id)
            if not session:
                return f"Session {session_id} not found."
            return json.dumps(session.get("messages", []), indent=2)
    except Exception as e:
        return f"Error reading history: {e}"


@mcp.resource("agent://attachments")
def get_attachments() -> str:
    """Lists all current file attachments."""
    # This is a simplified version as the actual AttachmentManager is per-session
    # In a real MCP server, we might need a way to link this to a session
    return "Attachment list is currently session-dependent."


# --- MCP Prompts ---


@mcp.prompt("code-review")
def code_review_prompt(code: str) -> str:
    """Prompt for performing a professional code review."""
    return f"Please perform a comprehensive code review of the following Python code, focusing on PEP 8 compliance, security vulnerabilities, and performance optimizations:\n\n```python\n{code}\n```"


@mcp.prompt("system-debug")
def system_debug_prompt(error_log: str) -> str:
    """Prompt for debugging system errors."""
    return f"Analyze the following error log and provide a root cause analysis and potential solutions:\n\n```\n{error_log}\n```"


# --- Tool Registration Helper ---


def register_mcp_tool(func):
    """Helper to register a function as an MCP tool."""
    try:
        # FastMCP tool decorator can be used as a function
        mcp.tool()(func)
        # logger.info(f"Registered MCP tool: {func.__name__}")
    except Exception:
        # logger.error(f"Failed to register MCP tool {func.__name__}: {e}")
        pass
    return func


if __name__ == "__main__":
    # This allows running the server standalone via stdio
    mcp.run()
