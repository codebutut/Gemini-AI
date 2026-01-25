import inspect
import logging
from collections.abc import Callable
from functools import wraps
from typing import Any

from google.genai import types
from pydantic import BaseModel, ValidationError

try:
    from utils.introspection import auto_generate_declaration
except ImportError:
    from gemini_agent.utils.introspection import auto_generate_declaration

# Configure logging
logger = logging.getLogger(__name__)

# --- Tool Registry ---

TOOL_REGISTRY: dict[str, Callable] = {}


def tool(func: Callable) -> Callable:
    """Decorator to register a function as a tool."""
    TOOL_REGISTRY[func.__name__] = func

    # MCP Integration: Register tool with FastMCP if available
    try:
        from gemini_agent.mcp.server import register_mcp_tool

        register_mcp_tool(func)
    except (ImportError, Exception):
        pass

    return func


def validate_args(model: type[BaseModel]):
    """Decorator to validate function arguments using a Pydantic model."""

    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                # Bind args/kwargs to the function's signature
                sig = inspect.signature(func)
                bound_args = sig.bind(*args, **kwargs)
                bound_args.apply_defaults()

                # Validate using the model
                validated_args = model(**bound_args.arguments)
                return func(**validated_args.model_dump())
            except ValidationError as e:
                return f"Validation Error: {str(e)}"
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {e}", exc_info=True)
                return f"Error: {str(e)}"

        return wrapper

    return decorator


def get_tool_config(
    extra_declarations: list[types.FunctionDeclaration] | None = None,
) -> types.Tool:
    """
    Returns the complete tool configuration for Gemini API, ensuring no duplicate function names.
    """
    declarations_dict: dict[str, types.FunctionDeclaration] = {}

    # Add core tools from registry
    for func in TOOL_REGISTRY.values():
        decl = auto_generate_declaration(func)
        declarations_dict[decl.name] = decl

    # Add extra declarations (e.g. from plugins), overriding core tools if names collide
    if extra_declarations:
        for decl in extra_declarations:
            if decl.name in declarations_dict:
                logger.warning(
                    f"Duplicate tool declaration found for '{decl.name}'. Using the extra declaration."
                )
            declarations_dict[decl.name] = decl

    return types.Tool(function_declarations=list(declarations_dict.values()))


# Import tools from submodules to register them
from .file_tools import (
    list_files,
    read_file,
    write_file,
    search_files,
    find_in_files,
    read_pdf,
    read_docx,
    read_excel,
    read_pptx,
)
from .system_tools import (
    run_python,
    start_application,
    kill_process,
    list_processes,
    execute_command,
    get_clipboard,
    set_clipboard,
    get_process_details,
)
from .code_tools import (
    analyze_python_file,
    refactor_code,
    generate_tests,
    debug_python,
    profile_code,
    search_codebase,
    execute_python_with_env,
    render_mermaid,
    get_dependency_graph,
)
from .git_tools import git_operation
from .package_tools import install_package
from .web_tools import fetch_url, fetch_dynamic_url
from .agent_tools import (
    get_agent_capabilities,
    update_plan,
    update_specs,
    delegate_to_agent,
    get_execution_plan,
)
from .media_tools import generate_image, analyze_image, capture_screen, analyze_screen
from .extension_tools import manage_extension
from .db_tools import query_database, list_database_tables, get_database_schema
from .viz_tools import generate_chart, plot_data
from .knowledge_tools import (
    update_knowledge_graph,
    query_knowledge_graph,
    create_note,
    search_notes,
    map_document_relationships,
    analyze_transcript,
    summarize_research_paper,
)

# Re-export for backward compatibility
TOOL_FUNCTIONS = TOOL_REGISTRY

__all__ = [
    "TOOL_REGISTRY",
    "TOOL_FUNCTIONS",
    "tool",
    "validate_args",
    "get_tool_config",
    "list_files",
    "read_file",
    "write_file",
    "search_files",
    "find_in_files",
    "read_pdf",
    "read_docx",
    "read_excel",
    "read_pptx",
    "run_python",
    "start_application",
    "kill_process",
    "list_processes",
    "execute_command",
    "get_clipboard",
    "set_clipboard",
    "get_process_details",
    "analyze_python_file",
    "refactor_code",
    "generate_tests",
    "debug_python",
    "profile_code",
    "search_codebase",
    "execute_python_with_env",
    "render_mermaid",
    "get_dependency_graph",
    "git_operation",
    "install_package",
    "fetch_url",
    "fetch_dynamic_url",
    "get_agent_capabilities",
    "update_plan",
    "update_specs",
    "delegate_to_agent",
    "get_execution_plan",
    "generate_image",
    "analyze_image",
    "capture_screen",
    "analyze_screen",
    "manage_extension",
    "query_database",
    "list_database_tables",
    "get_database_schema",
    "generate_chart",
    "plot_data",
    "update_knowledge_graph",
    "query_knowledge_graph",
    "create_note",
    "search_notes",
    "map_document_relationships",
    "analyze_transcript",
    "summarize_research_paper",
]
