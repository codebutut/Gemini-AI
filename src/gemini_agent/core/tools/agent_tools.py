import os
import sys
import inspect
import asyncio
from pydantic import BaseModel, Field
from . import tool, validate_args, TOOL_REGISTRY
from .file_tools import write_file


class IntrospectionArgs(BaseModel):
    category: str = Field(
        "all",
        description="Category to inspect: 'capabilities', 'tools', 'config', 'all'.",
    )


class DelegateArgs(BaseModel):
    agent_name: str = Field(
        ...,
        description="Name/Role of the sub-agent (e.g., 'ResearchAgent', 'CodeAgent', 'FileAgent', 'VisionAgent', 'ReasoningAgent', 'PlanningAgent').",
    )
    objective: str = Field(
        ..., description="The specific task or objective for the sub-agent."
    )


@tool
@validate_args(IntrospectionArgs)
def get_agent_capabilities(category: str = "all") -> str:
    """
    Introspects the agent's own capabilities, configuration, and available tools.

    Args:
        category: 'capabilities', 'tools', 'config', 'all'

    Returns:
        str: Introspection report.
    """
    lines = []

    if category in ("tools", "all"):
        lines.append("## Available Tools")
        for name, func in TOOL_REGISTRY.items():
            doc = inspect.getdoc(func) or "No description."
            first_line = doc.split("\n")[0]
            lines.append(f"- **{name}**: {first_line}")
        lines.append("")

    if category in ("capabilities", "all"):
        lines.append("## Core Competencies")
        lines.append("1. System Operations (Files, Processes, Git)")
        lines.append("2. Python Engineering (Analysis, Refactoring, Testing)")
        lines.append("3. Deep Review & Orchestration")
        lines.append("4. Intelligent Search (Ripgrep/Native)")
        lines.append(
            "5. Multi-Agent System (Router, Research, Code, File, Vision, Reasoning, Planning Agents)"
        )
        lines.append(
            "6. Multi-modal Support (PDF, DOCX, Image Analysis, Mermaid Diagrams)"
        )
        lines.append(
            "7. Planning & Execution Engine (Goal Decomposition, Task Tracking)"
        )
        lines.append("")

    if category in ("config", "all"):
        lines.append("## Runtime Configuration")
        lines.append(f"Working Directory: {os.getcwd()}")
        lines.append(f"Python Version: {sys.version.split()[0]}")
        lines.append(f"Platform: {sys.platform}")

    return "\n".join(lines)


@tool
def update_plan(content: str) -> str:
    """
    Updates the 'plan.md' file to track project progress and next steps.

    Args:
        content: The full updated content for plan.md.

    Returns:
        str: Success or error message.
    """
    return write_file(filepath="plan.md", content=content)


@tool
def update_specs(content: str) -> str:
    """
    Updates the 'specs.md' file to document technical specifications and architecture.

    Args:
        content: The full updated content for specs.md.

    Returns:
        str: Success or error message.
    """
    return write_file(filepath="specs.md", content=content)


@tool
@validate_args(DelegateArgs)
def delegate_to_agent(agent_name: str, objective: str) -> str:
    """
    Delegates a complex task to a specialized sub-agent.

    Args:
        agent_name: Name of the specialized agent (e.g., "ResearchAgent", "CodeAgent", "FileAgent", "VisionAgent", "ReasoningAgent", "PlanningAgent").
        objective: The detailed goal for the sub-agent to achieve.

    Returns:
        str: The result or report from the sub-agent.
    """
    try:
        # Dynamic import to avoid circular dependency
        from gemini_agent.core.mas.agents import (
            ResearchAgent,
            CodeAgent,
            FileAgent,
            VisionAgent,
            ReasoningAgent,
            PlanningAgent,
        )
        from gemini_agent.core.sub_agent import SubAgent
        from gemini_agent.core.mas.orchestrator import get_current_context

        state_manager, session_id, task_id, event_bus = get_current_context()

        agent_map = {
            "ResearchAgent": ResearchAgent,
            "CodeAgent": CodeAgent,
            "FileAgent": FileAgent,
            "VisionAgent": VisionAgent,
            "ReasoningAgent": ReasoningAgent,
            "PlanningAgent": PlanningAgent,
            "Researcher": ResearchAgent,
            "Coder": CodeAgent,
            "Filer": FileAgent,
            "Vision": VisionAgent,
            "Reasoner": ReasoningAgent,
            "Planner": PlanningAgent,
        }

        if agent_name in agent_map:
            agent = agent_map[agent_name](
                state_manager=state_manager, event_bus=event_bus
            )
        else:
            agent = SubAgent(name=agent_name)

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        if loop.is_running():
            import nest_asyncio

            nest_asyncio.apply()

        result = asyncio.run(
            agent.run(objective, session_id=session_id, task_id=task_id)
        )
        return f"Sub-Agent '{agent_name}' Result:\n{result}"
    except Exception as e:
        import traceback

        return f"Error delegating to agent: {str(e)}\n{traceback.format_exc()}"


@tool
def get_execution_plan() -> str:
    """
    Retrieves the current structured execution plan for the session.

    Returns:
        str: JSON representation of the plan or error message.
    """
    from gemini_agent.core.mas.orchestrator import get_current_context

    state_manager, session_id, _, _ = get_current_context()
    if state_manager and session_id:
        plan = state_manager.load_plan(session_id)
        if plan:
            return plan.model_dump_json(indent=2)
    return "No active structured plan found for this session."
