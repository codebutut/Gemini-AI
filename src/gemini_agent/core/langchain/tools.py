from typing import List
from langchain_core.tools import StructuredTool
from gemini_agent.core.tools import TOOL_REGISTRY


def get_langchain_tools() -> List[StructuredTool]:
    """
    Converts all registered tools in TOOL_REGISTRY to LangChain StructuredTools.
    """
    lc_tools = []
    for name, func in TOOL_REGISTRY.items():
        # LangChain's StructuredTool.from_function uses the function's docstring and signature
        # to create the tool's description and schema.
        tool = StructuredTool.from_function(
            func=func,
            name=name,
            description=func.__doc__ or "No description provided.",
        )
        lc_tools.append(tool)
    return lc_tools
