from typing import Any, Dict, List, Optional, Union
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.agents import AgentAction, AgentFinish
from PyQt6.QtCore import pyqtSignal

class LangChainStatusCallbackHandler(BaseCallbackHandler):
    """
    Callback handler that bridges LangChain events to PyQt signals.
    """
    def __init__(self, status_signal: pyqtSignal, terminal_signal: pyqtSignal):
        self.status_signal = status_signal
        self.terminal_signal = terminal_signal

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        self.status_signal.emit("🔄 LLM Thinking...")

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        """Run on agent action."""
        tool_name = action.tool
        tool_input = action.tool_input
        self.status_signal.emit(f"🛠️ Using tool: {tool_name}...")
        self.terminal_signal.emit(f"🛠️ Executing {tool_name} with input: {tool_input}\n", "info")

    def on_tool_end(self, output: str, **kwargs: Any) -> Any:
        """Run when tool ends running."""
        self.terminal_signal.emit(f"✅ Tool output: {str(output)[:500]}...\n", "success")

    def on_tool_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
        """Run when tool errors."""
        self.terminal_signal.emit(f"❌ Tool error: {str(error)}\n", "error")

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> Any:
        """Run on agent end."""
        self.status_signal.emit("🏁 Agent finished.")
