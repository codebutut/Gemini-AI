import os
from typing import Any, List, Optional, Union
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from PyQt6.QtCore import pyqtSignal

from gemini_agent.core.langchain.tools import get_langchain_tools
# from gemini_agent.core.langchain.callbacks import LangChainStatusCallbackHandler # Might need to adapt this


class LangChainAgent:
    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-2.0-flash",
        system_instruction: Optional[str] = None,
        status_signal: Optional[pyqtSignal] = None,
        terminal_signal: Optional[pyqtSignal] = None,
    ):
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=0.7,
        )

        self.tools = get_langchain_tools()
        self.system_instruction = system_instruction
        self.status_signal = status_signal
        self.terminal_signal = terminal_signal

        # Create the agent graph using the new LangChain 1.x API
        self.graph = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=self.system_instruction,
        )

    async def run(self, input_text: str) -> str:
        inputs = {"messages": [HumanMessage(content=input_text)]}

        final_output = ""
        # Using stream to capture updates and emit signals
        async for chunk in self.graph.astream(inputs, stream_mode="updates"):
            # chunk is a dict of node updates
            for node_name, update in chunk.items():
                if self.status_signal:
                    self.status_signal.emit(f"Node {node_name} active...")

                # Check for tool calls in the update
                if "messages" in update:
                    for msg in update["messages"]:
                        if isinstance(msg, AIMessage) and msg.tool_calls:
                            for tc in msg.tool_calls:
                                if self.terminal_signal:
                                    self.terminal_signal.emit(
                                        f"🛠️ Calling tool: {tc['name']} with {tc['args']}\n",
                                        "info",
                                    )
                        if isinstance(msg, AIMessage) and msg.content:
                            final_output = msg.content

        # If we didn't get a final output from the stream, try to get it from the final state
        if not final_output:
            state = await self.graph.ainvoke(inputs)
            if "messages" in state and state["messages"]:
                final_output = state["messages"][-1].content

        return final_output
