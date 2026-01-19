import json
import re
import time
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple, Union
from google import genai
from google.genai import types
from gemini_agent.config.app_config import AppConfig
from gemini_agent.core import tools
from gemini_agent.core.state_manager import StateManager
from gemini_agent.core.memory_manager import MemoryManager
from gemini_agent.core.cache_manager import CacheManager
from gemini_agent.core.models import ToolCall, ReflectionResult, Plan, Task, TaskStatus
from gemini_agent.core.comms import AsyncEventBus, AgentEvent
from gemini_agent.utils.logger import get_logger

logger = get_logger(__name__)

class BaseAgent:
    """Base class for all agents with Advanced Reasoning and Hierarchical Memory."""
    
    def __init__(
        self, 
        name: str, 
        system_instruction: str, 
        model: Optional[str] = None, 
        api_key: Optional[str] = None,
        state_manager: Optional[StateManager] = None,
        memory_manager: Optional[MemoryManager] = None,
        cache_manager: Optional[CacheManager] = None,
        event_bus: Optional[AsyncEventBus] = None,
        reasoning_level: str = "COT" # BASIC, COT, REFLECTIVE
    ):
        self.name = name
        self.system_instruction = system_instruction
        self.config = AppConfig()
        self.api_key = api_key or self.config.api_key
        self.model = model or self.config.model
        self.client = genai.Client(api_key=self.api_key)
        self.history: List[types.Content] = []
        self.state_manager = state_manager
        self.memory_manager = memory_manager
        self.cache_manager = cache_manager
        self.event_bus = event_bus
        self.reasoning_level = reasoning_level
        self.session_id: Optional[str] = None
        self.task_id: Optional[str] = None

    async def _emit(self, event_type: str, data: Dict[str, Any] = None):
        if self.event_bus and self.session_id:
            event = AgentEvent(
                event_type=event_type,
                session_id=self.session_id,
                task_id=self.task_id,
                agent_name=self.name,
                data=data or {}
            )
            await self.event_bus.publish(event)

    def _load_history(self):
        if self.state_manager and self.session_id:
            state = self.state_manager.load_agent_state(self.session_id, self.name)
            if state and state.history:
                self.history = [types.Content(**h) for h in state.history]
                logger.info(f"Agent '{self.name}' loaded {len(self.history)} messages from history.")
            else:
                self.history = []

    def _save_history(self):
        if self.state_manager and self.session_id:
            history_dicts = [h.model_dump(exclude_none=True) for h in self.history]
            self.state_manager.save_agent_state(self.session_id, self.name, history_dicts)

    def _get_system_instruction(self, current_prompt: Optional[str] = None) -> str:
        instruction = self.system_instruction
        
        if self.reasoning_level in ["COT", "REFLECTIVE"]:
            instruction += (
                "\n\nREASONING GUIDELINES:\n"
                "1. Always think step-by-step before acting or answering.\n"
                "2. Use <thought> tags to wrap your internal reasoning process.\n"
                "3. If you are unsure, express your uncertainty in the thought process.\n"
                "4. After executing a tool, evaluate the result in your next thought block."
            )

        if self.state_manager:
            profile = self.state_manager.get_user_profile()
            instruction += f"\n\nUser Preferences:\n{profile.preferences}"
            
            if self.task_id:
                context = self.state_manager.get_execution_context(self.task_id)
                if context.shared_data:
                    instruction += f"\n\nShared Context:\n{context.shared_data}"

        if self.memory_manager:
            semantic = self.memory_manager.semantic_memory
            if semantic.learned_patterns:
                instruction += f"\n\nLearned Patterns:\n{list(semantic.learned_patterns.keys())[:10]}"
            if semantic.entities:
                instruction += f"\n\nKnown Entities:\n{semantic.entities[:20]}"
            
            # Procedural memory - tool expertise
            procedural = self.memory_manager.procedural_memory
            if procedural.tool_expertise:
                expertise_summary = {k: f"Success: {v.success_count}, Fail: {v.failure_count}" for k, v in procedural.tool_expertise.items()}
                instruction += f"\n\nTool Expertise:\n{expertise_summary}"
            
            # Multi-stage Retrieval Context
            if current_prompt:
                retrieved_context = self.memory_manager.get_relevant_context(current_prompt, limit=3)
                if retrieved_context and "No relevant context found" not in retrieved_context:
                    instruction += f"\n\n{retrieved_context}"

        return instruction

    async def reflect(self, last_output: str) -> ReflectionResult:
        """Self-reflect on the last output to identify errors or improvements."""
        reflection_prompt = (
            f"Review your last response for accuracy, completeness, and adherence to instructions.\n"
            f"Last Response: {last_output}\n\n"
            "Provide your reflection in JSON format with keys: 'is_satisfactory' (bool), 'critique' (string), 'suggestions' (list of strings), 'score' (float 0-1)."
        )
        
        try:
            response = await self.client.aio.models.generate_content(
                model=self.model,
                contents=[types.Content(role="user", parts=[types.Part.from_text(text=reflection_prompt)])],
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )
            data = json.loads(response.text)
            return ReflectionResult(**data)
        except Exception as e:
            logger.error(f"Reflection failed: {e}")
            return ReflectionResult(is_satisfactory=True, critique="Reflection failed, assuming OK.", score=1.0)

    async def run(self, prompt: Union[str, List[types.Part]], session_id: Optional[str] = None, task_id: Optional[str] = None, max_turns: int = 10) -> str:
        self.session_id = session_id
        self.task_id = task_id
        
        prompt_text = prompt if isinstance(prompt, str) else "[Multi-modal Prompt]"
        await self._emit("agent.started", {"prompt": prompt_text, "reasoning_level": self.reasoning_level})
        self._load_history()
        
        if self.memory_manager:
            self.memory_manager.update_working_memory(self.name, active_task=prompt_text)
            # Simple pattern learning from prompt
            if isinstance(prompt, str):
                words = re.findall(r'\w+', prompt.lower())
                for word in words:
                    if len(word) > 5: # Simple heuristic for "interesting" words
                        self.memory_manager.learn_pattern(word)

        if isinstance(prompt, str):
            parts = [types.Part.from_text(text=prompt)]
        else:
            parts = prompt

        self.history.append(types.Content(role="user", parts=parts))
        
        tools_config = self.get_tools()
        system_instruction = self._get_system_instruction(current_prompt=prompt_text if isinstance(prompt, str) else None)
        
        gen_config = types.GenerateContentConfig(
            temperature=0.7,
            top_p=0.95,
            tools=[tools_config] if tools_config else None,
            system_instruction=system_instruction
        )

        turn = 0
        final_response = ""

        while turn < max_turns:
            turn += 1
            try:
                await self._emit("agent.thinking", {"turn": turn})
                
                # LLM Caching
                prompt_data = {
                    "model": self.model,
                    "history": [h.model_dump(exclude_none=True) for h in self.history],
                    "gen_config": gen_config.model_dump(exclude_none=True),
                    "system_instruction": system_instruction
                }
                
                response = None
                if self.cache_manager:
                    cached_response = self.cache_manager.get_llm_response(prompt_data)
                    if cached_response:
                        await self._emit("agent.cache_hit", {"type": "llm"})
                        response = types.GenerateContentResponse(**cached_response)

                if not response:
                    response = await self.client.aio.models.generate_content(
                        model=self.model, contents=self.history, config=gen_config
                    )
                    if self.cache_manager and response:
                        self.cache_manager.set_llm_response(prompt_data, response.model_dump(exclude_none=True))

                if not response.candidates or not response.candidates[0].content:
                    await self._emit("agent.error", {"error": "Empty response from model"})
                    return "Error: Empty response from model."

                candidate = response.candidates[0]
                self.history.append(candidate.content)

                parts = candidate.content.parts
                if parts is None:
                    parts = []
                    
                function_responses = []
                text_content = ""

                for part in parts:
                    if part.text:
                        text_content += part.text
                    if part.function_call:
                        fn_name = part.function_call.name
                        fn_args = part.function_call.args
                        await self._emit("tool.called", {"tool": fn_name, "args": fn_args})

                        # Tool Caching
                        result = None
                        success = True
                        if self.cache_manager:
                            result = self.cache_manager.get_tool_result(fn_name, fn_args)
                            if result:
                                await self._emit("tool.cache_hit", {"tool": fn_name})

                        if result is None:
                            start_time = time.time()
                            try:
                                if fn_name in tools.TOOL_FUNCTIONS:
                                    result = tools.TOOL_FUNCTIONS[fn_name](**fn_args)
                                    success = True
                                    # Cache deterministic tools
                                    if self.cache_manager and fn_name in ["read_file", "list_files", "search_files", "read_pdf", "read_docx", "read_excel", "read_pptx", "search_codebase", "get_dependency_graph"]:
                                        self.cache_manager.set_tool_result(fn_name, fn_args, result)
                                else:
                                    result = f"Error: Tool '{fn_name}' not found."
                                    success = False
                            except Exception as e:
                                result = f"Error executing {fn_name}: {e}"
                                success = False
                            
                            duration = time.time() - start_time
                            await self._emit("tool.result", {"tool": fn_name, "result": str(result), "success": success})
                            
                            tool_call = ToolCall(
                                agent_name=self.name, tool_name=fn_name, arguments=fn_args, result=str(result), success=success
                            )
                            
                            if self.state_manager:
                                self.state_manager.log_tool_call(tool_call)
                            
                            if self.memory_manager:
                                self.memory_manager.update_tool_expertise(tool_call, duration)
                                self.memory_manager.update_working_memory(self.name, current_tool_state={"last_tool": fn_name, "success": success})

                        function_responses.append(types.Part.from_function_response(name=fn_name, response={"result": result}))

                # Extract thoughts if present
                thoughts = re.findall(r"<thought>(.*?)</thought>", text_content, re.DOTALL)
                for thought in thoughts:
                    await self._emit("agent.thought", {"thought": thought.strip()})
                    if self.memory_manager:
                        wm = self.memory_manager.get_working_memory(self.name)
                        wm.intermediate_steps.append({"thought": thought.strip()})

                if function_responses:
                    self.history.append(types.Content(role="user", parts=function_responses))
                else:
                    final_response = text_content
                    
                    # Reflection Loop
                    if self.reasoning_level == "REFLECTIVE":
                        await self._emit("agent.reflecting", {"content": final_response})
                        reflection = await self.reflect(final_response)
                        await self._emit("agent.reflection", reflection.model_dump())
                        
                        if not reflection.is_satisfactory and turn < max_turns - 1:
                            correction_prompt = f"Your previous response was critiqued: {reflection.critique}\nSuggestions: {', '.join(reflection.suggestions)}\nPlease provide a corrected response."
                            await self._emit("agent.correction", {"feedback": reflection.critique})
                            self.history.append(types.Content(role="user", parts=[types.Part.from_text(text=correction_prompt)]))
                            continue
                    
                    break

            except Exception as e:
                logger.error(f"Agent '{self.name}' error: {e}")
                await self._emit("agent.error", {"error": str(e)})
                return f"Agent '{self.name}' crashed: {e}"

        self._save_history()
        if self.memory_manager and self.session_id:
            self.memory_manager.commit_episodic_event(self.session_id, {
                "agent": self.name,
                "prompt": prompt_text,
                "response": final_response,
                "timestamp": datetime.now().isoformat()
            })

        await self._emit("agent.finished", {"response": final_response})
        return final_response

    def get_tools(self) -> Optional[types.Tool]:
        return None

class RouterAgent(BaseAgent):
    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None, state_manager: Optional[StateManager] = None, memory_manager: Optional[MemoryManager] = None, cache_manager: Optional[CacheManager] = None, event_bus: Optional[AsyncEventBus] = None):
        instruction = (
            "You are the Router Agent. Your job is to analyze the user's request and delegate it to the most appropriate specialized agent.\n"
            "Available agents:\n"
            "- PlanningAgent: For decomposing complex goals into structured plans.\n"
            "- ResearchAgent: For information gathering, web search, codebase analysis, reading documents (PDF, DOCX), database querying, data visualization, and academic analysis.\n"
            "- CodeAgent: For writing, refactoring, debugging, explaining Python code, visualizing architectures, database management, and generating charts.\n"
            "- FileAgent: For file system operations (list, read, write, search, read_pdf, read_docx).\n"
            "- VisionAgent: For analyzing images, OCR, and diagram understanding.\n"
            "- ReasoningAgent: For complex logic, verification, and plan evaluation.\n"
            "- KnowledgeAgent: For managing the personal knowledge graph, note-taking, and document relationship mapping.\n"
            "For complex requests, use the PlanningAgent to create a plan first."
        )
        super().__init__("RouterAgent", instruction, model, api_key, state_manager, memory_manager, cache_manager, event_bus, reasoning_level="REFLECTIVE")

    def get_tools(self) -> types.Tool:
        return tools.get_tool_config()

class PlanningAgent(BaseAgent):
    """Specialized agent for goal decomposition and plan generation."""
    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None, state_manager: Optional[StateManager] = None, memory_manager: Optional[MemoryManager] = None, cache_manager: Optional[CacheManager] = None, event_bus: Optional[AsyncEventBus] = None):
        instruction = (
            "You are the Planning Agent. Your job is to take a complex goal and break it down into a structured plan.\n"
            "A plan consists of multiple tasks. Each task should have:\n"
            "- description: Clear instruction of what to do.\n"
            "- assigned_agent: One of [ResearchAgent, CodeAgent, FileAgent, VisionAgent, ReasoningAgent, KnowledgeAgent].\n"
            "- dependencies: List of task IDs that must be completed before this task can start.\n"
            "- priority: 1 (highest) to 5 (lowest).\n\n"
            "Output the plan in JSON format that matches the Plan model."
        )
        super().__init__("PlanningAgent", instruction, model, api_key, state_manager, memory_manager, cache_manager, event_bus, reasoning_level="COT")

    async def generate_plan(self, goal: str) -> Plan:
        # Check procedural memory for workflow templates
        if self.memory_manager:
            template = self.memory_manager.get_workflow_template(goal)
            if template:
                await self._emit("plan.template_found", {"goal": goal})
                return template

        prompt = f"Generate a detailed plan to achieve this goal: {goal}"
        await self._emit("plan.generating", {"goal": goal})
        
        response = await self.client.aio.models.generate_content(
            model=self.model,
            contents=[types.Content(role="user", parts=[types.Part.from_text(text=prompt)])],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                system_instruction=self._get_system_instruction(current_prompt=goal)
            )
        )
        try:
            data = json.loads(response.text)
            data["goal"] = goal
            plan = Plan(**data)
            await self._emit("plan.created", plan.model_dump())
            return plan
        except Exception as e:
            logger.error(f"Failed to parse plan: {e}")
            plan = Plan(goal=goal, tasks=[Task(description=goal, assigned_agent="RouterAgent")])
            await self._emit("plan.created", plan.model_dump())
            return plan

class ResearchAgent(BaseAgent):
    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None, state_manager: Optional[StateManager] = None, memory_manager: Optional[MemoryManager] = None, cache_manager: Optional[CacheManager] = None, event_bus: Optional[AsyncEventBus] = None):
        instruction = "You are the Research Agent. Gather information and provide deep insights using CoT reasoning. You can read PDF/DOCX, access the clipboard, query databases, plot data, and analyze academic papers or transcripts."
        super().__init__("ResearchAgent", instruction, model, api_key, state_manager, memory_manager, cache_manager, event_bus, reasoning_level="COT")

    def get_tools(self) -> types.Tool:
        research_tools = ["fetch_url", "search_files", "find_in_files", "list_files", "read_file", "search_codebase", "read_pdf", "read_docx", "get_clipboard", "query_database", "list_database_tables", "get_database_schema", "plot_data", "map_document_relationships", "analyze_transcript", "summarize_research_paper"]
        declarations = [tools.auto_generate_declaration(tools.TOOL_FUNCTIONS[name]) for name in research_tools if name in tools.TOOL_FUNCTIONS]
        return types.Tool(function_declarations=declarations)

class CodeAgent(BaseAgent):
    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None, state_manager: Optional[StateManager] = None, memory_manager: Optional[MemoryManager] = None, cache_manager: Optional[CacheManager] = None, event_bus: Optional[AsyncEventBus] = None):
        instruction = "You are the Code Agent. Write and analyze code with high precision. Use reflection to ensure code quality. You can execute system commands, monitor processes, manage databases, and generate charts."
        super().__init__("CodeAgent", instruction, model, api_key, state_manager, memory_manager, cache_manager, event_bus, reasoning_level="REFLECTIVE")

    def get_tools(self) -> types.Tool:
        code_tools = ["run_python", "analyze_python_file", "refactor_code", "generate_tests", "debug_python", "profile_code", "execute_python_with_env", "render_mermaid", "execute_command", "list_processes", "get_process_details", "kill_process", "query_database", "list_database_tables", "get_database_schema", "generate_chart"]
        declarations = [tools.auto_generate_declaration(tools.TOOL_FUNCTIONS[name]) for name in code_tools if name in tools.TOOL_FUNCTIONS]
        return types.Tool(function_declarations=declarations)

class FileAgent(BaseAgent):
    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None, state_manager: Optional[StateManager] = None, memory_manager: Optional[MemoryManager] = None, cache_manager: Optional[CacheManager] = None, event_bus: Optional[AsyncEventBus] = None):
        instruction = "You are the File Agent. Manage the file system reliably. You can read PDF and DOCX files, use the clipboard, and capture the screen."
        super().__init__("FileAgent", instruction, model, api_key, state_manager, memory_manager, cache_manager, event_bus, reasoning_level="BASIC")

    def get_tools(self) -> types.Tool:
        file_tools = ["list_files", "read_file", "write_file", "search_files", "find_in_files", "git_operation", "install_package", "read_pdf", "read_docx", "get_clipboard", "set_clipboard", "capture_screen"]
        declarations = [tools.auto_generate_declaration(tools.TOOL_FUNCTIONS[name]) for name in file_tools if name in tools.TOOL_FUNCTIONS]
        return types.Tool(function_declarations=declarations)

class VisionAgent(BaseAgent):
    """Specialized agent for visual analysis and OCR."""
    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None, state_manager: Optional[StateManager] = None, memory_manager: Optional[MemoryManager] = None, cache_manager: Optional[CacheManager] = None, event_bus: Optional[AsyncEventBus] = None):
        instruction = (
            "You are the Vision Agent. Your expertise is in analyzing images, performing OCR, and understanding diagrams.\n"
            "Use the analyze_image tool to process visual data. You can also capture and analyze the current screen."
        )
        super().__init__("VisionAgent", instruction, model, api_key, state_manager, memory_manager, cache_manager, event_bus, reasoning_level="COT")

    def get_tools(self) -> types.Tool:
        vision_tools = ["analyze_image", "generate_image", "capture_screen", "analyze_screen"]
        declarations = [tools.auto_generate_declaration(tools.TOOL_FUNCTIONS[name]) for name in vision_tools if name in tools.TOOL_FUNCTIONS]
        return types.Tool(function_declarations=declarations)

class ReasoningAgent(BaseAgent):
    """Specialized agent for complex logic, verification, and ToT evaluation."""
    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None, state_manager: Optional[StateManager] = None, memory_manager: Optional[MemoryManager] = None, cache_manager: Optional[CacheManager] = None, event_bus: Optional[AsyncEventBus] = None):
        instruction = (
            "You are the Reasoning Agent. Your expertise is in formal logic, plan evaluation, and error detection.\n"
            "You help other agents by critiquing their plans and verifying their conclusions."
        )
        super().__init__("ReasoningAgent", instruction, model, api_key, state_manager, memory_manager, cache_manager, event_bus, reasoning_level="REFLECTIVE")

class KnowledgeAgent(BaseAgent):
    """Specialized agent for knowledge management, note-taking, and relationship mapping."""
    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None, state_manager: Optional[StateManager] = None, memory_manager: Optional[MemoryManager] = None, cache_manager: Optional[CacheManager] = None, event_bus: Optional[AsyncEventBus] = None):
        instruction = (
            "You are the Knowledge Agent. Your job is to build and maintain a personal knowledge base.\n"
            "You can create and search notes, update the knowledge graph, and map relationships between documents."
        )
        super().__init__("KnowledgeAgent", instruction, model, api_key, state_manager, memory_manager, cache_manager, event_bus, reasoning_level="COT")

    def get_tools(self) -> types.Tool:
        knowledge_tools = ["update_knowledge_graph", "query_knowledge_graph", "create_note", "search_notes", "map_document_relationships"]
        declarations = [tools.auto_generate_declaration(tools.TOOL_FUNCTIONS[name]) for name in knowledge_tools if name in tools.TOOL_FUNCTIONS]
        return types.Tool(function_declarations=declarations)
