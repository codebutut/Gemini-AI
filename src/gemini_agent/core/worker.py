import asyncio
import uuid
from dataclasses import dataclass
from typing import Any

from google import genai
from google.genai import types, errors
from PyQt6.QtCore import QObject, QThread, pyqtSignal

from gemini_agent.config.app_config import ModelRegistry
from gemini_agent.core import tools
from gemini_agent.core.context_manager import ContextManager
from gemini_agent.core.extension_manager import ExtensionManager
from gemini_agent.core.mode_detector import ModeDetector
from gemini_agent.core.tool_executor import ToolExecutor
from gemini_agent.core.comms import AgentEvent
from gemini_agent.utils.helpers import RateLimiter
from gemini_agent.utils.logger import AgentLoggerAdapter, get_logger

logger = get_logger(__name__)


@dataclass
class WorkerConfig:
    api_key: str
    prompt: str
    model: str
    file_paths: list[str]
    history_context: list[dict[str, str]]
    use_grounding: bool = False
    system_instruction: str | None = None
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int = 40
    max_turns: int = 20
    thinking_enabled: bool = False
    thinking_budget: int = 4096
    session_id: str | None = None
    initial_plan: str = ""
    initial_specs: str = ""
    extension_manager: ExtensionManager | None = None
    use_langchain: bool = False
    use_mas: bool = False


class GeminiWorker(QObject):
    """
    Async worker to handle Gemini API requests using asyncio.
    Communicates with the UI via signals.
    """

    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    status_update = pyqtSignal(str)
    terminal_output = pyqtSignal(str, str)
    request_confirmation = pyqtSignal(str, dict, str)
    plan_updated = pyqtSignal(str)
    specs_updated = pyqtSignal(str)
    usage_updated = pyqtSignal(str, int, int)
    rate_limit_updated = pyqtSignal(str, int, int)  # model_id, remaining, limit

    # Shared rate limiters per model to persist across worker instances
    _RATE_LIMITERS: dict[str, RateLimiter] = {}

    def __init__(self, config: WorkerConfig):
        super().__init__()
        self.config = config
        self.log = AgentLoggerAdapter(logger, {"session_id": config.session_id})

        self._confirmation_event = asyncio.Event()
        self._confirmation_result: bool | None = None
        self._confirmation_modified_args: dict[str, Any] | None = None
        self._current_confirmation_id: str | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

        self.mode_detector = ModeDetector()
        self.tool_executor: ToolExecutor | None = None
        self.context_manager: ContextManager | None = None
        self._is_cancelled = False

        # Initialize or retrieve rate limiter for this model
        self.rate_limiter = self._get_rate_limiter(config.model)

    def _get_rate_limiter(self, model_id: str) -> RateLimiter:
        """Retrieves or creates a rate limiter for the specific model."""
        if model_id not in GeminiWorker._RATE_LIMITERS:
            attrs = ModelRegistry.get_attributes(model_id)
            GeminiWorker._RATE_LIMITERS[model_id] = RateLimiter(
                max_requests=attrs.get("rate_limit_requests", 20),
                period=attrs.get("rate_limit_period", 60),
                auto_refill=True,
            )
        return GeminiWorker._RATE_LIMITERS[model_id]

    def cancel(self):
        self._is_cancelled = True

    def confirm_tool(self, confirmation_id: str, allowed: bool, modified_args: dict[str, Any] | None = None) -> None:
        """Called by the UI thread to provide confirmation result."""
        if self._current_confirmation_id == confirmation_id:
            self._confirmation_result = allowed
            self._confirmation_modified_args = modified_args

            if self._loop and self._loop.is_running():
                self._loop.call_soon_threadsafe(self._confirmation_event.set)
            else:
                self.log.warning(
                    "Worker loop not available/running to set confirmation event. Falling back to direct set."
                )
                # Fallback for tests or non-running loop
                try:
                    self._confirmation_event.set()
                except Exception as e:
                    self.log.error(f"Failed to set confirmation event: {e}")

    async def _request_tool_confirmation(
        self, fn_name: str, fn_args: dict[str, Any]
    ) -> tuple[bool, dict[str, Any] | None]:
        """Requests user confirmation for dangerous tools (async)."""
        confirmation_id = str(uuid.uuid4())
        self._current_confirmation_id = confirmation_id
        self._confirmation_event.clear()
        self._confirmation_modified_args = None

        self.status_update.emit(f"⚠️ Waiting for confirmation: {fn_name}...")
        self.terminal_output.emit(f"⚠️ Requesting confirmation for: {fn_name}\n", "info")
        self.request_confirmation.emit(fn_name, fn_args, confirmation_id)

        await self._confirmation_event.wait()
        return self._confirmation_result, self._confirmation_modified_args

    def _create_config(self, tools_config: types.Tool | None = None) -> types.GenerateContentConfig:
        config_args = {
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "top_k": self.config.top_k,
            "system_instruction": self.config.system_instruction,
        }
        if tools_config:
            config_args["tools"] = [tools_config]

        # Check model attributes for thinking support
        model_attrs = ModelRegistry.get_attributes(self.config.model)
        supports_thinking = model_attrs.get("supports_thinking", False)

        if self.config.thinking_enabled and supports_thinking:
            try:
                if hasattr(types, "ThinkingConfig"):
                    config_args["thinking_config"] = types.ThinkingConfig(include_thoughts=True)
                if self.config.thinking_budget > 0:
                    config_args["max_output_tokens"] = self.config.thinking_budget
            except Exception as e:
                self.log.warning(f"Failed to configure thinking: {e}")
        elif self.config.thinking_enabled and not supports_thinking:
            self.log.warning(f"Thinking enabled but model {self.config.model} does not support it. Ignoring.")

        return types.GenerateContentConfig(**config_args)

    async def run_async(self) -> None:
        """Main async execution loop."""
        if not self.config.api_key:
            self.error.emit("API Key is missing.")
            return

        try:
            client = genai.Client(api_key=self.config.api_key)
            self.context_manager = ContextManager(client)

            # Capture the loop for thread-safe callbacks from the executor thread
            loop = asyncio.get_running_loop()
            self._loop = loop

            def sync_confirmation_callback(fn_name: str, fn_args: dict[str, Any]) -> tuple[bool, dict[str, Any] | None]:
                future = asyncio.run_coroutine_threadsafe(self._request_tool_confirmation(fn_name, fn_args), loop)
                return future.result()

            self.tool_executor = ToolExecutor(
                status_callback=self.status_update.emit,
                terminal_callback=self.terminal_output.emit,
                confirmation_callback=sync_confirmation_callback,
                extension_manager=self.config.extension_manager,
            )
            self.tool_executor.current_plan = self.config.initial_plan
            self.tool_executor.current_specs = self.config.initial_specs

            if self.config.use_langchain:
                await self._run_with_langchain()
                return

            if self.config.use_mas:
                await self._run_with_mas()
                return

            # 1. Prepare History
            gemini_contents = self.context_manager.prepare_history(self.config.history_context)

            # 2. Prepare Current Turn Content
            current_turn_parts = self.context_manager.prepare_current_turn(
                self.config.prompt,
                self.config.file_paths,
                self.tool_executor.current_plan,
                self.tool_executor.current_specs,
            )
            gemini_contents.append(types.Content(role="user", parts=current_turn_parts))

            # 3. Detect mode
            mode = self.mode_detector.detect_mode(self.config.prompt, self.config.use_grounding)
            self.log.info(f"Using mode: {mode}")

            # 4. Execute
            if mode == "grounding":
                await self._run_grounding_mode(client, gemini_contents)
            else:
                await self._handle_function_calls(client, gemini_contents)

        except Exception as e:
            self._handle_run_error(e)
        finally:
            pass

    async def _run_with_mas(self) -> None:
        from gemini_agent.core.mas.orchestrator import MultiAgentOrchestrator
        
        self.status_update.emit("🤖 Initializing Multi-Agent System...")
        self.terminal_output.emit("🤖 Multi-Agent System (MAS) Mode Active\n", "info")
        
        orchestrator = MultiAgentOrchestrator(
            api_key=self.config.api_key,
            model=self.config.model
        )
        
        # Bridge EventBus to PyQt signals
        async def event_bridge(event: AgentEvent):
            if event.event_type == "agent.started":
                self.status_update.emit(f"🤖 {event.agent_name} started...")
            elif event.event_type == "tool.called":
                self.status_update.emit(f"🛠️ {event.agent_name} calling {event.data.get('tool')}...")
                self.terminal_output.emit(f"🛠️ [{event.agent_name}] Executing {event.data.get('tool')}({event.data.get('args')})\n", "info")
            elif event.event_type == "tool.result":
                status = "✅" if event.data.get("success") else "❌"
                self.terminal_output.emit(f"{status} [{event.agent_name}] Result: {str(event.data.get('result'))[:200]}...\n", "success" if event.data.get("success") else "error")
            elif event.event_type == "agent.thought":
                self.terminal_output.emit(f"💭 [{event.agent_name}] {event.data.get('thought')}\n", "thought")
            elif event.event_type == "agent.error":
                self.terminal_output.emit(f"❌ [{event.agent_name}] Error: {event.data.get('error')}\n", "error")
            elif event.event_type == "task.finished":
                self.status_update.emit("✅ Task completed.")

        await orchestrator.event_bus.subscribe_all(event_bridge)
        
        try:
            # Pass session_id to orchestrator for state management
            result = await orchestrator.run(self.config.prompt, session_id=self.config.session_id)
            self.finished.emit(result)
        except Exception as e:
            self._handle_run_error(e)

    async def _run_with_langchain(self) -> None:
        from gemini_agent.core.langchain.agent import LangChainAgent
        
        self.status_update.emit("🚀 Initializing LangChain Agent...")
        
        agent = LangChainAgent(
            api_key=self.config.api_key,
            model_name=self.config.model,
            system_instruction=self.config.system_instruction,
            status_signal=self.status_update,
            terminal_signal=self.terminal_output
        )
        
        # Load history into memory
        for msg in self.config.history_context:
            if msg["role"] == "user":
                agent.memory.chat_history.add_user_message(msg["text"])
            else:
                agent.memory.chat_history.add_ai_message(msg["text"])
        
        try:
            result = await agent.run(self.config.prompt)
            self.finished.emit(result)
        except Exception as e:
            self._handle_run_error(e)

    async def _run_grounding_mode(self, client: genai.Client, gemini_contents: list[types.Content]) -> None:
        self.status_update.emit("🔍 Searching the web...")

        await self.rate_limiter.acquire_async()
        if self._is_cancelled:
            return

        config = self._create_config(tools_config=types.Tool(google_search=types.GoogleSearch()))

        try:
            response = await client.aio.models.generate_content(
                model=self.config.model, contents=gemini_contents, config=config
            )

            self._update_usage(response)

            if response.candidates:
                try:
                    final_text = response.text
                except Exception:
                    final_text = "[Text response unavailable from web search]"
                self.finished.emit(final_text or "[No text response from web search]")
            else:
                self.error.emit("API returned no candidates in grounding mode.")

        except Exception as api_error:
            if "Tool use with function calling is unsupported" in str(api_error):
                self.status_update.emit("⚠️ Web search not available, using local tools...")
                await self._handle_function_calls(client, gemini_contents)
            else:
                raise api_error

    def _update_usage(self, response: Any) -> None:
        if response.usage_metadata and self.config.session_id:
            self.usage_updated.emit(
                self.config.session_id,
                response.usage_metadata.prompt_token_count or 0,
                response.usage_metadata.candidates_token_count or 0,
            )

        # Update rate limit telemetry
        try:
            remaining = self.rate_limiter.remaining()
            limit = self.rate_limiter.max_requests

            # Attempt to extract real-time telemetry from response headers if available
            if hasattr(response, "_response") and hasattr(response._response, "headers"):
                headers = response._response.headers
                # Gemini API headers for rate limits (RPM)
                # Note: These header names might vary slightly depending on the specific API endpoint
                remaining_header = headers.get("x-ratelimit-remaining-requests")
                limit_header = headers.get("x-ratelimit-limit-requests")

                if remaining_header is not None and limit_header is not None:
                    remaining = int(remaining_header)
                    limit = int(limit_header)
                    self.rate_limiter.update_limits(remaining, limit)

            self.rate_limit_updated.emit(self.config.model, remaining, limit)
        except Exception as e:
            self.log.warning(f"Failed to update rate limit telemetry: {e}")

    def _handle_run_error(self, e: Exception) -> None:
        error_str = str(e)
        self.log.error(f"Worker Error: {error_str}", exc_info=True)
        if "ResourceExhausted" in error_str or "429" in error_str:
            self.error.emit(f"Rate limit exceeded. Please wait. Details: {e}")
        elif "GoogleAuthError" in error_str or "401" in error_str or "Unauthenticated" in error_str:
            self.error.emit(f"Authentication error. Check API key. Details: {e}")
        else:
            self.error.emit(f"An unexpected error occurred: {e}")

    async def _handle_function_calls(self, client: genai.Client, gemini_contents: list[types.Content]) -> None:
        final_response_text = ""
        loop_active = True
        turn_count = 0
        progress_metrics = []
        last_output = None

        extra_tools = self.config.extension_manager.get_all_tools() if self.config.extension_manager else []
        tools_config = tools.get_tool_config(extra_declarations=extra_tools)

        while loop_active and turn_count < self.config.max_turns and not self._is_cancelled:
            turn_count += 1
            self.status_update.emit(f"🔄 Thinking (Turn {turn_count}/{self.config.max_turns})...")

            await self.rate_limiter.acquire_async()
            if self._is_cancelled:
                return

            config = self._create_config(tools_config=tools_config)

            max_retries = 3
            retry_delay = 2
            for attempt in range(max_retries + 1):
                try:
                    response = await client.aio.models.generate_content(
                        model=self.config.model, contents=gemini_contents, config=config
                    )
                    break # Success, exit retry loop
                except errors.ClientError as e:
                    if e.code == 429 and attempt < max_retries:
                        self.log.warning(f"Rate limit hit (429). Retrying in {retry_delay}s... (Attempt {attempt+1}/{max_retries})")
                        self.status_update.emit(f"⏳ Rate limit hit. Retrying in {retry_delay}s...")
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 2 # Exponential backoff
                        continue
                    
                    # Not 429 or retries exhausted
                    self.log.error(f"API ClientError: {e.code} - {e.message}")
                    error_message = e.message or "Unknown API Error"
                    self.error.emit(f"API Request Failed ({e.code}): {error_message}")
                    return

            self._update_usage(response)

            if not response.candidates or response.candidates[0] is None:
                self.error.emit("API returned no candidates or an empty candidate.")
                return

            candidate = response.candidates[0]
            if not self._is_valid_candidate(candidate):
                return

            model_parts = candidate.content.parts
            gemini_contents.append(candidate.content)

            function_responses = []
            function_calls = [p.function_call for p in model_parts if p.function_call]

            if function_calls:
                # Execute all tools in parallel for this turn
                tasks = []
                for fc in function_calls:
                    tasks.append(
                        asyncio.to_thread(self.tool_executor.execute, fc.name, {k: v for k, v in fc.args.items()})
                    )

                results = await asyncio.gather(*tasks)

                for fc, result in zip(function_calls, results, strict=False):
                    # Check if plan or specs were updated
                    if (
                        fc.name in ["update_plan", "write_file"]
                        and self.tool_executor.current_plan != self.config.initial_plan
                    ):
                        self.plan_updated.emit(self.tool_executor.current_plan)
                    if (
                        fc.name in ["update_specs", "write_file"]
                        and self.tool_executor.current_specs != self.config.initial_specs
                    ):
                        self.specs_updated.emit(self.tool_executor.current_specs)

                    current_output = f"{fc.name}:{str(result)[:50]}"
                    progress_metrics.append("progress_made" if current_output != last_output else "no_progress")
                    last_output = current_output

                    # Safely handle result content to avoid API errors (400 Bad Request)
                    # Convert to string to ensure JSON serializability and truncate if excessive
                    safe_result = str(result)
                    if len(safe_result) > 500000: # 500k char limit
                        safe_result = safe_result[:500000] + "... [Output Truncated]"

                    function_responses.append(
                        types.Part.from_function_response(name=fc.name, response={"result": safe_result})
                    )

                if self._is_stuck(progress_metrics):
                    final_response_text = "[System: Agent stuck in repetitive loop. Process stopped.]"
                    loop_active = False
                else:
                    gemini_contents.append(types.Content(role="user", parts=function_responses))
                    self.status_update.emit(f"🔄 Processing results (Turn {turn_count}/{self.config.max_turns})...")
            else:
                try:
                    final_response_text = response.text or "(Task completed silently)"
                except Exception:
                    final_response_text = "(Task completed, but text response was unavailable)"
                loop_active = False

        if turn_count >= self.config.max_turns:
            max_turn_msg = f"[System: Max agent turns reached ({self.config.max_turns}). Process stopped.]"
            final_response_text = f"{final_response_text}\n\n{max_turn_msg}" if final_response_text else max_turn_msg

        if not self._is_cancelled:
            self.finished.emit(final_response_text)

    def _is_valid_candidate(self, candidate: Any) -> bool:
        if (
            not hasattr(candidate, "content")
            or candidate.content is None
            or not hasattr(candidate.content, "parts")
            or candidate.content.parts is None
        ):
            finish_reason = getattr(candidate, "finish_reason", "UNKNOWN")
            error_msg = f"API returned an empty response (Finish Reason: {finish_reason})."
            if finish_reason == "SAFETY":
                error_msg = "Response blocked by safety filters."
            self.error.emit(error_msg)
            return False
        return True

    def _is_stuck(self, progress_metrics: list[str]) -> bool:
        return len(progress_metrics) >= 3 and all(p == "no_progress" for p in progress_metrics[-3:])


class GeminiWorkerThread(QThread):
    """
    A QThread that runs an asyncio event loop.
    Used to run GeminiWorker without blocking the UI.
    """

    def __init__(self, worker: GeminiWorker):
        super().__init__()
        self.worker = worker
        self.loop = None

    def run(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_until_complete(self.worker.run_async())
        finally:
            self.loop.close()

    def stop(self):
        self.worker.cancel()
        if self.loop and self.loop.is_running():
            self.loop.call_soon_threadsafe(self.loop.stop)
