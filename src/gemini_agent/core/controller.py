import asyncio
import logging
from typing import Any

from PyQt6.QtCore import QObject, pyqtSignal

from gemini_agent.config.app_config import AppConfig, Role
from gemini_agent.core.attachment_manager import AttachmentManager
from gemini_agent.core.checkpoint_manager import CheckpointManager
from gemini_agent.core.conductor_manager import ConductorManager
from gemini_agent.core.extension_manager import ExtensionManager
from gemini_agent.core.indexer import Indexer
from gemini_agent.core.session_manager import SessionManager
from gemini_agent.core.vector_store import VectorStore
from gemini_agent.core.worker import GeminiWorker, GeminiWorkerThread, WorkerConfig

logger = logging.getLogger(__name__)


class ChatController(QObject):
    """
    Controller handling the business logic of the chat application.
    Separates UI from gemini_agent.core logic and worker coordination.
    """

    status_updated = pyqtSignal(str)
    response_received = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    usage_updated = pyqtSignal(str, int, int)
    rate_limit_updated = pyqtSignal(str, int, int)
    terminal_output = pyqtSignal(str, str)
    tool_confirmation_requested = pyqtSignal(str, dict, str)

    def __init__(
        self,
        app_config: AppConfig,
        session_manager: SessionManager,
        attachment_manager: AttachmentManager,
        conductor_manager: ConductorManager,
        indexer: Indexer,
        extension_manager: ExtensionManager,
        checkpoint_manager: CheckpointManager,
        vector_store: VectorStore,
    ):
        super().__init__()
        self.app_config = app_config
        self.session_manager = session_manager
        self.attachment_manager = attachment_manager
        self.conductor_manager = conductor_manager
        self.indexer = indexer
        self.extension_manager = extension_manager
        self.checkpoint_manager = checkpoint_manager
        self.vector_store = vector_store
        self.worker: GeminiWorker | None = None
        self.worker_thread: GeminiWorkerThread | None = None

    def stop_worker(self) -> None:
        """Safely stops any running worker thread."""
        if self.worker_thread and self.worker_thread.isRunning():
            self.worker_thread.stop()
            self.worker_thread.wait(3000)  # Wait up to 3 seconds
            if self.worker_thread.isRunning():
                self.worker_thread.terminate()

        self.worker = None
        self.worker_thread = None

    def send_message(
        self, prompt: str, system_instruction_override: str | None = None
    ) -> None:
        """Starts the Gemini worker to process the user request."""
        # Ensure previous worker is stopped
        self.stop_worker()

        attachments = self.attachment_manager.get_attachments()

        if not self.app_config.api_key:
            self.error_occurred.emit("Enter API Key in Settings.")
            return

        session_id = self.session_manager.current_session_id
        session = self.session_manager.get_session(session_id)

        if not session:
            return

        if not session.messages:
            new_title = prompt[:25] if prompt else "Analysis"
            self.session_manager.update_session_title(session_id, new_title)

        self.session_manager.add_message(
            session_id, Role.USER.value, prompt or "[Files]"
        )

        # Convert messages to dict for WorkerConfig compatibility
        history_context = [m.model_dump() for m in session.messages[:-1]]

        # Get session-specific config
        sess_config = session.config
        model = sess_config.get("model", self.app_config.model)
        temp = sess_config.get("temperature", self.app_config.get("temperature", 0.8))
        top_p = sess_config.get("top_p", self.app_config.get("top_p", 0.95))
        top_k = sess_config.get("top_k", self.app_config.get("top_k", 40))
        max_turns = sess_config.get("max_turns", self.app_config.get("max_turns", 20))
        thinking_enabled = sess_config.get(
            "thinking_enabled", self.app_config.get("thinking_enabled", False)
        )
        thinking_budget = sess_config.get(
            "thinking_budget", self.app_config.get("thinking_budget", 4096)
        )

        # Check for MAS flag in prompt or config
        use_mas = "/mas" in prompt or self.app_config.get("use_mas", False)
        if "/mas" in prompt:
            prompt = prompt.replace("/mas", "").strip()

        config = WorkerConfig(
            api_key=self.app_config.api_key,
            prompt=prompt,
            model=model,
            file_paths=list(attachments),
            history_context=history_context,
            use_grounding=self.app_config.get("use_search", False),
            system_instruction=system_instruction_override
            or self.app_config.get("system_instruction"),
            temperature=temp,
            top_p=top_p,
            top_k=top_k,
            max_turns=max_turns,
            thinking_enabled=thinking_enabled,
            thinking_budget=thinking_budget,
            session_id=session_id,
            initial_plan=session.plan,
            initial_specs=session.specs,
            extension_manager=self.extension_manager,
            use_mas=use_mas,
        )

        self.worker = GeminiWorker(config)
        self.worker.finished.connect(self._on_worker_finished)
        self.worker.error.connect(self.error_occurred.emit)
        self.worker.status_update.connect(self.status_updated.emit)
        self.worker.terminal_output.connect(self.terminal_output.emit)
        self.worker.request_confirmation.connect(self.tool_confirmation_requested.emit)
        self.worker.plan_updated.connect(
            lambda p: self.session_manager.update_session_plan(session_id, p)
        )
        self.worker.specs_updated.connect(
            lambda s: self.session_manager.update_session_specs(session_id, s)
        )
        self.worker.usage_updated.connect(self.usage_updated.emit)
        self.worker.rate_limit_updated.connect(self.rate_limit_updated.emit)

        try:
            self.worker_thread = GeminiWorkerThread(self.worker)
            # Use a member variable to keep the thread alive
            self.worker_thread.finished.connect(self._on_thread_finished)
            self.worker_thread.start()
        except Exception as e:
            self.error_occurred.emit(f"Failed to start worker: {str(e)}")

    def _on_thread_finished(self) -> None:
        """Cleanup when the thread finishes."""
        finished_thread = self.sender()
        if finished_thread:
            finished_thread.deleteLater()

            # Only clear the reference if it still points to the finished thread
            if self.worker_thread == finished_thread:
                self.worker_thread = None

    def _on_worker_finished(self, text: str) -> None:
        self.session_manager.add_message(
            self.session_manager.current_session_id, Role.MODEL.value, text
        )
        self.attachment_manager.clear_attachments()
        self.response_received.emit(text)

        # Async indexing to ChromaDB after response
        asyncio.create_task(self._index_response_to_chroma(text))

    async def _index_response_to_chroma(self, text: str):
        """Indexes the AI response into ChromaDB asynchronously."""
        session_id = self.session_manager.current_session_id
        session = self.session_manager.get_session(session_id)
        if not session:
            return
        doc_id = f"{session_id}_{len(session.messages)}"
        self.vector_store.add_documents(
            documents=[text],
            metadatas=[{"session_id": session_id, "role": "model"}],
            ids=[doc_id],
        )

    def confirm_tool(
        self,
        confirmation_id: str,
        allowed: bool,
        modified_args: dict[str, Any] | None = None,
    ) -> None:
        if self.worker:
            self.worker.confirm_tool(confirmation_id, allowed, modified_args)
