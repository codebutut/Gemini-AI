import argparse
import asyncio
import contextlib
import json
import logging
import multiprocessing
import os
import sys
import threading
from pathlib import Path
from typing import Any

from PyQt6.QtCore import Q_ARG, QMetaObject, QObject, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QDockWidget,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)
from qasync import QEventLoop, asyncSlot

from gemini_agent.config.app_config import AppConfig, ModelRegistry, Role, Theme, setup_logging
from gemini_agent.core.attachment_manager import AttachmentManager
from gemini_agent.core.checkpoint_manager import CheckpointManager
from gemini_agent.core.conductor_manager import ConductorManager
from gemini_agent.core.exporter import Exporter
from gemini_agent.core.extension_manager import ExtensionManager
from gemini_agent.core.indexer import Indexer
from gemini_agent.core.recent_manager import RecentManager
from gemini_agent.core.session_manager import SessionManager
from gemini_agent.core.tools import TOOL_REGISTRY
from gemini_agent.core.vector_store import VectorStore
from gemini_agent.core.worker import GeminiWorker, GeminiWorkerThread, WorkerConfig
from gemini_agent.ui.components import ChatHeader, SidebarContainer
from gemini_agent.ui.conductor_dialog import ConductorDialog
from gemini_agent.ui.deep_review import DeepReviewDialog
from gemini_agent.ui.plugin_dialog import PluginDialog
from gemini_agent.ui.project_explorer import ProjectExplorer
from gemini_agent.ui.settings_dialog import SettingsDialog
from gemini_agent.ui.status_widget import StatusWidget
from gemini_agent.ui.symbol_browser import SymbolBrowser
from gemini_agent.ui.terminal_widget import TerminalWidget
from gemini_agent.ui.theme_manager import ThemeManager
from gemini_agent.ui.widgets import AttachmentItem, AutoResizingTextEdit, MessageBubble

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

    def send_message(self, prompt: str, system_instruction_override: str | None = None) -> None:
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

        self.session_manager.add_message(session_id, Role.USER.value, prompt or "[Files]")

        # Convert messages to dict for WorkerConfig compatibility
        history_context = [m.model_dump() for m in session.messages[:-1]]

        # Get session-specific config
        sess_config = session.config
        model = sess_config.get("model", self.app_config.model)
        temp = sess_config.get("temperature", self.app_config.get("temperature", 0.8))
        top_p = sess_config.get("top_p", self.app_config.get("top_p", 0.95))
        top_k = sess_config.get("top_k", self.app_config.get("top_k", 40))
        max_turns = sess_config.get("max_turns", self.app_config.get("max_turns", 20))
        thinking_enabled = sess_config.get("thinking_enabled", self.app_config.get("thinking_enabled", False))
        thinking_budget = sess_config.get("thinking_budget", self.app_config.get("thinking_budget", 4096))
        
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
            system_instruction=system_instruction_override or self.app_config.get("system_instruction"),
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
            use_mas=use_mas
        )

        self.worker = GeminiWorker(config)
        self.worker.finished.connect(self._on_worker_finished)
        self.worker.error.connect(self.error_occurred.emit)
        self.worker.status_update.connect(self.status_updated.emit)
        self.worker.terminal_output.connect(self.terminal_output.emit)
        self.worker.request_confirmation.connect(self.tool_confirmation_requested.emit)
        self.worker.plan_updated.connect(lambda p: self.session_manager.update_session_plan(session_id, p))
        self.worker.specs_updated.connect(lambda s: self.session_manager.update_session_specs(session_id, s))
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
        self.session_manager.add_message(self.session_manager.current_session_id, Role.MODEL.value, text)
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
            documents=[text], metadatas=[{"session_id": session_id, "role": "model"}], ids=[doc_id]
        )

    def confirm_tool(self, confirmation_id: str, allowed: bool, modified_args: dict[str, Any] | None = None) -> None:
        if self.worker:
            self.worker.confirm_tool(confirmation_id, allowed, modified_args)


class GeminiBrowser(QMainWindow):
    """
    Main window for the Gemini AI Agent application.
    Handles UI layout and delegates logic to ChatController.
    """

    def __init__(
        self,
        app_config: AppConfig,
        theme_manager: ThemeManager,
        session_manager: SessionManager,
        attachment_manager: AttachmentManager,
        conductor_manager: ConductorManager,
        indexer: Indexer,
        extension_manager: ExtensionManager,
        checkpoint_manager: CheckpointManager,
        vector_store: VectorStore,
        recent_manager: RecentManager,
    ):
        super().__init__()
        self.app_config = app_config
        self.theme_manager = theme_manager
        self.session_manager = session_manager
        self.attachment_manager = attachment_manager
        self.conductor_manager = conductor_manager
        self.indexer = indexer
        self.extension_manager = extension_manager
        self.checkpoint_manager = checkpoint_manager
        self.vector_store = vector_store
        self.recent_manager = recent_manager

        self.controller = ChatController(
            app_config,
            session_manager,
            attachment_manager,
            conductor_manager,
            indexer,
            extension_manager,
            checkpoint_manager,
            vector_store,
        )

        self.status_widget: StatusWidget | None = None

        self.init_ui()
        self._connect_controller()
        self.update_sidebar()
        self.create_new_session()
        self.refresh_index()

        # Apply initial theme
        self.theme_manager.apply_theme(self.app_config.theme)

    def _connect_controller(self) -> None:
        self.controller.status_updated.connect(self.on_status_update)
        self.controller.response_received.connect(self.on_response_success)
        self.controller.error_occurred.connect(self.on_response_error)
        self.controller.usage_updated.connect(self.on_usage_updated)
        self.controller.rate_limit_updated.connect(self.on_rate_limit_updated)
        self.controller.terminal_output.connect(self.on_terminal_output)
        self.controller.tool_confirmation_requested.connect(self.show_tool_confirmation)

    def init_ui(self) -> None:
        """Initializes the main user interface components."""
        self.setWindowTitle("Gemini AI Agent - Professional Edition")
        self.resize(1200, 800)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        self._setup_sidebar()
        self._setup_chat_area()
        self._setup_terminal()

        # Splitter for Sidebar and Chat Area
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.addWidget(self.sidebar_container)
        self.splitter.addWidget(self.chat_area)
        self.splitter.setStretchFactor(1, 4)
        self.splitter.setHandleWidth(1)

        main_layout.addWidget(self.splitter)
        self.header.set_mode(self.app_config.get("use_search", False))

        self.setup_settings_menu()

    def _setup_sidebar(self) -> None:
        """Initializes the sidebar components."""
        self.sidebar_container = SidebarContainer()
        self.sidebar = self.sidebar_container.chat_sidebar
        self.sidebar.new_chat_requested.connect(self.create_new_session)
        self.sidebar.session_selected.connect(self.load_session_from_list)
        self.sidebar.context_menu_requested.connect(self.show_context_menu)
        self.sidebar.recent_item_selected.connect(self.add_attachment)

        # Project Explorer
        self.project_explorer = ProjectExplorer(root_path=".")
        self.project_explorer.file_attached.connect(self.add_attachment)
        self.project_explorer.folder_attached.connect(lambda p: self.add_attachment(p, "project"))
        self.sidebar_container.tabs.addTab(self.project_explorer, "📁 Project")

        # Symbol Browser
        self.symbol_browser = SymbolBrowser()
        self.symbol_browser.btn_refresh.clicked.connect(self.refresh_index)
        self.symbol_browser.symbol_selected.connect(self.on_symbol_selected)
        self.sidebar_container.tabs.addTab(self.symbol_browser, "🔍 Symbols")

    def _setup_chat_area(self) -> None:
        """Initializes the central chat area."""
        self.chat_area = QWidget()
        self.chat_area.setObjectName("ChatArea")
        chat_layout = QVBoxLayout(self.chat_area)
        chat_layout.setContentsMargins(0, 0, 0, 0)
        chat_layout.setSpacing(0)

        # Header
        self.header = ChatHeader()
        self.header.toggle_sidebar_requested.connect(self.toggle_sidebar)
        self.header.terminal_toggle_requested.connect(self.toggle_terminal)
        chat_layout.addWidget(self.header)

        # Splitter for Messages
        self.chat_splitter = QSplitter(Qt.Orientation.Vertical)
        self.chat_splitter.setHandleWidth(1)

        # Messages Scroll Area
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setObjectName("ScrollArea")

        self.messages_container = QWidget()
        self.messages_layout = QVBoxLayout(self.messages_container)
        self.messages_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.messages_layout.setSpacing(20)
        self.messages_layout.setContentsMargins(50, 20, 50, 20)

        self.scroll_area.setWidget(self.messages_container)
        self.chat_splitter.addWidget(self.scroll_area)

        chat_layout.addWidget(self.chat_splitter, 1)

        self._setup_input_area(chat_layout)

    def _setup_terminal(self) -> None:
        """Initializes the terminal dock."""
        self.terminal = TerminalWidget()
        self.terminal_dock = QDockWidget("Terminal", self)
        self.terminal_dock.setWidget(self.terminal)
        self.terminal_dock.setAllowedAreas(Qt.DockWidgetArea.BottomDockWidgetArea | Qt.DockWidgetArea.TopDockWidgetArea)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.terminal_dock)
        self.terminal_dock.hide()
        self.terminal_dock.visibilityChanged.connect(self.on_terminal_visibility_changed)

    def _setup_input_area(self, layout: QVBoxLayout) -> None:
        """Initializes the bottom input area."""
        bottom_container = QWidget()
        bottom_layout = QVBoxLayout(bottom_container)
        bottom_layout.setContentsMargins(20, 10, 20, 30)

        # Attachment List
        self.attachment_list_layout = QHBoxLayout()
        self.attachment_list_layout.setContentsMargins(10, 0, 10, 5)
        self.attachment_list_layout.setSpacing(5)
        self.attachment_list_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        bottom_layout.addLayout(self.attachment_list_layout)

        self.input_frame = QFrame()
        self.input_frame.setObjectName("InputFrame")
        pill_layout = QHBoxLayout(self.input_frame)
        pill_layout.setContentsMargins(10, 5, 10, 5)

        self.btn_file = QPushButton("+")
        self.btn_file.setFixedSize(36, 36)
        self.btn_file.clicked.connect(self.show_attach_menu)

        self.input_field = AutoResizingTextEdit(self)
        self.input_field.returnPressed.connect(self.send_message)

        # Populate completer with tools, commands, and files
        keywords = ["/clear", "/help", "/reset", "/conductor", "/search", "/mas"]
        keywords.extend(list(TOOL_REGISTRY.keys()))
        keywords.extend(self.conductor_manager.get_available_commands())
        with contextlib.suppress(OSError):
            keywords.extend([f for f in os.listdir(".") if not f.startswith(".")])
        self.input_field.update_keywords(keywords)

        btn_send = QPushButton("➤")
        btn_send.setFixedSize(36, 36)
        btn_send.clicked.connect(self.send_message)

        pill_layout.addWidget(self.btn_file)
        pill_layout.addWidget(self.input_field)
        pill_layout.addWidget(btn_send)

        bottom_layout.addWidget(self.input_frame)
        layout.addWidget(bottom_container, 0)

    def setup_settings_menu(self) -> None:
        """Sets up the unified settings menu on the header button."""
        self.settings_menu = QMenu(self)

        # 1. Settings Dialog
        self.settings_menu.addAction("⚙️ Application Settings").triggered.connect(self.open_settings)
        self.settings_menu.addSeparator()

        # 2. Conductor Orchestrator
        self.settings_menu.addAction("🚀 Conductor Orchestrator").triggered.connect(self.open_conductor)

        # 3. Conductor Submenu
        conductor_submenu = self.settings_menu.addMenu("📋 Conductor Commands")
        self._populate_conductor_menu(conductor_submenu)

        self.settings_menu.addSeparator()

        # 4. Plugins
        self.settings_menu.addAction("🔌 Manage Plugins").triggered.connect(self.open_plugins)

        self.settings_menu.addSeparator()

        # 5. History
        history_menu = self.settings_menu.addMenu("📜 History Management")
        history_menu.addAction("Backup History").triggered.connect(self.backup_history)
        history_menu.addAction("Restore History").triggered.connect(self.restore_history)

        self.settings_menu.addSeparator()
        self.settings_menu.addAction("🧹 Clear Vector Cache").triggered.connect(self.vector_store.delete_collection)

        # Attach menu to button
        self.header.btn_settings.setMenu(self.settings_menu)

    def _populate_conductor_menu(self, menu: QMenu) -> None:
        """Populates a QMenu with conductor commands."""
        menu.clear()
        commands = self.conductor_manager.get_available_commands()
        if not commands:
            action = menu.addAction("No commands found")
            action.setEnabled(False)
            return

        for cmd in commands:
            action = menu.addAction(cmd.capitalize())
            action.triggered.connect(lambda checked, c=cmd: self.run_conductor_command(c))

    def toggle_terminal(self) -> None:
        """Toggles the visibility of the terminal dock."""
        if self.terminal_dock.isVisible():
            self.terminal_dock.hide()
        else:
            self.terminal_dock.show()

    def on_terminal_visibility_changed(self, visible: bool) -> None:
        """Updates the terminal button state when dock visibility changes."""
        self.header.btn_terminal.setChecked(visible)

    def refresh_index(self) -> None:
        """Refreshes the project index for symbol browsing in a background thread."""

        def _bg_index():
            self.indexer.index_project()
            # Use QMetaObject to safely update UI from background thread
            QMetaObject.invokeMethod(
                self.symbol_browser,
                "set_symbols",
                Qt.ConnectionType.QueuedConnection,
                Q_ARG(list, self.indexer.get_all_symbols()),
            )

        threading.Thread(target=_bg_index, daemon=True).start()

    def on_symbol_selected(self, symbol: Any) -> None:
        """Handles symbol selection from the symbol browser."""
        full_path = os.path.join(self.indexer.root_dir, symbol.file_path)
        self.add_attachment(full_path)
        QMessageBox.information(
            self,
            "Symbol Selected",
            f"Attached {symbol.file_path}\nSymbol: {symbol.name} (line {symbol.line})",
        )

    def run_conductor_command(self, command_name: str) -> None:
        """Executes a specific conductor command."""
        prompt = f"Execute Conductor command: {command_name}"
        system_instruction = self.conductor_manager.get_command_prompt(command_name)

        if not system_instruction:
            QMessageBox.warning(self, "Error", f"Command {command_name} not found.")
            return

        self.create_new_session()
        self.session_manager.update_session_title(self.session_manager.current_session_id, f"Conductor: {command_name}")
        self.update_sidebar()

        self._start_worker(prompt, system_instruction_override=system_instruction)

    def toggle_sidebar(self) -> None:
        """Toggles the visibility of the sidebar."""
        is_visible = self.sidebar_container.isVisible()
        self.sidebar_container.setVisible(not is_visible)
        self.header.btn_toggle_sidebar.setText("≡" if is_visible else "✕")

    def show_attach_menu(self) -> None:
        """Shows the attachment menu."""
        menu = QMenu(self)
        menu.addAction("Attach Files").triggered.connect(self.attach_files_dialog)
        menu.addAction("Attach Folder").triggered.connect(self.attach_folder_dialog)
        menu.exec(self.btn_file.mapToGlobal(self.btn_file.rect().topLeft()))

    def attach_files_dialog(self) -> None:
        """Opens a dialog to select files for attachment."""
        files, _ = QFileDialog.getOpenFileNames(self, "Select Files", "", "All Files (*)")
        if files:
            for f in files:
                self.add_attachment(f)

    def attach_folder_dialog(self) -> None:
        """Opens a dialog to select a folder for attachment."""
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            self.add_attachment(folder, "project")

    def add_attachment(self, path: str, item_type: str = "file") -> None:
        """Adds a file or folder to the current session's attachments."""
        self.attachment_manager.add_attachment(path)
        self.recent_manager.add_item(path, item_type)
        self._update_attachment_ui()
        self.update_sidebar()

    def remove_attachment(self, path: str) -> None:
        """Removes an attachment from the current session."""
        if path in self.attachment_manager.attachments:
            self.attachment_manager.attachments.remove(path)
            self._update_attachment_ui()

    def _update_attachment_ui(self) -> None:
        """Updates the attachment list UI."""
        # Clear current list
        while self.attachment_list_layout.count():
            child = self.attachment_list_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        # Add items
        attachments = self.attachment_manager.get_attachments()
        # Limit display to first 5 to avoid clutter
        for path in attachments[:5]:
            item = AttachmentItem(path, self.app_config.theme)
            item.remove_requested.connect(self.remove_attachment)
            self.attachment_list_layout.addWidget(item)

        if len(attachments) > 5:
            more_lbl = QLabel(f"+{len(attachments) - 5} more")
            more_lbl.setStyleSheet("color: #888; font-size: 11px;")
            self.attachment_list_layout.addWidget(more_lbl)

    def apply_theme(self) -> None:
        """Applies the current theme to the entire application."""
        theme = self.app_config.theme
        self.theme_manager.apply_theme(theme)

        # Custom styling for specific buttons that don't follow palette perfectly
        btn_style = """
            QPushButton { 
                background-color: transparent; 
                border: 1px solid %s; 
                border-radius: 8px; 
                color: %s; 
                font-size: 20px;
            }
            QPushButton:hover { background-color: %s; }
        """
        if theme == Theme.DARK.value:
            self.header.btn_toggle_sidebar.setStyleSheet(btn_style % ("#555", "#888", "#333"))
            self.header.btn_settings.setStyleSheet(
                "QPushButton { background-color: transparent; border: 1px solid #555; border-radius: 4px; color: #888; } QPushButton:hover { background-color: #333; color: #EEE; }"
            )
        else:
            self.header.btn_toggle_sidebar.setStyleSheet(btn_style % ("#CCC", "#555", "#EEE"))
            self.header.btn_settings.setStyleSheet(
                "QPushButton { background-color: transparent; border: 1px solid #CCC; border-radius: 4px; color: #555; } QPushButton:hover { background-color: #F0F0F0; color: #000; }"
            )

        self.project_explorer.apply_theme(theme)
        self._update_attachment_ui()

        if self.session_manager.current_session_id:
            self.load_session_from_list(None)  # Reload current session UI

    def create_new_session(self) -> None:
        """Creates a new chat session."""
        # Use global default model for new sessions
        config = {"model": self.app_config.model}
        self.session_manager.create_session(config=config)
        self.clear_chat_ui()
        self.attachment_manager.clear_attachments()
        self._update_attachment_ui()
        self.update_sidebar()

        self._refresh_usage_display()

    def update_sidebar(self) -> None:
        """Updates the sidebar with the latest session list."""
        self.sidebar.populate_sessions(self.session_manager.get_all_sessions(), self.session_manager.current_session_id)
        self.sidebar.update_recent_items(self.recent_manager.get_recent_items())

    def _get_session_id_from_item(self, item: QListWidgetItem) -> str | None:
        """Extracts the session ID from a list item's user data."""
        data = item.data(Qt.ItemDataRole.UserRole)
        if isinstance(data, dict):
            return data.get("id")
        return data

    def load_session_from_list(self, item: QListWidgetItem | None) -> None:
        """Loads a session from the sidebar list."""
        if item:
            session_id = self._get_session_id_from_item(item)
        else:
            session_id = self.session_manager.current_session_id

        if not session_id:
            return

        self.session_manager.current_session_id = session_id

        self.clear_chat_ui()
        session = self.session_manager.get_session(session_id)
        if session:
            messages = session.messages
            # Limit to last 50 messages for performance
            display_messages = messages[-50:] if len(messages) > 50 else messages

            if len(messages) > 50:
                info_lbl = QLabel(f"Showing last 50 of {len(messages)} messages.")
                info_lbl.setStyleSheet("color: #888; font-style: italic; margin-left: 50px;")
                self.messages_layout.addWidget(info_lbl)

            for msg in display_messages:
                self.messages_layout.addWidget(
                    MessageBubble(
                        msg.text,
                        is_user=(msg.role == Role.USER.value),
                        theme_mode=self.app_config.theme,
                    )
                )

            self._refresh_usage_display()

        self.scroll_to_bottom()

    def clear_chat_ui(self) -> None:
        """Clears all message bubbles from the chat area."""
        while self.messages_layout.count():
            child = self.messages_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        self.status_widget = None

    def show_context_menu(self, pos: Qt.AlignmentFlag) -> None:
        """Shows the context menu for a session in the sidebar."""
        item = self.sidebar.chat_list.itemAt(pos)
        if not item:
            return
        data = item.data(Qt.ItemDataRole.UserRole)
        if not data or (isinstance(data, dict) and data.get("type") != "chat"):
            return

        menu = QMenu()
        menu.addAction("Rename").triggered.connect(lambda: self.rename_session(item))
        menu.addAction("Export to Markdown").triggered.connect(lambda: self.export_session_to_markdown(item))
        menu.addSeparator()
        menu.addAction("Delete").triggered.connect(lambda: self.delete_session(item))
        menu.exec(self.sidebar.chat_list.mapToGlobal(pos))

    def delete_session(self, item: QListWidgetItem) -> None:
        """Deletes a session."""
        sess_id = self._get_session_id_from_item(item)
        if sess_id and self.session_manager.delete_session(sess_id):
            self.update_sidebar()
            if not self.session_manager.current_session_id:
                self.create_new_session()

    def rename_session(self, item: QListWidgetItem) -> None:
        """Renames a session."""
        sess_id = self._get_session_id_from_item(item)
        if not sess_id:
            return
            
        session = self.session_manager.get_session(sess_id)
        current_name = session.title if session else item.text().strip()
        new_name, ok = QInputDialog.getText(self, "Rename", "Chat Title:", text=current_name)
        if ok and new_name:
            self.session_manager.update_session_title(sess_id, new_name)
            self.update_sidebar()

    def export_session_to_markdown(self, item: QListWidgetItem) -> None:
        """Exports a session to a Markdown file."""
        sess_id = self._get_session_id_from_item(item)
        if not sess_id:
            return
            
        session = self.session_manager.get_session(sess_id)
        if not session:
            return

        safe_title = "".join(
            [c for c in session.title if c.isalnum() or c in (" ", "_")]
        ).rstrip()
        default_name = f"{safe_title}.md"
        file_path, _ = QFileDialog.getSaveFileName(self, "Export Session", default_name, "Markdown Files (*.md)")

        if file_path:
            if Exporter.export_to_file(session, Path(file_path)):
                QMessageBox.information(self, "Export Successful", f"Session exported to {file_path}")
            else:
                QMessageBox.critical(self, "Export Failed", "Failed to export session.")

    def backup_history(self) -> None:
        """Creates a backup of all chat history."""
        file_path, _ = QFileDialog.getSaveFileName(self, "Backup History", "conductor_backup.zip", "ZIP Files (*.zip)")
        if file_path:
            if Exporter.create_backup(self.session_manager.get_all_sessions(), Path(file_path)):
                QMessageBox.information(self, "Backup Successful", f"History backed up to {file_path}")
            else:
                QMessageBox.critical(self, "Backup Failed", "Failed to create backup.")

    def restore_history(self) -> None:
        """Restores chat history from a backup file."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Restore History", "", "ZIP Files (*.zip)")
        if file_path:
            sessions = Exporter.restore_backup(Path(file_path))
            if sessions:
                reply = QMessageBox.question(
                    self,
                    "Restore History",
                    "This will replace your current history. Continue?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                )
                if reply == QMessageBox.StandardButton.Yes:
                    self.session_manager.sessions = sessions
                    self.session_manager.save_history()
                    self.update_sidebar()
                    self.create_new_session()
                    QMessageBox.information(self, "Restore Successful", "History restored successfully.")
            else:
                QMessageBox.critical(self, "Restore Failed", "Failed to restore backup or backup is empty.")

    def open_settings(self) -> None:
        """Opens the settings dialog."""
        settings_dialog = SettingsDialog(self, self.app_config._config)
        settings_dialog.exec()
        self.app_config.save()
        self.header.set_mode(self.app_config.get("use_search", False))
        self.apply_theme()

    def open_conductor(self) -> None:
        """Opens the conductor orchestrator dialog."""
        dialog = ConductorDialog(self, self.conductor_manager)
        dialog.exec()

    def open_plugins(self) -> None:
        """Opens the plugin management dialog."""
        dialog = PluginDialog(self.extension_manager, self, self.app_config.theme)
        dialog.exec()

    @asyncSlot()
    async def send_message(self) -> None:
        """Sends the user message to the AI model."""
        prompt = self.input_field.toPlainText().strip()
        if not prompt and not self.attachment_manager.get_attachments():
            return

        # Handle special commands
        if prompt.startswith("/search "):
            query = prompt[8:].strip()
            await self._perform_semantic_search(query)
            self.input_field.clear()
            return

        self._start_worker(prompt)
        self.input_field.clear()

        # Index user prompt
        session_id = self.session_manager.current_session_id
        session = self.session_manager.get_session(session_id)
        if session:
            doc_id = f"{session_id}_{len(session.messages)}_user"
            self.vector_store.add_documents(
                documents=[prompt], metadatas=[{"session_id": session_id, "role": "user"}], ids=[doc_id]
            )

    async def _perform_semantic_search(self, query: str):
        """Performs a semantic search and displays results in the chat."""
        self.messages_layout.addWidget(
            MessageBubble(f"🔍 Searching for: {query}", is_user=True, theme_mode=self.app_config.theme)
        )

        results = self.vector_store.query(query)
        docs = results.get("documents", [[]])[0]

        if not docs:
            response = "No relevant information found in vector cache."
        else:
            response = "**Semantic Search Results:**\n\n" + "\n\n---\n\n".join(docs)

        self.messages_layout.addWidget(MessageBubble(response, is_user=False, theme_mode=self.app_config.theme))
        self.scroll_to_bottom()

    def _start_worker(self, prompt: str, system_instruction_override: str | None = None) -> None:
        """Starts the Gemini worker to process the user request."""
        attachments = self.attachment_manager.get_attachments()

        display_text = prompt + (f" [Files: {len(attachments)}]" if attachments else "")
        self.messages_layout.addWidget(MessageBubble(display_text, is_user=True, theme_mode=self.app_config.theme))
        self.scroll_to_bottom()

        self.status_widget = StatusWidget()
        self.status_widget.set_status("Gemini is thinking...")
        self.status_widget.start_loading()
        self.messages_layout.addWidget(self.status_widget)

        self.controller.send_message(prompt, system_instruction_override)

    def on_terminal_output(self, text: str, output_type: str) -> None:
        """Handles terminal output from the worker."""
        if output_type == "error":
            self.terminal.append_error(text)
        elif output_type == "success":
            self.terminal.append_success(text)
        elif output_type == "info":
            self.terminal.append_info(text)
        else:
            self.terminal.append_text(text)

    def show_tool_confirmation(self, tool_name: str, args: dict, confirmation_id: str) -> None:
        """Shows a confirmation dialog for dangerous tool execution."""
        dialog = DeepReviewDialog(tool_name, args, parent=self, theme_mode=self.app_config.theme)
        result = dialog.exec()
        allowed = result == QDialog.DialogCode.Accepted
        # Pass modified args if approved
        modified_args = dialog.get_args() if allowed else None
        self.controller.confirm_tool(confirmation_id, allowed, modified_args)

    def on_response_success(self, text: str) -> None:
        """Handles successful AI response."""
        if self.status_widget:
            self.status_widget.stop_loading()
            self.status_widget.deleteLater()
            self.status_widget = None

        self.messages_layout.addWidget(MessageBubble(text, is_user=False, theme_mode=self.app_config.theme))
        self.scroll_to_bottom()

        self._update_attachment_ui()

    def on_response_error(self, err: str) -> None:
        """Handles AI response error."""
        if self.status_widget:
            self.status_widget.stop_loading()
            self.status_widget.deleteLater()
            self.status_widget = None

        self.messages_layout.addWidget(
            MessageBubble(f"**Error:** {err}", is_user=False, theme_mode=self.app_config.theme)
        )
        self.scroll_to_bottom()

    def on_status_update(self, status_message: str) -> None:
        """Updates the status widget with the latest worker status."""
        if self.status_widget:
            try:
                self.status_widget.set_status(status_message)
            except RuntimeError:
                # Handle case where C++ object is deleted but Python reference remains
                self.status_widget = None

    def on_usage_updated(self, session_id: str, input_tokens: int, output_tokens: int) -> None:
        """Updates the session usage data."""
        self.session_manager.update_session_usage(session_id, input_tokens, output_tokens)
        if session_id == self.session_manager.current_session_id:
            self._refresh_usage_display()

    def on_rate_limit_updated(self, model_id: str, remaining: int, limit: int) -> None:
        """Updates the rate limit indicator in the status widget."""
        if self.status_widget:
            try:
                self.status_widget.update_rate_limit(remaining, limit)
            except RuntimeError:
                self.status_widget = None

    def _refresh_usage_display(self) -> None:
        """Refreshes the usage display in the header."""
        session_id = self.session_manager.current_session_id
        if not session_id:
            return

        session = self.session_manager.get_session(session_id)
        if not session:
            return

        usage = session.usage

        # Calculate cost
        sess_config = session.config
        model_id = sess_config.get("model", self.app_config.model)

        pricing = ModelRegistry.MODEL_PRICING.get(model_id, (0.0, 0.0))
        input_cost = (usage.input_tokens / 1_000_000) * pricing[0]
        output_cost = (usage.output_tokens / 1_000_000) * pricing[1]
        total_cost = input_cost + output_cost

        self.header.update_usage(usage.total_tokens, total_cost)

    def scroll_to_bottom(self) -> None:
        """Scrolls the chat area to the bottom."""
        QApplication.processEvents()
        self.scroll_area.verticalScrollBar().setValue(self.scroll_area.verticalScrollBar().maximum())

    def closeEvent(self, event: Any) -> None:
        """Handles the window close event."""
        self.controller.stop_worker()
        self.attachment_manager.cleanup()
        event.accept()


def main() -> None:
    """Main entry point for the Gemini CLI."""
    if sys.platform != "win32":
        try:
            multiprocessing.set_start_method("spawn", force=True)
        except RuntimeError:
            pass

    parser = argparse.ArgumentParser(description="Gemini Agent CLI")
    subparsers = parser.add_subparsers(dest="command")

    # Extension CLI
    extension_parser = subparsers.add_parser("extension", help="Manage extensions (plugins and MCP servers)")
    ext_subparsers = extension_parser.add_subparsers(dest="ext_command")

    # List
    ext_subparsers.add_parser("list", help="List all extensions")

    # Install Plugin
    install_plugin_parser = ext_subparsers.add_parser("install-plugin", help="Install a plugin from PyPI")
    install_plugin_parser.add_argument("package_name", help="The name of the plugin package")

    # Uninstall Plugin
    uninstall_plugin_parser = ext_subparsers.add_parser("uninstall-plugin", help="Uninstall a plugin")
    uninstall_plugin_parser.add_argument("plugin_name", help="The name of the plugin")

    # Add MCP
    add_mcp_parser = ext_subparsers.add_parser("add-mcp", help="Add an MCP server")
    add_mcp_parser.add_argument("name", help="Name of the MCP server")
    add_mcp_parser.add_argument("command", help="Command to run")
    add_mcp_parser.add_argument("--args", nargs="*", help="Arguments for the command")

    # Remove MCP
    remove_mcp_parser = ext_subparsers.add_parser("remove-mcp", help="Remove an MCP server")
    remove_mcp_parser.add_argument("name", help="Name of the MCP server")

    args = parser.parse_args()

    extension_mgr = ExtensionManager()
    extension_mgr.discover_plugins()

    if args.command == "extension":
        if args.ext_command == "list":
            print(json.dumps(extension_mgr.list_extensions(), indent=2))
        elif args.ext_command == "install-plugin":
            print(extension_mgr.install_plugin(args.package_name))
        elif args.ext_command == "uninstall-plugin":
            print(extension_mgr.uninstall_plugin(args.plugin_name))
        elif args.ext_command == "add-mcp":
            print(extension_mgr.add_mcp_server(args.name, args.command, args.args or []))
        elif args.ext_command == "remove-mcp":
            print(extension_mgr.remove_mcp_server(args.name))
        sys.exit(0)

    app = QApplication(sys.argv)

    # Initialize qasync event loop
    loop = QEventLoop(app)
    asyncio.set_event_loop(loop)

    # Initialize services (Dependency Injection)
    config = AppConfig()
    theme_mgr = ThemeManager(app)
    session_mgr = SessionManager(AppConfig.HISTORY_FILE)
    attachment_mgr = AttachmentManager()
    conductor_mgr = ConductorManager(extension_path=config.conductor_path)
    indexer = Indexer(root_dir=".")
    checkpoint_mgr = CheckpointManager()
    vector_store = VectorStore()
    recent_mgr = RecentManager()

    # Inject into main window
    window = GeminiBrowser(
        config,
        theme_mgr,
        session_mgr,
        attachment_mgr,
        conductor_mgr,
        indexer,
        extension_mgr,
        checkpoint_mgr,
        vector_store,
        recent_mgr,
    )
    window.show()

    with loop:
        loop.run_forever()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
