import asyncio
import logging
import os
from typing import Any, List, Dict, Optional

from PyQt6.QtCore import (
    Qt,
    pyqtSignal,
    QSize,
    QPoint,
    QTimer,
    QPropertyAnimation,
    QEasingCurve,
)
from PyQt6.QtGui import QIcon, QKeyEvent
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QStackedWidget,
    QApplication,
    QScrollArea,
    QSpacerItem,
    QSizePolicy,
    QLabel,
    QFileDialog,
    QGraphicsOpacityEffect,
)
from qfluentwidgets import (
    FluentWindow,
    NavigationItemPosition,
    FluentIcon as FIF,
    setTheme,
    Theme,
    SearchLineEdit,
    PrimaryPushButton,
    TransparentToolButton,
    SubtitleLabel,
    BodyLabel,
    ScrollArea,
    Action,
    RoundMenu,
    InfoBar,
    InfoBarPosition,
)
from qasync import asyncSlot

from gemini_agent.config.app_config import AppConfig, Role, ModelRegistry
from gemini_agent.ui.fluent_components import (
    CommandPalette,
    ModernMessageBubble,
    ThoughtBlock,
    GraphView,
)
from gemini_agent.ui.widgets import AutoResizingTextEdit, MessageBubble
from gemini_agent.core.session_manager import SessionManager
from gemini_agent.core.attachment_manager import AttachmentManager
from gemini_agent.core.conductor_manager import ConductorManager
from gemini_agent.core.indexer import Indexer
from gemini_agent.core.extension_manager import ExtensionManager
from gemini_agent.core.checkpoint_manager import CheckpointManager
from gemini_agent.core.vector_store import VectorStore
from gemini_agent.core.recent_manager import RecentManager
from gemini_agent.ui.theme_manager import ThemeManager
from gemini_agent.core.controller import ChatController

logger = logging.getLogger(__name__)


class ChatInterface(QWidget):
    """
    The main chat interface using modern components.
    """

    send_message_requested = pyqtSignal(str)
    attach_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("chatInterface")
        self.init_ui()

    def init_ui(self):
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        # Header
        self.header = QWidget()
        self.header_layout = QHBoxLayout(self.header)
        self.header_layout.setContentsMargins(20, 10, 20, 10)

        self.title_label = SubtitleLabel("New Chat")
        self.header_layout.addWidget(self.title_label)
        self.header_layout.addStretch()

        self.usage_label = BodyLabel("Usage: 0 tokens")
        self.usage_label.setStyleSheet("color: gray;")
        self.header_layout.addWidget(self.usage_label)

        self.main_layout.addWidget(self.header)

        # Chat Area
        self.scroll_area = ScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setObjectName("ChatScrollArea")
        self.scroll_area.setStyleSheet(
            "QScrollArea { border: none; background: transparent; }"
        )

        self.messages_container = QWidget()
        self.messages_layout = QVBoxLayout(self.messages_container)
        self.messages_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        # Reduced spacing from 20 to 12 and margins from (50, 20, 50, 20) to (40, 10, 40, 10)
        self.messages_layout.setSpacing(12)
        self.messages_layout.setContentsMargins(40, 10, 40, 10)

        self.scroll_area.setWidget(self.messages_container)
        self.main_layout.addWidget(self.scroll_area, 1)

        # Input Area
        self.input_container = QWidget()
        self.input_layout = QVBoxLayout(self.input_container)
        self.input_layout.setContentsMargins(50, 10, 50, 30)

        self.input_frame = QWidget()
        self.input_frame.setObjectName("InputFrame")
        self.input_frame.setStyleSheet("""
            #InputFrame {
                background: rgba(255, 255, 255, 0.05); 
                border-radius: 15px; 
                border: 1px solid rgba(255, 255, 255, 0.1);
            }
        """)
        self.pill_layout = QHBoxLayout(self.input_frame)
        self.pill_layout.setContentsMargins(15, 5, 15, 5)

        self.btn_attach = TransparentToolButton(FIF.ADD, self.input_frame)
        self.btn_attach.clicked.connect(self.attach_requested.emit)

        self.input_field = AutoResizingTextEdit(self.input_frame)
        self.input_field.setPlaceholderText("Message Gemini...")
        self.input_field.setStyleSheet(
            "background: transparent; border: none; color: white; font-size: 14px;"
        )
        self.input_field.returnPressed.connect(self._on_send_clicked)

        self.btn_send = TransparentToolButton(FIF.SEND, self.input_frame)
        self.btn_send.clicked.connect(self._on_send_clicked)

        self.pill_layout.addWidget(self.btn_attach)
        self.pill_layout.addWidget(self.input_field, 1)
        self.pill_layout.addWidget(self.btn_send)

        self.input_layout.addWidget(self.input_frame)
        self.main_layout.addWidget(self.input_container)

    def _on_send_clicked(self):
        text = self.input_field.toPlainText().strip()
        if text:
            self.send_message_requested.emit(text)
            self.input_field.clear()

    def add_message(self, text: str, is_user: bool = False, theme_mode: str = "Dark"):
        bubble = MessageBubble(text, is_user, theme_mode)
        self.messages_layout.addWidget(bubble)
        self._animate_widget(bubble)
        self.scroll_to_bottom()

    def add_thought(self, text: str):
        thought = ThoughtBlock("Gemini's Reasoning", self)
        thought.set_text(text)
        self.messages_layout.addWidget(thought)
        self._animate_widget(thought)
        self.scroll_to_bottom()

    def _animate_widget(self, widget: QWidget):
        opacity_effect = QGraphicsOpacityEffect(widget)
        widget.setGraphicsEffect(opacity_effect)

        self.animation = QPropertyAnimation(opacity_effect, b"opacity")
        self.animation.setDuration(500)
        self.animation.setStartValue(0)
        self.animation.setEndValue(1)
        self.animation.setEasingCurve(QEasingCurve.Type.InOutQuad)
        self.animation.start()

    def clear_chat(self):
        while self.messages_layout.count():
            child = self.messages_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

    def scroll_to_bottom(self):
        QTimer.singleShot(
            50,
            lambda: self.scroll_area.verticalScrollBar().setValue(
                self.scroll_area.verticalScrollBar().maximum()
            ),
        )


class ModernGeminiBrowser(FluentWindow):
    """
    Modernized main window using Fluent Design.
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

        self.init_window()
        self.init_navigation()
        self.init_command_palette()
        self._connect_controller()

        # Apply theme
        setTheme(Theme.DARK if self.app_config.theme == "Dark" else Theme.LIGHT)

        # Load initial session
        self.create_new_session()

    def init_window(self):
        self.setWindowTitle("Gemini AI Agent")
        self.resize(1200, 850)
        self.setMinimumSize(800, 600)

    def init_navigation(self):
        self.chat_interface = ChatInterface(self)
        self.chat_interface.send_message_requested.connect(self.send_message)
        self.chat_interface.attach_requested.connect(self.attach_files)

        self.addSubInterface(self.chat_interface, FIF.CHAT, "Chat")

        # Graph View Interface
        self.graph_view = GraphView(self)
        self.graph_view.setObjectName("graphView")
        self.addSubInterface(self.graph_view, FIF.TILES, "Knowledge Graph")

        # Settings Interface (Placeholder)
        self.settings_interface = QWidget()
        self.settings_interface.setObjectName("settingsInterface")
        self.addSubInterface(
            self.settings_interface,
            FIF.SETTING,
            "Settings",
            NavigationItemPosition.BOTTOM,
        )

    def init_command_palette(self):
        self.command_palette = CommandPalette(self)
        self.command_palette.set_commands(
            [
                {
                    "id": "new_chat",
                    "name": "New Chat",
                    "description": "Start a fresh conversation",
                    "icon": FIF.ADD,
                    "shortcut": "Ctrl+N",
                },
                {
                    "id": "settings",
                    "name": "Settings",
                    "description": "Open application settings",
                    "icon": FIF.SETTING,
                    "shortcut": "Ctrl+,",
                },
                {
                    "id": "clear_history",
                    "name": "Clear History",
                    "description": "Delete all chat sessions",
                    "icon": FIF.DELETE,
                },
                {
                    "id": "export_md",
                    "name": "Export to Markdown",
                    "description": "Save current chat as .md",
                    "icon": FIF.DOCUMENT,
                },
                {
                    "id": "toggle_theme",
                    "name": "Toggle Theme",
                    "description": "Switch between Light and Dark mode",
                    "icon": FIF.BRUSH,
                },
                {
                    "id": "show_graph",
                    "name": "Show Graph",
                    "description": "View project knowledge graph",
                    "icon": FIF.TILES,
                },
            ]
        )
        self.command_palette.command_selected.connect(self._on_command_selected)

    def _connect_controller(self) -> None:
        self.controller.response_received.connect(self.on_response_success)
        self.controller.error_occurred.connect(self.on_response_error)
        self.controller.usage_updated.connect(self.on_usage_updated)
        self.controller.status_updated.connect(self.on_status_update)

    def _on_command_selected(self, cmd_id: str, cmd_data: dict):
        if cmd_id == "new_chat":
            self.create_new_session()
        elif cmd_id == "toggle_theme":
            new_theme = Theme.LIGHT if self.app_config.theme == "Dark" else Theme.DARK
            self.app_config.theme = "Light" if new_theme == Theme.LIGHT else "Dark"
            setTheme(new_theme)
            InfoBar.success(
                "Theme Updated",
                f"Switched to {self.app_config.theme} mode",
                parent=self,
            )
        elif cmd_id == "show_graph":
            self.switchTo(self.graph_view)

    def keyPressEvent(self, event: QKeyEvent):
        if (
            event.modifiers() == Qt.KeyboardModifier.ControlModifier
            and event.key() == Qt.Key.Key_P
        ):
            self.command_palette.show_at_center()
        elif (
            event.modifiers() == Qt.KeyboardModifier.ControlModifier
            and event.key() == Qt.Key.Key_N
        ):
            self.create_new_session()
        else:
            super().keyPressEvent(event)

    def create_new_session(self):
        self.session_manager.create_session(config={"model": self.app_config.model})
        self.chat_interface.clear_chat()
        self.chat_interface.title_label.setText("New Chat")
        self.attachment_manager.clear_attachments()

    def send_message(self, prompt: str):
        self.chat_interface.add_message(
            prompt, is_user=True, theme_mode=self.app_config.theme
        )
        self.controller.send_message(prompt)

    def attach_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Files", "", "All Files (*)"
        )
        if files:
            for f in files:
                self.attachment_manager.add_attachment(f)
            InfoBar.info("Files Attached", f"Attached {len(files)} files", parent=self)

    def on_response_success(self, text: str):
        # Check if there's thinking content
        if "<thought>" in text and "</thought>" in text:
            parts = text.split("</thought>")
            thought = parts[0].replace("<thought>", "").strip()
            actual_response = parts[1].strip()
            self.chat_interface.add_thought(thought)
            self.chat_interface.add_message(
                actual_response, is_user=False, theme_mode=self.app_config.theme
            )
        else:
            self.chat_interface.add_message(
                text, is_user=False, theme_mode=self.app_config.theme
            )

    def on_response_error(self, err: str):
        InfoBar.error("Error", err, parent=self, position=InfoBarPosition.TOP)

    def on_status_update(self, status: str):
        pass

    def on_usage_updated(self, session_id: str, input_tokens: int, output_tokens: int):
        session = self.session_manager.get_session(session_id)
        if session:
            total = session.usage.total_tokens
            self.chat_interface.usage_label.setText(f"Usage: {total:,} tokens")
