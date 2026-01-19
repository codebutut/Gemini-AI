from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSizePolicy,
    QSpacerItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from gemini_agent.ui.recent_widget import RecentWidget
from gemini_agent.core.models import Session


class Sidebar(QFrame):
    """
    Sidebar widget displaying unified recent items (chats, files, folders).
    """

    new_chat_requested = pyqtSignal()
    session_selected = pyqtSignal(object)
    context_menu_requested = pyqtSignal(object)
    recent_item_selected = pyqtSignal(str, str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("Sidebar")
        self.sessions: Dict[str, Session] = {}
        self.recent_items: List[dict] = []
        self.current_session_id: Optional[str] = None
        self.init_ui()

    def init_ui(self) -> None:
        """Initializes the sidebar user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        self.btn_new_chat = QPushButton(" +  New chat")
        self.btn_new_chat.setObjectName("BtnNewChat")
        self.btn_new_chat.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_new_chat.clicked.connect(self.new_chat_requested.emit)
        layout.addWidget(self.btn_new_chat)

        layout.addSpacing(5)

        # Search Bar
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search chats and files...")
        self.search_input.textChanged.connect(self.filter_items)
        layout.addWidget(self.search_input)

        # Unified Recent Items Bubble
        self.recent_widget = RecentWidget()
        self.recent_widget.item_selected.connect(self._handle_item_selected)
        
        # Expose the list widget for context menu and other operations
        self.chat_list = self.recent_widget.list_widget
        self.chat_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.chat_list.customContextMenuRequested.connect(self.context_menu_requested.emit)
        
        layout.addWidget(self.recent_widget)

    def populate_sessions(self, sessions: Dict[str, Session], current_session_id: Optional[str] = None) -> None:
        """
        Populates the sidebar with session items.
        """
        self.sessions = sessions
        self.current_session_id = current_session_id
        self.filter_items(self.search_input.text())

    def update_recent_items(self, items: List[dict]) -> None:
        """Updates the list of recent files/folders."""
        self.recent_items = items
        self.filter_items(self.search_input.text())

    def filter_items(self, text: str) -> None:
        """
        Filters and displays sessions and recent items.
        """
        self.chat_list.clear()
        text = text.lower()

        combined_items = []

        # Add Chats
        for sess_id, session in self.sessions.items():
            if text and text not in session.title.lower():
                continue
            
            # Use last message timestamp or creation date
            last_activity = session.created_at
            if session.messages:
                last_activity = session.messages[-1].timestamp
                
            combined_items.append({
                "name": session.title,
                "id": sess_id,
                "type": "chat",
                "timestamp": last_activity
            })

        # Add Recent Files/Folders
        for item in self.recent_items:
            if text and text not in item["name"].lower():
                continue
            combined_items.append(item)

        # Sort everything by timestamp
        combined_items.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

        # Update the RecentWidget
        self.recent_widget.update_items(combined_items)
        
        # Highlight current session
        if self.current_session_id:
            for i in range(self.chat_list.count()):
                item = self.chat_list.item(i)
                data = item.data(Qt.ItemDataRole.UserRole)
                if data and data.get("type") == "chat" and data.get("id") == self.current_session_id:
                    self.chat_list.setCurrentItem(item)
                    break

    def _handle_item_selected(self, path_or_id: str, item_type: str):
        """Handles item selection from the RecentWidget."""
        if item_type == "chat":
            # Find the QListWidgetItem to emit session_selected
            for i in range(self.chat_list.count()):
                item = self.chat_list.item(i)
                data = item.data(Qt.ItemDataRole.UserRole)
                if data and data.get("type") == "chat" and data.get("id") == path_or_id:
                    self.session_selected.emit(item)
                    break
        else:
            self.recent_item_selected.emit(path_or_id, item_type)


class SidebarContainer(QFrame):
    """
    Container for the sidebar that uses tabs to switch between
    Chat History and Project Explorer.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedWidth(280)
        self.setObjectName("SidebarContainer")
        self.init_ui()

    def init_ui(self) -> None:
        """Initializes the container with tabbed navigation."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.tabs = QTabWidget()
        self.tabs.setObjectName("SidebarTabs")

        self.chat_sidebar = Sidebar()
        self.tabs.addTab(self.chat_sidebar, "💬 Chats")

        layout.addWidget(self.tabs)


class ChatHeader(QWidget):
    """
    Header widget for the chat area, containing session-level controls and status.
    """

    toggle_sidebar_requested = pyqtSignal()
    settings_requested = pyqtSignal()
    terminal_toggle_requested = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.init_ui()

    def init_ui(self) -> None:
        """Initializes the header UI components."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(15, 10, 20, 0)

        self.btn_toggle_sidebar = QPushButton("≡")
        self.btn_toggle_sidebar.setFixedSize(40, 40)
        self.btn_toggle_sidebar.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_toggle_sidebar.clicked.connect(self.toggle_sidebar_requested.emit)
        layout.addWidget(self.btn_toggle_sidebar)

        layout.addSpacing(10)
        title_lbl = QLabel("Gemini AI Agent")
        title_lbl.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(title_lbl)

        layout.addSpacing(15)

        self.mode_indicator = QLabel("🔧 Local Tools")
        self.mode_indicator.setStyleSheet(
            "color: #0B57D0; font-size: 12px; padding: 4px 8px; background-color: rgba(11, 87, 208, 0.1); border-radius: 4px;"
        )
        layout.addWidget(self.mode_indicator)

        # Usage Label - Increased font size to 13px
        self.usage_label = QLabel("Usage: 0 tokens ($0.00)")
        self.usage_label.setStyleSheet("color: #888; font-size: 13px; margin-left: 10px;")
        layout.addWidget(self.usage_label)

        layout.addSpacing(20)

        layout.addSpacerItem(QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum))

        # Terminal Toggle Button
        self.btn_terminal = QPushButton("💻 Terminal")
        self.btn_terminal.setObjectName("BtnTerminal")
        self.btn_terminal.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_terminal.setCheckable(True)
        self.btn_terminal.setFixedHeight(32)
        self.btn_terminal.clicked.connect(self.terminal_toggle_requested.emit)
        layout.addWidget(self.btn_terminal)

        layout.addSpacing(10)

        # Settings Button
        self.btn_settings = QPushButton("⚙️ Settings")
        self.btn_settings.setObjectName("BtnSettings")
        self.btn_settings.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_settings.setFixedHeight(32)
        self.btn_settings.setMinimumWidth(110)
        self.btn_settings.clicked.connect(self.settings_requested.emit)
        layout.addWidget(self.btn_settings)

    def set_mode(self, use_search: bool) -> None:
        """
        Updates the mode indicator (Web Search vs Local Tools).
        """
        if use_search:
            self.mode_indicator.setText("🔍 Web Search")
            self.mode_indicator.setStyleSheet(
                "color: #ff9800; font-size: 12px; padding: 4px 8px; background-color: rgba(255, 152, 0, 0.1); border-radius: 4px;"
            )
        else:
            self.mode_indicator.setText("🔧 Local Tools")
            self.mode_indicator.setStyleSheet(
                "color: #0B57D0; font-size: 12px; padding: 4px 8px; background-color: rgba(11, 87, 208, 0.1); border-radius: 4px;"
            )

    def update_usage(self, total_tokens: int, estimated_cost: float) -> None:
        """
        Updates the usage label with token count and cost.
        """
        self.usage_label.setText(f"Usage: {total_tokens:,} tokens (${estimated_cost:.4f})")
