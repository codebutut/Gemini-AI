from PyQt6.QtCore import Qt, pyqtSignal, QSize
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
    QWidget,
    QSizePolicy,
)


class RecentItemWidget(QWidget):
    """
    Custom widget for a single recent item in the list.
    Supports word wrap for long titles and ensures full visibility.
    Features a grey bubble background for the text.
    """
    def __init__(self, name: str, path: str, item_type: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.layout = QHBoxLayout(self)
        # Minimal margins for the outer widget
        self.layout.setContentsMargins(2, 2, 2, 2)
        self.layout.setSpacing(6)
        self.layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        if item_type == "file":
            icon = "📄"
        elif item_type == "folder" or item_type == "project":
            icon = "📁"
        elif item_type == "chat":
            icon = "💬"
        else:
            icon = "📄"

        self.icon_label = QLabel(icon)
        self.icon_label.setFixedWidth(24)
        self.icon_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)
        self.icon_label.setStyleSheet("font-size: 14px; margin-top: 6px;")
        
        # Bubble container for the text
        self.bubble_frame = QFrame()
        self.bubble_frame.setObjectName("RecentBubble")
        self.bubble_layout = QVBoxLayout(self.bubble_frame)
        self.bubble_layout.setContentsMargins(10, 8, 10, 8)
        self.bubble_layout.setSpacing(0)
        
        self.name_label = QLabel(name)
        self.name_label.setStyleSheet("""
            font-weight: 500; 
            font-size: 12px; 
            color: #E0E0E0; 
            background: transparent; 
            border: none;
        """)
        self.name_label.setWordWrap(True)
        # Allow the label to expand vertically as needed
        self.name_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        
        self.bubble_layout.addWidget(self.name_label)
        
        # Grey background for the bubble - using a slightly lighter grey than the sidebar
        self.bubble_frame.setStyleSheet("""
            QFrame#RecentBubble {
                background-color: #333333;
                border-radius: 8px;
                border: 1px solid #444444;
            }
        """)
        
        self.layout.addWidget(self.icon_label)
        self.layout.addWidget(self.bubble_frame, 1)
        
        if path:
            self.setToolTip(path)

    def sizeHint(self) -> QSize:
        """
        Returns an accurate size hint by ensuring the layout is activated.
        """
        self.layout.activate()
        hint = super().sizeHint()
        # Ensure a minimum height for the bubble
        hint.setHeight(max(hint.height(), 42))
        return hint


class RecentWidget(QFrame):
    """
    Widget displaying a list of recent items, including files, folders, and chats.
    """
    item_selected = pyqtSignal(str, str)  # path/id, type

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("RecentWidget")
        self.init_ui()

    def init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)

        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(10, 5, 10, 5)
        self.title_label = QLabel("Recent")
        self.title_label.setStyleSheet("font-weight: bold; color: #888; font-size: 12px;")
        header_layout.addWidget(self.title_label)
        header_layout.addStretch()
        layout.addLayout(header_layout)

        self.list_widget = QListWidget()
        self.list_widget.setObjectName("RecentList")
        self.list_widget.setWordWrap(True)
        self.list_widget.setVerticalScrollMode(QListWidget.ScrollMode.ScrollPerPixel)
        self.list_widget.setStyleSheet("""
            QListWidget#RecentList {
                background-color: transparent;
                border: none;
                outline: none;
            }
            QListWidget#RecentList::item {
                padding: 2px;
                margin-bottom: 2px;
                background-color: transparent;
            }
            QListWidget#RecentList::item:selected {
                background-color: transparent;
                color: white;
            }
        """)
        self.list_widget.itemClicked.connect(self._on_item_clicked)
        layout.addWidget(self.list_widget)

    def update_items(self, items: list[dict]):
        """
        Updates the list with a mix of items.
        Each item should be a dict with: 'name', 'path' (or 'id'), 'type'.
        """
        self.list_widget.clear()
        
        # SidebarContainer width is 280. 
        # Subtracting Sidebar margins and ListWidget padding/scrollbars.
        # We use a slightly smaller width to ensure word wrap triggers correctly.
        available_width = 230 

        for item in items:
            list_item = QListWidgetItem(self.list_widget)
            list_item.setData(Qt.ItemDataRole.UserRole, item)
            
            path = item.get("path") or item.get("id")
            custom_widget = RecentItemWidget(item["name"], path, item["type"])
            
            # Force the widget to a specific width so sizeHint() can calculate 
            # the correct height based on word wrapping.
            custom_widget.setFixedWidth(available_width)
            
            # Set the size hint for the list item based on the calculated widget size
            list_item.setSizeHint(custom_widget.sizeHint())
            
            self.list_widget.addItem(list_item)
            self.list_widget.setItemWidget(list_item, custom_widget)

    def _on_item_clicked(self, item: QListWidgetItem):
        data = item.data(Qt.ItemDataRole.UserRole)
        if data:
            path = data.get("path") or data.get("id")
            self.item_selected.emit(path, data["type"])
