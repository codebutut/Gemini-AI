import logging
from typing import List, Optional

from PyQt6.QtCore import (
    Qt,
    pyqtSignal,
    QRect,
    QPropertyAnimation,
    QEasingCurve,
    QPoint,
    QSize,
)
from PyQt6.QtGui import QColor, QPainter, QKeyEvent, QBrush, QPen
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QGraphicsDropShadowEffect,
    QApplication,
)
from qtawesome import icon
from qfluentwidgets import (
    SearchLineEdit,
    BodyLabel,
    CaptionLabel,
    CardWidget,
    FluentIcon as FIF,
    IconWidget,
    setTheme,
    Theme,
    ToolButton,
    TransparentToolButton,
)

logger = logging.getLogger(__name__)


class CommandPalette(QWidget):
    """
    A modern command palette overlay for quick actions and navigation.
    """

    command_selected = pyqtSignal(str, dict)
    closed = pyqtSignal()

    def __init__(self, parent: QWidget = None):
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.ToolTip)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        self.commands = []
        self.init_ui()
        self._setup_shadow()

        # Install event filter on parent to handle resizing and positioning
        if parent:
            parent.installEventFilter(self)

    def init_ui(self):
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(10, 10, 10, 10)

        self.container = CardWidget(self)
        self.container_layout = QVBoxLayout(self.container)
        self.container_layout.setContentsMargins(10, 10, 10, 10)
        self.container_layout.setSpacing(10)

        # Search Input
        self.search_input = SearchLineEdit(self.container)
        self.search_input.setPlaceholderText("Type a command or search...")
        self.search_input.textChanged.connect(self._on_text_changed)
        self.search_input.returnPressed.connect(self._on_return_pressed)
        self.container_layout.addWidget(self.search_input)

        # Results List
        self.list_widget = QListWidget(self.container)
        self.list_widget.setObjectName("CommandList")
        self.list_widget.setStyleSheet(
            "QListWidget { border: none; background: transparent; }"
        )
        self.list_widget.itemClicked.connect(self._on_item_clicked)
        self.container_layout.addWidget(self.list_widget)

        self.main_layout.addWidget(self.container)

        self.setFixedSize(600, 400)

    def _setup_shadow(self):
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(20)
        shadow.setXOffset(0)
        shadow.setYOffset(5)
        shadow.setColor(QColor(0, 0, 0, 80))
        self.container.setGraphicsEffect(shadow)

    def show_at_center(self):
        if self.parentWidget():
            parent_rect = self.parentWidget().geometry()
            # Get global position of parent
            global_pos = self.parentWidget().mapToGlobal(QPoint(0, 0))
            x = global_pos.x() + (parent_rect.width() - self.width()) // 2
            y = global_pos.y() + 100  # Show near top
            self.move(x, y)

        self.show()
        self.search_input.setFocus()
        self.search_input.clear()

    def set_commands(self, commands: List[dict]):
        self.commands = commands
        self._update_list(commands)

    def _update_list(self, commands: List[dict]):
        self.list_widget.clear()
        for cmd in commands:
            item = QListWidgetItem(self.list_widget)
            item.setData(Qt.ItemDataRole.UserRole, cmd)

            widget = QWidget()
            layout = QHBoxLayout(widget)
            layout.setContentsMargins(10, 5, 10, 5)

            icon_label = IconWidget(cmd.get("icon", FIF.COMMAND_PROMPT), widget)
            icon_label.setFixedSize(16, 16)
            layout.addWidget(icon_label)

            text_layout = QVBoxLayout()
            text_layout.setSpacing(0)

            name_label = BodyLabel(cmd.get("name", "Unknown"), widget)
            desc_label = CaptionLabel(cmd.get("description", ""), widget)
            desc_label.setStyleSheet("color: gray;")

            text_layout.addWidget(name_label)
            text_layout.addWidget(desc_label)

            layout.addLayout(text_layout)
            layout.addStretch()

            if "shortcut" in cmd:
                shortcut_label = CaptionLabel(cmd["shortcut"], widget)
                shortcut_label.setStyleSheet(
                    "background: rgba(128, 128, 128, 0.2); padding: 2px 5px; border-radius: 3px;"
                )
                layout.addWidget(shortcut_label)

            item.setSizeHint(widget.sizeHint())
            self.list_widget.setItemWidget(item, widget)

    def _on_text_changed(self, text: str):
        filtered = [
            cmd
            for cmd in self.commands
            if text.lower() in cmd["name"].lower()
            or text.lower() in cmd.get("description", "").lower()
        ]
        self._update_list(filtered)
        if self.list_widget.count() > 0:
            self.list_widget.setCurrentRow(0)

    def _on_return_pressed(self):
        item = self.list_widget.currentItem()
        if item:
            self._on_item_clicked(item)

    def _on_item_clicked(self, item: QListWidgetItem):
        cmd = item.data(Qt.ItemDataRole.UserRole)
        self.command_selected.emit(cmd["id"], cmd)
        self.hide()
        self.closed.emit()

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key.Key_Escape:
            self.hide()
            self.closed.emit()
        elif event.key() == Qt.Key.Key_Down:
            self.list_widget.setCurrentRow(
                (self.list_widget.currentRow() + 1) % self.list_widget.count()
            )
        elif event.key() == Qt.Key.Key_Up:
            self.list_widget.setCurrentRow(
                (self.list_widget.currentRow() - 1 + self.list_widget.count())
                % self.list_widget.count()
            )
        else:
            super().keyPressEvent(event)


class ModernMessageBubble(CardWidget):
    """
    Enhanced message bubble with better styling and interactive elements.
    """

    action_triggered = pyqtSignal(str, str)

    def __init__(self, text: str, is_user: bool = False, parent=None):
        super().__init__(parent)
        self.is_user = is_user
        self.text = text
        self.init_ui(text)

    def init_ui(self, text: str):
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(15, 10, 15, 10)

        if self.is_user:
            self.setBorderRadius(12)
            self.setStyleSheet(
                "ModernMessageBubble { background-color: rgba(0, 120, 212, 0.1); border: 1px solid rgba(0, 120, 212, 0.2); }"
            )
        else:
            self.setBorderRadius(12)
            self.setStyleSheet(
                "ModernMessageBubble { background-color: rgba(255, 255, 255, 0.05); border: 1px solid rgba(255, 255, 255, 0.1); }"
            )

        self.content_label = BodyLabel(text, self)
        self.content_label.setWordWrap(True)
        self.content_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.main_layout.addWidget(self.content_label)

        # Action buttons container
        if not self.is_user:
            self.actions_layout = QHBoxLayout()
            self.actions_layout.setContentsMargins(0, 5, 0, 0)
            self.actions_layout.setSpacing(5)
            self.actions_layout.addStretch()

            self.copy_btn = TransparentToolButton(FIF.COPY, self)
            self.copy_btn.setToolTip("Copy to clipboard")
            self.copy_btn.clicked.connect(
                lambda: self.action_triggered.emit("copy", self.text)
            )

            self.explain_btn = TransparentToolButton(FIF.INFO, self)
            self.explain_btn.setToolTip("Explain this")
            self.explain_btn.clicked.connect(
                lambda: self.action_triggered.emit("explain", self.text)
            )

            self.actions_layout.addWidget(self.explain_btn)
            self.actions_layout.addWidget(self.copy_btn)

            self.main_layout.addLayout(self.actions_layout)


class ThoughtBlock(CardWidget):
    """
    Collapsible widget for AI reasoning/thinking process.
    """

    def __init__(self, title: str = "Thinking...", parent=None):
        super().__init__(parent)
        self.init_ui(title)
        self.is_expanded = False
        self.content_widget.hide()

    def init_ui(self, title: str):
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(10, 5, 10, 5)
        self.main_layout.setSpacing(0)

        self.header = QWidget(self)
        self.header_layout = QHBoxLayout(self.header)
        self.header_layout.setContentsMargins(0, 0, 0, 0)

        self.icon_label = IconWidget(FIF.INFO, self.header)
        self.icon_label.setFixedSize(14, 14)
        self.header_layout.addWidget(self.icon_label)

        self.title_label = CaptionLabel(title, self.header)
        self.title_label.setStyleSheet("font-weight: bold; color: gray;")
        self.header_layout.addWidget(self.title_label)

        self.header_layout.addStretch()

        self.expand_icon = IconWidget(FIF.CHEVRON_RIGHT, self.header)
        self.expand_icon.setFixedSize(12, 12)
        self.header_layout.addWidget(self.expand_icon)

        self.main_layout.addWidget(self.header)

        self.content_widget = QWidget(self)
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(20, 5, 0, 5)

        self.content_label = CaptionLabel("", self.content_widget)
        self.content_label.setWordWrap(True)
        self.content_label.setStyleSheet("color: gray;")
        self.content_layout.addWidget(self.content_label)

        self.main_layout.addWidget(self.content_widget)

        self.header.setCursor(Qt.CursorShape.PointingHandCursor)
        self.header.mousePressEvent = self.toggle_expand

    def toggle_expand(self, event):
        self.is_expanded = not self.is_expanded
        self.content_widget.setVisible(self.is_expanded)
        self.expand_icon.setIcon(
            FIF.CHEVRON_DOWN if self.is_expanded else FIF.CHEVRON_RIGHT
        )

    def set_text(self, text: str):
        self.content_label.setText(text)


class GraphView(QWidget):
    """
    A simple visual graph view for project structure.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.nodes = []
        self.edges = []
        self.setMinimumSize(400, 400)

    def set_data(self, nodes, edges):
        self.nodes = nodes
        self.edges = edges
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Draw edges
        painter.setPen(QPen(QColor(100, 100, 100, 100), 1))
        for start_node, end_node in self.edges:
            painter.drawLine(start_node["pos"], end_node["pos"])

        # Draw nodes
        for node in self.nodes:
            painter.setBrush(QBrush(QColor(0, 120, 212)))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(node["pos"], 10, 10)
            painter.setPen(QPen(Qt.GlobalColor.white))
            painter.drawText(node["pos"].x() + 15, node["pos"].y() + 5, node["name"])
