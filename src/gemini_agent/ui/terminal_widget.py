from PyQt6.QtGui import QColor, QTextCharFormat, QTextCursor
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QPlainTextEdit, QPushButton, QVBoxLayout, QWidget


class TerminalWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Toolbar
        toolbar = QWidget()
        toolbar.setStyleSheet("background-color: #222; border-bottom: 1px solid #333;")
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(10, 2, 10, 2)

        title = QLabel("TERMINAL")
        title.setStyleSheet("color: #888; font-weight: bold; font-size: 10px;")
        toolbar_layout.addWidget(title)

        toolbar_layout.addStretch()

        self.btn_clear = QPushButton("Clear")
        self.btn_clear.setFixedWidth(50)
        self.btn_clear.setStyleSheet(
            "QPushButton { background: transparent; color: #888; border: none; font-size: 10px; } QPushButton:hover { color: #EEE; }"
        )
        self.btn_clear.clicked.connect(self.clear)
        toolbar_layout.addWidget(self.btn_clear)

        layout.addWidget(toolbar)

        # Terminal Output
        self.output = QPlainTextEdit()
        self.output.setReadOnly(True)
        self.output.setStyleSheet("""
            QPlainTextEdit {
                background-color: #1e1e1e;
                color: #d4d4d4;
                font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
                font-size: 12px;
                border: none;
            }
        """)
        layout.addWidget(self.output)

    def append_text(self, text: str, color: str = None):
        cursor = self.output.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)

        if color:
            fmt = QTextCharFormat()
            fmt.setForeground(QColor(color))
            cursor.setCharFormat(fmt)
        else:
            cursor.setCharFormat(QTextCharFormat())  # Reset to default

        cursor.insertText(text)
        self.output.setTextCursor(cursor)
        self.output.ensureCursorVisible()

    def append_error(self, text: str):
        self.append_text(text, "#f44747")

    def append_success(self, text: str):
        self.append_text(text, "#6a9955")

    def append_info(self, text: str):
        self.append_text(text, "#3794ff")

    def clear(self):
        self.output.clear()
