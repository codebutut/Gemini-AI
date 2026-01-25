import os
import re
from typing import Any

import qtawesome as qta
from markdown_it import MarkdownIt
from mdit_py_plugins.texmath import texmath_plugin
from mdit_py_plugins.tasklists import tasklists_plugin

# Pygments imports for Syntax Highlighting
from pygments import lex
from pygments.lexers import TextLexer, get_lexer_by_name
from pygments.token import Token
from PyQt6.QtCore import QStringListModel, Qt, QTimer, pyqtSignal, QSize
from PyQt6.QtGui import (
    QColor,
    QFontDatabase,
    QSyntaxHighlighter,
    QTextCharFormat,
    QTextCursor,
    QTextDocument,
)
from PyQt6.QtWidgets import (
    QApplication,
    QCompleter,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QSizePolicy,
)

from gemini_agent.ui.styles import SYNTAX_COLORS
from gemini_agent.ui.theme_manager import ThemeManager

# Shared MarkdownIt instance with plugins
MD_PARSER = (
    MarkdownIt("commonmark", {"breaks": True, "html": True})
    .enable("table")
    .use(texmath_plugin)
    .use(tasklists_plugin)
)


class AttachmentItem(QFrame):
    """Small widget to show an attached file with a remove button."""

    remove_requested = pyqtSignal(str)

    def __init__(self, file_path: str, theme_mode: str = "Dark"):
        super().__init__()
        self.file_path = file_path
        self.theme_mode = theme_mode
        self.init_ui()

    def init_ui(self) -> None:
        """Initializes the attachment item UI."""
        self.setObjectName("AttachmentItem")
        is_dark = self.theme_mode == "Dark"
        bg = "#333" if is_dark else "#E0E0E0"
        fg = "#EEE" if is_dark else "#333"

        self.setStyleSheet(f"""
            QFrame#AttachmentItem {{
                background-color: {bg};
                border-radius: 4px;
                padding: 2px 5px;
            }}
            QLabel {{ color: {fg}; font-size: 11px; }}
            QPushButton {{ 
                background: transparent; 
                color: {fg}; 
                border: none; 
                font-weight: bold;
                font-size: 12px;
            }}
            QPushButton:hover {{ color: #ff4444; }}
        """)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 2, 5, 2)
        layout.setSpacing(5)

        name = os.path.basename(self.file_path)

        # Add file icon
        icon_lbl = QLabel()
        icon_lbl.setPixmap(qta.icon("fa5s.file", color=fg).pixmap(12, 12))
        layout.addWidget(icon_lbl)

        lbl = QLabel(name)
        lbl.setToolTip(self.file_path)
        layout.addWidget(lbl)

        btn_remove = QPushButton()
        btn_remove.setIcon(qta.icon("fa5s.times-circle", color=fg))
        btn_remove.setIconSize(QSize(14, 14))
        btn_remove.setFixedSize(16, 16)
        btn_remove.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_remove.clicked.connect(lambda: self.remove_requested.emit(self.file_path))

        layout.addWidget(btn_remove)


class GeminiHighlighter(QSyntaxHighlighter):
    """
    A Qt Syntax Highlighter that uses Pygments to tokenize code.
    """

    def __init__(
        self, document: QTextDocument, language: str, theme_mode: str = "Dark"
    ):
        super().__init__(document)
        self.theme_mode = theme_mode
        self.styles = SYNTAX_COLORS.get(theme_mode, SYNTAX_COLORS["Dark"])

        try:
            self.lexer = get_lexer_by_name(language)
        except Exception:
            self.lexer = TextLexer()

        self.formats = self._create_formats()

    def _create_formats(self) -> dict[Any, QTextCharFormat]:
        """Maps Pygments tokens to QTextCharFormat based on current theme."""
        formats = {}

        def _fmt(color_key: str) -> QTextCharFormat:
            color = self.styles.get(color_key, "#dcdcdc")
            f = QTextCharFormat()
            f.setForeground(QColor(color))
            return f

        formats[Token.Keyword] = _fmt("keyword")
        formats[Token.Name.Function] = _fmt("function")
        formats[Token.Name.Class] = _fmt("class")
        formats[Token.String] = _fmt("string")
        formats[Token.Comment] = _fmt("comment")
        formats[Token.Number] = _fmt("number")
        formats[Token.Operator] = _fmt("operator")
        formats[Token.Name] = _fmt("variable")

        return formats

    def highlightBlock(self, text: str):
        """Applied to every block of text in the document."""
        tokens = lex(text, self.lexer)

        index = 0
        for token_type, value in tokens:
            length = len(value)
            fmt = self.formats.get(token_type)
            if not fmt:
                if token_type in Token.Keyword:
                    fmt = self.formats.get(Token.Keyword)
                elif token_type in Token.Name:
                    fmt = self.formats.get(Token.Name)
                elif token_type in Token.String:
                    fmt = self.formats.get(Token.String)
                elif token_type in Token.Comment:
                    fmt = self.formats.get(Token.Comment)

            if fmt:
                self.setFormat(index, length, fmt)

            index += length


class CodeBlock(QFrame):
    """
    A widget to display code with a header, Copy button, and auto-sizing.
    """

    def __init__(self, code: str, language: str = "text", theme_mode: str = "Dark"):
        super().__init__()
        self.code = code
        self.language = language if language else "text"
        self.theme_mode = theme_mode
        self.initUI()

    def initUI(self):
        self.setObjectName("CodeBlock")
        bg_color = "#1e1e1e" if self.theme_mode == "Dark" else "#f6f8fa"
        border_color = "#333" if self.theme_mode == "Dark" else "#d0d7de"

        self.setStyleSheet(f"""
            QFrame#CodeBlock {{ 
                background-color: {bg_color}; 
                border-radius: 6px; 
                border: 1px solid {border_color}; 
                margin-top: 4px; 
                margin-bottom: 4px; 
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        header = QFrame()
        header.setObjectName("CodeHeader")
        header_bg = "#2d2d2d" if self.theme_mode == "Dark" else "#eaeef2"
        header.setStyleSheet(f"""
            QFrame#CodeHeader {{ 
                background-color: {header_bg}; 
                border-top-left-radius: 6px; 
                border-top-right-radius: 6px;
                border-bottom: 1px solid {border_color};
            }}
        """)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(10, 4, 10, 4)

        lang_lbl = QLabel(self.language.upper())
        lang_lbl.setStyleSheet(
            f"color: {'#aaa' if self.theme_mode == 'Dark' else '#57606a'}; font-weight: bold; font-size: 10px; font-family: 'Segoe UI', sans-serif;"
        )

        copy_btn = QPushButton(" Copy")
        copy_btn.setIcon(
            qta.icon(
                "fa5s.copy", color="#aaa" if self.theme_mode == "Dark" else "#57606a"
            )
        )
        copy_btn.setIconSize(QSize(10, 10))
        copy_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        copy_btn.setStyleSheet(f"""
            QPushButton {{
                background: transparent; 
                color: {"#aaa" if self.theme_mode == "Dark" else "#57606a"}; 
                border: none; 
                font-size: 10px;
            }}
            QPushButton:hover {{ color: {"#fff" if self.theme_mode == "Dark" else "#0969da"}; }}
        """)
        copy_btn.clicked.connect(self.copy_code)

        header_layout.addWidget(lang_lbl)
        header_layout.addStretch()
        header_layout.addWidget(copy_btn)

        layout.addWidget(header)

        self.editor = QTextEdit()
        self.editor.setReadOnly(True)
        self.editor.setPlainText(self.code)

        font = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
        font.setPointSize(10)
        self.editor.setFont(font)

        text_color = "#e3e3e3" if self.theme_mode == "Dark" else "#1f2328"
        self.editor.setStyleSheet(f"""
            QTextEdit {{
                background-color: transparent;
                color: {text_color};
                border: none;
                padding: 5px;
            }}
        """)

        self.editor.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.editor.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.editor.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)

        self.highlighter = GeminiHighlighter(
            self.editor.document(), self.language, self.theme_mode
        )

        layout.addWidget(self.editor)

        # Connect document changes to height adjustment
        self.editor.document().contentsChanged.connect(self.adjust_height)

        # Use a timer to adjust height after the widget is laid out
        QTimer.singleShot(50, self.adjust_height)

    def adjust_height(self):
        """Adjusts the height of the code block based on its content and width."""
        doc = self.editor.document()
        # Set document width to match editor width for accurate height calculation with wrapping
        width = self.editor.viewport().width()
        if width <= 0:
            width = self.editor.width() if self.editor.width() > 0 else 500

        doc.setTextWidth(width)

        doc_height = doc.size().height()
        # Header (~25px) + document height + padding/margins
        new_height = int(doc_height) + 40

        # Apply height constraints - Reduced minimum height from 80 to 50
        self.setFixedHeight(min(max(new_height, 50), 1000))
        self.updateGeometry()

    def resizeEvent(self, event):
        """Handle resize events to re-calculate height if text wraps."""
        super().resizeEvent(event)
        QTimer.singleShot(10, self.adjust_height)

    def copy_code(self):
        clipboard = QApplication.clipboard()
        clipboard.setText(self.code)
        btn = self.findChild(QPushButton)
        if btn:
            original_text = btn.text()
            original_icon = btn.icon()
            btn.setText(" Copied!")
            btn.setIcon(qta.icon("fa5s.check", color="#50fa7b"))
            QTimer.singleShot(
                1500, lambda: (btn.setText(original_text), btn.setIcon(original_icon))
            )


class MessageBubble(QFrame):
    """
    A bubble for rendering Markdown text and Code Blocks nicely with justified alignment.
    """

    def __init__(self, text: str, is_user: bool = False, theme_mode: str = "Dark"):
        super().__init__()
        self.raw_text = text
        self.is_user = is_user
        self.theme_mode = theme_mode
        self.initUI()

    def initUI(self):
        self.setObjectName("UserBubble" if self.is_user else "AIBubble")

        # Apply bubble style using ThemeManager's centralized logic
        self.setStyleSheet(
            ThemeManager.get_bubble_style_static(self.theme_mode, self.is_user)
        )

        layout = QVBoxLayout(self)
        # Reduced margins from (10, 10, 10, 10) to (8, 4, 8, 4)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(8)

        # Use shared parser
        tokens = MD_PARSER.parse(self.raw_text)

        lines = self.raw_text.splitlines(keepends=True)
        last_line_idx = 0

        for token in tokens:
            if token.type == "fence":
                start_line, end_line = token.map
                if start_line > last_line_idx:
                    text_chunk = "".join(lines[last_line_idx:start_line])
                    if text_chunk.strip():
                        self._add_markdown_text(text_chunk, layout)

                self._add_code_block(token.content, token.info, layout)
                last_line_idx = end_line

        if last_line_idx < len(lines):
            text_chunk = "".join(lines[last_line_idx:])
            if text_chunk.strip():
                self._add_markdown_text(text_chunk, layout)

        if not self.is_user:
            self._add_toolbar(layout)

        # Ensure the bubble can grow and doesn't get squashed
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)

    def _add_markdown_text(self, text: str, layout: QVBoxLayout):
        """Renders markdown text using QLabel with justified rich text."""
        html_content = self._markdown_to_html(text)

        lbl = QLabel(html_content)
        lbl.setObjectName("BubbleText")
        lbl.setWordWrap(True)
        lbl.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        lbl.setOpenExternalLinks(True)

        # Changed from MinimumExpanding to Preferred to avoid excessive vertical growth
        lbl.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred
        )

        text_color = "#E3E3E3" if self.theme_mode == "Dark" else "#1F1F1F"
        lbl.setStyleSheet(
            f"color: {text_color}; border: none; font-size: 15px; background: transparent;"
        )

        layout.addWidget(lbl)

    def _add_code_block(self, code_content: str, language: str, layout: QVBoxLayout):
        """Creates a CodeBlock widget."""
        code = code_content[:-1] if code_content.endswith("\n") else code_content
        block = CodeBlock(code, (language or "text").strip(), self.theme_mode)
        layout.addWidget(block)

    def _add_toolbar(self, layout: QVBoxLayout) -> None:
        """Adds a toolbar with Copy Plain Text and Copy Markdown buttons."""
        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(0, 2, 0, 2)
        toolbar.setSpacing(8)
        toolbar.addStretch()

        btn_style = """
            QPushButton {
                border: none;
                color: #888;
                font-size: 10px;
                padding: 2px 6px;
                background-color: transparent;
            }
            QPushButton:hover { 
                color: #4da6ff; 
                background-color: rgba(77, 166, 255, 0.1); 
                border-radius: 4px; 
            }
        """

        copy_plain_btn = QPushButton(" Copy Plain Text")
        copy_plain_btn.setIcon(qta.icon("fa5s.file-alt", color="#888"))
        copy_plain_btn.setIconSize(QSize(10, 10))
        copy_plain_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        copy_plain_btn.setStyleSheet(btn_style)
        copy_plain_btn.clicked.connect(lambda: self.copy_plain_text(copy_plain_btn))

        copy_md_btn = QPushButton(" Copy Markdown")
        copy_md_btn.setIcon(qta.icon("fa5b.markdown", color="#888"))
        copy_md_btn.setIconSize(QSize(12, 12))
        copy_md_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        copy_md_btn.setStyleSheet(btn_style)
        copy_md_btn.clicked.connect(lambda: self.copy_markdown(copy_md_btn))

        toolbar.addWidget(copy_plain_btn)
        toolbar.addWidget(copy_md_btn)
        layout.addLayout(toolbar)

    def _markdown_to_html(self, text: str) -> str:
        """Converts Markdown to basic HTML for QLabel with justification wrapper."""
        try:
            # Use shared parser for rendering
            html = MD_PARSER.render(text)
            link_color = "#4da6ff" if self.theme_mode == "Dark" else "#0969da"
            html = html.replace(
                "<a href=",
                f'<a style="color: {link_color}; text-decoration: none;" href=',
            )

            if "<table>" in html:
                border_color = "#555555" if self.theme_mode == "Dark" else "#dddddd"
                header_bg = "#333333" if self.theme_mode == "Dark" else "#f0f0f0"

                # Add border, spacing, and width
                html = html.replace(
                    "<table>",
                    f'<table border="1" cellspacing="0" cellpadding="5" width="100%" style="border-collapse: collapse; border-color: {border_color};">',
                )

                # Style headers (background color) - Use regex to avoid corrupting <thead>
                html = re.sub(r"<th(\s|>)", f'<th bgcolor="{header_bg}"\\1', html)

            return f"<div>{html}</div>"
        except Exception:
            return f"<div>{text}</div>"

    def copy_plain_text(self, btn: QPushButton) -> None:
        """
        Copies the content as plain text, removing markdown formatting
        using QTextDocument for robust stripping.
        """
        html = self._markdown_to_html(self.raw_text)
        doc = QTextDocument()
        doc.setHtml(html)
        text = doc.toPlainText()

        clipboard = QApplication.clipboard()
        clipboard.setText(text.strip())

        self._show_copied_feedback(btn)

    def copy_markdown(self, btn: QPushButton) -> None:
        """Copies the raw markdown content to the clipboard."""
        clipboard = QApplication.clipboard()
        clipboard.setText(self.raw_text.strip())

        self._show_copied_feedback(btn)

    def _show_copied_feedback(self, btn: QPushButton) -> None:
        """Temporarily changes button text to provide visual feedback."""
        original_text = btn.text()
        original_icon = btn.icon()
        btn.setText(" Copied!")
        btn.setIcon(qta.icon("fa5s.check", color="#50fa7b"))
        QTimer.singleShot(
            1500, lambda: (btn.setText(original_text), btn.setIcon(original_icon))
        )


class AutoResizingTextEdit(QTextEdit):
    """
    A QTextEdit that expands vertically based on content.
    """

    returnPressed = pyqtSignal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("InputBox")
        self.setPlaceholderText("Enter a prompt here (Shift+Enter for new line)...")
        self.setFixedHeight(50)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.textChanged.connect(self.adjust_height)

        # specific completion setup
        self._completer = None
        self._setup_completer()

    def _setup_completer(self):
        """Sets up the auto-completer with initial default keywords."""
        keywords = [
            "/clear",
            "/help",
            "/reset",
            "/conductor",
            "analyze",
            "refactor",
            "explain",
            "fix",
            "search",
        ]
        self.update_keywords(keywords)

    def update_keywords(self, keywords: list[str]):
        """Updates the completer model with a new list of keywords."""
        # Ensure unique and sorted
        unique_keywords = sorted(list(set(keywords)))

        if self._completer is None:
            self._completer = QCompleter(unique_keywords, self)
            self._completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
            self._completer.setWidget(self)
            self._completer.activated.connect(self._insert_completion)
        else:
            self._completer.setModel(QStringListModel(unique_keywords))

    def _insert_completion(self, completion: str):
        """Inserts the selected completion into the text."""
        tc = self.textCursor()
        extra = len(completion) - len(self._completer.completionPrefix())
        tc.movePosition(QTextCursor.MoveOperation.Left)
        tc.movePosition(QTextCursor.MoveOperation.EndOfWord)
        tc.insertText(completion[-extra:])
        self.setTextCursor(tc)

    def keyPressEvent(self, event):
        if (
            self._completer
            and self._completer.popup().isVisible()
            and event.key()
            in (
                Qt.Key.Key_Enter,
                Qt.Key.Key_Return,
                Qt.Key.Key_Escape,
                Qt.Key.Key_Tab,
                Qt.Key.Key_Backtab,
            )
        ):
            event.ignore()
            return

        if event.key() == Qt.Key.Key_Return and not (
            event.modifiers() & Qt.KeyboardModifier.ShiftModifier
        ):
            self.returnPressed.emit()
            return

        # Trigger completer
        is_shortcut = (
            event.modifiers() & Qt.KeyboardModifier.ControlModifier
        ) and event.key() == Qt.Key.Key_E

        super().keyPressEvent(event)

        completion_prefix = self._text_under_cursor()

        if not is_shortcut and (len(completion_prefix) < 1 or event.text() == ""):
            if self._completer:
                self._completer.popup().hide()
            return

        if self._completer and completion_prefix != self._completer.completionPrefix():
            self._completer.setCompletionPrefix(completion_prefix)
            self._completer.popup().setCurrentIndex(
                self._completer.completionModel().index(0, 0)
            )

        cr = self.cursorRect()
        cr.setWidth(
            self._completer.popup().sizeHintForColumn(0)
            + self._completer.popup().verticalScrollBar().sizeHint().width()
        )
        self._completer.complete(cr)

    def _text_under_cursor(self):
        tc = self.textCursor()
        tc.select(QTextCursor.SelectionType.WordUnderCursor)
        return tc.selectedText()

    def adjust_height(self):
        doc_height = self.document().size().height()
        new_height = min(max(50, int(doc_height + 10)), 150)
        self.setFixedHeight(new_height)
