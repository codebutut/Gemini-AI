import json
import os

from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from gemini_agent.core.review_engine import ReviewEngine
from gemini_agent.ui.widgets import GeminiHighlighter


class DeepReviewDialog(QDialog):
    def __init__(self, tool_name, args, parent=None, theme_mode="Dark"):
        super().__init__(parent)
        self.setWindowTitle(f"Deep Review: {tool_name}")
        self.resize(900, 700)
        self.tool_name = tool_name
        self.args = args
        self.theme_mode = theme_mode
        self.review_engine = ReviewEngine()

        self.init_ui()
        self.apply_theme_to_dialog()
        self.load_data()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # Header
        self.header = QLabel(f"Reviewing Action: <b>{self.tool_name}</b>")
        self.header.setStyleSheet("font-size: 16px; margin-bottom: 10px;")
        layout.addWidget(self.header)

        # Tabs
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        # Tab 1: Visual Diff
        self.diff_viewer = QTextEdit()
        self.diff_viewer.setReadOnly(True)
        self.diff_viewer.setFont(QFont("Courier New", 10))
        self.tabs.addTab(self.diff_viewer, "Visual Diff")

        # Tab 2: Source Code (Proposed)
        source_container = QWidget()
        source_layout = QVBoxLayout(source_container)
        source_layout.setContentsMargins(0, 0, 0, 0)

        self.source_viewer = QTextEdit()
        self.source_viewer.setReadOnly(False)
        self.source_viewer.setFont(QFont("Courier New", 10))
        source_layout.addWidget(self.source_viewer)

        self.btn_copy_source = QPushButton("Copy to Clipboard")
        self.btn_copy_source.clicked.connect(self.copy_source)
        source_layout.addWidget(self.btn_copy_source)

        self.tabs.addTab(source_container, "Proposed Source")

        # Tab 3: Analysis
        self.analysis_list = QListWidget()
        self.tabs.addTab(self.analysis_list, "Analysis & Security")

        # Tab 4: Raw Arguments
        self.raw_viewer = QTextEdit()
        self.raw_viewer.setReadOnly(True)
        self.tabs.addTab(self.raw_viewer, "Raw Arguments")

        # Footer Buttons
        btn_layout = QHBoxLayout()

        self.btn_save = QPushButton("Save Changes")
        self.btn_save.clicked.connect(self.save_changes)

        self.btn_reject = QPushButton("Reject")
        self.btn_reject.clicked.connect(self.reject)

        self.btn_approve = QPushButton("Approve")
        self.btn_approve.clicked.connect(self.accept)

        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_save)
        btn_layout.addWidget(self.btn_reject)
        btn_layout.addWidget(self.btn_approve)

        layout.addLayout(btn_layout)

    def save_changes(self):
        """Updates internal args with the content from the source viewer."""
        new_content = self.source_viewer.toPlainText()

        if "content" in self.args:
            self.args["content"] = new_content
        elif "code" in self.args:
            self.args["code"] = new_content

        # Re-run analysis on new content
        self.review_engine.analyze_code(
            new_content
        )  # Just to refresh internal state if needed

        # Visual feedback
        self.btn_save.setText("Saved ✓")
        from PyQt6.QtCore import QTimer

        QTimer.singleShot(2000, lambda: self.btn_save.setText("Save Changes"))

    def get_args(self):
        """Returns the potentially modified arguments."""
        return self.args

    def apply_theme_to_dialog(self):
        is_dark = self.theme_mode == "Dark"
        bg = "#1E1F20" if is_dark else "#FFFFFF"
        fg = "#E3E3E3" if is_dark else "#000000"
        input_bg = "#282A2C" if is_dark else "#F0F4F9"
        tab_bg = "#131314" if is_dark else "#F0F4F9"

        self.setStyleSheet(f"""
            QDialog {{ 
                background-color: {bg}; 
                color: {fg}; 
            }} 
            QLabel, QListWidget {{ 
                color: {fg}; 
            }}
            QTextEdit, QListWidget {{ 
                background-color: {input_bg}; 
                color: {fg}; 
                border: 1px solid {"#444" if is_dark else "#CCC"}; 
                border-radius: 8px;
                padding: 5px;
            }}
            QTabWidget::pane {{
                border: 1px solid {"#444" if is_dark else "#CCC"};
                background-color: {tab_bg};
            }}
            QTabBar::tab {{
                background-color: {"#2D2E30" if is_dark else "#E0E0E0"};
                color: {fg};
                padding: 8px 12px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                margin-right: 2px;
            }}
            QTabBar::tab:selected {{
                background-color: {input_bg};
                border: 1px solid {"#444" if is_dark else "#CCC"};
                border-bottom-color: {input_bg};
            }}
        """)

        # Style buttons
        self.btn_reject.setStyleSheet(f"""
            QPushButton {{
                background-color: {"#442a2a" if is_dark else "#ffcccc"};
                color: {"#ffa3a3" if is_dark else "#cc0000"};
                border: 1px solid {"#663333" if is_dark else "#ff9999"};
                border-radius: 4px;
                padding: 8px 20px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {"#553333" if is_dark else "#ffb3b3"};
            }}
        """)

        self.btn_approve.setStyleSheet(f"""
            QPushButton {{
                background-color: {"#1e3a1e" if is_dark else "#ccffcc"};
                color: {"#afffbe" if is_dark else "#006600"};
                border: 1px solid {"#336633" if is_dark else "#99ff99"};
                border-radius: 4px;
                padding: 8px 20px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {"#2a4a2a" if is_dark else "#b3ffb3"};
            }}
        """)

        self.btn_copy_source.setStyleSheet(f"""
            QPushButton {{
                background-color: {"#333" if is_dark else "#EEE"};
                color: {fg};
                border: 1px solid {"#555" if is_dark else "#CCC"};
                border-radius: 4px;
                padding: 5px;
            }}
        """)

        self.btn_save.setStyleSheet(f"""
            QPushButton {{
                background-color: {"#333" if is_dark else "#EEE"};
                color: {fg};
                border: 1px solid {"#555" if is_dark else "#CCC"};
                border-radius: 4px;
                padding: 8px 20px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {"#444" if is_dark else "#DDD"};
            }}
        """)

    def copy_source(self):
        from PyQt6.QtWidgets import QApplication

        QApplication.clipboard().setText(self.source_viewer.toPlainText())
        QMessageBox.information(self, "Copied", "Source code copied to clipboard.")

    def load_data(self):
        # 1. Populate Raw Args
        self.raw_viewer.setText(json.dumps(self.args, indent=2))

        # 2. Determine Content and Filepath
        content = self.args.get("content", "")
        filepath = self.args.get("filepath", "")

        # Handle 'code' argument for python execution tools
        if not content and "code" in self.args:
            content = self.args["code"]
            filepath = "InMemory Script"

        # 3. Load Existing Content (if applicable)
        existing_content = ""
        if filepath and os.path.exists(filepath) and os.path.isfile(filepath):
            try:
                with open(filepath, encoding="utf-8") as f:
                    existing_content = f.read()
            except Exception as e:
                existing_content = f"Error reading file: {e}"

        # 4. Generate Diff
        diff_html = self.review_engine.generate_diff_html(
            existing_content, content, self.theme_mode
        )
        self.diff_viewer.setHtml(diff_html)

        # 5. Set Source View
        # Determine language
        lang = "python"
        if filepath:
            ext = os.path.splitext(filepath)[1].lower()
            if ext == ".js":
                lang = "javascript"
            elif ext == ".html":
                lang = "html"
            elif ext == ".css":
                lang = "css"
            elif ext == ".json":
                lang = "json"
            elif ext == ".md":
                lang = "markdown"
            elif ext == ".sh":
                lang = "bash"

        self.highlighter = GeminiHighlighter(
            self.source_viewer.document(), lang, self.theme_mode
        )
        self.source_viewer.setPlainText(content)

        # 6. Run Analysis
        issues = self.review_engine.analyze_code(content)
        security_risks = self.review_engine.scan_security(content)

        if not issues and not security_risks:
            self.analysis_list.addItem(
                "✅ No syntax errors or obvious security risks found."
            )
        else:
            # Add Security Risks first (higher priority)
            for risk in security_risks:
                item = QListWidgetItem(risk)
                item.setForeground(QColor("#ff5555"))  # Red
                font = item.font()
                font.setBold(True)
                item.setFont(font)
                self.analysis_list.addItem(item)

            # Add Linting Issues
            for issue in issues:
                item = QListWidgetItem(issue)
                if "CRITICAL" in issue:
                    item.setForeground(QColor("#ff5555"))
                    font = item.font()
                    font.setBold(True)
                    item.setFont(font)
                elif "LINT" in issue:
                    item.setForeground(QColor("#ffb86c"))  # Orange
                self.analysis_list.addItem(item)

            # Highlight the tab if there are issues
            total_issues = len(issues) + len(security_risks)
            self.tabs.setTabText(2, f"Analysis ({total_issues} Issues)")
            self.tabs.tabBar().setTabTextColor(2, QColor("#ff5555"))
