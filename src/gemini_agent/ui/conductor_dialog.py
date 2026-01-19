from pathlib import Path

from PyQt6.QtCore import Qt
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


class ConductorDialog(QDialog):
    """
    Dialog for managing Conductor files and commands.
    Provides a centralized interface for project orchestration.
    """

    def __init__(self, parent, conductor_manager):
        super().__init__(parent)
        self.main_window = parent
        self.conductor_manager = conductor_manager
        self.setWindowTitle("Conductor - Project Orchestration")
        self.setMinimumSize(800, 600)
        self.init_ui()
        self.apply_theme()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # Header
        header_lbl = QLabel("🚀 Conductor Orchestrator")
        header_lbl.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(header_lbl)

        description = QLabel("Manage project orchestration files and execute specialized commands.")
        description.setStyleSheet("color: #888; margin-bottom: 10px;")
        layout.addWidget(description)

        # Tabs for different files
        self.tabs = QTabWidget()

        self.product_edit = QTextEdit()
        self.tech_stack_edit = QTextEdit()
        self.workflow_edit = QTextEdit()

        self.tabs.addTab(self.product_edit, "Product (product.md)")
        self.tabs.addTab(self.tech_stack_edit, "Tech Stack (tech-stack.md)")
        self.tabs.addTab(self.workflow_edit, "Workflow (workflow.md)")

        layout.addWidget(self.tabs)

        # Commands Section
        cmd_group = QWidget()
        cmd_layout = QVBoxLayout(cmd_group)
        cmd_layout.addWidget(QLabel("Available Commands:"))

        self.cmd_list = QListWidget()
        self.refresh_commands()
        cmd_layout.addWidget(self.cmd_list)

        btn_run_cmd = QPushButton("Run Selected Command")
        btn_run_cmd.clicked.connect(self.run_selected_command)
        cmd_layout.addWidget(btn_run_cmd)

        self.tabs.addTab(cmd_group, "Commands")

        # Bottom Buttons
        bottom_layout = QHBoxLayout()

        btn_load = QPushButton("Load Files")
        btn_load.clicked.connect(self.load_files)
        bottom_layout.addWidget(btn_load)

        btn_save = QPushButton("Save Files")
        btn_save.clicked.connect(self.save_files)
        bottom_layout.addWidget(btn_save)

        btn_templates = QPushButton("Generate Templates")
        btn_templates.clicked.connect(self.generate_templates)
        bottom_layout.addWidget(btn_templates)

        bottom_layout.addStretch(1)

        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        bottom_layout.addWidget(btn_close)

        layout.addLayout(bottom_layout)

        self.load_files()

    def refresh_commands(self):
        """Refreshes the list of available conductor commands."""
        self.cmd_list.clear()
        commands = self.conductor_manager.get_available_commands()
        for cmd in commands:
            item = QListWidgetItem(cmd.capitalize())
            item.setData(Qt.ItemDataRole.UserRole, cmd)
            self.cmd_list.addItem(item)

    def load_files(self):
        """Loads conductor files from the project directory."""
        project_path = Path(".")
        conductor_dir = project_path / "conductor"

        files = {
            "product.md": self.product_edit,
            "tech-stack.md": self.tech_stack_edit,
            "workflow.md": self.workflow_edit,
        }

        for filename, edit in files.items():
            file_path = conductor_dir / filename
            if file_path.exists():
                try:
                    with open(file_path, encoding="utf-8") as f:
                        edit.setText(f.read())
                except Exception as e:
                    edit.setText(f"Error loading {filename}: {e}")
            else:
                edit.setText(f"# {filename}\n\nFile not found. Click 'Generate Templates' or 'Save' to create it.")

    def save_files(self):
        """Saves the current content of the text edits to conductor files."""
        project_path = Path(".")
        conductor_dir = project_path / "conductor"
        conductor_dir.mkdir(exist_ok=True)

        files = {
            "product.md": self.product_edit.toPlainText(),
            "tech-stack.md": self.tech_stack_edit.toPlainText(),
            "workflow.md": self.workflow_edit.toPlainText(),
        }

        try:
            for filename, content in files.items():
                file_path = conductor_dir / filename
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)
            QMessageBox.information(self, "Success", "Conductor files saved successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save files: {e}")

    def generate_templates(self):
        """Fills the text edits with default templates if they are empty or requested."""
        templates = {
            "product.md": "# Product Vision\n\n## Overview\nDescribe the product here.\n\n## Target Audience\nWho is this for?\n\n## Key Features\n- Feature 1\n- Feature 2",
            "tech-stack.md": "# Tech Stack\n\n## Languages\n- Python 3.10+\n\n## Frameworks\n- PyQt6\n\n## Tools\n- Pytest",
            "workflow.md": "# Development Workflow\n\n## Branching Strategy\nMain branch for production.\n\n## Testing\nRun `pytest` before committing.",
        }

        if not self.product_edit.toPlainText() or "File not found" in self.product_edit.toPlainText():
            self.product_edit.setText(templates["product.md"])
        if not self.tech_stack_edit.toPlainText() or "File not found" in self.tech_stack_edit.toPlainText():
            self.tech_stack_edit.setText(templates["tech-stack.md"])
        if not self.workflow_edit.toPlainText() or "File not found" in self.workflow_edit.toPlainText():
            self.workflow_edit.setText(templates["workflow.md"])

    def run_selected_command(self):
        """Executes the selected conductor command in the main window."""
        item = self.cmd_list.currentItem()
        if not item:
            QMessageBox.warning(self, "Warning", "Please select a command to run.")
            return

        command_name = item.data(Qt.ItemDataRole.UserRole)
        self.accept()
        self.main_window.run_conductor_command(command_name)

    def apply_theme(self):
        """Applies the current application theme to the dialog."""
        theme = self.main_window.app_config.theme
        is_dark = theme == "Dark"
        bg = "#1E1F20" if is_dark else "#FFFFFF"
        fg = "#E3E3E3" if is_dark else "#000000"
        input_bg = "#282A2C" if is_dark else "#F0F0F0"

        self.setStyleSheet(f"""
            QDialog {{ background-color: {bg}; color: {fg}; }}
            QLabel {{ color: {fg}; }}
            QTextEdit, QListWidget {{ 
                background-color: {input_bg}; color: {fg}; border: 1px solid #444; padding: 5px;
                font-family: 'Consolas', 'Monaco', monospace;
            }}
            QPushButton {{
                background-color: {"#2D2E30" if is_dark else "#E0E0E0"};
                color: {fg};
                border: 1px solid {"#444" if is_dark else "#CCC"};
                border-radius: 4px;
                padding: 8px 16px;
            }}
            QPushButton:hover {{ background-color: {"#3C4043" if is_dark else "#D0D0D0"}; }}
            QTabBar::tab {{
                background: {"#2D2E30" if is_dark else "#E0E0E0"};
                color: {fg};
                padding: 8px 12px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }}
            QTabBar::tab:selected {{
                background: {"#0B57D0" if is_dark else "#1A73E8"};
                color: white;
            }}
        """)
