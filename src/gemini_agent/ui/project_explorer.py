import os

import qtawesome as qta
from PyQt6.QtCore import QDir, Qt, pyqtSignal
from PyQt6.QtGui import QAction, QFileSystemModel
from PyQt6.QtWidgets import (
    QLineEdit,
    QMenu,
    QTreeView,
    QVBoxLayout,
    QWidget,
)


class ProjectExplorer(QWidget):
    """
    A widget that displays the project file structure and allows
    attaching files/folders to the chat.
    """

    file_attached = pyqtSignal(str)
    folder_attached = pyqtSignal(str)
    file_opened = pyqtSignal(str)

    def __init__(self, root_path: str = ".", parent=None):
        super().__init__(parent)
        self.root_path = os.path.abspath(root_path)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)

        # Search/Filter
        self.filter_input = QLineEdit()
        self.filter_input.setPlaceholderText("Filter files...")
        self.filter_input.setStyleSheet("""
            QLineEdit {
                border: 1px solid #444;
                border-radius: 4px;
                padding: 4px;
                background-color: #2d2d2d;
                color: #eee;
                margin: 5px;
            }
            QLineEdit:focus { border: 1px solid #0B57D0; }
        """)
        self.filter_input.textChanged.connect(self.filter_files)
        layout.addWidget(self.filter_input)

        # File System Model
        self.model = QFileSystemModel()
        self.model.setRootPath(self.root_path)
        self.model.setFilter(QDir.Filter.AllEntries | QDir.Filter.NoDotAndDotDot)

        # Tree View
        self.tree = QTreeView()
        self.tree.setModel(self.model)
        self.tree.setRootIndex(self.model.index(self.root_path))

        # Hide columns except Name
        for i in range(1, self.model.columnCount()):
            self.tree.hideColumn(i)

        self.tree.header().hide()
        self.tree.setAnimated(True)
        self.tree.setIndentation(20)
        self.tree.setSortingEnabled(True)
        self.tree.sortByColumn(0, Qt.SortOrder.AscendingOrder)

        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self.show_context_menu)
        self.tree.doubleClicked.connect(self.on_double_clicked)

        layout.addWidget(self.tree)

    def filter_files(self, text):
        if not text:
            self.model.setNameFilters([])
            self.model.setNameFilterDisables(True)
        else:
            self.model.setNameFilters([f"*{text}*"])
            self.model.setNameFilterDisables(False)

    def show_context_menu(self, position):
        index = self.tree.indexAt(position)
        if not index.isValid():
            return

        path = self.model.filePath(index)
        is_dir = self.model.isDir(index)

        menu = QMenu()

        attach_icon = qta.icon("fa5s.paperclip", color="#888")
        attach_action = QAction(
            attach_icon, f" Attach {'Folder' if is_dir else 'File'}", self
        )
        attach_action.triggered.connect(lambda: self.emit_attach(path, is_dir))
        menu.addAction(attach_action)

        if not is_dir:
            open_icon = qta.icon("fa5s.book-open", color="#888")
            open_action = QAction(open_icon, " Open in Editor", self)
            open_action.triggered.connect(lambda: self.file_opened.emit(path))
            menu.addAction(open_action)

        menu.addSeparator()

        copy_icon = qta.icon("fa5s.copy", color="#888")
        copy_path_action = QAction(copy_icon, " Copy Path", self)
        copy_path_action.triggered.connect(lambda: self.copy_to_clipboard(path))
        menu.addAction(copy_path_action)

        menu.exec(self.tree.viewport().mapToGlobal(position))

    def emit_attach(self, path, is_dir):
        if is_dir:
            self.folder_attached.emit(path)
        else:
            self.file_attached.emit(path)

    def on_double_clicked(self, index):
        path = self.model.filePath(index)
        if not self.model.isDir(index):
            self.file_attached.emit(path)

    def copy_to_clipboard(self, text):
        from PyQt6.QtWidgets import QApplication

        QApplication.clipboard().setText(text)

    def apply_theme(self, theme_mode: str):
        is_dark = theme_mode == "Dark"
        bg = "#1E1F20" if is_dark else "#FFFFFF"
        fg = "#E3E3E3" if is_dark else "#000000"
        input_bg = "#2d2d2d" if is_dark else "#F0F0F0"

        self.filter_input.setStyleSheet(f"""
            QLineEdit {{
                border: 1px solid {"#444" if is_dark else "#CCC"};
                border-radius: 4px;
                padding: 4px;
                background-color: {input_bg};
                color: {fg};
                margin: 5px;
            }}
            QLineEdit:focus {{ border: 1px solid #0B57D0; }}
        """)

        self.tree.setStyleSheet(f"""
            QTreeView {{
                background-color: {bg};
                color: {fg};
                border: none;
            }}
            QTreeView::item:hover {{
                background-color: {"#333" if is_dark else "#F0F0F0"};
            }}
            QTreeView::item:selected {{
                background-color: {"#0B57D0" if is_dark else "#E8F0FE"};
                color: {"white" if is_dark else "#0B57D0"};
            }}
        """)
