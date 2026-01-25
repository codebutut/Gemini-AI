import qtawesome as qta
from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot, QSize
from PyQt6.QtWidgets import (
    QLineEdit,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from gemini_agent.core.indexer import Symbol


class SymbolBrowser(QWidget):
    symbol_selected = pyqtSignal(Symbol)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.symbols = []
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # Search Bar
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search symbols (classes, functions)...")
        self.search_input.setStyleSheet("""
            QLineEdit {
                border: 1px solid #444;
                border-radius: 4px;
                padding: 6px;
                background-color: #2d2d2d;
                color: #eee;
            }
            QLineEdit:focus { border: 1px solid #0B57D0; }
        """)
        self.search_input.textChanged.connect(self.filter_symbols)
        layout.addWidget(self.search_input)

        # Symbol Tree
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Symbol", "File"])
        self.tree.setColumnWidth(0, 150)
        self.tree.itemDoubleClicked.connect(self._on_item_double_clicked)
        self.tree.setStyleSheet("""
            QTreeWidget {
                background-color: #1e1e1e;
                border: none;
                color: #ccc;
            }
            QTreeWidget::item:hover { background-color: #2a2a2a; }
            QTreeWidget::item:selected { background-color: #37373d; color: #fff; }
        """)
        layout.addWidget(self.tree)

        # Refresh Button
        self.btn_refresh = QPushButton(" Refresh Index")
        self.btn_refresh.setIcon(qta.icon("fa5s.sync-alt", color="#eeeeee"))
        self.btn_refresh.setIconSize(QSize(14, 14))
        self.btn_refresh.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_refresh.setStyleSheet("""
            QPushButton {
                background-color: #333;
                color: #eee;
                border: 1px solid #444;
                padding: 5px;
                border-radius: 4px;
            }
            QPushButton:hover { background-color: #444; }
        """)
        layout.addWidget(self.btn_refresh)

    @pyqtSlot(list)
    def set_symbols(self, symbols: list[Symbol]):
        self.symbols = symbols
        self.filter_symbols(self.search_input.text())

    def filter_symbols(self, text: str):
        self.tree.clear()
        text = text.lower()

        # Group symbols by file
        files = {}
        for s in self.symbols:
            if text and text not in s.name.lower() and text not in s.file_path.lower():
                continue

            if s.file_path not in files:
                files[s.file_path] = []
            files[s.file_path].append(s)

        for file_path, file_symbols in files.items():
            file_item = QTreeWidgetItem([file_path])
            file_item.setData(0, Qt.ItemDataRole.UserRole, "file")
            file_item.setIcon(0, qta.icon("fa5s.file-code", color="#A0A0A0"))
            self.tree.addTopLevelItem(file_item)

            # Group by class if applicable
            classes = {}
            standalone = []
            for s in file_symbols:
                if s.kind == "class":
                    if s.name not in classes:
                        classes[s.name] = {"item": None, "symbols": []}
                    classes[s.name]["symbols"].append(s)
                elif s.parent:
                    if s.parent not in classes:
                        classes[s.parent] = {"item": None, "symbols": []}
                    classes[s.parent]["symbols"].append(s)
                else:
                    standalone.append(s)

            for class_name, data in classes.items():
                # Find the class symbol itself if it exists in this file
                class_symbol = next(
                    (
                        s
                        for s in data["symbols"]
                        if s.kind == "class" and s.name == class_name
                    ),
                    None,
                )

                class_item = QTreeWidgetItem([f"class {class_name}"])
                class_item.setData(0, Qt.ItemDataRole.UserRole, class_symbol)
                class_item.setIcon(0, qta.icon("fa5s.cube", color="#8be9fd"))
                file_item.addChild(class_item)

                for s in data["symbols"]:
                    if s.kind != "class":
                        icon_name = (
                            "fa5s.terminal" if s.kind == "function" else "fa5s.box"
                        )
                        icon_color = "#50fa7b" if s.kind == "function" else "#bd93f9"
                        item = QTreeWidgetItem([s.name])
                        item.setIcon(0, qta.icon(icon_name, color=icon_color))
                        item.setData(0, Qt.ItemDataRole.UserRole, s)
                        class_item.addChild(item)

                class_item.setExpanded(True)

            for s in standalone:
                icon_name = "fa5s.terminal" if s.kind == "function" else "fa5s.box"
                icon_color = "#50fa7b" if s.kind == "function" else "#bd93f9"
                item = QTreeWidgetItem([s.name])
                item.setIcon(0, qta.icon(icon_name, color=icon_color))
                item.setData(0, Qt.ItemDataRole.UserRole, s)
                file_item.addChild(item)

            file_item.setExpanded(True)

    def _on_item_double_clicked(self, item, column):
        data = item.data(0, Qt.ItemDataRole.UserRole)
        if isinstance(data, Symbol):
            self.symbol_selected.emit(data)
