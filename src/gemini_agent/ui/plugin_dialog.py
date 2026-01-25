from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QTextBrowser,
    QVBoxLayout,
)

from gemini_agent.core.extension_manager import ExtensionManager


class PluginDialog(QDialog):
    def __init__(
        self, extension_manager: ExtensionManager, parent=None, theme_mode="Dark"
    ):
        super().__init__(parent)
        self.extension_manager = extension_manager
        self.theme_mode = theme_mode
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Extension Manager")
        self.resize(600, 400)

        layout = QHBoxLayout(self)

        # Left side: Plugin List
        left_layout = QVBoxLayout()
        self.plugin_list = QListWidget()
        self.plugin_list.itemSelectionChanged.connect(self.on_selection_changed)
        left_layout.addWidget(QLabel("Installed Plugins:"))
        left_layout.addWidget(self.plugin_list)

        btn_refresh = QPushButton("Refresh Extensions")
        btn_refresh.clicked.connect(self.refresh_plugins)
        left_layout.addWidget(btn_refresh)

        layout.addLayout(left_layout, 1)

        # Right side: Plugin Details
        self.details_frame = QFrame()
        self.details_frame.setFrameShape(QFrame.Shape.StyledPanel)
        details_layout = QVBoxLayout(self.details_frame)

        self.lbl_name = QLabel("Select an extension")
        self.lbl_name.setStyleSheet("font-weight: bold; font-size: 14px;")
        details_layout.addWidget(self.lbl_name)

        self.lbl_version = QLabel("")
        details_layout.addWidget(self.lbl_version)

        self.txt_description = QTextBrowser()
        details_layout.addWidget(self.txt_description)

        self.chk_enabled = QCheckBox("Enabled")
        self.chk_enabled.stateChanged.connect(self.on_enabled_changed)
        details_layout.addWidget(self.chk_enabled)

        layout.addWidget(self.details_frame, 2)

        self.refresh_plugins()

    def refresh_plugins(self):
        self.extension_manager.discover_plugins()
        self.plugin_list.clear()
        for name, plugin in self.extension_manager.plugins.items():
            item = QListWidgetItem(name)
            item.setData(Qt.ItemDataRole.UserRole, plugin)
            self.plugin_list.addItem(item)

    def on_selection_changed(self):
        items = self.plugin_list.selectedItems()
        if not items:
            self.lbl_name.setText("Select an extension")
            self.lbl_version.setText("")
            self.txt_description.clear()
            self.chk_enabled.setEnabled(False)
            return

        plugin = items[0].data(Qt.ItemDataRole.UserRole)
        self.lbl_name.setText(plugin.name)
        self.lbl_version.setText(f"Version: {plugin.version} | Author: {plugin.author}")
        self.txt_description.setText(plugin.description)

        self.chk_enabled.setEnabled(True)
        self.chk_enabled.blockSignals(True)
        self.chk_enabled.setChecked(plugin.enabled)
        self.chk_enabled.blockSignals(False)

    def on_enabled_changed(self, state):
        items = self.plugin_list.selectedItems()
        if items:
            plugin = items[0].data(Qt.ItemDataRole.UserRole)
            plugin.enabled = state == Qt.CheckState.Checked.value
