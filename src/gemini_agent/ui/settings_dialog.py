from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from gemini_agent.config.app_config import AppConfig, ModelRegistry, save_json


class SettingsDialog(QDialog):
    """
    Modal dialog for application settings with scrollable content.
    """

    def __init__(self, parent, config):
        super().__init__(parent)
        self.main_window = parent
        self.config = config
        self.checkpoint_manager = getattr(parent, "checkpoint_manager", None)
        self.setWindowTitle("Settings")
        self.setMinimumWidth(550)
        self.setMinimumHeight(500)
        self.setMaximumHeight(800)
        self.initUI()
        self.apply_theme_to_dialog()
        self.refresh_checkpoints()

    def initUI(self):
        # Create main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Create scroll area
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)

        # Create container widget for scroll area
        self.scroll_widget = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_widget)
        self.scroll_layout.setSpacing(15)
        self.scroll_layout.setContentsMargins(20, 20, 20, 20)

        # 1. Model Selection
        self.scroll_layout.addWidget(QLabel("Gemini Model:"))
        self.model_combo = QComboBox()
        for display_name, model_id in ModelRegistry.GEMINI_MODELS:
            self.model_combo.addItem(display_name, model_id)
        current_model = self.config.get("model", ModelRegistry.DEFAULT_MODEL_ID)
        found_index = -1
        for i in range(self.model_combo.count()):
            if self.model_combo.itemData(i) == current_model:
                found_index = i
                break
        if found_index >= 0:
            self.model_combo.setCurrentIndex(found_index)
        else:
            self.model_combo.addItem(f"Custom: {current_model}", current_model)
            self.model_combo.setCurrentIndex(self.model_combo.count() - 1)
        self.model_combo.currentIndexChanged.connect(self.save_general_settings)
        self.scroll_layout.addWidget(self.model_combo)

        model_info = QLabel(
            "💡 Gemini 3 Series: Advanced reasoning | Gemini 2.5 Series: Current stable | Flash: Fast & efficient"
        )
        model_info.setStyleSheet("color: #888; font-size: 11px; font-style: italic;")
        self.scroll_layout.addWidget(model_info)

        # 2. API Key
        self.scroll_layout.addWidget(QLabel("API Key:"))
        self.api_input = QLineEdit(self.config.get("api_key", ""))
        self.api_input.setPlaceholderText("Paste Gemini API Key")
        self.api_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_input.textChanged.connect(self.save_general_settings)
        self.scroll_layout.addWidget(self.api_input)

        # 3. Grounding
        self.chk_grounding = QCheckBox("Enable Google Search Grounding (Web queries only)")
        self.chk_grounding.setChecked(self.config.get("use_search", False))
        self.chk_grounding.toggled.connect(self.save_general_settings)
        self.scroll_layout.addWidget(self.chk_grounding)

        # --- Appearance ---
        appearance_group = QGroupBox("Appearance")
        appearance_layout = QVBoxLayout()
        appearance_layout.addWidget(QLabel("Theme:"))
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Dark", "Light"])
        self.theme_combo.setCurrentText(self.config.get("theme", "Dark"))
        self.theme_combo.currentTextChanged.connect(self.on_theme_changed)
        appearance_layout.addWidget(self.theme_combo)
        appearance_group.setLayout(appearance_layout)
        self.scroll_layout.addWidget(appearance_group)

        # --- Checkpointing ---
        if self.checkpoint_manager:
            checkpoint_group = QGroupBox("Project Checkpointing")
            checkpoint_layout = QVBoxLayout()

            checkpoint_layout.addWidget(QLabel("Create New Checkpoint:"))
            create_layout = QHBoxLayout()
            self.checkpoint_name_input = QLineEdit()
            self.checkpoint_name_input.setPlaceholderText("Checkpoint Name (e.g., Before refactoring)")
            self.btn_create_checkpoint = QPushButton("Create")
            self.btn_create_checkpoint.clicked.connect(self.create_checkpoint)
            create_layout.addWidget(self.checkpoint_name_input)
            create_layout.addWidget(self.btn_create_checkpoint)
            checkpoint_layout.addLayout(create_layout)

            checkpoint_layout.addWidget(QLabel("Existing Checkpoints:"))
            self.checkpoint_list = QListWidget()
            self.checkpoint_list.setFixedHeight(150)
            checkpoint_layout.addWidget(self.checkpoint_list)

            btn_layout = QHBoxLayout()
            self.btn_restore_checkpoint = QPushButton("Restore Selected")
            self.btn_restore_checkpoint.clicked.connect(self.restore_checkpoint)
            self.btn_delete_checkpoint = QPushButton("Delete Selected")
            self.btn_delete_checkpoint.clicked.connect(self.delete_checkpoint)
            btn_layout.addWidget(self.btn_restore_checkpoint)
            btn_layout.addWidget(self.btn_delete_checkpoint)
            checkpoint_layout.addLayout(btn_layout)

            checkpoint_group.setLayout(checkpoint_layout)
            self.scroll_layout.addWidget(checkpoint_group)

        # --- Conductor Settings ---
        conductor_group = QGroupBox("Conductor Settings")
        conductor_layout = QVBoxLayout()
        conductor_layout.addWidget(QLabel("Conductor Extension Path:"))
        path_layout = QHBoxLayout()
        self.conductor_path_input = QLineEdit(self.config.get("conductor_path", ""))
        self.conductor_path_input.setReadOnly(True)
        btn_browse_conductor = QPushButton("Browse")
        btn_browse_conductor.clicked.connect(self.browse_conductor_path)
        path_layout.addWidget(self.conductor_path_input)
        path_layout.addWidget(btn_browse_conductor)
        conductor_layout.addLayout(path_layout)
        self.btn_open_conductor = QPushButton("🚀 Open Conductor Orchestrator")
        self.btn_open_conductor.clicked.connect(self.open_conductor_dialog)
        conductor_layout.addWidget(self.btn_open_conductor)
        conductor_group.setLayout(conductor_layout)
        self.scroll_layout.addWidget(conductor_group)

        # --- Thinking Parameters ---
        thinking_group = QGroupBox("Advanced Thinking")
        thinking_layout = QVBoxLayout()
        self.chk_thinking = QCheckBox("Enable Thinking Process")
        self.chk_thinking.setChecked(self.config.get("thinking_enabled", False))
        thinking_layout.addWidget(self.chk_thinking)
        thinking_budget_layout = QHBoxLayout()
        thinking_budget_layout.addWidget(QLabel("Thinking Budget (Tokens):"))
        self.spin_thinking_budget = QSpinBox()
        self.spin_thinking_budget.setRange(1024, 65536)
        self.spin_thinking_budget.setSingleStep(1024)
        self.spin_thinking_budget.setValue(self.config.get("thinking_budget", 4096))
        self.spin_thinking_budget.setSuffix(" tokens")
        thinking_budget_layout.addWidget(self.spin_thinking_budget)
        thinking_layout.addLayout(thinking_budget_layout)
        self.btn_save_thinking = QPushButton("Save Thinking Settings")
        self.btn_save_thinking.clicked.connect(self.save_thinking_params)
        thinking_layout.addWidget(self.btn_save_thinking)
        thinking_group.setLayout(thinking_layout)
        self.scroll_layout.addWidget(thinking_group)

        # --- Generation Parameters ---
        param_group = QGroupBox("Generation Parameters")
        param_layout = QVBoxLayout()
        hbox_params = QHBoxLayout()
        temp_layout = QVBoxLayout()
        temp_layout.addWidget(QLabel("Temperature:"))
        self.spin_temp = QDoubleSpinBox()
        self.spin_temp.setRange(0.0, 2.0)
        self.spin_temp.setValue(self.config.get("temperature", 0.8))
        temp_layout.addWidget(self.spin_temp)
        hbox_params.addLayout(temp_layout)
        top_p_layout = QVBoxLayout()
        top_p_layout.addWidget(QLabel("Top P:"))
        self.spin_top_p = QDoubleSpinBox()
        self.spin_top_p.setRange(0.0, 1.0)
        self.spin_top_p.setValue(self.config.get("top_p", 0.95))
        top_p_layout.addWidget(self.spin_top_p)
        hbox_params.addLayout(top_p_layout)

        top_k_layout = QVBoxLayout()
        top_k_layout.addWidget(QLabel("Top K:"))
        self.spin_top_k = QSpinBox()
        self.spin_top_k.setRange(1, 100)
        self.spin_top_k.setValue(self.config.get("top_k", 40))
        top_k_layout.addWidget(self.spin_top_k)
        hbox_params.addLayout(top_k_layout)

        param_layout.addLayout(hbox_params)
        turns_layout = QVBoxLayout()
        turns_layout.addWidget(QLabel("Max Agent Turns:"))
        hbox_turns = QHBoxLayout()
        self.spin_max_turns = QSpinBox()
        self.spin_max_turns.setRange(5, 50)
        self.spin_max_turns.setValue(self.config.get("max_turns", 20))
        self.spin_max_turns.setSuffix(" turns")
        hbox_turns.addWidget(self.spin_max_turns)
        turns_layout.addLayout(hbox_turns)
        param_layout.addLayout(turns_layout)
        self.btn_save_params = QPushButton("Save Parameters")
        self.btn_save_params.clicked.connect(self.save_generation_params)
        param_layout.addWidget(self.btn_save_params)
        param_group.setLayout(param_layout)
        self.scroll_layout.addWidget(param_group)

        # 4. System Instruction
        self.scroll_layout.addWidget(QLabel("System Instructions:"))
        self.txt_system_instruction = QTextEdit()
        self.txt_system_instruction.setFixedHeight(120)
        self.txt_system_instruction.setText(self.config.get("system_instruction", ""))
        self.scroll_layout.addWidget(self.txt_system_instruction)
        self.btn_save_sys = QPushButton("Save System Instruction")
        self.btn_save_sys.clicked.connect(self.save_system_instruction)
        self.scroll_layout.addWidget(self.btn_save_sys)

        self.scroll_layout.addStretch(1)
        self.scroll_area.setWidget(self.scroll_widget)
        main_layout.addWidget(self.scroll_area)

        bottom_frame = QFrame()
        bottom_layout = QHBoxLayout(bottom_frame)
        bottom_layout.setContentsMargins(20, 10, 20, 20)
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        bottom_layout.addStretch(1)
        bottom_layout.addWidget(btn_close)
        main_layout.addWidget(bottom_frame)

    def refresh_checkpoints(self):
        if not self.checkpoint_manager:
            return
        self.checkpoint_list.clear()
        checkpoints = self.checkpoint_manager.list_checkpoints()
        for cp in reversed(checkpoints):
            item = QListWidgetItem(f"{cp['name']} ({cp['timestamp'][:19].replace('T', ' ')})")
            item.setData(Qt.ItemDataRole.UserRole, cp["id"])
            self.checkpoint_list.addItem(item)

    def create_checkpoint(self):
        name = self.checkpoint_name_input.text().strip()
        if not name:
            QMessageBox.warning(self, "Warning", "Please enter a name for the checkpoint.")
            return

        cp = self.checkpoint_manager.create_checkpoint(name)
        if cp:
            QMessageBox.information(self, "Success", f"Checkpoint '{name}' created successfully.")
            self.checkpoint_name_input.clear()
            self.refresh_checkpoints()
        else:
            QMessageBox.critical(self, "Error", "Failed to create checkpoint.")

    def restore_checkpoint(self):
        selected_item = self.checkpoint_list.currentItem()
        if not selected_item:
            QMessageBox.warning(self, "Warning", "Please select a checkpoint to restore.")
            return

        checkpoint_id = selected_item.data(Qt.ItemDataRole.UserRole)
        checkpoint_name = selected_item.text()

        reply = QMessageBox.question(
            self,
            "Restore Checkpoint",
            f"Are you sure you want to restore '{checkpoint_name}'?\n\n"
            "This will overwrite current project files. A safety backup will be created.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            if self.checkpoint_manager.restore_checkpoint(checkpoint_id):
                QMessageBox.information(
                    self,
                    "Success",
                    "Project restored successfully. The application will now close to apply changes.",
                )
                self.main_window.close()
            else:
                QMessageBox.critical(self, "Error", "Failed to restore checkpoint.")

    def delete_checkpoint(self):
        selected_item = self.checkpoint_list.currentItem()
        if not selected_item:
            QMessageBox.warning(self, "Warning", "Please select a checkpoint to delete.")
            return

        checkpoint_id = selected_item.data(Qt.ItemDataRole.UserRole)
        checkpoint_name = selected_item.text()

        reply = QMessageBox.question(
            self,
            "Delete Checkpoint",
            f"Are you sure you want to delete '{checkpoint_name}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            if self.checkpoint_manager.delete_checkpoint(checkpoint_id):
                self.refresh_checkpoints()
            else:
                QMessageBox.critical(self, "Error", "Failed to delete checkpoint.")

    def browse_conductor_path(self):
        path = QFileDialog.getExistingDirectory(
            self, "Select Conductor Extension Directory", self.conductor_path_input.text()
        )
        if path:
            self.conductor_path_input.setText(path)
            self.config["conductor_path"] = path
            save_json(AppConfig.CONFIG_FILE, self.config)
            if hasattr(self.main_window, "conductor_manager"):
                from gemini_agent.core.conductor_manager import ConductorManager

                self.main_window.conductor_manager = ConductorManager(extension_path=path)

    def open_conductor_dialog(self):
        from gemini_agent.ui.conductor_dialog import ConductorDialog

        dialog = ConductorDialog(self.main_window, self.main_window.conductor_manager)
        dialog.exec()

    def save_general_settings(self):
        self.config["api_key"] = self.api_input.text().strip()
        current_index = self.model_combo.currentIndex()
        model_id = self.model_combo.itemData(current_index)
        self.config["model"] = model_id
        self.config["use_search"] = self.chk_grounding.isChecked()
        save_json(AppConfig.CONFIG_FILE, self.config)

    def on_theme_changed(self, theme_name):
        self.config["theme"] = theme_name
        save_json(AppConfig.CONFIG_FILE, self.config)
        self.main_window.apply_theme()
        self.apply_theme_to_dialog()

    def save_thinking_params(self):
        self.config["thinking_enabled"] = self.chk_thinking.isChecked()
        self.config["thinking_budget"] = self.spin_thinking_budget.value()
        save_json(AppConfig.CONFIG_FILE, self.config)
        QMessageBox.information(self, "Saved", "Thinking parameters saved.")

    def save_generation_params(self):
        self.config["temperature"] = round(self.spin_temp.value(), 2)
        self.config["top_p"] = round(self.spin_top_p.value(), 2)
        self.config["top_k"] = self.spin_top_k.value()
        self.config["max_turns"] = self.spin_max_turns.value()
        save_json(AppConfig.CONFIG_FILE, self.config)
        QMessageBox.information(self, "Saved", "Parameters saved.")

    def save_system_instruction(self):
        self.config["system_instruction"] = self.txt_system_instruction.toPlainText().strip()
        save_json(AppConfig.CONFIG_FILE, self.config)
        QMessageBox.information(self, "Saved", "Instructions saved.")

    def apply_theme_to_dialog(self):
        is_dark = self.config.get("theme") == "Dark"
        bg = "#1E1F20" if is_dark else "#FFFFFF"
        fg = "#E3E3E3" if is_dark else "#000000"
        input_bg = "#282A2C" if is_dark else "#F0F0F0"
        scroll_bg = "#1A1A1A" if is_dark else "#F5F5F5"

        self.setStyleSheet(f"""
            QDialog {{ background-color: {bg}; color: {fg}; }} 
            QLabel, QCheckBox, QGroupBox {{ color: {fg}; }}
            QTextEdit, QLineEdit, QComboBox, QDoubleSpinBox, QSpinBox, QListWidget {{ 
                background-color: {input_bg}; color: {fg}; border: 1px solid #444; padding: 5px;
            }}
            QPushButton {{
                background-color: {"#2D2E30" if is_dark else "#E0E0E0"};
                color: {fg};
                border: 1px solid {"#444" if is_dark else "#CCC"};
                border-radius: 4px;
                padding: 8px 16px;
            }}
            QPushButton:hover {{ background-color: {"#3C4043" if is_dark else "#D0D0D0"}; }}
            QGroupBox {{
                font-weight: bold;
                border: 1px solid {"#444" if is_dark else "#CCC"};
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }}
            QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 5px 0 5px; }}
            QScrollArea {{ background-color: {scroll_bg}; border: none; }}
            QScrollArea > QWidget > QWidget {{ background-color: {scroll_bg}; }}
        """)

        btn_primary_style = f"""
            QPushButton {{
                background-color: {"#0B57D0" if is_dark else "#1A73E8"};
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }}
            QPushButton:hover {{ background-color: {"#1C71D8" if is_dark else "#1669D6"}; }}
        """
        self.btn_save_params.setStyleSheet(btn_primary_style)
        self.btn_save_sys.setStyleSheet(btn_primary_style)
        self.btn_save_thinking.setStyleSheet(btn_primary_style)
        self.btn_open_conductor.setStyleSheet(btn_primary_style)
        self.btn_create_checkpoint.setStyleSheet(btn_primary_style)
        self.btn_restore_checkpoint.setStyleSheet(btn_primary_style)
