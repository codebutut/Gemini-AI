import logging

from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import QApplication


class ThemeManager:
    """
    Manages application themes and custom CSS overrides.
    """

    DARK_PALETTE = {
        "Window": "#131314",
        "WindowText": "#E3E3E3",
        "Base": "#1E1F20",
        "AlternateBase": "#282A2C",
        "ToolTipBase": "#E3E3E3",
        "ToolTipText": "#E3E3E3",
        "Text": "#E3E3E3",
        "Button": "#1E1F20",
        "ButtonText": "#E3E3E3",
        "BrightText": "#ff0000",
        "Link": "#4da6ff",
        "Highlight": "#004A77",
        "HighlightedText": "#C2E7FF",
    }

    LIGHT_PALETTE = {
        "Window": "#FFFFFF",
        "WindowText": "#1F1F1F",
        "Base": "#F0F4F9",
        "AlternateBase": "#E0E0E0",
        "ToolTipBase": "#1F1F1F",
        "ToolTipText": "#1F1F1F",
        "Text": "#1F1F1F",
        "Button": "#D3E3FD",
        "ButtonText": "#041E49",
        "BrightText": "#ff0000",
        "Link": "#0969da",
        "Highlight": "#D3E3FD",
        "HighlightedText": "#041E49",
    }

    DARK_STYLE = """
    QMainWindow, QWidget { background-color: #131314; color: #E3E3E3; }
    QTextEdit, QLineEdit { background-color: #282A2C; border: none; border-radius: 12px; padding: 10px; color: #E3E3E3; }
    QListWidget { background-color: #1E1F20; border: none; }
    QListWidget::item:selected { background-color: #004A77; color: #C2E7FF; border-radius: 20px; }
    QLabel { color: #E3E3E3; }
    QPushButton#BtnNewChat { background-color: #1E1F20; color: #E3E3E3; border-radius: 15px; font-weight: bold; padding: 12px; text-align: left; padding-left: 20px; border: none; }
    QPushButton#BtnNewChat:hover { background-color: #2D2E30; }
    
    QPushButton#BtnTerminal, QPushButton#BtnSettings { 
        background-color: #1E1F20; 
        color: #E3E3E3; 
        border-radius: 16px; 
        padding: 0px 15px; 
        border: 1px solid #333;
        font-size: 12px;
    }
    QPushButton#BtnTerminal:hover, QPushButton#BtnSettings:hover { background-color: #2D2E30; }
    QPushButton#BtnTerminal:checked { background-color: #004A77; color: #C2E7FF; }

    QFrame#Sidebar { background-color: #1E1F20; border-right: none; }
    QFrame#SidebarContainer { background-color: #1E1F20; border-right: 1px solid #333; }
    QTabWidget::pane { border: none; }
    QTabBar::tab { background: #1E1F20; color: #888; padding: 10px 20px; border: none; }
    QTabBar::tab:selected { color: #C2E7FF; border-bottom: 2px solid #C2E7FF; }
    QScrollArea { border: none; background: transparent; }
    QScrollBar:vertical { border: none; background: transparent; width: 8px; margin: 0px; }
    QScrollBar::handle:vertical { background: #333; min-height: 20px; border-radius: 4px; }
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
    QSplitter::handle { background-color: #333; }
    """

    LIGHT_STYLE = """
    QMainWindow, QWidget { background-color: #FFFFFF; color: #1F1F1F; }
    QTextEdit, QLineEdit { background-color: #F0F4F9; border: none; border-radius: 12px; padding: 10px; color: #1F1F1F; }
    QListWidget { background-color: #F0F4F9; border: none; }
    QListWidget::item:selected { background-color: #D3E3FD; color: #041E49; border-radius: 20px; }
    QLabel { color: #1F1F1F; }
    QPushButton#BtnNewChat { background-color: #D3E3FD; color: #041E49; border-radius: 15px; font-weight: bold; padding: 12px; text-align: left; padding-left: 20px; border: none; }
    QPushButton#BtnNewChat:hover { background-color: #C2E7FF; }

    QPushButton#BtnTerminal, QPushButton#BtnSettings { 
        background-color: #F0F4F9; 
        color: #1F1F1F; 
        border-radius: 16px; 
        padding: 0px 15px; 
        border: 1px solid #DDD;
        font-size: 12px;
    }
    QPushButton#BtnTerminal:hover, QPushButton#BtnSettings:hover { background-color: #E0E0E0; }
    QPushButton#BtnTerminal:checked { background-color: #D3E3FD; color: #041E49; }

    QFrame#Sidebar { background-color: #F0F4F9; border-right: none; }
    QFrame#SidebarContainer { background-color: #F0F4F9; border-right: 1px solid #DDD; }
    QTabWidget::pane { border: none; }
    QTabBar::tab { background: #F0F4F9; color: #555; padding: 10px 20px; border: none; }
    QTabBar::tab:selected { color: #041E49; border-bottom: 2px solid #041E49; }
    QScrollArea { border: none; background: transparent; }
    QScrollBar:vertical { border: none; background: transparent; width: 8px; margin: 0px; }
    QScrollBar::handle:vertical { background: #CCC; min-height: 20px; border-radius: 4px; }
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
    QSplitter::handle { background-color: #DDD; }
    """

    def __init__(self, app: QApplication):
        self.app = app
        self.current_theme = "Dark"

    def apply_theme(self, theme_name: str):
        """Applies the specified theme to the application."""
        self.current_theme = theme_name
        palette = QPalette()

        theme_data = self.DARK_PALETTE if theme_name == "Dark" else self.LIGHT_PALETTE

        for role_name, color_hex in theme_data.items():
            role = getattr(QPalette.ColorRole, role_name)
            palette.setColor(role, QColor(color_hex))

        self.app.setPalette(palette)

        style = self.DARK_STYLE if theme_name == "Dark" else self.LIGHT_STYLE
        self.app.setStyleSheet(style)
        logging.info(f"Applied theme: {theme_name}")

    def get_bubble_style(self, is_user: bool) -> str:
        """Returns the CSS style for chat bubbles based on the current theme."""
        return self.get_bubble_style_static(self.current_theme, is_user)

    @staticmethod
    def get_bubble_style_static(theme_mode: str, is_user: bool) -> str:
        """Returns the CSS style for chat bubbles based on the provided theme mode."""
        # Reduced padding for a more compact look
        if theme_mode == "Dark":
            if is_user:
                return "background-color: #2D2E30; border: 1px solid #4CAF50; border-radius: 12px; padding: 6px 12px; color: #E3E3E3;"
            else:
                return "background-color: #1E1F20; border: 1px solid #2196F3; border-radius: 12px; padding: 8px 16px; color: #E3E3E3;"
        else:
            if is_user:
                return "background-color: #E8F5E9; border: 1px solid #4CAF50; border-radius: 12px; padding: 6px 12px; color: #1F1F1F;"
            else:
                return "background-color: #E3F2FD; border: 1px solid #2196F3; border-radius: 12px; padding: 8px 16px; color: #1F1F1F;"

    def get_text_color(self) -> str:
        return "#E3E3E3" if self.current_theme == "Dark" else "#1F1F1F"

    def get_link_color(self) -> str:
        return "#4da6ff" if self.current_theme == "Dark" else "#0969da"
