from PyQt6.QtCore import QSize
from PyQt6.QtGui import QMovie
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QProgressBar, QWidget

from gemini_agent.config.app_config import AppConfig


class StatusWidget(QWidget):
    """
    Widget to display status text, a 'Thinking' GIF animation, and rate limit telemetry.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.initUI()

    def initUI(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(15)

        # Status Label
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #888; font-style: italic;")
        layout.addWidget(self.status_label)

        # GIF Label
        self.gif_label = QLabel()
        self.gif_label.setFixedSize(30, 30)
        self.gif_label.setScaledContents(True)
        self.gif_label.hide()  # Hidden by default
        layout.addWidget(self.gif_label)

        # Load Movie
        gif_path = AppConfig.RESOURCES_DIR / "thinking.gif"
        if gif_path.exists():
            self.movie = QMovie(str(gif_path))
            self.movie.setScaledSize(QSize(30, 30))
            self.gif_label.setMovie(self.movie)
        else:
            self.movie = None

        layout.addStretch()

        # Rate Limit Container
        self.rate_limit_container = QWidget()
        rate_layout = QHBoxLayout(self.rate_limit_container)
        rate_layout.setContentsMargins(0, 0, 0, 0)
        rate_layout.setSpacing(8)

        self.rate_limit_label = QLabel("RPM: -/-")
        self.rate_limit_label.setStyleSheet("color: #aaa; font-size: 11px; font-family: 'Segoe UI', sans-serif;")

        self.rate_limit_bar = QProgressBar()
        self.rate_limit_bar.setFixedSize(120, 10)
        self.rate_limit_bar.setTextVisible(False)
        self.rate_limit_bar.setStyleSheet(
            """
            QProgressBar {
                border: 1px solid #333;
                border-radius: 5px;
                background-color: #1e1e1e;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                border-radius: 4px;
            }
        """
        )

        rate_layout.addWidget(self.rate_limit_label)
        rate_layout.addWidget(self.rate_limit_bar)
        layout.addWidget(self.rate_limit_container)

    def set_status(self, text: str):
        """Updates the status text safely."""
        try:
            if hasattr(self, "status_label") and self.status_label:
                self.status_label.setText(text)
        except (RuntimeError, AttributeError):
            pass

    def start_loading(self):
        """Shows and starts the GIF animation."""
        try:
            if self.movie and self.gif_label:
                self.gif_label.show()
                self.movie.start()
        except (RuntimeError, AttributeError):
            pass

    def stop_loading(self):
        """Stops and hides the GIF animation."""
        try:
            if self.movie and self.gif_label:
                self.movie.stop()
                self.gif_label.hide()
        except (RuntimeError, AttributeError):
            pass

    def update_rate_limit(self, remaining: int, limit: int):
        """Updates the rate limit indicator with real-time telemetry."""
        try:
            self.rate_limit_label.setText(f"RPM: {remaining}/{limit}")
            self.rate_limit_bar.setMaximum(limit)
            self.rate_limit_bar.setValue(remaining)

            # Dynamic coloring based on remaining capacity
            percentage = (remaining / limit) * 100 if limit > 0 else 0
            if percentage < 20:
                color = "#f44336"  # Red
            elif percentage < 50:
                color = "#ffeb3b"  # Yellow
            else:
                color = "#4CAF50"  # Green

            self.rate_limit_bar.setStyleSheet(
                f"""
                QProgressBar {{
                    border: 1px solid #333;
                    border-radius: 5px;
                    background-color: #1e1e1e;
                }}
                QProgressBar::chunk {{
                    background-color: {color};
                    border-radius: 4px;
                }}
            """
            )
        except (RuntimeError, AttributeError):
            pass
