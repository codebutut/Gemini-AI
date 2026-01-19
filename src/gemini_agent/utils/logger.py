import json
import logging
import sys
from datetime import datetime
from typing import Any


class StructuredFormatter(logging.Formatter):
    """
    Custom formatter for structured logging.
    Outputs logs in a consistent format, optionally as JSON.
    """

    def __init__(self, use_json: bool = False):
        super().__init__()
        self.use_json = use_json

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "module": record.module,
            "message": record.getMessage(),
        }

        if hasattr(record, "extra_data"):
            log_data.update(record.extra_data)

        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        if self.use_json:
            return json.dumps(log_data)

        # Fallback to a nice string format for console
        color = ""
        reset = "\033[0m"
        if record.levelno >= logging.ERROR:
            color = "\033[91m"  # Red
        elif record.levelno >= logging.WARNING:
            color = "\033[93m"  # Yellow
        elif record.levelno >= logging.INFO:
            color = "\033[94m"  # Blue

        return f"{color}[{log_data['timestamp']}] {log_data['level']:7} - {log_data['module']}: {log_data['message']}{reset}"


def setup_agent_logging(level: int = logging.INFO, log_file: str | None = None):
    """Sets up the global logging configuration."""
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(StructuredFormatter(use_json=False))
    root_logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(StructuredFormatter(use_json=True))
        root_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """Returns a logger with the given name."""
    return logging.getLogger(name)


class AgentLoggerAdapter(logging.LoggerAdapter):
    """Adapter to easily add extra data to log records."""

    def process(self, msg: Any, kwargs: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        extra = self.extra.copy()
        if "extra" in kwargs:
            extra.update(kwargs.pop("extra"))
        kwargs["extra"] = {"extra_data": extra}
        return msg, kwargs
