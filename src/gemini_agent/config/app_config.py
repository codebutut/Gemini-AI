import json
import logging
import os
import threading
from enum import Enum
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Load environment variables immediately upon import
load_dotenv()

# --- Enums ---


class Theme(Enum):
    DARK = "Dark"
    LIGHT = "Light"


class Role(Enum):
    USER = "user"
    MODEL = "model"
    SYSTEM = "system"


# --- Constants & Registry ---


class ModelRegistry:
    # Latest Gemini Models with Technical API Identifiers
    GEMINI_MODELS: list[tuple[str, str]] = [
        # Gemini 3 Series (Latest)
        ("Gemini 3 Pro (Preview)", "gemini-3-pro-preview"),
        ("Gemini 3 Flash (Preview)", "gemini-3-flash-preview"),
        # Gemini 2.5 Series (Current Stable)
        ("Gemini 2.5 Pro", "gemini-2.5-pro"),
        ("Gemini 2.5 Flash", "gemini-2.5-flash"),
        # Dynamic Latest Pointers
        ("Gemini Pro (Latest)", "gemini-pro-latest"),
        ("Gemini Flash (Latest)", "gemini-flash-latest"),
        # Specialized Models
        ("Gemini Nano Banana Pro (Image Preview)", "gemini-3-pro-image-preview"),
        # Legacy Models (for backward compatibility)
        ("Gemini 2.0 Flash (Thinking)", "gemini-2.0-flash-thinking-exp-01-21"),
        ("Gemini 2.0 Flash", "gemini-2.0-flash"),
        ("Gemini 2.0 Pro (Experimental)", "gemini-2.0-pro-exp-02-05"),
        ("Gemini 1.5 Pro", "gemini-1.5-pro"),
        ("Gemini 1.5 Flash", "gemini-1.5-flash"),
        ("Gemini 1.5 Flash 8B", "gemini-1.5-flash-8b"),
    ]

    # Pricing per 1M tokens (Input, Output) in USD
    MODEL_PRICING: dict[str, tuple[float, float]] = {
        "gemini-3-pro-preview": (5.00, 15.00),
        "gemini-3-flash-preview": (0.20, 0.80),
        "gemini-2.5-pro": (3.50, 10.50),
        "gemini-2.5-flash": (0.10, 0.40),
        "gemini-pro-latest": (3.50, 10.50),
        "gemini-flash-latest": (0.10, 0.40),
        "gemini-3-pro-image-preview": (5.00, 15.00),
        "gemini-2.0-flash-thinking-exp-01-21": (0.10, 0.40),
        "gemini-2.0-flash": (0.10, 0.40),
        "gemini-2.0-pro-exp-02-05": (3.50, 10.50),
        "gemini-1.5-pro": (3.50, 10.50),
        "gemini-1.5-flash": (0.075, 0.30),
        "gemini-1.5-flash-8b": (0.0375, 0.15),
    }

    # Specific attributes for models (Rate limits, capabilities)
    MODEL_ATTRIBUTES: dict[str, dict[str, Any]] = {
        "gemini-3-flash-preview": {
            "rate_limit_requests": 100,  # High throughput for Flash
            "rate_limit_period": 60,
            "context_window": 2_000_000,
            "supports_thinking": True,
            "is_flash": True,
            "description": "Next-gen high-speed multimodal model.",
        },
        "gemini-3-pro-preview": {
            "rate_limit_requests": 50,
            "rate_limit_period": 60,
            "context_window": 2_000_000,
            "supports_thinking": True,
            "is_flash": False,
        },
        "gemini-2.0-flash-thinking-exp-01-21": {
            "rate_limit_requests": 60,
            "rate_limit_period": 60,
            "context_window": 1_000_000,
            "supports_thinking": True,
            "is_flash": True,
        },
        # Default fallback
        "default": {
            "rate_limit_requests": 20,
            "rate_limit_period": 60,
            "context_window": 1_000_000,
            "supports_thinking": False,
            "is_flash": False,
        },
    }

    DEFAULT_MODEL_ID = "gemini-3-flash-preview"

    @classmethod
    def get_attributes(cls, model_id: str) -> dict[str, Any]:
        """Retrieves attributes for a specific model, falling back to default."""
        return cls.MODEL_ATTRIBUTES.get(model_id, cls.MODEL_ATTRIBUTES["default"])


class AppConfig:
    """
    Manages application configuration with validation and persistence.
    """

    BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
    HISTORY_FILE = BASE_DIR / "history.json"
    CONFIG_FILE = BASE_DIR / "settings.json"
    RESOURCES_DIR = Path(__file__).resolve().parent.parent / "resources"
    SYSTEM_INSTRUCTION_FILE = RESOURCES_DIR / "system_instruction.txt"

    DANGEROUS_TOOLS = [
        "write_file",
        "run_python",
        "start_application",
        "kill_process",
        "git_operation",
        "install_package",
        "execute_python_with_env",
        "refactor_code",
        "generate_tests",
        "update_plan",
        "update_specs",
        "generate_image",
        "analyze_image",
        "query_database",
        "execute_command",
    ]

    DEFAULT_CONFIG = {
        "api_key": "",
        "model": ModelRegistry.DEFAULT_MODEL_ID,
        "theme": Theme.DARK.value,
        "use_search": False,
        "system_instruction": "",  # Loaded dynamically
        "temperature": 0.8,
        "top_p": 0.95,
        "top_k": 40,
        "max_turns": 20,
        "thinking_enabled": False,
        "thinking_budget": 4096,
        "conductor_path": str(BASE_DIR / "conductor"),
        "recent_items": [],  # List of recently opened files/folders
    }

    def __init__(self, config_file: Path | None = None):
        self.config_file = config_file or self.CONFIG_FILE
        self._config = self.load()
        self._lock = threading.Lock()

        # Sync API Key with Environment
        env_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if env_key and not self._config.get("api_key"):
            self._config["api_key"] = env_key
        elif self._config.get("api_key"):
            os.environ["GOOGLE_API_KEY"] = self._config["api_key"]
            os.environ["GEMINI_API_KEY"] = self._config["api_key"]

        if not self._config.get("system_instruction"):
            self._config["system_instruction"] = self._load_system_instruction()

    def _load_system_instruction(self) -> str:
        """Loads the default system instruction from file."""
        if self.SYSTEM_INSTRUCTION_FILE.exists():
            try:
                return self.SYSTEM_INSTRUCTION_FILE.read_text(encoding="utf-8")
            except OSError as e:
                logging.error(
                    f"Failed to load system instruction from {self.SYSTEM_INSTRUCTION_FILE}: {e}"
                )

        return "You are a helpful AI assistant."

    def load(self) -> dict[str, Any]:
        if not self.config_file.exists():
            return self.DEFAULT_CONFIG.copy()
        try:
            with self.config_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
                # Ensure model is valid or fallback
                config = {**self.DEFAULT_CONFIG, **data}
                # Optional: Validate model exists in registry, else warn?
                return config
        except (json.JSONDecodeError, OSError) as e:
            logging.error(f"Failed to load {self.config_file}: {e}")
            return self.DEFAULT_CONFIG.copy()

    def save(self, sync: bool = False) -> None:
        """Saves configuration. By default, saves asynchronously to avoid blocking the UI."""
        config_copy = self._config.copy()
        if sync:
            self._save_sync(config_copy)
        else:
            threading.Thread(
                target=self._save_sync, args=(config_copy,), daemon=True
            ).start()

    def _save_sync(self, config_data: dict[str, Any]) -> None:
        with self._lock:
            try:
                with self.config_file.open("w", encoding="utf-8") as f:
                    json.dump(config_data, f, indent=4, ensure_ascii=False)
            except OSError as e:
                logging.error(f"Failed to save {self.config_file}: {e}")

    def get(self, key: str, default: Any = None) -> Any:
        return self._config.get(key, default)

    def set(self, key: str, value: Any, sync: bool = False) -> None:
        self._config[key] = value
        self.save(sync=sync)

    @property
    def api_key(self) -> str:
        return self.get("api_key", "")

    @api_key.setter
    def api_key(self, value: str):
        self.set("api_key", value)
        # Update environment variables immediately
        os.environ["GOOGLE_API_KEY"] = value
        os.environ["GEMINI_API_KEY"] = value

    @property
    def model(self) -> str:
        return self.get("model", ModelRegistry.DEFAULT_MODEL_ID)

    @model.setter
    def model(self, value: str):
        self.set("model", value)

    @property
    def theme(self) -> str:
        return self.get("theme", Theme.DARK.value)

    @theme.setter
    def theme(self, value: str):
        self.set("theme", value)

    @property
    def conductor_path(self) -> str:
        path = self.get("conductor_path")
        if path and Path(path).exists():
            return path
        return str(self.BASE_DIR / "conductor")

    @conductor_path.setter
    def conductor_path(self, value: str):
        self.set("conductor_path", value)

    @property
    def recent_items(self) -> list[dict[str, Any]]:
        return self.get("recent_items", [])

    @recent_items.setter
    def recent_items(self, value: list[dict[str, Any]]):
        self.set("recent_items", value)


def setup_logging(level=logging.INFO):
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")


def load_json(filepath: Path, default: Any = None) -> Any:
    if not filepath.exists():
        return default if default is not None else {}
    try:
        with filepath.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logging.error(f"Failed to load {filepath}: {e}")
        return default if default is not None else {}


def save_json(filepath: Path, data: Any) -> None:
    try:
        with filepath.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
    except OSError as e:
        logging.error(f"Failed to save {filepath}: {e}")
