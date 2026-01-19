import logging
from pathlib import Path
from typing import Any

import tomllib


class ConductorManager:
    """
    Manages Conductor commands and templates.
    """

    def __init__(self, extension_path: str | None = None) -> None:
        """
        Initializes the ConductorManager.

        Args:
            extension_path: Optional path to the conductor extension.
                             Defaults to a 'conductor' directory sibling to 'core'.
        """
        # Use provided path if it exists, otherwise fallback to default project structure
        if extension_path and Path(extension_path).exists():
            self.extension_path = Path(extension_path)
        else:
            # Points to the root project directory's conductor folder
            # Path(__file__) is src/gemini_agent/core/conductor_manager.py
            # .parent.parent.parent.parent is the project root
            self.extension_path = Path(__file__).resolve().parent.parent.parent.parent / "conductor"

        self.commands_path: Path = self.extension_path / "commands" / "conductor"
        self.templates_path: Path = self.extension_path / "templates"
        self.commands: dict[str, dict[str, Any]] = {}
        self._load_commands()

    def _load_commands(self) -> None:
        """Loads all TOML command definitions from the commands directory."""
        if not self.commands_path.exists():
            logging.warning(f"Conductor commands path does not exist: {self.commands_path}")
            return

        for toml_file in self.commands_path.glob("*.toml"):
            try:
                with open(toml_file, "rb") as f:
                    data = tomllib.load(f)
                    self.commands[toml_file.stem] = data
            except (tomllib.TOMLDecodeError, OSError) as e:
                logging.error(f"Error loading conductor command {toml_file}: {e}")

    def get_command_prompt(self, command_name: str) -> str | None:
        """
        Returns the system prompt for a given command.

        Args:
            command_name: The name of the command.

        Returns:
            Optional[str]: The command prompt, or None if not found.
        """
        command = self.commands.get(command_name)
        if command:
            return command.get("prompt")
        return None

    def get_available_commands(self) -> list[str]:
        """
        Returns a list of available command names.

        Returns:
            List[str]: A list of command names.
        """
        return list(self.commands.keys())

    def is_setup(self, project_path: str = ".") -> bool:
        """
        Checks if the project has the required conductor files.

        Args:
            project_path: The project directory to check. Defaults to current dir.

        Returns:
            bool: True if project is set up with conductor, False otherwise.
        """
        conductor_dir = Path(project_path) / "conductor"
        required_files = ["product.md", "tech-stack.md", "workflow.md"]
        return conductor_dir.exists() and all((conductor_dir / f).exists() for f in required_files)

    def get_setup_state(self, project_path: str = ".") -> dict[str, Any] | None:
        """
        Loads the setup state from the project's conductor directory.

        Args:
            project_path: The project directory. Defaults to current dir.

        Returns:
            Optional[Dict[str, Any]]: The setup state dictionary, or None if not found.
        """
        state_file = Path(project_path) / "conductor" / "setup_state.json"
        if state_file.exists():
            import json

            try:
                with open(state_file) as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                logging.error(f"Error loading setup state from {state_file}: {e}")
        return None
