import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any


class RecentManager:
    """
    Manages the list of recently accessed files and projects.
    """

    def __init__(self, storage_path: str = "recent.json", max_items: int = 10):
        self.storage_path = storage_path
        self.max_items = max_items
        self.recent_items: list[dict[str, Any]] = []
        self.load()

    def load(self) -> None:
        """Loads recent items from the JSON file."""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, "r") as f:
                    data = json.load(f)
                    self.recent_items = data.get("recent_items", [])
            except (json.JSONDecodeError, IOError):
                self.recent_items = []
        else:
            self.recent_items = []

    def save(self) -> None:
        """Saves recent items to the JSON file."""
        try:
            with open(self.storage_path, "w") as f:
                json.dump({"recent_items": self.recent_items}, f, indent=4)
        except IOError:
            pass

    def add_item(self, path: str, item_type: str = "file") -> None:
        """
        Adds an item to the recent list. If it already exists, moves it to the top.

        Args:
            path: The absolute path to the file or project.
            item_type: Either 'file' or 'project'.
        """
        path = os.path.abspath(path)
        name = os.path.basename(path)

        # Remove if already exists to move it to the top
        self.recent_items = [item for item in self.recent_items if item["path"] != path]

        new_item = {
            "path": path,
            "name": name,
            "timestamp": datetime.now().isoformat(),
            "type": item_type,
        }

        self.recent_items.insert(0, new_item)

        # Limit the number of items
        if len(self.recent_items) > self.max_items:
            self.recent_items = self.recent_items[: self.max_items]

        self.save()

    def get_recent_items(self) -> list[dict[str, Any]]:
        """Returns the list of recent items."""
        return self.recent_items

    def clear(self) -> None:
        """Clears the recent items list."""
        self.recent_items = []
        self.save()
