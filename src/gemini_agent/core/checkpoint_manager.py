import json
import logging
import os
import shutil
import uuid
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Manages project checkpoints (save states) by creating ZIP archives of the workspace.
    """

    def __init__(self, root_dir: str = ".") -> None:
        """
        Initializes the CheckpointManager.

        Args:
            root_dir: The project root directory.
        """
        self.root_dir: Path = Path(root_dir).resolve()
        self.checkpoint_dir: Path = self.root_dir / ".checkpoints"
        self.metadata_file: Path = self.checkpoint_dir / "checkpoints.json"
        self.exclude_dirs: set[str] = {
            "env",
            "venv",
            ".venv",
            "__pycache__",
            ".git",
            ".checkpoints",
            "node_modules",
            ".pytest_cache",
            ".mypy_cache",
            ".ruff_cache",
            "dist",
            "build",
            "*.egg-info",
        }
        self.exclude_files: set[str] = {
            "history.json",
            "checkpoints.json",
            ".DS_Store",
            "Thumbs.db",
        }

        self._ensure_dir()

    def _ensure_dir(self) -> None:
        """Ensures the checkpoint directory exists and initializes metadata."""
        if not self.checkpoint_dir.exists():
            self.checkpoint_dir.mkdir(parents=True)
        if not self.metadata_file.exists():
            self._save_metadata([])

    def _load_metadata(self) -> list[dict[str, Any]]:
        """
        Loads checkpoint metadata from the JSON file.

        Returns:
            List[Dict[str, Any]]: A list of checkpoint metadata dictionaries.
        """
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file, encoding="utf-8") as f:
                    return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.error(f"Error loading checkpoint metadata: {e}")
        return []

    def _save_metadata(self, checkpoints: list[dict[str, Any]]) -> None:
        """
        Saves checkpoint metadata to the JSON file.

        Args:
            checkpoints: A list of checkpoint metadata dictionaries.
        """
        try:
            with open(self.metadata_file, "w", encoding="utf-8") as f:
                json.dump(checkpoints, f, indent=4)
        except OSError as e:
            logger.error(f"Error saving checkpoint metadata: {e}")

    def create_checkpoint(self, name: str) -> dict[str, Any] | None:
        """
        Creates a new checkpoint of the project.

        Args:
            name: A descriptive name for the checkpoint.

        Returns:
            Optional[Dict[str, Any]]: The created checkpoint metadata, or None if it failed.
        """
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        checkpoint_id = f"{timestamp_str}_{unique_id}"
        filename = f"checkpoint_{checkpoint_id}.zip"
        filepath = self.checkpoint_dir / filename

        try:
            with zipfile.ZipFile(filepath, "w", zipfile.ZIP_DEFLATED) as zipf:
                for root, dirs, files in os.walk(self.root_dir):
                    # Filter directories
                    dirs[:] = [d for d in dirs if not self._is_excluded(d, is_dir=True)]

                    rel_root = Path(root).relative_to(self.root_dir)

                    for file in files:
                        if self._is_excluded(file, is_dir=False):
                            continue
                        if file.endswith((".pyc", ".pyo", ".pyd")):
                            continue

                        file_path = Path(root) / file
                        arcname = rel_root / file
                        zipf.write(file_path, arcname)

            checkpoint = {
                "id": checkpoint_id,
                "name": name or f"Checkpoint {checkpoint_id}",
                "timestamp": datetime.now().isoformat(),
                "filename": filename,
            }

            checkpoints = self._load_metadata()
            checkpoints.append(checkpoint)
            self._save_metadata(checkpoints)
            return checkpoint

        except (zipfile.BadZipFile, OSError) as e:
            logger.error(f"Error creating checkpoint: {e}")
            if filepath.exists():
                filepath.unlink()
            return None

    def _is_excluded(self, name: str, is_dir: bool) -> bool:
        """Checks if a file or directory should be excluded."""
        import fnmatch

        patterns = self.exclude_dirs if is_dir else self.exclude_files
        return any(fnmatch.fnmatch(name, pattern) for pattern in patterns)

    def list_checkpoints(self) -> list[dict[str, Any]]:
        """
        Returns a list of all checkpoints.

        Returns:
            List[Dict[str, Any]]: List of all checkpoint metadata.
        """
        return self._load_metadata()

    def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """
        Deletes a checkpoint and its associated ZIP file.

        Args:
            checkpoint_id: The ID of the checkpoint to delete.

        Returns:
            bool: True if deleted successfully, False otherwise.
        """
        checkpoints = self._load_metadata()
        checkpoint = next((c for c in checkpoints if c["id"] == checkpoint_id), None)

        if not checkpoint:
            return False

        filepath = self.checkpoint_dir / checkpoint["filename"]
        try:
            if filepath.exists():
                filepath.unlink()

            checkpoints = [c for c in checkpoints if c["id"] != checkpoint_id]
            self._save_metadata(checkpoints)
            return True
        except OSError as e:
            logger.error(f"Error deleting checkpoint: {e}")
            return False

    def restore_checkpoint(self, checkpoint_id: str) -> bool:
        """
        Restores the project from a checkpoint.
        Creates a safety backup before restoring.

        Args:
            checkpoint_id: The ID of the checkpoint to restore.

        Returns:
            bool: True if restored successfully, False otherwise.
        """
        checkpoints = self._load_metadata()
        checkpoint = next((c for c in checkpoints if c["id"] == checkpoint_id), None)

        if not checkpoint:
            return False

        filepath = self.checkpoint_dir / checkpoint["filename"]
        if not filepath.exists():
            return False

        try:
            # 1. Create a safety backup before restoring
            self.create_checkpoint(f"Pre-restore backup ({checkpoint['name']})")

            # 2. Clean up current directory (except excluded ones)
            for item in self.root_dir.iterdir():
                if self._is_excluded(item.name, is_dir=item.is_dir()):
                    continue

                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()

            # 3. Extract checkpoint
            with zipfile.ZipFile(filepath, "r") as zipf:
                zipf.extractall(self.root_dir)

            return True
        except (zipfile.BadZipFile, OSError) as e:
            logger.error(f"Error restoring checkpoint: {e}")
            return False
