import logging
import os
import shutil
import tarfile
import tempfile
import zipfile
from pathlib import Path


class AttachmentManager:
    """
    Manages file attachments, including processing folders/archives and cleaning up temp files.
    """

    def __init__(self) -> None:
        """Initializes the AttachmentManager and creates a temporary directory."""
        self.temp_dir: Path = Path(tempfile.mkdtemp(prefix="gemini_attachments_"))
        self.attachments: list[str] = []

    def add_attachment(self, file_path: str) -> list[str]:
        """
        Adds a file or folder to attachments.

        If the path is a folder or archive, it extracts/walks and returns a list
        of processed file paths.

        Args:
            file_path: The path to the file, folder, or archive.

        Returns:
            List[str]: A list of all processed file paths.
        """
        path = Path(file_path)
        processed_files: list[str] = []

        if path.is_dir():
            processed_files.extend(self._process_directory(path))
        elif zipfile.is_zipfile(path) or tarfile.is_tarfile(path):
            processed_files.extend(self._extract_archive(path))
        else:
            processed_files.append(str(path))

        self.attachments.extend(processed_files)
        return processed_files

    def _process_directory(self, dir_path: Path) -> list[str]:
        """
        Recursively finds all files in a directory.

        Args:
            dir_path: The directory path to scan.

        Returns:
            List[str]: A list of file paths.
        """
        files: list[str] = []
        for root, _, filenames in os.walk(dir_path):
            for filename in filenames:
                files.append(os.path.join(root, filename))
        return files

    def _extract_archive(self, archive_path: Path) -> list[str]:
        """
        Extracts zip/tar archives to temp dir and returns file paths.

        Args:
            archive_path: The path to the archive file.

        Returns:
            List[str]: A list of extracted file paths.
        """
        extract_path = self.temp_dir / archive_path.stem
        extract_path.mkdir(exist_ok=True)

        extracted_files: list[str] = []
        try:
            if zipfile.is_zipfile(archive_path):
                with zipfile.ZipFile(archive_path, "r") as zip_ref:
                    zip_ref.extractall(extract_path)
            elif tarfile.is_tarfile(archive_path):
                with tarfile.open(archive_path, "r") as tar_ref:
                    tar_ref.extractall(extract_path)

            extracted_files = self._process_directory(extract_path)
        except (zipfile.BadZipFile, tarfile.TarError, OSError) as e:
            logging.error(f"Failed to extract archive {archive_path}: {e}")

        return extracted_files

    def clear_attachments(self) -> None:
        """Clears the list of attachments."""
        self.attachments = []

    def cleanup(self) -> None:
        """Removes the temporary directory and all its contents."""
        if self.temp_dir.exists():
            try:
                shutil.rmtree(self.temp_dir)
            except OSError as e:
                logging.error(f"Failed to cleanup temp dir {self.temp_dir}: {e}")

    def get_attachments(self) -> list[str]:
        """
        Returns the current list of attachments.

        Returns:
            List[str]: The current list of attachment paths.
        """
        return self.attachments
