import json
import logging
import zipfile
from pathlib import Path
from typing import Any, Dict

from .models import Session

logger = logging.getLogger(__name__)


class Exporter:
    """
    Handles exporting sessions to Markdown and managing history backups in ZIP format.
    """

    @staticmethod
    def session_to_markdown(session: Session) -> str:
        """
        Converts a single session's data to a Markdown string.

        Args:
            session: The Session object.

        Returns:
            str: The session data formatted as Markdown.
        """
        title = session.title
        created_at = session.created_at
        messages = session.messages
        plan = session.plan
        specs = session.specs

        md = f"# {title}\n\n"
        md += f"*Created at: {created_at}*\n\n"

        if plan:
            md += "## Plan\n\n"
            md += f"{plan}\n\n"

        if specs:
            md += "## Specifications\n\n"
            md += f"{specs}\n\n"

        md += "## Chat History\n\n"
        for msg in messages:
            role = msg.role.capitalize()
            text = msg.text
            timestamp = msg.timestamp

            md += f"### {role} ({timestamp})\n\n"
            md += f"{text}\n\n"

            if msg.images:
                md += "**Attachments (Images):**\n"
                for img in msg.images:
                    md += f"- {img}\n"
                md += "\n"

        return md

    @staticmethod
    def export_to_file(session: Session, filepath: Path) -> bool:
        """
        Exports a session to a Markdown file.

        Args:
            session: The Session object.
            filepath: The destination Path.

        Returns:
            bool: True if export was successful, False otherwise.
        """
        try:
            content = Exporter.session_to_markdown(session)
            filepath.write_text(content, encoding="utf-8")
            return True
        except OSError as e:
            logger.error(f"Error exporting to file: {e}")
            return False

    @staticmethod
    def create_backup(sessions: Dict[str, Session], backup_path: Path) -> bool:
        """
        Creates a ZIP backup of all sessions, including raw JSON and individual Markdown files.

        Args:
            sessions: Dictionary of all session data.
            backup_path: The destination Path for the ZIP backup.

        Returns:
            bool: True if backup was successful, False otherwise.
        """
        try:
            with zipfile.ZipFile(backup_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                # Save the raw JSON
                data = {sid: s.model_dump() for sid, s in sessions.items()}
                zipf.writestr(
                    "history.json", json.dumps(data, indent=4, ensure_ascii=False)
                )

                # Save individual markdown files for convenience
                for session_id, session in sessions.items():
                    # Sanitize title for filename
                    safe_title = "".join(
                        [c for c in session.title if c.isalnum() or c in (" ", "_")]
                    ).rstrip()
                    safe_title = safe_title.replace(" ", "_")
                    filename = f"sessions/{safe_title}_{session_id[:8]}.md"
                    md_content = Exporter.session_to_markdown(session)
                    zipf.writestr(filename, md_content)
            return True
        except (zipfile.BadZipFile, OSError) as e:
            logger.error(f"Error creating backup: {e}")
            return False

    @staticmethod
    def restore_backup(backup_path: Path) -> Dict[str, Session]:
        """
        Restores sessions from a ZIP backup.

        Args:
            backup_path: The Path to the ZIP backup.

        Returns:
            Dict[str, Session]: The restored session data, or an empty dict if it failed.
        """
        try:
            with zipfile.ZipFile(backup_path, "r") as zipf:
                if "history.json" in zipf.namelist():
                    with zipf.open("history.json") as f:
                        data = json.load(f)
                        return {sid: Session(**sdata) for sid, sdata in data.items()}
        except (zipfile.BadZipFile, json.JSONDecodeError, OSError) as e:
            logger.error(f"Error restoring backup: {e}")
        return {}
