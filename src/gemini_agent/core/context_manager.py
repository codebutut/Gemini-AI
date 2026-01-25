import logging
import mimetypes
import time
from pathlib import Path

from google import genai
from google.genai import types


class ContextManager:
    """
    Handles preparation of history and current turn content for the Gemini API.
    """

    def __init__(self, client: genai.Client):
        self.client = client
        mimetypes.init()

    def prepare_history(
        self, history_context: list[dict[str, str]]
    ) -> list[types.Content]:
        """Convert simplified history to proper types.Content objects."""
        gemini_contents = []
        for msg in history_context:
            role = "user" if msg["role"] == "user" else "model"
            gemini_contents.append(
                types.Content(role=role, parts=[types.Part.from_text(text=msg["text"])])
            )
        return gemini_contents

    def prepare_current_turn(
        self,
        prompt: str,
        file_paths: list[str],
        current_plan: str = "",
        current_specs: str = "",
        prompt_flags: str = "",
    ) -> list[types.Part]:
        """Prepare parts for the current turn, including files and prompt."""
        current_turn_parts = []

        # Inject session-specific plan and specs if they exist
        if current_plan:
            current_turn_parts.append(
                types.Part.from_text(text=f"Current plan.md:\n{current_plan}")
            )
        if current_specs:
            current_turn_parts.append(
                types.Part.from_text(text=f"Current specs.md:\n{current_specs}")
            )

        for file_path in file_paths:
            part = self._load_file_content(file_path)
            if isinstance(part, str):
                current_turn_parts.append(types.Part.from_text(text=part))
            else:
                current_turn_parts.append(part)

        # Ensure prompt is not empty
        if not prompt:
            prompt = "[System: No user prompt provided, follow default instructions.]"

        # Append prompt flags if provided and not already in prompt
        if prompt_flags and prompt_flags.strip() not in prompt:
            prompt = f"{prompt}\n\n{prompt_flags}"

        current_turn_parts.append(types.Part.from_text(text=prompt))
        return current_turn_parts

    def _load_file_content(self, file_path: str) -> types.Part | str:
        """
        Loads file content, handling images/PDFs via upload and text directly.
        """
        path = Path(file_path)
        mime_type, _ = mimetypes.guess_type(path)

        if not mime_type:
            mime_type = "application/octet-stream"

        # Binary/Media files -> Upload to File API
        if (
            mime_type.startswith("image/")
            or mime_type.startswith("audio/")
            or mime_type == "application/pdf"
        ):
            try:
                uploaded_file = self.client.files.upload(path=path)
                # Wait for processing if necessary
                while uploaded_file.state.name == "PROCESSING":
                    time.sleep(1)
                    uploaded_file = self.client.files.get(name=uploaded_file.name)

                return types.Part.from_uri(
                    file_uri=uploaded_file.uri, mime_type=uploaded_file.mime_type
                )
            except Exception as e:
                logging.error(f"Failed to upload file {path}: {e}")
                return f"[Error uploading {path.name}: {e}]"

        # Text/Code files -> Read content directly
        else:
            try:
                with open(path, encoding="utf-8", errors="replace") as f:
                    text_content = f.read()
                return types.Part.from_text(
                    text=f"File: {path.name}\nContent:\n{text_content}"
                )
            except Exception as e:
                logging.error(f"Failed to read text file {path}: {e}")
                return f"[Error reading {path.name}: {e}]"
