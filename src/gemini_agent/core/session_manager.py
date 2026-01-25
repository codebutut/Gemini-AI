import json
import logging
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import ValidationError
from .models import Session, Message, Usage


class SessionManager:
    """
    Manages chat sessions using Pydantic models for data integrity and
    thread-safe throttled persistence.
    """

    def __init__(self, history_file: Path):
        self.history_file = history_file
        self._lock = threading.RLock()
        self.sessions: Dict[str, Session] = self._load_history()
        self.current_session_id: Optional[str] = None

        self._save_timer: Optional[threading.Timer] = None
        self._last_save_time = 0.0
        self._save_interval = 2.0  # Minimum seconds between saves

    def _load_history(self) -> Dict[str, Session]:
        """Loads chat history from JSON file and validates with Pydantic."""
        if not self.history_file.exists():
            return {}
        try:
            with self.history_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
                return {sid: Session(**sdata) for sid, sdata in data.items()}
        except (json.JSONDecodeError, OSError, ValidationError) as e:
            logging.error(f"Failed to load history: {e}")
            return {}

    def save_history(self, sync: bool = False) -> None:
        """Saves current sessions to JSON file with throttling."""
        if sync:
            self._perform_save()
            return

        with self._lock:
            if self._save_timer is not None:
                return

            now = time.time()
            elapsed = now - self._last_save_time
            if elapsed >= self._save_interval:
                threading.Thread(target=self._perform_save_async, daemon=True).start()
            else:
                delay = self._save_interval - elapsed
                self._save_timer = threading.Timer(delay, self._perform_save_async)
                self._save_timer.daemon = True
                self._save_timer.start()

    def _perform_save_async(self) -> None:
        with self._lock:
            self._save_timer = None
            self._perform_save()

    def _perform_save(self) -> None:
        """Performs the actual save operation."""
        try:
            with self._lock:
                # Convert models to dict for serialization
                data = {
                    sid: session.model_dump() for sid, session in self.sessions.items()
                }
                data_str = json.dumps(data, indent=4, ensure_ascii=False)
                self._last_save_time = time.time()

            with self.history_file.open("w", encoding="utf-8") as f:
                f.write(data_str)
        except Exception as e:
            logging.error(f"Failed to save history: {e}")

    def create_session(
        self,
        title: str = "New Chat",
        config: Optional[Dict[str, Any]] = None,
        sync: bool = False,
    ) -> str:
        """Creates a new session and returns its ID."""
        session_id = str(uuid.uuid4())
        session = Session(title=title, config=config or {})

        with self._lock:
            self.sessions[session_id] = session
            self.current_session_id = session_id

        self.save_history(sync=sync)
        return session_id

    def delete_session(self, session_id: str, sync: bool = False) -> bool:
        """Deletes a session by ID."""
        with self._lock:
            if session_id in self.sessions:
                del self.sessions[session_id]
                if self.current_session_id == session_id:
                    self.current_session_id = None
                self.save_history(sync=sync)
                return True
        return False

    def get_session(self, session_id: str) -> Optional[Session]:
        """Returns session data."""
        return self.sessions.get(session_id)

    def get_all_sessions(self) -> Dict[str, Session]:
        """Returns all sessions."""
        return self.sessions

    def add_message(
        self,
        session_id: str,
        role: str,
        text: str,
        images: Optional[list[str]] = None,
        sync: bool = False,
    ) -> None:
        """Adds a message to a session."""
        session = self.sessions.get(session_id)
        if not session:
            return

        message = Message(role=role, text=text, images=images)
        with self._lock:
            session.messages.append(message)
        self.save_history(sync=sync)

    def update_session_title(
        self, session_id: str, new_title: str, sync: bool = False
    ) -> None:
        """Updates the title of a session."""
        session = self.sessions.get(session_id)
        if session:
            session.title = new_title
            self.save_history(sync=sync)

    def update_session_plan(
        self, session_id: str, plan: str, sync: bool = False
    ) -> None:
        """Updates the plan of a session."""
        session = self.sessions.get(session_id)
        if session:
            session.plan = plan
            self.save_history(sync=sync)

    def update_session_specs(
        self, session_id: str, specs: str, sync: bool = False
    ) -> None:
        """Updates the specs of a session."""
        session = self.sessions.get(session_id)
        if session:
            session.specs = specs
            self.save_history(sync=sync)

    def update_session_config(
        self, session_id: str, config: Dict[str, Any], sync: bool = False
    ) -> None:
        """Updates the configuration overrides for a session."""
        session = self.sessions.get(session_id)
        if session:
            session.config.update(config)
            self.save_history(sync=sync)

    def update_session_usage(
        self, session_id: str, input_tokens: int, output_tokens: int, sync: bool = False
    ) -> None:
        """Updates the cumulative token usage for a session."""
        session = self.sessions.get(session_id)
        if session:
            with self._lock:
                session.usage.input_tokens += input_tokens
                session.usage.output_tokens += output_tokens
                session.usage.total_tokens = (
                    session.usage.input_tokens + session.usage.output_tokens
                )
            self.save_history(sync=sync)

    def clear_current_session(self, sync: bool = False) -> None:
        """Clears messages, plan, and specs in the current session."""
        if self.current_session_id:
            session = self.sessions.get(self.current_session_id)
            if session:
                with self._lock:
                    session.messages = []
                    session.plan = ""
                    session.specs = ""
                self.save_history(sync=sync)
