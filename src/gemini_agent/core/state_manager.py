import json
import logging
import threading
import base64
from pathlib import Path
from typing import Any, Dict, List, Optional
from .models import AgentState, ToolCall, UserProfile, ExecutionContext, Plan
from .cache_manager import CacheManager

logger = logging.getLogger(__name__)


def decode_base64_recursive(data: Any) -> Any:
    """Recursively decode strings starting with 'base64:' back to bytes."""
    if isinstance(data, dict):
        return {k: decode_base64_recursive(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [decode_base64_recursive(v) for v in data]
    elif isinstance(data, str) and data.startswith("base64:"):
        try:
            return base64.b64decode(data[7:])
        except Exception:
            return data
    return data


class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, bytes):
            return f"base64:{base64.b64encode(obj).decode('utf-8')}"
        if hasattr(obj, "model_dump"):
            return obj.model_dump(exclude_none=True)
        if hasattr(obj, "value") and isinstance(obj.value, str):  # For Enums
            return obj.value
        return super().default(obj)


class StateManager:
    """
    Manages persistent state for the Multi-Agent System, including
    agent history, tool usage, user profiles, execution context, and plans.
    """

    def __init__(self, state_dir: Path, cache_manager: Optional[CacheManager] = None):
        self.state_dir = state_dir
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.cache_manager = cache_manager

        self.agent_states_file = self.state_dir / "agent_states.json"
        self.tool_calls_file = self.state_dir / "tool_calls.json"
        self.user_profiles_file = self.state_dir / "user_profiles.json"
        self.plans_file = self.state_dir / "plans.json"

        self._lock = threading.RLock()

        self.agent_states: Dict[str, AgentState] = self._load_json(
            self.agent_states_file, AgentState
        )
        self.tool_calls: List[ToolCall] = self._load_json_list(
            self.tool_calls_file, ToolCall
        )
        self.user_profiles: Dict[str, UserProfile] = self._load_json(
            self.user_profiles_file, UserProfile
        )
        self.plans: Dict[str, Plan] = self._load_json(self.plans_file, Plan)

        self.execution_contexts: Dict[str, ExecutionContext] = {}

    def _load_json(self, file_path: Path, model_class: Any) -> Dict[str, Any]:
        if not file_path.exists():
            return {}
        try:
            with file_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
                decoded_data = decode_base64_recursive(data)
                return {k: model_class(**v) for k, v in decoded_data.items()}
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")
            return {}

    def _load_json_list(self, file_path: Path, model_class: Any) -> List[Any]:
        if not file_path.exists():
            return []
        try:
            with file_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
                decoded_data = decode_base64_recursive(data)
                return [model_class(**v) for v in decoded_data]
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")
            return []

    def _save_json(self, file_path: Path, data: Any):
        try:
            with self._lock:
                with file_path.open("w", encoding="utf-8") as f:
                    json.dump(
                        data, f, indent=4, ensure_ascii=False, cls=EnhancedJSONEncoder
                    )
        except Exception as e:
            logger.error(f"Failed to save {file_path}: {e}")

    def save_agent_state(
        self, session_id: str, agent_name: str, history: List[Dict[str, Any]]
    ):
        key = f"{session_id}_{agent_name}"
        with self._lock:
            state = AgentState(
                agent_name=agent_name, session_id=session_id, history=history
            )
            self.agent_states[key] = state
            self._save_json(self.agent_states_file, self.agent_states)
            if self.cache_manager:
                self.cache_manager.set_state(
                    f"agent_state_{key}", state.model_dump(exclude_none=True)
                )

    def load_agent_state(
        self, session_id: str, agent_name: str
    ) -> Optional[AgentState]:
        key = f"{session_id}_{agent_name}"
        if self.cache_manager:
            cached = self.cache_manager.get_state(f"agent_state_{key}")
            if cached:
                return AgentState(**cached)
        return self.agent_states.get(key)

    def log_tool_call(self, tool_call: ToolCall):
        with self._lock:
            self.tool_calls.append(tool_call)
            if len(self.tool_calls) > 1000:
                self.tool_calls = self.tool_calls[-1000:]
            self._save_json(self.tool_calls_file, self.tool_calls)

    def get_user_profile(self, user_id: str = "default_user") -> UserProfile:
        with self._lock:
            if self.cache_manager:
                cached = self.cache_manager.get_state(f"user_profile_{user_id}")
                if cached:
                    return UserProfile(**cached)

            if user_id not in self.user_profiles:
                self.user_profiles[user_id] = UserProfile(user_id=user_id)
                self._save_json(self.user_profiles_file, self.user_profiles)

            profile = self.user_profiles[user_id]
            if self.cache_manager:
                self.cache_manager.set_state(
                    f"user_profile_{user_id}", profile.model_dump(exclude_none=True)
                )
            return profile

    def update_user_profile(self, user_id: str, preferences: Dict[str, Any]):
        with self._lock:
            profile = self.get_user_profile(user_id)
            profile.preferences.update(preferences)
            self.user_profiles[user_id] = profile
            self._save_json(self.user_profiles_file, self.user_profiles)
            if self.cache_manager:
                self.cache_manager.set_state(
                    f"user_profile_{user_id}", profile.model_dump(exclude_none=True)
                )

    def get_execution_context(self, task_id: str) -> ExecutionContext:
        with self._lock:
            if self.cache_manager:
                cached = self.cache_manager.get_state(f"exec_context_{task_id}")
                if cached:
                    return ExecutionContext(**cached)

            if task_id not in self.execution_contexts:
                self.execution_contexts[task_id] = ExecutionContext(task_id=task_id)

            context = self.execution_contexts[task_id]
            if self.cache_manager:
                self.cache_manager.set_state(
                    f"exec_context_{task_id}", context.model_dump(exclude_none=True)
                )
            return context

    def update_execution_context(self, task_id: str, data: Dict[str, Any]):
        with self._lock:
            context = self.get_execution_context(task_id)
            context.shared_data.update(data)
            self.execution_contexts[task_id] = context
            if self.cache_manager:
                self.cache_manager.set_state(
                    f"exec_context_{task_id}", context.model_dump(exclude_none=True)
                )

    def save_plan(self, session_id: str, plan: Plan):
        with self._lock:
            self.plans[session_id] = plan
            self._save_json(self.plans_file, self.plans)
            if self.cache_manager:
                self.cache_manager.set_state(
                    f"plan_{session_id}", plan.model_dump(exclude_none=True)
                )

    def load_plan(self, session_id: str) -> Optional[Plan]:
        if self.cache_manager:
            cached = self.cache_manager.get_state(f"plan_{session_id}")
            if cached:
                return Plan(**cached)
        return self.plans.get(session_id)
