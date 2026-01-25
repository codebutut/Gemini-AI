from datetime import datetime
from typing import Any, Optional, Dict, List, Union
from pydantic import BaseModel, Field
from enum import Enum
import uuid


class Message(BaseModel):
    role: str
    text: str
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    images: Optional[List[str]] = None

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        return super().model_dump(**kwargs)


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class Session(BaseModel):
    title: str = "New Chat"
    messages: List[Message] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    plan: str = ""
    specs: str = ""
    config: Dict[str, Any] = Field(default_factory=dict)
    usage: Usage = Field(default_factory=Usage)

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        return super().model_dump(**kwargs)


class ToolCall(BaseModel):
    call_id: str = Field(default_factory=lambda: str(datetime.now().timestamp()))
    agent_name: str
    tool_name: str
    arguments: Dict[str, Any]
    result: Any
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    success: bool = True


class AgentState(BaseModel):
    agent_name: str
    session_id: str
    history: List[Dict[str, Any]] = Field(default_factory=list)
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat())


class UserProfile(BaseModel):
    user_id: str = "default_user"
    preferences: Dict[str, Any] = Field(
        default_factory=lambda: {
            "verbosity": "normal",
            "coding_style": "PEP8",
            "preferred_language": "Python",
        }
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ExecutionContext(BaseModel):
    task_id: str
    shared_data: Dict[str, Any] = Field(default_factory=dict)
    active_agent: Optional[str] = None
    status: str = "idle"
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())


class ReflectionResult(BaseModel):
    is_satisfactory: bool
    critique: str
    suggestions: List[str] = Field(default_factory=list)
    score: float = 0.0  # 0.0 to 1.0


class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


class Task(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    description: str
    assigned_agent: Optional[str] = None
    dependencies: List[str] = Field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    priority: int = 1
    result: Optional[str] = None
    error: Optional[str] = None


class Plan(BaseModel):
    goal: str
    tasks: List[Task] = Field(default_factory=list)
    status: str = "created"
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat())


# --- Hierarchical Memory Models ---


class WorkingMemory(BaseModel):
    """Volatile memory for the current turn."""

    active_task: Optional[str] = None
    intermediate_steps: List[Dict[str, Any]] = Field(default_factory=list)
    current_tool_state: Dict[str, Any] = Field(default_factory=dict)


class EpisodicMemory(BaseModel):
    """Persistent record of experiences."""

    session_id: str
    events: List[Dict[str, Any]] = Field(
        default_factory=list
    )  # Messages, tool calls, reflections
    summary: Optional[str] = None


class SemanticMemory(BaseModel):
    """Persistent general knowledge."""

    learned_patterns: Dict[str, int] = Field(
        default_factory=dict
    )  # Pattern -> Frequency
    entities: List[str] = Field(default_factory=list)
    user_preferences: Dict[str, Any] = Field(default_factory=dict)


class ToolExpertise(BaseModel):
    tool_name: str
    success_count: int = 0
    failure_count: int = 0
    avg_execution_time: float = 0.0
    common_errors: List[str] = Field(default_factory=list)


class ProceduralMemory(BaseModel):
    """Persistent 'how-to' knowledge."""

    tool_expertise: Dict[str, ToolExpertise] = Field(default_factory=dict)
    workflow_templates: Dict[str, Plan] = Field(
        default_factory=dict
    )  # Goal -> Successful Plan
