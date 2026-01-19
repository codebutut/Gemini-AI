import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime
from .state_manager import StateManager
from .models import (
    WorkingMemory, EpisodicMemory, SemanticMemory, 
    ProceduralMemory, ToolExpertise, Plan, ToolCall
)
from .retrieval import MultiStageRetriever
from .vector_store import VectorStore

logger = logging.getLogger(__name__)

class MemoryManager:
    """
    Orchestrates the Hierarchical Memory System:
    - Working Memory (Volatile)
    - Episodic Memory (Persistent)
    - Semantic Memory (Persistent)
    - Procedural Memory (Persistent)
    
    Integrated with MultiStageRetriever for advanced context retrieval.
    """

    def __init__(self, state_dir: Path, state_manager: StateManager):
        self.state_dir = state_dir
        self.state_manager = state_manager
        
        self.episodic_dir = self.state_dir / "episodic"
        self.episodic_dir.mkdir(parents=True, exist_ok=True)
        
        self.semantic_file = self.state_dir / "semantic_memory.json"
        self.procedural_file = self.state_dir / "procedural_memory.json"
        
        self._lock = threading.RLock()
        
        # Initialize Vector Store with CacheManager from StateManager
        self.vector_store = VectorStore(
            persist_directory=str(self.state_dir / "chroma_db"),
            cache_manager=self.state_manager.cache_manager
        )
        
        # Initialize Multi-stage Retriever
        self.retriever = MultiStageRetriever(vector_store=self.vector_store)
        
        # Load persistent memories
        self.semantic_memory: SemanticMemory = self._load_semantic()
        self.procedural_memory: ProceduralMemory = self._load_procedural()
        
        # Working memory is volatile and per-session/agent
        self.working_memories: Dict[str, WorkingMemory] = {}
        
        # Initial index update
        self._refresh_retrieval_index()

    def _load_semantic(self) -> SemanticMemory:
        data = self.state_manager._load_json(self.semantic_file, SemanticMemory)
        if not data:
            return SemanticMemory()
        return list(data.values())[0] if data else SemanticMemory()

    def _load_procedural(self) -> ProceduralMemory:
        data = self.state_manager._load_json(self.procedural_file, ProceduralMemory)
        return list(data.values())[0] if data else ProceduralMemory()

    def _save_semantic(self):
        with self._lock:
            self.state_manager._save_json(self.semantic_file, {"main": self.semantic_memory})
            self._refresh_retrieval_index()

    def _save_procedural(self):
        with self._lock:
            self.state_manager._save_json(self.procedural_file, {"main": self.procedural_memory})

    def _refresh_retrieval_index(self):
        """Refreshes the BM25 index with current episodic and semantic data."""
        docs = []
        metas = []
        
        # Add semantic entities
        for entity in self.semantic_memory.entities:
            docs.append(f"Entity: {entity}")
            metas.append({"type": "semantic", "entity": entity})
            
        # Add episodic events from all sessions
        for session_file in self.episodic_dir.glob("*.json"):
            try:
                events = self.state_manager._load_json_list(session_file, dict)
                for event in events:
                    if "text" in event:
                        docs.append(event["text"])
                        metas.append({
                            "type": "episodic", 
                            "session_id": session_file.stem,
                            "timestamp": event.get("timestamp")
                        })
            except Exception as e:
                logger.error(f"Failed to load episodic events from {session_file}: {e}")
                
        self.retriever.update_index(docs, metas)

    # --- Retrieval ---

    def get_relevant_context(self, query: str, context: Optional[str] = None, limit: int = 5) -> str:
        """Retrieves and formats relevant context for the agent."""
        results = self.retriever.retrieve(query, context=context, limit=limit)
        if not results:
            return "No relevant context found."
            
        formatted = ["Relevant Context:"]
        for res in results:
            source = res["source"]
            content = res["content"]
            formatted.append(f"- [{source}] {content}")
            
        return "\n".join(formatted)

    # --- Working Memory ---

    def get_working_memory(self, agent_id: str) -> WorkingMemory:
        if agent_id not in self.working_memories:
            self.working_memories[agent_id] = WorkingMemory()
        return self.working_memories[agent_id]

    def update_working_memory(self, agent_id: str, **kwargs):
        wm = self.get_working_memory(agent_id)
        for key, value in kwargs.items():
            if hasattr(wm, key):
                setattr(wm, key, value)

    # --- Episodic Memory ---

    def commit_episodic_event(self, session_id: str, event: Dict[str, Any]):
        file_path = self.episodic_dir / f"{session_id}.json"
        with self._lock:
            events = self.state_manager._load_json_list(file_path, dict)
            events.append(event)
            self.state_manager._save_json(file_path, events)
            
            # Also add to vector store for semantic search
            if "text" in event:
                self.vector_store.add_documents(
                    documents=[event["text"]],
                    metadatas=[{"session_id": session_id, "timestamp": event.get("timestamp")}],
                    ids=[f"episodic_{session_id}_{len(events)}"]
                )
            
            # Refresh BM25 index periodically or on commit
            self._refresh_retrieval_index()

    # --- Semantic Memory ---

    def learn_pattern(self, pattern: str):
        with self._lock:
            count = self.semantic_memory.learned_patterns.get(pattern, 0)
            self.semantic_memory.learned_patterns[pattern] = count + 1
            self._save_semantic()

    def add_entity(self, entity: str):
        with self._lock:
            if entity not in self.semantic_memory.entities:
                self.semantic_memory.entities.append(entity)
                
                # Also add to vector store
                self.vector_store.add_documents(
                    documents=[f"Entity: {entity}"],
                    metadatas=[{"type": "semantic", "entity": entity}],
                    ids=[f"semantic_entity_{entity}"]
                )
                
                self._save_semantic()

    # --- Procedural Memory ---

    def update_tool_expertise(self, tool_call: ToolCall, execution_time: float):
        with self._lock:
            name = tool_call.tool_name
            if name not in self.procedural_memory.tool_expertise:
                self.procedural_memory.tool_expertise[name] = ToolExpertise(tool_name=name)
            
            exp = self.procedural_memory.tool_expertise[name]
            if tool_call.success:
                exp.success_count += 1
            else:
                exp.failure_count += 1
                if isinstance(tool_call.result, str):
                    exp.common_errors.append(tool_call.result[:100])
            
            total_calls = exp.success_count + exp.failure_count
            exp.avg_execution_time = ((exp.avg_execution_time * (total_calls - 1)) + execution_time) / total_calls
            
            self._save_procedural()

    def save_workflow_template(self, goal: str, plan: Plan):
        with self._lock:
            self.procedural_memory.workflow_templates[goal] = plan
            self._save_procedural()

    def get_workflow_template(self, goal: str) -> Optional[Plan]:
        return self.procedural_memory.workflow_templates.get(goal)
