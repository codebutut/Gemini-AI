import uuid
import asyncio
from typing import Optional, Dict, List, Union
from pathlib import Path
from google.genai import types
from gemini_agent.core.mas.agents import (
    RouterAgent,
    ResearchAgent,
    CodeAgent,
    FileAgent,
    ReasoningAgent,
    VisionAgent,
    PlanningAgent,
    KnowledgeAgent,
)
from gemini_agent.core.state_manager import StateManager
from gemini_agent.core.memory_manager import MemoryManager
from gemini_agent.core.cache_manager import CacheManager
from gemini_agent.core.models import Plan, Task, TaskStatus
from gemini_agent.core.comms import AsyncEventBus, RealTimeServer, AgentEvent
from gemini_agent.utils.logger import get_logger

logger = get_logger(__name__)

# Global context to share state with tools
_current_state_manager: Optional[StateManager] = None
_current_memory_manager: Optional[MemoryManager] = None
_current_session_id: Optional[str] = None
_current_task_id: Optional[str] = None
_current_event_bus: Optional[AsyncEventBus] = None
_current_cache_manager: Optional[CacheManager] = None


class MultiAgentOrchestrator:
    """Orchestrates multiple specialized agents with Planning & Execution Engine and Hierarchical Memory."""

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        state_dir: str = ".agent_state",
        enable_server: bool = True,
        server_port: int = 8000,
    ):
        self.state_dir = Path(state_dir)
        self.cache_manager = CacheManager(self.state_dir / "cache")
        self.state_manager = StateManager(
            self.state_dir, cache_manager=self.cache_manager
        )
        self.memory_manager = MemoryManager(self.state_dir, self.state_manager)
        self.event_bus = AsyncEventBus()
        self.model = model
        self.api_key = api_key

        self.router = RouterAgent(
            model=model,
            api_key=api_key,
            state_manager=self.state_manager,
            memory_manager=self.memory_manager,
            cache_manager=self.cache_manager,
            event_bus=self.event_bus,
        )
        self.planning_agent = PlanningAgent(
            model=model,
            api_key=api_key,
            state_manager=self.state_manager,
            memory_manager=self.memory_manager,
            cache_manager=self.cache_manager,
            event_bus=self.event_bus,
        )
        self.agents = {
            "ResearchAgent": ResearchAgent(
                model=model,
                api_key=api_key,
                state_manager=self.state_manager,
                memory_manager=self.memory_manager,
                cache_manager=self.cache_manager,
                event_bus=self.event_bus,
            ),
            "CodeAgent": CodeAgent(
                model=model,
                api_key=api_key,
                state_manager=self.state_manager,
                memory_manager=self.memory_manager,
                cache_manager=self.cache_manager,
                event_bus=self.event_bus,
            ),
            "FileAgent": FileAgent(
                model=model,
                api_key=api_key,
                state_manager=self.state_manager,
                memory_manager=self.memory_manager,
                cache_manager=self.cache_manager,
                event_bus=self.event_bus,
            ),
            "ReasoningAgent": ReasoningAgent(
                model=model,
                api_key=api_key,
                state_manager=self.state_manager,
                memory_manager=self.memory_manager,
                cache_manager=self.cache_manager,
                event_bus=self.event_bus,
            ),
            "VisionAgent": VisionAgent(
                model=model,
                api_key=api_key,
                state_manager=self.state_manager,
                memory_manager=self.memory_manager,
                cache_manager=self.cache_manager,
                event_bus=self.event_bus,
            ),
            "KnowledgeAgent": KnowledgeAgent(
                model=model,
                api_key=api_key,
                state_manager=self.state_manager,
                memory_manager=self.memory_manager,
                cache_manager=self.cache_manager,
                event_bus=self.event_bus,
            ),
            "PlanningAgent": self.planning_agent,
        }

        self.server = None
        if enable_server:
            self.server = RealTimeServer(self.event_bus, port=server_port)

    async def execute_plan(self, plan: Plan, session_id: str, task_id: str) -> str:
        """Executes a structured plan by delegating tasks to agents."""
        completed_tasks = {}

        await self.event_bus.publish(
            AgentEvent(
                event_type="plan.execution_started",
                session_id=session_id,
                task_id=task_id,
                data={"goal": plan.goal, "num_tasks": len(plan.tasks)},
            )
        )

        while any(
            t.status in [TaskStatus.PENDING, TaskStatus.IN_PROGRESS] for t in plan.tasks
        ):
            # Find tasks that are pending and have all dependencies met
            ready_tasks = [
                t
                for t in plan.tasks
                if t.status == TaskStatus.PENDING
                and all(dep in completed_tasks for dep in t.dependencies)
            ]

            if not ready_tasks:
                if any(t.status == TaskStatus.IN_PROGRESS for t in plan.tasks):
                    await asyncio.sleep(1)
                    continue
                else:
                    # Blocked or failed
                    logger.warning(f"Plan execution blocked for session {session_id}")
                    break

            # Sort by priority
            ready_tasks.sort(key=lambda x: x.priority)

            for task in ready_tasks:
                task.status = TaskStatus.IN_PROGRESS
                await self.event_bus.publish(
                    AgentEvent(
                        event_type="task.started",
                        session_id=session_id,
                        task_id=task_id,
                        data={
                            "task_id": task.id,
                            "description": task.description,
                            "agent": task.assigned_agent,
                        },
                    )
                )

                agent = self.get_agent(task.assigned_agent) or self.router
                try:
                    result = await agent.run(
                        task.description, session_id=session_id, task_id=task_id
                    )
                    task.result = result
                    task.status = TaskStatus.COMPLETED
                    completed_tasks[task.id] = result

                    await self.event_bus.publish(
                        AgentEvent(
                            event_type="task.completed",
                            session_id=session_id,
                            task_id=task_id,
                            data={"task_id": task.id, "success": True},
                        )
                    )
                except Exception as e:
                    task.status = TaskStatus.FAILED
                    task.error = str(e)
                    await self.event_bus.publish(
                        AgentEvent(
                            event_type="task.failed",
                            session_id=session_id,
                            task_id=task_id,
                            data={"task_id": task.id, "error": str(e)},
                        )
                    )

                self.state_manager.save_plan(session_id, plan)

        success = all(t.status == TaskStatus.COMPLETED for t in plan.tasks)
        if success:
            # Save successful plan to procedural memory as a template
            self.memory_manager.save_workflow_template(plan.goal, plan)
            await self.event_bus.publish(
                AgentEvent(
                    event_type="plan.template_saved",
                    session_id=session_id,
                    task_id=task_id,
                    data={"goal": plan.goal},
                )
            )

        final_result = (
            list(completed_tasks.values())[-1]
            if completed_tasks
            else "Plan execution failed or produced no results."
        )

        await self.event_bus.publish(
            AgentEvent(
                event_type="plan.execution_finished",
                session_id=session_id,
                task_id=task_id,
                data={"final_result": final_result},
            )
        )

        return final_result

    async def run(
        self, prompt: Union[str, List[types.Part]], session_id: Optional[str] = None
    ) -> str:
        global \
            _current_state_manager, \
            _current_memory_manager, \
            _current_session_id, \
            _current_task_id, \
            _current_event_bus, \
            _current_cache_manager

        task_id = str(uuid.uuid4())
        session_id = session_id or "default_session"

        # Set global context for tools
        _current_state_manager = self.state_manager
        _current_memory_manager = self.memory_manager
        _current_session_id = session_id
        _current_task_id = task_id
        _current_event_bus = self.event_bus
        _current_cache_manager = self.cache_manager

        logger.info(f"Orchestrator starting task {task_id} in session {session_id}")

        if self.server:
            await self.server.start()

        try:
            self.state_manager.get_execution_context(task_id)
            prompt_text = prompt if isinstance(prompt, str) else "[Multi-modal Prompt]"
            self.state_manager.update_execution_context(
                task_id, {"original_prompt": prompt_text}
            )

            await self.event_bus.publish(
                AgentEvent(
                    event_type="task.started",
                    session_id=session_id,
                    task_id=task_id,
                    data={"prompt": prompt_text},
                )
            )

            # Check for explicit planning command
            if isinstance(prompt, str) and prompt.startswith("/plan"):
                goal = prompt.replace("/plan", "").strip()
                plan = await self.planning_agent.generate_plan(goal)
                self.state_manager.save_plan(session_id, plan)
                result = await self.execute_plan(plan, session_id, task_id)
            else:
                # Default to router
                result = await self.router.run(
                    prompt, session_id=session_id, task_id=task_id
                )

            await self.event_bus.publish(
                AgentEvent(
                    event_type="task.finished",
                    session_id=session_id,
                    task_id=task_id,
                    data={"result": result},
                )
            )

            return result
        finally:
            if self.server:
                await asyncio.sleep(1)
                await self.server.stop()

    def get_agent(self, name: str):
        return self.agents.get(name)

    def close(self):
        """Closes all resources."""
        if self.cache_manager:
            self.cache_manager.close()


def get_current_context():
    return (
        _current_state_manager,
        _current_memory_manager,
        _current_session_id,
        _current_task_id,
        _current_event_bus,
        _current_cache_manager,
    )
