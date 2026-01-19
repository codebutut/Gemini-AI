import asyncio
import json
import uvicorn
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Set
from pydantic import BaseModel, Field
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse

class AgentEvent(BaseModel):
    event_type: str
    session_id: str
    task_id: Optional[str] = None
    agent_name: Optional[str] = None
    data: Dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

class AsyncEventBus:
    """A simple asynchronous event bus for internal pub-sub."""
    
    def __init__(self):
        self._subscribers: Dict[str, Set[Callable]] = {}
        self._global_subscribers: Set[Callable] = set()
        self._lock = asyncio.Lock()

    async def subscribe(self, event_type: str, callback: Callable[[AgentEvent], Any]):
        async with self._lock:
            if event_type not in self._subscribers:
                self._subscribers[event_type] = set()
            self._subscribers[event_type].add(callback)

    async def subscribe_all(self, callback: Callable[[AgentEvent], Any]):
        async with self._lock:
            self._global_subscribers.add(callback)

    async def unsubscribe(self, event_type: str, callback: Callable[[AgentEvent], Any]):
        async with self._lock:
            if event_type in self._subscribers:
                self._subscribers[event_type].discard(callback)
            self._global_subscribers.discard(callback)

    async def publish(self, event: AgentEvent):
        # Call specific subscribers
        callbacks = []
        async with self._lock:
            if event.event_type in self._subscribers:
                callbacks.extend(list(self._subscribers[event.event_type]))
            callbacks.extend(list(self._global_subscribers))

        if callbacks:
            # Run callbacks concurrently
            await asyncio.gather(*(self._run_callback(cb, event) for cb in callbacks))

    async def _run_callback(self, callback: Callable, event: AgentEvent):
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(event)
            else:
                callback(event)
        except Exception as e:
            print(f"Error in event bus callback: {e}")

class RealTimeServer:
    """FastAPI server to expose events via WebSocket and SSE."""
    
    def __init__(self, event_bus: AsyncEventBus, host: str = "127.0.0.1", port: int = 8000):
        self.event_bus = event_bus
        self.host = host
        self.port = port
        self.app = FastAPI(title="Gemini Agent Real-time API")
        self._setup_routes()
        self._active_connections: Dict[str, Set[WebSocket]] = {}
        self._server_task: Optional[asyncio.Task] = None

    def _setup_routes(self):
        @self.app.get("/health")
        async def health():
            return {"status": "ok"}

        @self.app.websocket("/ws/{session_id}")
        async def websocket_endpoint(websocket: WebSocket, session_id: str):
            await websocket.accept()
            if session_id not in self._active_connections:
                self._active_connections[session_id] = set()
            self._active_connections[session_id].add(websocket)
            
            try:
                while True:
                    # Keep connection alive
                    await websocket.receive_text()
            except WebSocketDisconnect:
                self._active_connections[session_id].remove(websocket)

        @self.app.get("/events/{session_id}")
        async def sse_endpoint(session_id: str):
            async def event_generator():
                queue = asyncio.Queue()
                
                async def on_event(event: AgentEvent):
                    if event.session_id == session_id:
                        await queue.put(event)
                
                await self.event_bus.subscribe_all(on_event)
                try:
                    while True:
                        event = await queue.get()
                        yield f"data: {event.model_dump_json()}\n\n"
                finally:
                    await self.event_bus.unsubscribe("all", on_event) # Simplified unsubscribe

            return StreamingResponse(event_generator(), media_type="text/event-stream")

    async def start(self):
        """Starts the server in the background."""
        config = uvicorn.Config(self.app, host=self.host, port=self.port, log_level="info")
        server = uvicorn.Server(config)
        self._server_task = asyncio.create_task(server.serve())
        
        # Subscribe to all events to broadcast to WebSockets
        await self.event_bus.subscribe_all(self._broadcast_to_websockets)

    async def stop(self):
        if self._server_task:
            self._server_task.cancel()
            try:
                await self._server_task
            except asyncio.CancelledError:
                pass

    async def _broadcast_to_websockets(self, event: AgentEvent):
        session_id = event.session_id
        if session_id in self._active_connections:
            disconnected = set()
            for ws in self._active_connections[session_id]:
                try:
                    await ws.send_json(event.model_dump())
                except Exception:
                    disconnected.add(ws)
            
            for ws in disconnected:
                self._active_connections[session_id].remove(ws)
