"""
FastAPI application for the Hive Dashboard.
"""

import logging
from typing import Any
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from framework.runner.runner import AgentRunner

logger = logging.getLogger(__name__)


def create_app(runner: AgentRunner) -> FastAPI:
    """
    Create the FastAPI app.

    Args:
        runner: The AgentRunner instance.
    """
    app = FastAPI(title="Hive Dashboard")

    # Setup templates
    base_dir = Path(__file__).parent
    templates = Jinja2Templates(directory=str(base_dir / "templates"))

    # Store active websocket connections
    active_connections: list[WebSocket] = []

    async def broadcast(message: dict[str, Any]):
        """Broadcast a message to all connected clients."""
        for connection in active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                # Connection might be dead
                pass

    # Subscribe to runner events if possible
    # We assume 'enable_dashboard' was called on the runner or it's in a mode that supports events
    if hasattr(runner, "subscribe_to_events"):

        async def on_event(event):
            await broadcast(event.to_dict())

        try:
            # Subscribe to all event types
            # Real types are Enums in EventBus, but we pass strings or the Enum values if we import them
            # For now let's try to subscribe to all if the method supports wildcard or list
            # We'll need to import EventType from event_bus if we want to be precise
            from framework.runtime.event_bus import EventType

            runner.subscribe_to_events(event_types=[e for e in EventType], handler=on_event)
            logger.info("Subscribed to agent events")
        except Exception as e:
            logger.warning(f"Could not subscribe to runner events: {e}")

    @app.get("/", response_class=HTMLResponse)
    async def get_root(request: Request):
        return templates.TemplateResponse(
            "index.html", {"request": request, "agent_name": runner.info().name}
        )

    @app.get("/api/graph")
    async def get_graph():
        """Get the agent graph structure."""
        info = runner.info()
        # Convert dataclass to dict
        return {
            "nodes": info.nodes,
            "edges": info.edges,
            "name": info.name,
            "description": info.description,
            "entry_node": info.entry_node,
        }

    @app.post("/api/run")
    async def run_agent(input_data: dict[str, Any]):
        """Trigger an agent run."""
        # We need to determine if we use trigger (async) or run (sync but we can background it?)
        # For dashboard, async trigger is best.
        # If the runner doesn't support trigger, we might need to wrap it.

        try:
            # Check if we can trigger
            if hasattr(runner, "trigger") and runner._uses_async_entry_points:
                # Use the first available entry point or 'default'
                entry_points = runner.get_entry_points()
                ep_id = entry_points[0].id if entry_points else "default"

                exec_id = await runner.trigger(ep_id, input_data)
                return {"status": "started", "execution_id": exec_id}
            else:
                # Fallback implementation if we forced event bus but not full async triggers?
                # Or just error out if not compatible
                return {"status": "error", "message": "Runner does not support async triggering"}
        except Exception as e:
            logger.error(f"Error running agent: {e}")
            return {"status": "error", "message": str(e)}

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await websocket.accept()
        active_connections.append(websocket)
        try:
            while True:
                # Keep alive
                await websocket.receive_text()
        except WebSocketDisconnect:
            active_connections.remove(websocket)

    return app
