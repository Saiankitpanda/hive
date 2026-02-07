"""
Streaming Mode - Real-time response streaming for agents.

Provides:
- Token-by-token streaming from LLMs
- Event-based streaming for agent execution
- WebSocket streaming support
- SSE (Server-Sent Events) support

From ROADMAP Phase 2: Streaming mode support
"""

import asyncio
import json
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class StreamEventType(Enum):
    """Types of streaming events."""

    # Content events
    TOKEN = "token"  # Single token from LLM
    CONTENT_START = "content_start"
    CONTENT_DELTA = "content_delta"
    CONTENT_END = "content_end"

    # Execution events
    EXECUTION_START = "execution_start"
    EXECUTION_END = "execution_end"
    NODE_START = "node_start"
    NODE_END = "node_end"

    # Tool events
    TOOL_CALL_START = "tool_call_start"
    TOOL_CALL_END = "tool_call_end"

    # State events
    STATE_UPDATE = "state_update"
    PROGRESS = "progress"

    # Control events
    HEARTBEAT = "heartbeat"
    ERROR = "error"
    DONE = "done"


@dataclass
class StreamEvent:
    """A streaming event."""

    event_type: StreamEventType
    data: Any
    timestamp: float = field(default_factory=time.time)
    execution_id: str | None = None
    node_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.event_type.value,
            "data": self.data,
            "timestamp": self.timestamp,
            "execution_id": self.execution_id,
            "node_id": self.node_id,
            "metadata": self.metadata,
        }

    def to_sse(self) -> str:
        """Format as Server-Sent Event."""
        data = json.dumps(self.to_dict())
        return f"event: {self.event_type.value}\ndata: {data}\n\n"

    def to_json(self) -> str:
        """Format as JSON line."""
        return json.dumps(self.to_dict()) + "\n"


class StreamBuffer:
    """Buffer for accumulating streamed content."""

    def __init__(self, max_size: int = 100000):
        self.buffer: list[str] = []
        self.max_size = max_size
        self._total_tokens = 0

    def append(self, content: str) -> None:
        """Append content to buffer."""
        self.buffer.append(content)
        self._total_tokens += 1
        if len(self.buffer) > self.max_size:
            self.buffer = self.buffer[-self.max_size :]

    def get_content(self) -> str:
        """Get accumulated content."""
        return "".join(self.buffer)

    def clear(self) -> None:
        """Clear the buffer."""
        self.buffer = []
        self._total_tokens = 0

    @property
    def token_count(self) -> int:
        return self._total_tokens


class StreamingHandler:
    """
    Handler for streaming responses.

    Usage:
        handler = StreamingHandler()

        # Subscribe to events
        handler.on_token(lambda t: print(t, end="", flush=True))
        handler.on_event(lambda e: log_event(e))

        # Stream from LLM
        async for event in handler.stream_llm_response(response):
            # Events are already dispatched to handlers
            pass

        # Get final content
        content = handler.get_content()
    """

    def __init__(self):
        self._buffer = StreamBuffer()
        self._token_handlers: list[Callable[[str], None]] = []
        self._event_handlers: list[Callable[[StreamEvent], None]] = []
        self._async_event_handlers: list[Callable[[StreamEvent], Any]] = []

    # === Handler Registration ===

    def on_token(self, handler: Callable[[str], None]) -> None:
        """Register a token handler."""
        self._token_handlers.append(handler)

    def on_event(self, handler: Callable[[StreamEvent], None]) -> None:
        """Register an event handler."""
        self._event_handlers.append(handler)

    def on_event_async(self, handler: Callable[[StreamEvent], Any]) -> None:
        """Register an async event handler."""
        self._async_event_handlers.append(handler)

    # === Event Emission ===

    def emit_event(self, event: StreamEvent) -> None:
        """Emit an event to all handlers."""
        for handler in self._event_handlers:
            try:
                handler(event)
            except Exception:
                pass

    async def emit_event_async(self, event: StreamEvent) -> None:
        """Emit an event to async handlers."""
        for handler in self._async_event_handlers:
            try:
                await handler(event)
            except Exception:
                pass

    def emit_token(self, token: str) -> None:
        """Emit a token to all token handlers."""
        self._buffer.append(token)
        for handler in self._token_handlers:
            try:
                handler(token)
            except Exception:
                pass

    # === Streaming Methods ===

    async def stream_tokens(
        self,
        token_iterator: AsyncIterator[str],
        execution_id: str | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Stream tokens from an async iterator."""
        # Start event
        start_event = StreamEvent(
            event_type=StreamEventType.CONTENT_START,
            data={"execution_id": execution_id},
            execution_id=execution_id,
        )
        self.emit_event(start_event)
        yield start_event

        try:
            async for token in token_iterator:
                self.emit_token(token)

                event = StreamEvent(
                    event_type=StreamEventType.TOKEN,
                    data={"token": token},
                    execution_id=execution_id,
                )
                self.emit_event(event)
                yield event

        except Exception as e:
            error_event = StreamEvent(
                event_type=StreamEventType.ERROR,
                data={"error": str(e)},
                execution_id=execution_id,
            )
            self.emit_event(error_event)
            yield error_event
            return

        # End event
        end_event = StreamEvent(
            event_type=StreamEventType.CONTENT_END,
            data={
                "content": self._buffer.get_content(),
                "token_count": self._buffer.token_count,
            },
            execution_id=execution_id,
        )
        self.emit_event(end_event)
        yield end_event

    async def stream_execution(
        self,
        execution_iterator: AsyncIterator[dict[str, Any]],
        execution_id: str,
    ) -> AsyncIterator[StreamEvent]:
        """Stream execution events."""
        start_event = StreamEvent(
            event_type=StreamEventType.EXECUTION_START,
            data={"execution_id": execution_id},
            execution_id=execution_id,
        )
        yield start_event

        try:
            async for event_data in execution_iterator:
                event_type = StreamEventType(event_data.get("type", "progress"))
                event = StreamEvent(
                    event_type=event_type,
                    data=event_data.get("data", {}),
                    execution_id=execution_id,
                    node_id=event_data.get("node_id"),
                )
                self.emit_event(event)
                await self.emit_event_async(event)
                yield event

        except Exception as e:
            error_event = StreamEvent(
                event_type=StreamEventType.ERROR,
                data={"error": str(e)},
                execution_id=execution_id,
            )
            yield error_event

        end_event = StreamEvent(
            event_type=StreamEventType.EXECUTION_END,
            data={"execution_id": execution_id},
            execution_id=execution_id,
        )
        yield end_event

    # === Utility Methods ===

    def get_content(self) -> str:
        """Get accumulated content."""
        return self._buffer.get_content()

    def get_token_count(self) -> int:
        """Get total token count."""
        return self._buffer.token_count

    def clear(self) -> None:
        """Clear the buffer."""
        self._buffer.clear()


class SSEStream:
    """
    Server-Sent Events stream for web clients.

    Usage (FastAPI):
        @app.get("/stream")
        async def stream():
            sse = SSEStream()

            async def generate():
                async for event in handler.stream_tokens(tokens):
                    yield sse.format(event)
                yield sse.done()

            return StreamingResponse(generate(), media_type="text/event-stream")
    """

    @staticmethod
    def format(event: StreamEvent) -> str:
        """Format event as SSE."""
        return event.to_sse()

    @staticmethod
    def done() -> str:
        """Send done event."""
        return "event: done\ndata: {}\n\n"

    @staticmethod
    def heartbeat() -> str:
        """Send heartbeat."""
        return ": heartbeat\n\n"

    @staticmethod
    def error(message: str) -> str:
        """Send error event."""
        data = json.dumps({"error": message})
        return f"event: error\ndata: {data}\n\n"


class WebSocketStream:
    """
    WebSocket stream for real-time bidirectional communication.

    Usage (FastAPI):
        @app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            ws_stream = WebSocketStream(websocket)

            async for event in handler.stream_execution(events, exec_id):
                await ws_stream.send(event)
    """

    def __init__(self, websocket: Any):
        self.websocket = websocket

    async def send(self, event: StreamEvent) -> None:
        """Send an event."""
        await self.websocket.send_json(event.to_dict())

    async def send_token(self, token: str) -> None:
        """Send a token."""
        await self.websocket.send_json(
            {
                "type": "token",
                "data": {"token": token},
                "timestamp": time.time(),
            }
        )

    async def send_error(self, message: str) -> None:
        """Send an error."""
        await self.websocket.send_json(
            {
                "type": "error",
                "data": {"error": message},
                "timestamp": time.time(),
            }
        )

    async def send_done(self) -> None:
        """Send done signal."""
        await self.websocket.send_json(
            {
                "type": "done",
                "data": {},
                "timestamp": time.time(),
            }
        )


# Helper for creating streaming response in FastAPI
def create_streaming_response(
    stream_handler: StreamingHandler,
    token_iterator: AsyncIterator[str],
    media_type: str = "text/event-stream",
):
    """
    Create a FastAPI StreamingResponse.

    Usage:
        from fastapi.responses import StreamingResponse

        @app.get("/stream")
        async def stream():
            handler = StreamingHandler()
            return create_streaming_response(handler, token_iterator)
    """
    try:
        from fastapi.responses import StreamingResponse
    except ImportError:
        raise ImportError("FastAPI required for streaming response")

    async def generate():
        async for event in stream_handler.stream_tokens(token_iterator):
            yield event.to_sse()

    return StreamingResponse(generate(), media_type=media_type)
