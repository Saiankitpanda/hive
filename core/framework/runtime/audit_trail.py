"""
Audit Trail - Decision timeline and execution history tracking.

Provides comprehensive logging of:
- Agent decisions and reasoning
- Node executions and state changes
- Tool usage and results
- Human interventions
- Goal progress updates

Use Cases:
- Debugging agent behavior
- Compliance and auditing
- Performance analysis
- Understanding agent reasoning
"""

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any
from collections.abc import Callable


class EventType(Enum):
    """Types of audit events."""

    # Execution events
    EXECUTION_START = "execution_start"
    EXECUTION_END = "execution_end"
    EXECUTION_ERROR = "execution_error"

    # Node events
    NODE_START = "node_start"
    NODE_END = "node_end"
    NODE_ERROR = "node_error"
    NODE_SKIP = "node_skip"

    # Decision events
    DECISION = "decision"
    REASONING = "reasoning"
    ROUTING = "routing"

    # State events
    STATE_CHANGE = "state_change"
    STATE_CHECKPOINT = "state_checkpoint"

    # Tool events
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    TOOL_ERROR = "tool_error"

    # Human events
    HUMAN_INPUT = "human_input"
    HUMAN_APPROVAL = "human_approval"
    HUMAN_REJECTION = "human_rejection"

    # Goal events
    GOAL_PROGRESS = "goal_progress"
    GOAL_COMPLETE = "goal_complete"
    GOAL_FAILED = "goal_failed"

    # Guardrail events
    GUARDRAIL_CHECK = "guardrail_check"
    GUARDRAIL_VIOLATION = "guardrail_violation"


@dataclass
class AuditEvent:
    """A single audit event."""

    id: str
    event_type: EventType
    timestamp: float
    data: dict[str, Any]
    execution_id: str | None = None
    node_id: str | None = None
    correlation_id: str | None = None
    duration_ms: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp,
            "datetime": datetime.fromtimestamp(self.timestamp).isoformat(),
            "data": self.data,
            "execution_id": self.execution_id,
            "node_id": self.node_id,
            "correlation_id": self.correlation_id,
            "duration_ms": self.duration_ms,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AuditEvent":
        return cls(
            id=data["id"],
            event_type=EventType(data["event_type"]),
            timestamp=data["timestamp"],
            data=data.get("data", {}),
            execution_id=data.get("execution_id"),
            node_id=data.get("node_id"),
            correlation_id=data.get("correlation_id"),
            duration_ms=data.get("duration_ms"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class DecisionRecord:
    """Record of a decision made by the agent."""

    id: str
    timestamp: float
    decision: str
    reasoning: str
    alternatives: list[str]
    confidence: float
    context: dict[str, Any]
    outcome: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "datetime": datetime.fromtimestamp(self.timestamp).isoformat(),
            "decision": self.decision,
            "reasoning": self.reasoning,
            "alternatives": self.alternatives,
            "confidence": self.confidence,
            "context": self.context,
            "outcome": self.outcome,
        }


class AuditTrail:
    """
    Audit trail for tracking agent decisions and actions.

    Usage:
        audit = AuditTrail(storage_path="./audit_logs")

        # Log events
        audit.log_execution_start("exec-123", {"input": "..."})
        audit.log_decision(
            decision="Use web search",
            reasoning="User asked about current events",
            alternatives=["Use knowledge base", "Ask for clarification"]
        )
        audit.log_node_execution("node-1", {"result": "..."})

        # Query events
        events = audit.get_events(event_type=EventType.DECISION)
        timeline = audit.get_timeline("exec-123")

        # Export
        audit.export_json("audit_export.json")
    """

    def __init__(
        self,
        storage_path: str | Path | None = None,
        max_events: int = 10000,
        auto_persist: bool = True,
    ):
        """
        Initialize audit trail.

        Args:
            storage_path: Path to store audit logs
            max_events: Maximum events to keep in memory
            auto_persist: Automatically persist events to disk
        """
        self.storage_path = Path(storage_path) if storage_path else None
        self.max_events = max_events
        self.auto_persist = auto_persist

        self._events: list[AuditEvent] = []
        self._decisions: list[DecisionRecord] = []
        self._event_handlers: list[Callable[[AuditEvent], None]] = []
        self._event_counter = 0

        if self.storage_path:
            self.storage_path.mkdir(parents=True, exist_ok=True)

    # === Logging Methods ===

    def log_event(
        self,
        event_type: EventType,
        data: dict[str, Any],
        execution_id: str | None = None,
        node_id: str | None = None,
        correlation_id: str | None = None,
        duration_ms: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AuditEvent:
        """Log a generic event."""
        self._event_counter += 1
        event = AuditEvent(
            id=f"evt-{self._event_counter:06d}",
            event_type=event_type,
            timestamp=time.time(),
            data=data,
            execution_id=execution_id,
            node_id=node_id,
            correlation_id=correlation_id,
            duration_ms=duration_ms,
            metadata=metadata or {},
        )

        self._events.append(event)

        # Trim if needed
        if len(self._events) > self.max_events:
            self._events = self._events[-self.max_events :]

        # Notify handlers
        for handler in self._event_handlers:
            try:
                handler(event)
            except Exception:
                pass

        # Auto-persist
        if self.auto_persist and self.storage_path:
            self._persist_event(event)

        return event

    def log_execution_start(
        self,
        execution_id: str,
        input_data: dict[str, Any],
        **kwargs,
    ) -> AuditEvent:
        """Log execution start."""
        return self.log_event(
            EventType.EXECUTION_START,
            {"input": input_data},
            execution_id=execution_id,
            **kwargs,
        )

    def log_execution_end(
        self,
        execution_id: str,
        output: dict[str, Any],
        duration_ms: float,
        **kwargs,
    ) -> AuditEvent:
        """Log execution end."""
        return self.log_event(
            EventType.EXECUTION_END,
            {"output": output},
            execution_id=execution_id,
            duration_ms=duration_ms,
            **kwargs,
        )

    def log_node_execution(
        self,
        node_id: str,
        result: Any,
        duration_ms: float | None = None,
        execution_id: str | None = None,
        **kwargs,
    ) -> AuditEvent:
        """Log node execution."""
        return self.log_event(
            EventType.NODE_END,
            {"result": result},
            node_id=node_id,
            execution_id=execution_id,
            duration_ms=duration_ms,
            **kwargs,
        )

    def log_decision(
        self,
        decision: str,
        reasoning: str,
        alternatives: list[str] | None = None,
        confidence: float = 1.0,
        context: dict[str, Any] | None = None,
        execution_id: str | None = None,
    ) -> DecisionRecord:
        """Log a decision with reasoning."""
        self._event_counter += 1
        record = DecisionRecord(
            id=f"dec-{self._event_counter:06d}",
            timestamp=time.time(),
            decision=decision,
            reasoning=reasoning,
            alternatives=alternatives or [],
            confidence=confidence,
            context=context or {},
        )

        self._decisions.append(record)

        # Also log as event
        self.log_event(
            EventType.DECISION,
            record.to_dict(),
            execution_id=execution_id,
        )

        return record

    def log_tool_call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        execution_id: str | None = None,
        **kwargs,
    ) -> AuditEvent:
        """Log a tool call."""
        return self.log_event(
            EventType.TOOL_CALL,
            {"tool": tool_name, "arguments": arguments},
            execution_id=execution_id,
            **kwargs,
        )

    def log_tool_result(
        self,
        tool_name: str,
        result: Any,
        duration_ms: float,
        execution_id: str | None = None,
        **kwargs,
    ) -> AuditEvent:
        """Log a tool result."""
        return self.log_event(
            EventType.TOOL_RESULT,
            {"tool": tool_name, "result": result},
            execution_id=execution_id,
            duration_ms=duration_ms,
            **kwargs,
        )

    def log_human_input(
        self,
        input_type: str,
        content: Any,
        execution_id: str | None = None,
        **kwargs,
    ) -> AuditEvent:
        """Log human input or intervention."""
        return self.log_event(
            EventType.HUMAN_INPUT,
            {"type": input_type, "content": content},
            execution_id=execution_id,
            **kwargs,
        )

    def log_guardrail_violation(
        self,
        guardrail_name: str,
        violation: dict[str, Any],
        execution_id: str | None = None,
        **kwargs,
    ) -> AuditEvent:
        """Log a guardrail violation."""
        return self.log_event(
            EventType.GUARDRAIL_VIOLATION,
            {"guardrail": guardrail_name, "violation": violation},
            execution_id=execution_id,
            **kwargs,
        )

    # === Query Methods ===

    def get_events(
        self,
        event_type: EventType | None = None,
        execution_id: str | None = None,
        node_id: str | None = None,
        start_time: float | None = None,
        end_time: float | None = None,
        limit: int = 100,
    ) -> list[AuditEvent]:
        """Query events with filters."""
        events = self._events

        if event_type:
            events = [e for e in events if e.event_type == event_type]

        if execution_id:
            events = [e for e in events if e.execution_id == execution_id]

        if node_id:
            events = [e for e in events if e.node_id == node_id]

        if start_time:
            events = [e for e in events if e.timestamp >= start_time]

        if end_time:
            events = [e for e in events if e.timestamp <= end_time]

        return events[-limit:]

    def get_timeline(self, execution_id: str) -> list[AuditEvent]:
        """Get timeline of events for an execution."""
        return sorted(
            [e for e in self._events if e.execution_id == execution_id],
            key=lambda e: e.timestamp,
        )

    def get_decisions(self, limit: int = 50) -> list[DecisionRecord]:
        """Get recent decisions."""
        return self._decisions[-limit:]

    def get_decision_by_id(self, decision_id: str) -> DecisionRecord | None:
        """Get a specific decision."""
        for d in self._decisions:
            if d.id == decision_id:
                return d
        return None

    # === Analytics ===

    def get_stats(self) -> dict[str, Any]:
        """Get audit statistics."""
        event_counts = {}
        for event in self._events:
            key = event.event_type.value
            event_counts[key] = event_counts.get(key, 0) + 1

        return {
            "total_events": len(self._events),
            "total_decisions": len(self._decisions),
            "event_counts": event_counts,
            "oldest_event": self._events[0].timestamp if self._events else None,
            "newest_event": self._events[-1].timestamp if self._events else None,
        }

    def get_execution_summary(self, execution_id: str) -> dict[str, Any]:
        """Get summary of an execution."""
        events = self.get_timeline(execution_id)

        if not events:
            return {"execution_id": execution_id, "found": False}

        start_event = next(
            (e for e in events if e.event_type == EventType.EXECUTION_START),
            None,
        )
        end_event = next(
            (e for e in reversed(events) if e.event_type == EventType.EXECUTION_END),
            None,
        )

        node_events = [e for e in events if e.node_id]
        tool_events = [
            e for e in events if e.event_type in (EventType.TOOL_CALL, EventType.TOOL_RESULT)
        ]
        decision_events = [e for e in events if e.event_type == EventType.DECISION]

        return {
            "execution_id": execution_id,
            "found": True,
            "start_time": start_event.timestamp if start_event else None,
            "end_time": end_event.timestamp if end_event else None,
            "duration_ms": end_event.duration_ms if end_event else None,
            "total_events": len(events),
            "nodes_executed": len(set(e.node_id for e in node_events if e.node_id)),
            "tool_calls": len([e for e in tool_events if e.event_type == EventType.TOOL_CALL]),
            "decisions_made": len(decision_events),
        }

    # === Export ===

    def export_json(self, filepath: str | Path) -> None:
        """Export audit trail to JSON file."""
        data = {
            "events": [e.to_dict() for e in self._events],
            "decisions": [d.to_dict() for d in self._decisions],
            "stats": self.get_stats(),
            "exported_at": datetime.now().isoformat(),
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def export_timeline_markdown(
        self,
        execution_id: str,
        filepath: str | Path,
    ) -> None:
        """Export execution timeline as markdown."""
        events = self.get_timeline(execution_id)
        summary = self.get_execution_summary(execution_id)

        lines = [
            f"# Execution Timeline: {execution_id}",
            "",
            "## Summary",
            f"- **Duration**: {summary.get('duration_ms', 'N/A')}ms",
            f"- **Nodes Executed**: {summary.get('nodes_executed', 0)}",
            f"- **Tool Calls**: {summary.get('tool_calls', 0)}",
            f"- **Decisions**: {summary.get('decisions_made', 0)}",
            "",
            "## Timeline",
            "",
        ]

        for event in events:
            dt = datetime.fromtimestamp(event.timestamp).strftime("%H:%M:%S.%f")[:-3]
            icon = self._get_event_icon(event.event_type)
            lines.append(f"### {icon} [{dt}] {event.event_type.value}")
            if event.node_id:
                lines.append(f"**Node**: {event.node_id}")
            if event.duration_ms:
                lines.append(f"**Duration**: {event.duration_ms:.2f}ms")
            lines.append(f"```json\n{json.dumps(event.data, indent=2)}\n```")
            lines.append("")

        with open(filepath, "w") as f:
            f.write("\n".join(lines))

    # === Event Handlers ===

    def on_event(self, handler: Callable[[AuditEvent], None]) -> None:
        """Register an event handler."""
        self._event_handlers.append(handler)

    # === Persistence ===

    def _persist_event(self, event: AuditEvent) -> None:
        """Persist a single event to disk."""
        if not self.storage_path:
            return

        date_str = datetime.fromtimestamp(event.timestamp).strftime("%Y-%m-%d")
        log_file = self.storage_path / f"audit_{date_str}.jsonl"

        with open(log_file, "a") as f:
            f.write(json.dumps(event.to_dict()) + "\n")

    def load_events(self, date: str | None = None) -> None:
        """Load events from disk."""
        if not self.storage_path:
            return

        pattern = f"audit_{date}.jsonl" if date else "audit_*.jsonl"

        for log_file in self.storage_path.glob(pattern):
            with open(log_file) as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        event = AuditEvent.from_dict(data)
                        self._events.append(event)
                    except (json.JSONDecodeError, KeyError):
                        continue

        self._events.sort(key=lambda e: e.timestamp)

    def _get_event_icon(self, event_type: EventType) -> str:
        """Get icon for event type."""
        icons = {
            EventType.EXECUTION_START: "🚀",
            EventType.EXECUTION_END: "✅",
            EventType.EXECUTION_ERROR: "❌",
            EventType.NODE_START: "▶️",
            EventType.NODE_END: "⏹️",
            EventType.DECISION: "🤔",
            EventType.TOOL_CALL: "🔧",
            EventType.TOOL_RESULT: "📤",
            EventType.HUMAN_INPUT: "👤",
            EventType.GUARDRAIL_VIOLATION: "🛡️",
        }
        return icons.get(event_type, "📌")
