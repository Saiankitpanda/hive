"""
Telemetry - Observability for agent execution.

Provides:
- Metrics collection (counters, gauges, histograms)
- Distributed tracing
- Event logging
- Performance profiling

From ROADMAP: Production observability features
"""

import time
from collections import defaultdict
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from threading import Lock
from typing import Any, TypeVar

T = TypeVar("T")


class MetricType(Enum):
    """Types of metrics."""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class Metric:
    """A single metric."""

    name: str
    metric_type: MetricType
    value: float = 0.0
    labels: dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": self.metric_type.value,
            "value": self.value,
            "labels": self.labels,
            "timestamp": self.timestamp,
        }


@dataclass
class Span:
    """A tracing span."""

    trace_id: str
    span_id: str
    name: str
    start_time: float
    end_time: float | None = None
    parent_id: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    status: str = "ok"

    @property
    def duration_ms(self) -> float:
        if self.end_time:
            return (self.end_time - self.start_time) * 1000
        return 0.0

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        self.events.append(
            {
                "name": name,
                "timestamp": time.time(),
                "attributes": attributes or {},
            }
        )

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "name": self.name,
            "parent_id": self.parent_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "attributes": self.attributes,
            "events": self.events,
            "status": self.status,
        }


class MetricsCollector:
    """
    Collects and manages metrics.

    Usage:
        metrics = MetricsCollector()

        # Counters
        metrics.increment("requests_total")
        metrics.increment("errors_total", labels={"type": "timeout"})

        # Gauges
        metrics.set_gauge("active_connections", 42)

        # Histograms
        metrics.observe("request_duration", 0.5)

        # Export
        all_metrics = metrics.export()
    """

    def __init__(self):
        self._counters: dict[str, float] = defaultdict(float)
        self._gauges: dict[str, float] = {}
        self._histograms: dict[str, list[float]] = defaultdict(list)
        self._lock = Lock()

    def increment(
        self,
        name: str,
        value: float = 1.0,
        labels: dict[str, str] | None = None,
    ) -> None:
        """Increment a counter."""
        key = self._make_key(name, labels)
        with self._lock:
            self._counters[key] += value

    def decrement(
        self,
        name: str,
        value: float = 1.0,
        labels: dict[str, str] | None = None,
    ) -> None:
        """Decrement a counter."""
        key = self._make_key(name, labels)
        with self._lock:
            self._counters[key] -= value

    def set_gauge(
        self,
        name: str,
        value: float,
        labels: dict[str, str] | None = None,
    ) -> None:
        """Set a gauge value."""
        key = self._make_key(name, labels)
        with self._lock:
            self._gauges[key] = value

    def observe(
        self,
        name: str,
        value: float,
        labels: dict[str, str] | None = None,
    ) -> None:
        """Observe a histogram value."""
        key = self._make_key(name, labels)
        with self._lock:
            self._histograms[key].append(value)

    @contextmanager
    def timer(self, name: str, labels: dict[str, str] | None = None):
        """Context manager to time operations."""
        start = time.time()
        try:
            yield
        finally:
            duration = time.time() - start
            self.observe(name, duration, labels)

    def _make_key(self, name: str, labels: dict[str, str] | None) -> str:
        """Create key from name and labels."""
        if not labels:
            return name
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"

    def get_counter(self, name: str, labels: dict[str, str] | None = None) -> float:
        """Get counter value."""
        key = self._make_key(name, labels)
        return self._counters.get(key, 0.0)

    def get_gauge(self, name: str, labels: dict[str, str] | None = None) -> float:
        """Get gauge value."""
        key = self._make_key(name, labels)
        return self._gauges.get(key, 0.0)

    def get_histogram_stats(
        self, name: str, labels: dict[str, str] | None = None
    ) -> dict[str, float]:
        """Get histogram statistics."""
        key = self._make_key(name, labels)
        values = self._histograms.get(key, [])

        if not values:
            return {"count": 0, "sum": 0, "mean": 0, "min": 0, "max": 0}

        return {
            "count": len(values),
            "sum": sum(values),
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
        }

    def export(self) -> dict[str, Any]:
        """Export all metrics."""
        with self._lock:
            return {
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "histograms": {k: self.get_histogram_stats(k) for k in self._histograms},
            }

    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()


class Tracer:
    """
    Distributed tracing support.

    Usage:
        tracer = Tracer()

        with tracer.span("process_request") as span:
            span.set_attribute("user_id", "123")
            # ... do work ...
            span.add_event("validation_complete")

        # Nested spans
        with tracer.span("parent") as parent:
            with tracer.span("child", parent=parent) as child:
                # ... nested work ...
                pass
    """

    def __init__(self, service_name: str = "agent"):
        self.service_name = service_name
        self._spans: list[Span] = []
        self._active_span: Span | None = None
        self._lock = Lock()

    def _generate_id(self) -> str:
        """Generate random ID."""
        import uuid

        return uuid.uuid4().hex[:16]

    @contextmanager
    def span(
        self,
        name: str,
        parent: Span | None = None,
        attributes: dict[str, Any] | None = None,
    ):
        """Create a span for tracing."""
        parent = parent or self._active_span

        trace_id = parent.trace_id if parent else self._generate_id()
        span_id = self._generate_id()
        parent_id = parent.span_id if parent else None

        span = Span(
            trace_id=trace_id,
            span_id=span_id,
            name=name,
            start_time=time.time(),
            parent_id=parent_id,
            attributes=attributes or {},
        )

        previous_span = self._active_span
        self._active_span = span

        try:
            yield span
            span.status = "ok"
        except Exception as e:
            span.status = "error"
            span.set_attribute("error.type", type(e).__name__)
            span.set_attribute("error.message", str(e))
            raise
        finally:
            span.end_time = time.time()
            with self._lock:
                self._spans.append(span)
            self._active_span = previous_span

    def get_spans(self) -> list[Span]:
        """Get all recorded spans."""
        with self._lock:
            return list(self._spans)

    def export(self) -> list[dict[str, Any]]:
        """Export all spans as dicts."""
        return [s.to_dict() for s in self.get_spans()]

    def clear(self) -> None:
        """Clear recorded spans."""
        with self._lock:
            self._spans.clear()


def trace(name: str | None = None, tracer: Tracer | None = None) -> Callable:
    """
    Decorator to trace function execution.

    Usage:
        @trace("process_data")
        def process(data):
            ...

        @trace()  # Uses function name
        async def async_process(data):
            ...
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        span_name = name or func.__name__
        t = tracer or _default_tracer

        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            with t.span(span_name):
                return func(*args, **kwargs)

        return wrapper

    return decorator


def timed(
    metric_name: str | None = None,
    metrics: MetricsCollector | None = None,
) -> Callable:
    """
    Decorator to time function execution.

    Usage:
        @timed("api_call_duration")
        def api_call():
            ...
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        name = metric_name or f"{func.__name__}_duration"
        m = metrics or _default_metrics

        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            with m.timer(name):
                return func(*args, **kwargs)

        return wrapper

    return decorator


# Global instances
_default_metrics = MetricsCollector()
_default_tracer = Tracer()


def get_metrics() -> MetricsCollector:
    """Get default metrics collector."""
    return _default_metrics


def get_tracer() -> Tracer:
    """Get default tracer."""
    return _default_tracer
