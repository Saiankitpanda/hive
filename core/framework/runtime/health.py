"""
Health Check API - Basic lifecycle and monitoring endpoints.

Provides health check, readiness, and liveness probes for:
- Kubernetes deployments
- Docker health checks
- Load balancer health monitoring
- CI/CD pipeline integration

Endpoints:
- /health - Basic health status
- /ready - Readiness probe (are all components ready?)
- /live - Liveness probe (is the service alive?)
- /metrics - Basic metrics (execution counts, latency)
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from framework.runtime.agent_runtime import AgentRuntime


class HealthStatus(Enum):
    """Health status levels."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class ComponentHealth:
    """Health status of a component."""

    name: str
    status: HealthStatus
    message: str = ""
    latency_ms: float = 0.0
    last_check: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "latency_ms": round(self.latency_ms, 2),
            "last_check": self.last_check,
        }


@dataclass
class HealthReport:
    """Complete health report."""

    status: HealthStatus
    version: str
    uptime_seconds: float
    components: list[ComponentHealth]
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "version": self.version,
            "uptime_seconds": round(self.uptime_seconds, 2),
            "components": [c.to_dict() for c in self.components],
            "timestamp": self.timestamp,
        }


@dataclass
class MetricsReport:
    """Basic metrics report."""

    total_executions: int
    successful_executions: int
    failed_executions: int
    avg_latency_ms: float
    active_streams: int
    uptime_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_executions": self.total_executions,
            "successful_executions": self.successful_executions,
            "failed_executions": self.failed_executions,
            "success_rate": (
                round(self.successful_executions / self.total_executions * 100, 2)
                if self.total_executions > 0
                else 0.0
            ),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "active_streams": self.active_streams,
            "uptime_seconds": round(self.uptime_seconds, 2),
        }


class HealthCheckService:
    """
    Health check service for agent monitoring.

    Provides health, readiness, and liveness probes for:
    - Kubernetes deployments
    - Docker health checks
    - Load balancer monitoring

    Usage:
        runtime = AgentRuntime(...)
        health = HealthCheckService(runtime)

        # Basic health check
        report = health.get_health()
        print(report.status)  # HealthStatus.HEALTHY

        # Kubernetes probes
        is_ready = health.is_ready()
        is_live = health.is_live()

        # Metrics
        metrics = health.get_metrics()
    """

    VERSION = "1.1.0"

    def __init__(
        self,
        runtime: AgentRuntime | None = None,
        version: str | None = None,
    ):
        """
        Initialize health check service.

        Args:
            runtime: AgentRuntime to monitor
            version: Application version string
        """
        self._runtime = runtime
        self._version = version or self.VERSION
        self._start_time = time.time()
        self._execution_latencies: list[float] = []
        self._max_latency_samples = 1000

    def get_health(self) -> HealthReport:
        """
        Get comprehensive health report.

        Returns:
            HealthReport with status of all components
        """
        components = []

        # Check runtime
        runtime_health = self._check_runtime()
        components.append(runtime_health)

        # Check event bus
        event_bus_health = self._check_event_bus()
        components.append(event_bus_health)

        # Check state manager
        state_health = self._check_state_manager()
        components.append(state_health)

        # Determine overall status
        if any(c.status == HealthStatus.UNHEALTHY for c in components):
            overall_status = HealthStatus.UNHEALTHY
        elif any(c.status == HealthStatus.DEGRADED for c in components):
            overall_status = HealthStatus.DEGRADED
        else:
            overall_status = HealthStatus.HEALTHY

        return HealthReport(
            status=overall_status,
            version=self._version,
            uptime_seconds=time.time() - self._start_time,
            components=components,
        )

    def is_ready(self) -> bool:
        """
        Readiness probe - is the service ready to accept traffic?

        Returns:
            True if all components are ready
        """
        report = self.get_health()
        return report.status != HealthStatus.UNHEALTHY

    def is_live(self) -> bool:
        """
        Liveness probe - is the service alive?

        Returns:
            True if the service is running
        """
        # Basic liveness - just check if we can respond
        return True

    def get_metrics(self) -> MetricsReport:
        """
        Get basic metrics.

        Returns:
            MetricsReport with execution statistics
        """
        stats = self._get_runtime_stats()

        return MetricsReport(
            total_executions=stats.get("total_executions", 0),
            successful_executions=stats.get("successful_executions", 0),
            failed_executions=stats.get("failed_executions", 0),
            avg_latency_ms=self._calculate_avg_latency(),
            active_streams=stats.get("active_streams", 0),
            uptime_seconds=time.time() - self._start_time,
        )

    def record_execution_latency(self, latency_ms: float) -> None:
        """Record an execution latency for metrics."""
        self._execution_latencies.append(latency_ms)
        # Keep only recent samples
        if len(self._execution_latencies) > self._max_latency_samples:
            self._execution_latencies = self._execution_latencies[-self._max_latency_samples :]

    def _check_runtime(self) -> ComponentHealth:
        """Check runtime health."""
        start = time.time()

        if self._runtime is None:
            return ComponentHealth(
                name="runtime",
                status=HealthStatus.UNHEALTHY,
                message="Runtime not configured",
            )

        try:
            # Check if runtime is running
            if hasattr(self._runtime, "_is_running") and self._runtime._is_running:
                status = HealthStatus.HEALTHY
                message = "Runtime is running"
            else:
                status = HealthStatus.DEGRADED
                message = "Runtime is not started"

            return ComponentHealth(
                name="runtime",
                status=status,
                message=message,
                latency_ms=(time.time() - start) * 1000,
            )
        except Exception as e:
            return ComponentHealth(
                name="runtime",
                status=HealthStatus.UNHEALTHY,
                message=f"Runtime check failed: {e}",
                latency_ms=(time.time() - start) * 1000,
            )

    def _check_event_bus(self) -> ComponentHealth:
        """Check event bus health."""
        start = time.time()

        if self._runtime is None:
            return ComponentHealth(
                name="event_bus",
                status=HealthStatus.DEGRADED,
                message="No runtime configured",
            )

        try:
            if hasattr(self._runtime, "_event_bus") and self._runtime._event_bus:
                return ComponentHealth(
                    name="event_bus",
                    status=HealthStatus.HEALTHY,
                    message="Event bus operational",
                    latency_ms=(time.time() - start) * 1000,
                )
            else:
                return ComponentHealth(
                    name="event_bus",
                    status=HealthStatus.DEGRADED,
                    message="Event bus not initialized",
                    latency_ms=(time.time() - start) * 1000,
                )
        except Exception as e:
            return ComponentHealth(
                name="event_bus",
                status=HealthStatus.UNHEALTHY,
                message=f"Event bus check failed: {e}",
                latency_ms=(time.time() - start) * 1000,
            )

    def _check_state_manager(self) -> ComponentHealth:
        """Check state manager health."""
        start = time.time()

        if self._runtime is None:
            return ComponentHealth(
                name="state_manager",
                status=HealthStatus.DEGRADED,
                message="No runtime configured",
            )

        try:
            if hasattr(self._runtime, "_state_manager") and self._runtime._state_manager:
                return ComponentHealth(
                    name="state_manager",
                    status=HealthStatus.HEALTHY,
                    message="State manager operational",
                    latency_ms=(time.time() - start) * 1000,
                )
            else:
                return ComponentHealth(
                    name="state_manager",
                    status=HealthStatus.DEGRADED,
                    message="State manager not initialized",
                    latency_ms=(time.time() - start) * 1000,
                )
        except Exception as e:
            return ComponentHealth(
                name="state_manager",
                status=HealthStatus.UNHEALTHY,
                message=f"State manager check failed: {e}",
                latency_ms=(time.time() - start) * 1000,
            )

    def _get_runtime_stats(self) -> dict[str, Any]:
        """Get stats from runtime."""
        if self._runtime is None:
            return {}

        try:
            if hasattr(self._runtime, "get_stats"):
                return self._runtime.get_stats()
        except Exception:
            pass

        return {}

    def _calculate_avg_latency(self) -> float:
        """Calculate average execution latency."""
        if not self._execution_latencies:
            return 0.0
        return sum(self._execution_latencies) / len(self._execution_latencies)


# FastAPI integration helper
def create_health_routes(app, health_service: HealthCheckService):
    """
    Add health check routes to a FastAPI app.

    Usage:
        from fastapi import FastAPI
        from framework.runtime.health import HealthCheckService, create_health_routes

        app = FastAPI()
        health = HealthCheckService(runtime)
        create_health_routes(app, health)
    """
    try:
        from fastapi import Response
        from fastapi.responses import JSONResponse
    except ImportError:
        return  # FastAPI not installed

    @app.get("/health")
    async def health():
        """Health check endpoint."""
        report = health_service.get_health()
        status_code = 200 if report.status == HealthStatus.HEALTHY else 503
        return JSONResponse(content=report.to_dict(), status_code=status_code)

    @app.get("/ready")
    async def ready():
        """Readiness probe for Kubernetes."""
        if health_service.is_ready():
            return {"status": "ready"}
        return Response(status_code=503, content='{"status": "not ready"}')

    @app.get("/live")
    async def live():
        """Liveness probe for Kubernetes."""
        if health_service.is_live():
            return {"status": "alive"}
        return Response(status_code=503, content='{"status": "dead"}')

    @app.get("/metrics")
    async def metrics():
        """Basic metrics endpoint."""
        return health_service.get_metrics().to_dict()
