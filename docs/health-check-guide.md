# Health Check API Guide

The Health Check API provides Kubernetes-compatible probes and metrics for agent monitoring.

## Quick Start

```python
from framework.runtime import HealthCheckService
from framework.runtime.agent_runtime import AgentRuntime

# Create runtime and health service
runtime = AgentRuntime(...)
health = HealthCheckService(runtime)

# Check health
report = health.get_health()
print(report.status)  # HealthStatus.HEALTHY
```

## Endpoints

| Endpoint | Description | Use Case |
|----------|-------------|----------|
| `/health` | Full health report | Monitoring dashboards |
| `/ready` | Readiness probe | Kubernetes readinessProbe |
| `/live` | Liveness probe | Kubernetes livenessProbe |
| `/metrics` | Basic metrics | Prometheus/monitoring |

## FastAPI Integration

```python
from fastapi import FastAPI
from framework.runtime.health import HealthCheckService, create_health_routes

app = FastAPI()
health = HealthCheckService(runtime)
create_health_routes(app, health)

# Routes automatically added:
# GET /health
# GET /ready
# GET /live
# GET /metrics
```

## Kubernetes Configuration

```yaml
apiVersion: v1
kind: Pod
spec:
  containers:
  - name: agent
    livenessProbe:
      httpGet:
        path: /live
        port: 8000
      initialDelaySeconds: 10
      periodSeconds: 5
    readinessProbe:
      httpGet:
        path: /ready
        port: 8000
      initialDelaySeconds: 5
      periodSeconds: 3
```

## Docker Health Check

```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1
```

## Metrics Response

```json
{
  "total_executions": 150,
  "successful_executions": 142,
  "failed_executions": 8,
  "success_rate": 94.67,
  "avg_latency_ms": 245.5,
  "active_streams": 3,
  "uptime_seconds": 3600.5
}
```

## Component Health

The `/health` endpoint checks:
- **runtime**: Agent runtime status
- **event_bus**: Event bus operational
- **state_manager**: State manager status

```json
{
  "status": "healthy",
  "version": "1.1.0",
  "uptime_seconds": 3600.5,
  "components": [
    {"name": "runtime", "status": "healthy", "latency_ms": 0.5},
    {"name": "event_bus", "status": "healthy", "latency_ms": 0.2},
    {"name": "state_manager", "status": "healthy", "latency_ms": 0.3}
  ]
}
```
