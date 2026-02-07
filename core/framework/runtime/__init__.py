"""Runtime core for agent execution."""

from framework.runtime.core import Runtime

__all__ = ["Runtime"]

# Health check service
try:
    from framework.runtime.health import (  # noqa: F401
        HealthCheckService,
        HealthReport,
        HealthStatus,
        MetricsReport,
    )

    __all__.extend(["HealthCheckService", "HealthReport", "HealthStatus", "MetricsReport"])
except ImportError:
    pass

# Guardrails system
try:
    from framework.runtime.guardrails import (  # noqa: F401
        Guardrail,
        GuardrailAction,
        GuardrailResult,
        GuardrailsManager,
        TokenLimitGuardrail,
        BudgetGuardrail,
        ContentFilterGuardrail,
        PIIGuardrail,
        TimeoutGuardrail,
        IterationLimitGuardrail,
        create_default_guardrails,
        create_strict_guardrails,
    )

    __all__.extend(
        [
            "Guardrail",
            "GuardrailAction",
            "GuardrailResult",
            "GuardrailsManager",
            "TokenLimitGuardrail",
            "BudgetGuardrail",
            "ContentFilterGuardrail",
            "PIIGuardrail",
            "TimeoutGuardrail",
            "IterationLimitGuardrail",
            "create_default_guardrails",
            "create_strict_guardrails",
        ]
    )
except ImportError:
    pass

# Audit trail
try:
    from framework.runtime.audit_trail import (  # noqa: F401
        AuditTrail,
        AuditEvent,
        EventType,
        DecisionRecord,
    )

    __all__.extend(["AuditTrail", "AuditEvent", "EventType", "DecisionRecord"])
except ImportError:
    pass

# Node discovery
try:
    from framework.runtime.node_discovery import (  # noqa: F401
        NodeDiscoveryService,
        AgentInfo,
        DiscoveryResult,
        NODE_DISCOVERY_TOOL,
    )

    __all__.extend(["NodeDiscoveryService", "AgentInfo", "DiscoveryResult", "NODE_DISCOVERY_TOOL"])
except ImportError:
    pass

# HITL (Human-in-the-Loop)
try:
    from framework.runtime.hitl import (  # noqa: F401
        HITLService,
        ApprovalRequest,
        ApprovalStatus,
        ApprovalType,
        HITL_TOOL,
    )

    __all__.extend(
        ["HITLService", "ApprovalRequest", "ApprovalStatus", "ApprovalType", "HITL_TOOL"]
    )
except ImportError:
    pass
