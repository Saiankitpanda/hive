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
