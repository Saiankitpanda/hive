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

# Streaming support
try:
    from framework.runtime.streaming import (  # noqa: F401
        StreamingHandler,
        StreamEvent,
        StreamEventType,
        StreamBuffer,
        SSEStream,
        WebSocketStream,
    )

    __all__.extend(
        [
            "StreamingHandler",
            "StreamEvent",
            "StreamEventType",
            "StreamBuffer",
            "SSEStream",
            "WebSocketStream",
        ]
    )
except ImportError:
    pass

# Caching
try:
    from framework.runtime.caching import (  # noqa: F401
        LLMCache,
        CacheEntry,
        CacheStats,
        ResponseCache,
        create_llm_cache,
    )

    __all__.extend(["LLMCache", "CacheEntry", "CacheStats", "ResponseCache", "create_llm_cache"])
except ImportError:
    pass

# Configuration
try:
    from framework.runtime.config import (  # noqa: F401
        AgentConfig,
        LLMConfig,
        RuntimeConfig,
        GuardrailsConfig,
        ConfigManager,
        get_config,
        load_config,
    )

    __all__.extend(
        [
            "AgentConfig",
            "LLMConfig",
            "RuntimeConfig",
            "GuardrailsConfig",
            "ConfigManager",
            "get_config",
            "load_config",
        ]
    )
except ImportError:
    pass

# Testing utilities
try:
    from framework.runtime.testing import (  # noqa: F401
        MockLLMProvider,
        MockResponse,
        RecordingProvider,
        PlaybackProvider,
        TestFixture,
        FixtureBuilder,
        AgentAssertions,
        AgentTestRunner,
    )

    __all__.extend(
        [
            "MockLLMProvider",
            "MockResponse",
            "RecordingProvider",
            "PlaybackProvider",
            "TestFixture",
            "FixtureBuilder",
            "AgentAssertions",
            "AgentTestRunner",
        ]
    )
except ImportError:
    pass

# State management
try:
    from framework.runtime.state import (  # noqa: F401
        StateManager,
        StateSnapshot,
        StateTransaction,
        create_state_manager,
    )

    __all__.extend(["StateManager", "StateSnapshot", "StateTransaction", "create_state_manager"])
except ImportError:
    pass

# Retry utilities
try:
    from framework.runtime.retry import (  # noqa: F401
        RetryConfig,
        RetryStrategy,
        RetryError,
        CircuitBreaker,
        CircuitBreakerOpen,
        retry,
        retry_async,
        with_timeout,
    )

    __all__.extend(
        [
            "RetryConfig",
            "RetryStrategy",
            "RetryError",
            "CircuitBreaker",
            "CircuitBreakerOpen",
            "retry",
            "retry_async",
            "with_timeout",
        ]
    )
except ImportError:
    pass

# Rate limiting
try:
    from framework.runtime.rate_limit import (  # noqa: F401
        TokenBucket,
        SlidingWindowLimiter,
        KeyedRateLimiter,
        AdaptiveRateLimiter,
        RateLimitConfig,
        rate_limit,
        create_rate_limiter,
    )

    __all__.extend(
        [
            "TokenBucket",
            "SlidingWindowLimiter",
            "KeyedRateLimiter",
            "AdaptiveRateLimiter",
            "RateLimitConfig",
            "rate_limit",
            "create_rate_limiter",
        ]
    )
except ImportError:
    pass

# Telemetry
try:
    from framework.runtime.telemetry import (  # noqa: F401
        MetricsCollector,
        Tracer,
        Span,
        Metric,
        MetricType,
        trace,
        timed,
        get_metrics,
        get_tracer,
    )

    __all__.extend(
        [
            "MetricsCollector",
            "Tracer",
            "Span",
            "Metric",
            "MetricType",
            "trace",
            "timed",
            "get_metrics",
            "get_tracer",
        ]
    )
except ImportError:
    pass

# Security
try:
    from framework.runtime.security import (  # noqa: F401
        InputValidator,
        SecretManager,
        PermissionManager,
        SecurityAuditor,
        SecurityContext,
        Permission,
        SecurityLevel,
        hash_password,
        verify_password,
        generate_csrf_token,
        get_validator,
        get_secret_manager,
        get_permission_manager,
        get_security_auditor,
    )

    __all__.extend(
        [
            "InputValidator",
            "SecretManager",
            "PermissionManager",
            "SecurityAuditor",
            "SecurityContext",
            "Permission",
            "SecurityLevel",
            "hash_password",
            "verify_password",
            "generate_csrf_token",
            "get_validator",
            "get_secret_manager",
            "get_permission_manager",
            "get_security_auditor",
        ]
    )
except ImportError:
    pass

# Plugins
try:
    from framework.runtime.plugins import (  # noqa: F401
        PluginManager,
        HookManager,
        Plugin,
        PluginMetadata,
        PluginState,
        create_plugin_template,
    )

    __all__.extend(
        [
            "PluginManager",
            "HookManager",
            "Plugin",
            "PluginMetadata",
            "PluginState",
            "create_plugin_template",
        ]
    )
except ImportError:
    pass

# Validation
try:
    from framework.runtime.validation import (  # noqa: F401
        Validator,
        ValidationResult,
        ValidationError,
        ValidationErrorType,
        ValidationException,
        validate_dict_schema,
        assert_type,
        assert_not_none,
        assert_not_empty,
    )

    __all__.extend(
        [
            "Validator",
            "ValidationResult",
            "ValidationError",
            "ValidationErrorType",
            "ValidationException",
            "validate_dict_schema",
            "assert_type",
            "assert_not_none",
            "assert_not_empty",
        ]
    )
except ImportError:
    pass
