"""
Guardrails - Safety and quality constraints for agent execution.

Provides configurable guardrails to:
- Limit costs and token usage
- Prevent harmful outputs
- Enforce output quality standards
- Set execution boundaries

Guardrail Types:
- Cost guardrails: Budget limits, token caps
- Safety guardrails: Content filtering, PII detection
- Quality guardrails: Response length, format validation
- Execution guardrails: Timeout, iteration limits
"""

import re
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class GuardrailAction(Enum):
    """Action to take when guardrail is triggered."""

    ALLOW = "allow"  # Let execution continue
    WARN = "warn"  # Log warning but continue
    BLOCK = "block"  # Block the action
    MODIFY = "modify"  # Modify the content


class GuardrailSeverity(Enum):
    """Severity level of guardrail violation."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class GuardrailResult:
    """Result of a guardrail check."""

    passed: bool
    action: GuardrailAction
    message: str = ""
    severity: GuardrailSeverity = GuardrailSeverity.LOW
    modified_content: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "action": self.action.value,
            "message": self.message,
            "severity": self.severity.value,
            "modified_content": self.modified_content,
            "metadata": self.metadata,
        }


class Guardrail(ABC):
    """Base class for guardrails."""

    def __init__(
        self,
        name: str,
        enabled: bool = True,
        action: GuardrailAction = GuardrailAction.BLOCK,
        severity: GuardrailSeverity = GuardrailSeverity.MEDIUM,
    ):
        self.name = name
        self.enabled = enabled
        self.default_action = action
        self.severity = severity

    @abstractmethod
    def check(self, content: str, context: dict[str, Any] | None = None) -> GuardrailResult:
        """
        Check content against this guardrail.

        Args:
            content: Content to check
            context: Additional context (e.g., execution state)

        Returns:
            GuardrailResult with pass/fail and action
        """
        pass


# === Cost Guardrails ===


class TokenLimitGuardrail(Guardrail):
    """Limit the number of tokens in requests/responses."""

    def __init__(
        self,
        max_tokens: int = 100000,
        name: str = "token_limit",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.max_tokens = max_tokens

    def check(self, content: str, context: dict[str, Any] | None = None) -> GuardrailResult:
        context = context or {}
        total_tokens = context.get("total_tokens", 0)

        if total_tokens > self.max_tokens:
            return GuardrailResult(
                passed=False,
                action=self.default_action,
                message=f"Token limit exceeded: {total_tokens}/{self.max_tokens}",
                severity=self.severity,
                metadata={"total_tokens": total_tokens, "limit": self.max_tokens},
            )

        return GuardrailResult(
            passed=True,
            action=GuardrailAction.ALLOW,
            message="Token usage within limits",
        )


class BudgetGuardrail(Guardrail):
    """Limit spending on API calls."""

    def __init__(
        self,
        max_cost_usd: float = 10.0,
        name: str = "budget_limit",
        **kwargs,
    ):
        super().__init__(name=name, severity=GuardrailSeverity.HIGH, **kwargs)
        self.max_cost = max_cost_usd
        self.current_cost = 0.0

    def check(self, content: str, context: dict[str, Any] | None = None) -> GuardrailResult:
        context = context or {}
        cost = context.get("cost_usd", 0.0)
        self.current_cost += cost

        if self.current_cost > self.max_cost:
            return GuardrailResult(
                passed=False,
                action=self.default_action,
                message=f"Budget exceeded: ${self.current_cost:.2f}/${self.max_cost:.2f}",
                severity=GuardrailSeverity.CRITICAL,
                metadata={"current_cost": self.current_cost, "limit": self.max_cost},
            )

        return GuardrailResult(
            passed=True,
            action=GuardrailAction.ALLOW,
            message=f"Budget OK: ${self.current_cost:.2f}/${self.max_cost:.2f}",
        )

    def reset(self):
        """Reset the budget counter."""
        self.current_cost = 0.0


# === Safety Guardrails ===


class ContentFilterGuardrail(Guardrail):
    """Filter harmful or inappropriate content."""

    # Default blocked patterns
    DEFAULT_PATTERNS = [
        r"(?i)(password|secret|api[_-]?key)\s*[:=]\s*['\"]?\w+",  # Credentials
        r"(?i)rm\s+-rf\s+/",  # Dangerous commands
        r"(?i)(drop\s+table|delete\s+from\s+\w+\s+where\s+1\s*=\s*1)",  # SQL injection
    ]

    def __init__(
        self,
        blocked_patterns: list[str] | None = None,
        name: str = "content_filter",
        **kwargs,
    ):
        super().__init__(name=name, severity=GuardrailSeverity.HIGH, **kwargs)
        patterns = blocked_patterns or self.DEFAULT_PATTERNS
        self.patterns = [re.compile(p) for p in patterns]

    def check(self, content: str, context: dict[str, Any] | None = None) -> GuardrailResult:
        for pattern in self.patterns:
            match = pattern.search(content)
            if match:
                return GuardrailResult(
                    passed=False,
                    action=self.default_action,
                    message="Blocked content pattern detected",
                    severity=self.severity,
                    metadata={"pattern": pattern.pattern, "match": match.group()[:50]},
                )

        return GuardrailResult(
            passed=True,
            action=GuardrailAction.ALLOW,
            message="Content passed safety filter",
        )


class PIIGuardrail(Guardrail):
    """Detect and optionally redact PII (Personally Identifiable Information)."""

    PII_PATTERNS = {
        "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
        "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
        "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
        "credit_card": r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
    }

    def __init__(
        self,
        redact: bool = True,
        name: str = "pii_filter",
        **kwargs,
    ):
        super().__init__(name=name, action=GuardrailAction.MODIFY, **kwargs)
        self.redact = redact
        self.patterns = {k: re.compile(v) for k, v in self.PII_PATTERNS.items()}

    def check(self, content: str, context: dict[str, Any] | None = None) -> GuardrailResult:
        found_pii = []
        modified = content

        for pii_type, pattern in self.patterns.items():
            if pattern.search(content):
                found_pii.append(pii_type)
                if self.redact:
                    modified = pattern.sub(f"[REDACTED_{pii_type.upper()}]", modified)

        if found_pii:
            return GuardrailResult(
                passed=False,
                action=GuardrailAction.MODIFY if self.redact else self.default_action,
                message=f"PII detected: {', '.join(found_pii)}",
                severity=GuardrailSeverity.HIGH,
                modified_content=modified if self.redact else None,
                metadata={"pii_types": found_pii},
            )

        return GuardrailResult(
            passed=True,
            action=GuardrailAction.ALLOW,
            message="No PII detected",
        )


# === Quality Guardrails ===


class ResponseLengthGuardrail(Guardrail):
    """Enforce response length constraints."""

    def __init__(
        self,
        min_length: int = 0,
        max_length: int = 50000,
        name: str = "response_length",
        **kwargs,
    ):
        super().__init__(name=name, severity=GuardrailSeverity.LOW, **kwargs)
        self.min_length = min_length
        self.max_length = max_length

    def check(self, content: str, context: dict[str, Any] | None = None) -> GuardrailResult:
        length = len(content)

        if length < self.min_length:
            return GuardrailResult(
                passed=False,
                action=GuardrailAction.WARN,
                message=f"Response too short: {length} < {self.min_length}",
                severity=GuardrailSeverity.LOW,
                metadata={"length": length, "min": self.min_length},
            )

        if length > self.max_length:
            return GuardrailResult(
                passed=False,
                action=self.default_action,
                message=f"Response too long: {length} > {self.max_length}",
                severity=self.severity,
                metadata={"length": length, "max": self.max_length},
            )

        return GuardrailResult(
            passed=True,
            action=GuardrailAction.ALLOW,
            message=f"Response length OK: {length}",
        )


class JSONFormatGuardrail(Guardrail):
    """Validate JSON format in responses."""

    def __init__(self, name: str = "json_format", **kwargs):
        super().__init__(name=name, severity=GuardrailSeverity.MEDIUM, **kwargs)

    def check(self, content: str, context: dict[str, Any] | None = None) -> GuardrailResult:
        import json

        try:
            json.loads(content)
            return GuardrailResult(
                passed=True,
                action=GuardrailAction.ALLOW,
                message="Valid JSON",
            )
        except json.JSONDecodeError as e:
            return GuardrailResult(
                passed=False,
                action=self.default_action,
                message=f"Invalid JSON: {str(e)[:100]}",
                severity=self.severity,
                metadata={"error": str(e)},
            )


# === Execution Guardrails ===


class TimeoutGuardrail(Guardrail):
    """Enforce execution timeout."""

    def __init__(
        self,
        timeout_seconds: float = 300.0,
        name: str = "timeout",
        **kwargs,
    ):
        super().__init__(name=name, severity=GuardrailSeverity.HIGH, **kwargs)
        self.timeout = timeout_seconds
        self.start_time: float | None = None

    def start(self):
        """Start the timeout timer."""
        self.start_time = time.time()

    def check(self, content: str, context: dict[str, Any] | None = None) -> GuardrailResult:
        if self.start_time is None:
            self.start()

        elapsed = time.time() - self.start_time

        if elapsed > self.timeout:
            return GuardrailResult(
                passed=False,
                action=self.default_action,
                message=f"Timeout exceeded: {elapsed:.1f}s > {self.timeout}s",
                severity=GuardrailSeverity.CRITICAL,
                metadata={"elapsed": elapsed, "timeout": self.timeout},
            )

        return GuardrailResult(
            passed=True,
            action=GuardrailAction.ALLOW,
            message=f"Timeout OK: {elapsed:.1f}s / {self.timeout}s",
        )

    def reset(self):
        """Reset the timer."""
        self.start_time = None


class IterationLimitGuardrail(Guardrail):
    """Limit the number of iterations/loops."""

    def __init__(
        self,
        max_iterations: int = 100,
        name: str = "iteration_limit",
        **kwargs,
    ):
        super().__init__(name=name, severity=GuardrailSeverity.MEDIUM, **kwargs)
        self.max_iterations = max_iterations
        self.current = 0

    def check(self, content: str, context: dict[str, Any] | None = None) -> GuardrailResult:
        self.current += 1

        if self.current > self.max_iterations:
            return GuardrailResult(
                passed=False,
                action=self.default_action,
                message=f"Iteration limit exceeded: {self.current}/{self.max_iterations}",
                severity=self.severity,
                metadata={"current": self.current, "limit": self.max_iterations},
            )

        return GuardrailResult(
            passed=True,
            action=GuardrailAction.ALLOW,
            message=f"Iteration OK: {self.current}/{self.max_iterations}",
        )

    def reset(self):
        """Reset the counter."""
        self.current = 0


# === Guardrails Manager ===


class GuardrailsManager:
    """
    Manage and apply multiple guardrails.

    Usage:
        manager = GuardrailsManager()

        # Add guardrails
        manager.add(TokenLimitGuardrail(max_tokens=50000))
        manager.add(ContentFilterGuardrail())
        manager.add(PIIGuardrail(redact=True))

        # Check content
        results = manager.check_all("Some content to check")

        # Check if any failed
        if manager.has_violations(results):
            print("Guardrails violated!")

        # Get modified content (if any guardrail modified it)
        modified = manager.get_modified_content(results, original_content)
    """

    def __init__(self):
        self._guardrails: list[Guardrail] = []
        self._violation_handlers: list[Callable[[GuardrailResult], None]] = []

    def add(self, guardrail: Guardrail) -> "GuardrailsManager":
        """Add a guardrail. Returns self for chaining."""
        self._guardrails.append(guardrail)
        return self

    def remove(self, name: str) -> bool:
        """Remove a guardrail by name. Returns True if found."""
        for i, g in enumerate(self._guardrails):
            if g.name == name:
                self._guardrails.pop(i)
                return True
        return False

    def check_all(
        self,
        content: str,
        context: dict[str, Any] | None = None,
    ) -> list[GuardrailResult]:
        """
        Check content against all guardrails.

        Args:
            content: Content to check
            context: Additional context

        Returns:
            List of GuardrailResults
        """
        results = []

        for guardrail in self._guardrails:
            if not guardrail.enabled:
                continue

            result = guardrail.check(content, context)
            results.append(result)

            # Call violation handlers
            if not result.passed:
                for handler in self._violation_handlers:
                    handler(result)

        return results

    def has_violations(self, results: list[GuardrailResult]) -> bool:
        """Check if any guardrail failed."""
        return any(not r.passed for r in results)

    def has_blocking_violations(self, results: list[GuardrailResult]) -> bool:
        """Check if any guardrail requires blocking."""
        return any(not r.passed and r.action == GuardrailAction.BLOCK for r in results)

    def get_modified_content(
        self,
        results: list[GuardrailResult],
        original: str,
    ) -> str:
        """Get content after modifications from guardrails."""
        content = original

        for result in results:
            if result.modified_content is not None:
                content = result.modified_content

        return content

    def on_violation(self, handler: Callable[[GuardrailResult], None]):
        """Register a violation handler."""
        self._violation_handlers.append(handler)

    def reset_all(self):
        """Reset all stateful guardrails."""
        for g in self._guardrails:
            if hasattr(g, "reset"):
                g.reset()


# === Preset Configurations ===


def create_default_guardrails() -> GuardrailsManager:
    """Create a manager with sensible default guardrails."""
    manager = GuardrailsManager()
    manager.add(TokenLimitGuardrail(max_tokens=100000))
    manager.add(BudgetGuardrail(max_cost_usd=10.0))
    manager.add(ContentFilterGuardrail())
    manager.add(ResponseLengthGuardrail(max_length=100000))
    manager.add(TimeoutGuardrail(timeout_seconds=300))
    manager.add(IterationLimitGuardrail(max_iterations=50))
    return manager


def create_strict_guardrails() -> GuardrailsManager:
    """Create a manager with strict guardrails for production."""
    manager = GuardrailsManager()
    manager.add(TokenLimitGuardrail(max_tokens=50000))
    manager.add(BudgetGuardrail(max_cost_usd=5.0))
    manager.add(ContentFilterGuardrail())
    manager.add(PIIGuardrail(redact=True))
    manager.add(ResponseLengthGuardrail(min_length=10, max_length=50000))
    manager.add(TimeoutGuardrail(timeout_seconds=120))
    manager.add(IterationLimitGuardrail(max_iterations=25))
    return manager
