# Guardrails Guide

Guardrails provide safety and quality constraints for agent execution.

## Quick Start

```python
from framework.runtime import create_default_guardrails, GuardrailsManager

# Use preset guardrails
manager = create_default_guardrails()

# Check content
results = manager.check_all("Some content", context={"total_tokens": 1000})

if manager.has_blocking_violations(results):
    print("Execution blocked!")
```

## Available Guardrails

### Cost Guardrails

| Guardrail | Description |
|-----------|-------------|
| `TokenLimitGuardrail` | Limit total token usage |
| `BudgetGuardrail` | Limit spending on API calls |

```python
from framework.runtime import TokenLimitGuardrail, BudgetGuardrail

token_limit = TokenLimitGuardrail(max_tokens=50000)
budget = BudgetGuardrail(max_cost_usd=5.0)
```

### Safety Guardrails

| Guardrail | Description |
|-----------|-------------|
| `ContentFilterGuardrail` | Block harmful patterns (credentials, SQL injection) |
| `PIIGuardrail` | Detect and redact PII (email, phone, SSN, credit card) |

```python
from framework.runtime import ContentFilterGuardrail, PIIGuardrail

content_filter = ContentFilterGuardrail()
pii_filter = PIIGuardrail(redact=True)

# Check for PII
result = pii_filter.check("Contact: john@example.com")
print(result.modified_content)  # "Contact: [REDACTED_EMAIL]"
```

### Quality Guardrails

| Guardrail | Description |
|-----------|-------------|
| `ResponseLengthGuardrail` | Enforce min/max response length |
| `JSONFormatGuardrail` | Validate JSON format |

### Execution Guardrails

| Guardrail | Description |
|-----------|-------------|
| `TimeoutGuardrail` | Enforce execution timeout |
| `IterationLimitGuardrail` | Limit loop iterations |

## Custom Guardrails

```python
from framework.runtime import Guardrail, GuardrailResult, GuardrailAction

class ProfanityGuardrail(Guardrail):
    def __init__(self):
        super().__init__(name="profanity_filter")
        self.blocked_words = ["badword1", "badword2"]

    def check(self, content, context=None):
        for word in self.blocked_words:
            if word in content.lower():
                return GuardrailResult(
                    passed=False,
                    action=GuardrailAction.BLOCK,
                    message=f"Profanity detected"
                )
        return GuardrailResult(passed=True, action=GuardrailAction.ALLOW)
```

## Presets

```python
# Default: Balanced safety and usability
manager = create_default_guardrails()

# Strict: Production-grade security
manager = create_strict_guardrails()
```

## Violation Handlers

```python
def log_violation(result):
    print(f"VIOLATION: {result.message}")

manager.on_violation(log_violation)
```
