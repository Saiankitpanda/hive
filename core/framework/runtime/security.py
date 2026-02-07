"""
Security - Comprehensive security utilities for agent execution.

Provides:
- Input validation and sanitization
- Secret management
- Permission controls
- Rate limiting for security
- Audit logging for compliance

From ROADMAP: Production security features
"""

import hashlib
import hmac
import re
import secrets
import time
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from threading import Lock
from typing import Any


class Permission(Enum):
    """Permission levels for actions."""

    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    ADMIN = "admin"
    DELETE = "delete"


class SecurityLevel(Enum):
    """Security levels for data classification."""

    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


@dataclass
class SecurityContext:
    """Security context for an operation."""

    user_id: str
    permissions: set[Permission] = field(default_factory=set)
    roles: set[str] = field(default_factory=set)
    attributes: dict[str, Any] = field(default_factory=dict)
    session_id: str = ""
    ip_address: str = ""
    created_at: float = field(default_factory=time.time)

    def has_permission(self, permission: Permission) -> bool:
        """Check if context has permission."""
        return permission in self.permissions or Permission.ADMIN in self.permissions

    def has_role(self, role: str) -> bool:
        """Check if context has role."""
        return role in self.roles or "admin" in self.roles


class InputValidator:
    """
    Validates and sanitizes input data.

    Usage:
        validator = InputValidator()

        # Validate string
        clean = validator.sanitize_string(user_input)

        # Validate with rules
        is_valid = validator.validate_email(email)
        is_valid = validator.validate_url(url)

        # Custom validation
        validator.add_rule("username", r"^[a-zA-Z0-9_]{3,20}$")
        is_valid = validator.validate("username", value)
    """

    # Common patterns
    PATTERNS = {
        "email": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
        "url": r"^https?://[^\s<>\"{}|\\^`\[\]]+$",
        "alphanumeric": r"^[a-zA-Z0-9]+$",
        "username": r"^[a-zA-Z0-9_]{3,30}$",
        "uuid": r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
        "ip_v4": r"^(\d{1,3}\.){3}\d{1,3}$",
        "phone": r"^\+?[1-9]\d{1,14}$",
    }

    # Dangerous patterns to detect
    INJECTION_PATTERNS = [
        r"<script[^>]*>",
        r"javascript:",
        r"on\w+\s*=",
        r"eval\s*\(",
        r"exec\s*\(",
        r"import\s+os",
        r"__import__",
        r"subprocess\.",
        r"\.\./",
        r";\s*rm\s",
        r";\s*cat\s",
        r"DROP\s+TABLE",
        r"DELETE\s+FROM",
        r"INSERT\s+INTO",
        r"UPDATE\s+.*\s+SET",
        r"UNION\s+SELECT",
    ]

    def __init__(self):
        self._custom_rules: dict[str, str] = {}
        self._compiled_injection = [re.compile(p, re.IGNORECASE) for p in self.INJECTION_PATTERNS]

    def sanitize_string(
        self,
        value: str,
        max_length: int = 10000,
        strip_html: bool = True,
        strip_scripts: bool = True,
    ) -> str:
        """Sanitize a string value."""
        if not isinstance(value, str):
            return str(value)

        # Truncate
        value = value[:max_length]

        # Strip null bytes
        value = value.replace("\x00", "")

        # Strip HTML tags if requested
        if strip_html:
            value = re.sub(r"<[^>]+>", "", value)

        # Strip script content
        if strip_scripts:
            value = re.sub(r"<script[^>]*>.*?</script>", "", value, flags=re.DOTALL | re.IGNORECASE)

        return value.strip()

    def detect_injection(self, value: str) -> list[str]:
        """Detect potential injection attacks."""
        threats = []

        for pattern in self._compiled_injection:
            if pattern.search(value):
                threats.append(pattern.pattern)

        return threats

    def is_safe(self, value: str) -> bool:
        """Check if value is safe from injection."""
        return len(self.detect_injection(value)) == 0

    def validate_pattern(self, pattern_name: str, value: str) -> bool:
        """Validate value against named pattern."""
        pattern = self.PATTERNS.get(pattern_name) or self._custom_rules.get(pattern_name)
        if not pattern:
            return False
        return bool(re.match(pattern, value))

    def validate_email(self, email: str) -> bool:
        """Validate email address."""
        return self.validate_pattern("email", email)

    def validate_url(self, url: str) -> bool:
        """Validate URL."""
        return self.validate_pattern("url", url)

    def add_rule(self, name: str, pattern: str) -> None:
        """Add custom validation rule."""
        self._custom_rules[name] = pattern

    def validate_length(
        self,
        value: str,
        min_length: int = 0,
        max_length: int = 10000,
    ) -> bool:
        """Validate string length."""
        return min_length <= len(value) <= max_length


class SecretManager:
    """
    Secure secret management.

    Usage:
        secrets_mgr = SecretManager()

        # Store secret
        secrets_mgr.set("api_key", "sk-xxx")

        # Retrieve secret
        key = secrets_mgr.get("api_key")

        # Mask secret in output
        masked = secrets_mgr.mask(api_key)  # "sk-x***"
    """

    def __init__(self, encryption_key: str | None = None):
        self._secrets: dict[str, str] = {}
        self._lock = Lock()
        self._encryption_key = encryption_key or secrets.token_hex(32)
        self._access_log: list[dict[str, Any]] = []

    def set(
        self,
        name: str,
        value: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Store a secret."""
        with self._lock:
            # In production, encrypt the value
            self._secrets[name] = value
            self._log_access(name, "set", metadata)

    def get(
        self,
        name: str,
        default: str | None = None,
    ) -> str | None:
        """Retrieve a secret."""
        with self._lock:
            value = self._secrets.get(name, default)
            self._log_access(name, "get")
            return value

    def delete(self, name: str) -> bool:
        """Delete a secret."""
        with self._lock:
            if name in self._secrets:
                del self._secrets[name]
                self._log_access(name, "delete")
                return True
            return False

    def exists(self, name: str) -> bool:
        """Check if secret exists."""
        return name in self._secrets

    def list_names(self) -> list[str]:
        """List secret names (not values)."""
        return list(self._secrets.keys())

    @staticmethod
    def mask(
        value: str,
        visible_chars: int = 4,
        mask_char: str = "*",
    ) -> str:
        """Mask a secret value for display."""
        if len(value) <= visible_chars:
            return mask_char * len(value)
        return value[:visible_chars] + mask_char * (len(value) - visible_chars)

    @staticmethod
    def generate_token(length: int = 32) -> str:
        """Generate a secure random token."""
        return secrets.token_urlsafe(length)

    @staticmethod
    def generate_api_key(prefix: str = "sk") -> str:
        """Generate an API key."""
        return f"{prefix}-{secrets.token_urlsafe(32)}"

    def _log_access(
        self,
        name: str,
        action: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Log secret access."""
        self._access_log.append(
            {
                "name": name,
                "action": action,
                "timestamp": time.time(),
                "metadata": metadata or {},
            }
        )

    def get_access_log(self) -> list[dict[str, Any]]:
        """Get access log."""
        return list(self._access_log)


class PermissionManager:
    """
    Manages permissions for agent actions.

    Usage:
        perms = PermissionManager()

        # Define roles
        perms.define_role("viewer", {Permission.READ})
        perms.define_role("editor", {Permission.READ, Permission.WRITE})

        # Check permissions
        ctx = SecurityContext(user_id="123", roles={"editor"})
        if perms.check_permission(ctx, Permission.WRITE):
            perform_write()

        # Use as decorator
        @perms.require(Permission.ADMIN)
        def admin_action():
            ...
    """

    def __init__(self):
        self._roles: dict[str, set[Permission]] = {}
        self._user_permissions: dict[str, set[Permission]] = {}
        self._lock = Lock()

        # Default roles
        self.define_role("viewer", {Permission.READ})
        self.define_role("editor", {Permission.READ, Permission.WRITE})
        self.define_role(
            "admin",
            {Permission.READ, Permission.WRITE, Permission.EXECUTE, Permission.ADMIN},
        )

    def define_role(self, role: str, permissions: set[Permission]) -> None:
        """Define a role with permissions."""
        with self._lock:
            self._roles[role] = permissions

    def get_role_permissions(self, role: str) -> set[Permission]:
        """Get permissions for a role."""
        return self._roles.get(role, set())

    def grant_permission(self, user_id: str, permission: Permission) -> None:
        """Grant permission to user."""
        with self._lock:
            if user_id not in self._user_permissions:
                self._user_permissions[user_id] = set()
            self._user_permissions[user_id].add(permission)

    def revoke_permission(self, user_id: str, permission: Permission) -> None:
        """Revoke permission from user."""
        with self._lock:
            if user_id in self._user_permissions:
                self._user_permissions[user_id].discard(permission)

    def check_permission(
        self,
        context: SecurityContext,
        permission: Permission,
    ) -> bool:
        """Check if context has permission."""
        # Check direct permissions
        if context.has_permission(permission):
            return True

        # Check role permissions
        for role in context.roles:
            role_perms = self.get_role_permissions(role)
            if permission in role_perms:
                return True

        # Check user-specific permissions
        user_perms = self._user_permissions.get(context.user_id, set())
        return permission in user_perms

    def require(self, permission: Permission):
        """Decorator to require permission."""

        def decorator(func):
            @wraps(func)
            def wrapper(*args, context: SecurityContext = None, **kwargs):
                if context is None:
                    raise PermissionError("Security context required")

                if not self.check_permission(context, permission):
                    raise PermissionError(f"Permission denied: {permission.value} required")

                return func(*args, context=context, **kwargs)

            return wrapper

        return decorator


class SecurityAuditor:
    """
    Security audit logging for compliance.

    Usage:
        auditor = SecurityAuditor()

        # Log events
        auditor.log_authentication("user123", success=True)
        auditor.log_authorization("user123", "read", "document", granted=True)
        auditor.log_data_access("user123", "customer_data", action="read")

        # Get audit trail
        events = auditor.get_events(user_id="user123")
    """

    def __init__(self, max_events: int = 10000):
        self._events: list[dict[str, Any]] = []
        self._max_events = max_events
        self._lock = Lock()

    def _log(self, event: dict[str, Any]) -> None:
        """Log an event."""
        event["timestamp"] = time.time()
        event["id"] = secrets.token_hex(8)

        with self._lock:
            self._events.append(event)
            if len(self._events) > self._max_events:
                self._events = self._events[-self._max_events :]

    def log_authentication(
        self,
        user_id: str,
        success: bool,
        method: str = "password",
        ip_address: str = "",
    ) -> None:
        """Log authentication attempt."""
        self._log(
            {
                "type": "authentication",
                "user_id": user_id,
                "success": success,
                "method": method,
                "ip_address": ip_address,
            }
        )

    def log_authorization(
        self,
        user_id: str,
        action: str,
        resource: str,
        granted: bool,
    ) -> None:
        """Log authorization decision."""
        self._log(
            {
                "type": "authorization",
                "user_id": user_id,
                "action": action,
                "resource": resource,
                "granted": granted,
            }
        )

    def log_data_access(
        self,
        user_id: str,
        resource: str,
        action: str = "read",
        details: dict[str, Any] | None = None,
    ) -> None:
        """Log data access."""
        self._log(
            {
                "type": "data_access",
                "user_id": user_id,
                "resource": resource,
                "action": action,
                "details": details or {},
            }
        )

    def log_security_event(
        self,
        event_type: str,
        severity: str = "info",
        details: dict[str, Any] | None = None,
    ) -> None:
        """Log generic security event."""
        self._log(
            {
                "type": event_type,
                "severity": severity,
                "details": details or {},
            }
        )

    def get_events(
        self,
        user_id: str | None = None,
        event_type: str | None = None,
        since: float | None = None,
    ) -> list[dict[str, Any]]:
        """Get filtered events."""
        with self._lock:
            events = self._events

            if user_id:
                events = [e for e in events if e.get("user_id") == user_id]

            if event_type:
                events = [e for e in events if e.get("type") == event_type]

            if since:
                events = [e for e in events if e.get("timestamp", 0) >= since]

            return events


def hash_password(password: str, salt: str | None = None) -> tuple[str, str]:
    """
    Hash a password securely.

    Returns:
        Tuple of (hash, salt)
    """
    if salt is None:
        salt = secrets.token_hex(16)

    key = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode(),
        salt.encode(),
        100000,
    )
    return key.hex(), salt


def verify_password(password: str, hash_value: str, salt: str) -> bool:
    """Verify a password against hash."""
    computed_hash, _ = hash_password(password, salt)
    return hmac.compare_digest(computed_hash, hash_value)


def generate_csrf_token() -> str:
    """Generate CSRF token."""
    return secrets.token_urlsafe(32)


def verify_csrf_token(token: str, expected: str) -> bool:
    """Verify CSRF token."""
    return hmac.compare_digest(token, expected)


# Default instances
_default_validator = InputValidator()
_default_secret_manager = SecretManager()
_default_permission_manager = PermissionManager()
_default_auditor = SecurityAuditor()


def get_validator() -> InputValidator:
    """Get default validator."""
    return _default_validator


def get_secret_manager() -> SecretManager:
    """Get default secret manager."""
    return _default_secret_manager


def get_permission_manager() -> PermissionManager:
    """Get default permission manager."""
    return _default_permission_manager


def get_security_auditor() -> SecurityAuditor:
    """Get default auditor."""
    return _default_auditor
