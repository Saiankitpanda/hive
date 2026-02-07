"""
State Manager - Persistent state management for agents.

Provides:
- State persistence across executions
- State versioning and history
- Atomic updates with transactions
- State synchronization

From ROADMAP: State management system
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from collections.abc import Callable
import copy


@dataclass
class StateSnapshot:
    """A snapshot of state at a point in time."""

    version: int
    data: dict[str, Any]
    timestamp: float
    source: str = "manual"  # manual, auto, checkpoint

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "data": self.data,
            "timestamp": self.timestamp,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StateSnapshot":
        return cls(
            version=data["version"],
            data=data["data"],
            timestamp=data["timestamp"],
            source=data.get("source", "manual"),
        )


class StateManager:
    """
    Manager for agent state.

    Usage:
        state = StateManager(storage_path="./state")

        # Get/set state
        state.set("user", {"name": "John"})
        user = state.get("user")

        # Nested access
        state.set("config.llm.model", "claude-3")
        model = state.get("config.llm.model")

        # Transactions
        with state.transaction() as tx:
            tx.set("count", tx.get("count", 0) + 1)
            tx.set("updated_at", time.time())

        # History
        history = state.get_history(limit=10)
        state.rollback(version=5)
    """

    def __init__(
        self,
        storage_path: str | Path | None = None,
        max_history: int = 100,
        auto_save: bool = True,
    ):
        """
        Initialize state manager.

        Args:
            storage_path: Path for persistent storage
            max_history: Maximum history snapshots to keep
            auto_save: Automatically save on changes
        """
        self.storage_path = Path(storage_path) if storage_path else None
        self.max_history = max_history
        self.auto_save = auto_save

        self._state: dict[str, Any] = {}
        self._version = 0
        self._history: list[StateSnapshot] = []
        self._change_handlers: list[Callable[[str, Any, Any], None]] = []
        self._in_transaction = False

        if self.storage_path:
            self.storage_path.mkdir(parents=True, exist_ok=True)
            self._load()

    # === Core State Operations ===

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a value from state.

        Args:
            key: Dot-notation key (e.g., "user.name")
            default: Default value if not found

        Returns:
            Value or default
        """
        parts = key.split(".")
        obj = self._state

        for part in parts:
            if isinstance(obj, dict) and part in obj:
                obj = obj[part]
            else:
                return default

        return obj

    def set(self, key: str, value: Any) -> None:
        """
        Set a value in state.

        Args:
            key: Dot-notation key
            value: Value to set
        """
        old_value = self.get(key)
        parts = key.split(".")

        # Navigate to parent
        obj = self._state
        for part in parts[:-1]:
            if part not in obj:
                obj[part] = {}
            obj = obj[part]

        # Set value
        obj[parts[-1]] = value

        # Notify handlers
        for handler in self._change_handlers:
            try:
                handler(key, old_value, value)
            except Exception:
                pass

        # Auto-save
        if self.auto_save and not self._in_transaction:
            self._create_snapshot("auto")
            self._save()

    def delete(self, key: str) -> bool:
        """
        Delete a key from state.

        Args:
            key: Dot-notation key

        Returns:
            True if deleted
        """
        parts = key.split(".")
        obj = self._state

        for part in parts[:-1]:
            if isinstance(obj, dict) and part in obj:
                obj = obj[part]
            else:
                return False

        if parts[-1] in obj:
            del obj[parts[-1]]
            if self.auto_save and not self._in_transaction:
                self._create_snapshot("auto")
                self._save()
            return True

        return False

    def exists(self, key: str) -> bool:
        """Check if key exists."""
        return self.get(key, None) is not None

    def keys(self, prefix: str = "") -> list[str]:
        """Get all keys with optional prefix."""
        all_keys = []
        self._collect_keys(self._state, "", all_keys)
        if prefix:
            return [k for k in all_keys if k.startswith(prefix)]
        return all_keys

    def _collect_keys(
        self,
        obj: dict[str, Any],
        prefix: str,
        keys: list[str],
    ) -> None:
        """Recursively collect all keys."""
        for key, value in obj.items():
            full_key = f"{prefix}.{key}" if prefix else key
            keys.append(full_key)
            if isinstance(value, dict):
                self._collect_keys(value, full_key, keys)

    # === Bulk Operations ===

    def update(self, data: dict[str, Any]) -> None:
        """Update multiple keys at once."""
        for key, value in data.items():
            self.set(key, value)

    def get_all(self) -> dict[str, Any]:
        """Get entire state."""
        return copy.deepcopy(self._state)

    def clear(self) -> None:
        """Clear all state."""
        self._state = {}
        if self.auto_save:
            self._create_snapshot("manual")
            self._save()

    # === Transactions ===

    def transaction(self) -> "StateTransaction":
        """
        Start a transaction.

        Usage:
            with state.transaction() as tx:
                tx.set("key", "value")
        """
        return StateTransaction(self)

    def begin_transaction(self) -> None:
        """Begin a transaction."""
        self._in_transaction = True
        self._transaction_snapshot = copy.deepcopy(self._state)

    def commit_transaction(self) -> None:
        """Commit the current transaction."""
        self._in_transaction = False
        self._create_snapshot("transaction")
        self._save()

    def rollback_transaction(self) -> None:
        """Rollback the current transaction."""
        if hasattr(self, "_transaction_snapshot"):
            self._state = self._transaction_snapshot
        self._in_transaction = False

    # === History & Versioning ===

    def get_version(self) -> int:
        """Get current version."""
        return self._version

    def get_history(self, limit: int = 10) -> list[StateSnapshot]:
        """Get state history."""
        return self._history[-limit:]

    def rollback(self, version: int) -> bool:
        """
        Rollback to a specific version.

        Args:
            version: Version number to rollback to

        Returns:
            True if successful
        """
        for snapshot in self._history:
            if snapshot.version == version:
                self._state = copy.deepcopy(snapshot.data)
                self._create_snapshot("rollback")
                self._save()
                return True
        return False

    def checkpoint(self, name: str = "") -> int:
        """
        Create a named checkpoint.

        Args:
            name: Optional checkpoint name

        Returns:
            Version number
        """
        snapshot = self._create_snapshot("checkpoint")
        if name:
            snapshot.source = f"checkpoint:{name}"
        self._save()
        return snapshot.version

    def _create_snapshot(self, source: str) -> StateSnapshot:
        """Create a state snapshot."""
        self._version += 1
        snapshot = StateSnapshot(
            version=self._version,
            data=copy.deepcopy(self._state),
            timestamp=time.time(),
            source=source,
        )
        self._history.append(snapshot)

        # Trim history
        if len(self._history) > self.max_history:
            self._history = self._history[-self.max_history :]

        return snapshot

    # === Change Handlers ===

    def on_change(
        self,
        handler: Callable[[str, Any, Any], None],
    ) -> None:
        """
        Register a change handler.

        Args:
            handler: Function(key, old_value, new_value)
        """
        self._change_handlers.append(handler)

    # === Persistence ===

    def _get_state_path(self) -> Path:
        """Get state file path."""
        if not self.storage_path:
            raise ValueError("No storage path configured")
        return self.storage_path / "state.json"

    def _get_history_path(self) -> Path:
        """Get history file path."""
        if not self.storage_path:
            raise ValueError("No storage path configured")
        return self.storage_path / "history.json"

    def _save(self) -> None:
        """Save state to disk."""
        if not self.storage_path:
            return

        # Save current state
        state_data = {
            "version": self._version,
            "state": self._state,
            "timestamp": time.time(),
        }
        with open(self._get_state_path(), "w") as f:
            json.dump(state_data, f, indent=2)

        # Save history
        history_data = [s.to_dict() for s in self._history]
        with open(self._get_history_path(), "w") as f:
            json.dump(history_data, f, indent=2)

    def _load(self) -> None:
        """Load state from disk."""
        if not self.storage_path:
            return

        state_path = self._get_state_path()
        if state_path.exists():
            with open(state_path) as f:
                data = json.load(f)
            self._version = data.get("version", 0)
            self._state = data.get("state", {})

        history_path = self._get_history_path()
        if history_path.exists():
            with open(history_path) as f:
                history_data = json.load(f)
            self._history = [StateSnapshot.from_dict(s) for s in history_data]


class StateTransaction:
    """Context manager for state transactions."""

    def __init__(self, manager: StateManager):
        self._manager = manager

    def __enter__(self) -> "StateTransaction":
        self._manager.begin_transaction()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self._manager.commit_transaction()
        else:
            self._manager.rollback_transaction()
        return False

    def get(self, key: str, default: Any = None) -> Any:
        """Get value within transaction."""
        return self._manager.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set value within transaction."""
        self._manager.set(key, value)

    def delete(self, key: str) -> bool:
        """Delete key within transaction."""
        return self._manager.delete(key)


def create_state_manager(
    storage_path: str | None = None,
    max_history: int = 100,
) -> StateManager:
    """
    Factory function to create state manager.

    Args:
        storage_path: Path for persistence
        max_history: Max history to keep

    Returns:
        Configured StateManager
    """
    return StateManager(
        storage_path=storage_path,
        max_history=max_history,
        auto_save=True,
    )
