"""
Caching Layer - LLM response caching for cost optimization.

Provides:
- In-memory caching with TTL
- Disk-based persistent cache
- Semantic similarity caching
- Cache statistics and management

From ROADMAP: Cost optimization via caching
"""

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class CacheEntry:
    """A single cache entry."""

    key: str
    value: Any
    created_at: float
    expires_at: float
    hits: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_expired(self) -> bool:
        """Check if entry is expired."""
        return time.time() > self.expires_at

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "value": self.value,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "hits": self.hits,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CacheEntry":
        return cls(
            key=data["key"],
            value=data["value"],
            created_at=data["created_at"],
            expires_at=data["expires_at"],
            hits=data.get("hits", 0),
            metadata=data.get("metadata", {}),
        )


@dataclass
class CacheStats:
    """Cache statistics."""

    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    total_entries: int = 0
    total_size_bytes: int = 0
    evictions: int = 0

    @property
    def hit_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.cache_hits / self.total_requests

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_requests": self.total_requests,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": round(self.hit_rate * 100, 2),
            "total_entries": self.total_entries,
            "total_size_bytes": self.total_size_bytes,
            "evictions": self.evictions,
        }


class LLMCache:
    """
    Cache for LLM responses.

    Usage:
        cache = LLMCache(ttl_seconds=3600)

        # Check cache before calling LLM
        key = cache.create_key(messages, model)
        cached = cache.get(key)

        if cached:
            return cached

        # Call LLM and cache result
        response = llm.complete(messages)
        cache.set(key, response)
    """

    def __init__(
        self,
        ttl_seconds: int = 3600,
        max_entries: int = 1000,
        storage_path: str | Path | None = None,
    ):
        """
        Initialize LLM cache.

        Args:
            ttl_seconds: Default TTL for cache entries
            max_entries: Maximum entries in memory
            storage_path: Path for persistent storage
        """
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self.storage_path = Path(storage_path) if storage_path else None

        self._cache: dict[str, CacheEntry] = {}
        self._stats = CacheStats()

        if self.storage_path:
            self.storage_path.mkdir(parents=True, exist_ok=True)
            self._load_from_disk()

    # === Core Methods ===

    def get(self, key: str) -> Any | None:
        """Get a value from cache."""
        self._stats.total_requests += 1

        entry = self._cache.get(key)

        if entry is None:
            # Try disk cache
            if self.storage_path:
                entry = self._load_entry(key)
                if entry:
                    self._cache[key] = entry

        if entry is None:
            self._stats.cache_misses += 1
            return None

        if entry.is_expired():
            self.delete(key)
            self._stats.cache_misses += 1
            return None

        entry.hits += 1
        self._stats.cache_hits += 1
        return entry.value

    def set(
        self,
        key: str,
        value: Any,
        ttl: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Set a value in cache."""
        self._evict_if_needed()

        now = time.time()
        entry = CacheEntry(
            key=key,
            value=value,
            created_at=now,
            expires_at=now + (ttl or self.ttl_seconds),
            metadata=metadata or {},
        )

        self._cache[key] = entry
        self._stats.total_entries = len(self._cache)

        if self.storage_path:
            self._save_entry(entry)

    def delete(self, key: str) -> bool:
        """Delete an entry from cache."""
        if key in self._cache:
            del self._cache[key]
            self._stats.total_entries = len(self._cache)

            if self.storage_path:
                self._delete_entry_file(key)

            return True
        return False

    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()
        self._stats = CacheStats()

        if self.storage_path:
            for f in self.storage_path.glob("*.json"):
                f.unlink()

    # === Key Generation ===

    def create_key(
        self,
        messages: list[dict[str, str]],
        model: str = "",
        **kwargs,
    ) -> str:
        """Create a cache key from messages."""
        key_data = {
            "messages": messages,
            "model": model,
            **kwargs,
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()[:32]

    def create_semantic_key(
        self,
        prompt: str,
        model: str = "",
    ) -> str:
        """Create a semantic key (simplified without embeddings)."""
        # Normalize prompt
        normalized = " ".join(prompt.lower().split())
        key_data = f"{model}:{normalized}"
        return hashlib.sha256(key_data.encode()).hexdigest()[:32]

    # === Statistics ===

    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        return self._stats

    def get_entry_count(self) -> int:
        """Get number of entries."""
        return len(self._cache)

    def get_size_bytes(self) -> int:
        """Estimate cache size in bytes."""
        total = 0
        for entry in self._cache.values():
            total += len(json.dumps(entry.to_dict()).encode())
        return total

    # === Eviction ===

    def _evict_if_needed(self) -> None:
        """Evict entries if cache is full."""
        # Remove expired entries first
        expired_keys = [k for k, v in self._cache.items() if v.is_expired()]
        for key in expired_keys:
            del self._cache[key]
            self._stats.evictions += 1

        # LRU eviction if still over limit
        while len(self._cache) >= self.max_entries:
            # Find least recently used (fewest hits)
            lru_key = min(self._cache.keys(), key=lambda k: self._cache[k].hits)
            del self._cache[lru_key]
            self._stats.evictions += 1

    # === Persistence ===

    def _get_entry_path(self, key: str) -> Path:
        """Get file path for entry."""
        if not self.storage_path:
            raise ValueError("No storage path configured")
        return self.storage_path / f"{key}.json"

    def _save_entry(self, entry: CacheEntry) -> None:
        """Save entry to disk."""
        if not self.storage_path:
            return

        path = self._get_entry_path(entry.key)
        with open(path, "w") as f:
            json.dump(entry.to_dict(), f)

    def _load_entry(self, key: str) -> CacheEntry | None:
        """Load entry from disk."""
        if not self.storage_path:
            return None

        path = self._get_entry_path(key)
        if not path.exists():
            return None

        try:
            with open(path) as f:
                data = json.load(f)
            return CacheEntry.from_dict(data)
        except (json.JSONDecodeError, KeyError):
            return None

    def _delete_entry_file(self, key: str) -> None:
        """Delete entry file from disk."""
        if not self.storage_path:
            return

        path = self._get_entry_path(key)
        if path.exists():
            path.unlink()

    def _load_from_disk(self) -> None:
        """Load all entries from disk."""
        if not self.storage_path:
            return

        for f in self.storage_path.glob("*.json"):
            key = f.stem
            entry = self._load_entry(key)
            if entry and not entry.is_expired():
                self._cache[key] = entry

        self._stats.total_entries = len(self._cache)


class ResponseCache:
    """
    Higher-level cache for complete LLM responses.

    Usage:
        cache = ResponseCache()

        @cache.cached
        async def get_completion(prompt: str) -> str:
            return await llm.complete(prompt)
    """

    def __init__(self, llm_cache: LLMCache | None = None):
        self._cache = llm_cache or LLMCache()

    def cached(self, func):
        """Decorator for caching function results."""

        def wrapper(*args, **kwargs):
            # Create key from function args
            key = self._cache.create_key(
                messages=[{"content": str(args)}],
                model=func.__name__,
            )

            cached = self._cache.get(key)
            if cached is not None:
                return cached

            result = func(*args, **kwargs)
            self._cache.set(key, result)
            return result

        return wrapper

    async def cached_async(self, func):
        """Async decorator for caching."""

        async def wrapper(*args, **kwargs):
            key = self._cache.create_key(
                messages=[{"content": str(args)}],
                model=func.__name__,
            )

            cached = self._cache.get(key)
            if cached is not None:
                return cached

            result = await func(*args, **kwargs)
            self._cache.set(key, result)
            return result

        return wrapper

    def get_stats(self) -> CacheStats:
        return self._cache.get_stats()

    def clear(self) -> None:
        self._cache.clear()


def create_llm_cache(
    ttl_hours: int = 24,
    max_entries: int = 1000,
    storage_path: str | None = None,
) -> LLMCache:
    """
    Factory function to create LLM cache.

    Args:
        ttl_hours: TTL in hours
        max_entries: Max entries to store
        storage_path: Path for persistent storage

    Returns:
        Configured LLMCache
    """
    return LLMCache(
        ttl_seconds=ttl_hours * 3600,
        max_entries=max_entries,
        storage_path=storage_path,
    )
