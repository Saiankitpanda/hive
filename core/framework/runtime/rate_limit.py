"""
Rate Limiting - Control throughput for API calls and agent actions.

Provides:
- Token bucket rate limiter
- Sliding window rate limiter
- Per-key rate limiting
- Async-compatible limiters

From ROADMAP: Production safety features
"""

import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import wraps
from threading import Lock
from typing import Any, TypeVar

T = TypeVar("T")


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""

    requests_per_second: float = 10.0
    burst_size: int = 10
    timeout: float | None = None  # Max wait time for token


class TokenBucket:
    """
    Token bucket rate limiter.

    Usage:
        limiter = TokenBucket(rate=10, capacity=20)

        # Check if request can proceed
        if limiter.acquire():
            make_request()

        # Wait for token
        limiter.acquire(blocking=True)
        make_request()
    """

    def __init__(
        self,
        rate: float = 10.0,
        capacity: int | None = None,
    ):
        """
        Initialize token bucket.

        Args:
            rate: Tokens added per second
            capacity: Maximum tokens (defaults to rate)
        """
        self.rate = rate
        self.capacity = capacity or int(rate)
        self._tokens = float(self.capacity)
        self._last_update = time.time()
        self._lock = Lock()

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self._last_update
        self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)
        self._last_update = now

    def acquire(
        self,
        tokens: int = 1,
        blocking: bool = False,
        timeout: float | None = None,
    ) -> bool:
        """
        Acquire tokens from the bucket.

        Args:
            tokens: Number of tokens to acquire
            blocking: Wait for tokens if not available
            timeout: Max wait time in seconds

        Returns:
            True if tokens acquired
        """
        start_time = time.time()

        while True:
            with self._lock:
                self._refill()

                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return True

            if not blocking:
                return False

            if timeout and (time.time() - start_time) >= timeout:
                return False

            # Wait before next check
            time.sleep(0.01)

    @property
    def available_tokens(self) -> float:
        """Get current available tokens."""
        with self._lock:
            self._refill()
            return self._tokens


class SlidingWindowLimiter:
    """
    Sliding window rate limiter for more accurate limiting.

    Usage:
        limiter = SlidingWindowLimiter(
            max_requests=100,
            window_seconds=60,
        )

        if limiter.is_allowed():
            make_request()
    """

    def __init__(
        self,
        max_requests: int = 100,
        window_seconds: float = 60.0,
    ):
        """
        Initialize sliding window limiter.

        Args:
            max_requests: Max requests per window
            window_seconds: Window size in seconds
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._timestamps: deque = deque()
        self._lock = Lock()

    def _cleanup(self) -> None:
        """Remove expired timestamps."""
        cutoff = time.time() - self.window_seconds
        while self._timestamps and self._timestamps[0] < cutoff:
            self._timestamps.popleft()

    def is_allowed(self) -> bool:
        """Check if request is allowed."""
        with self._lock:
            self._cleanup()

            if len(self._timestamps) < self.max_requests:
                self._timestamps.append(time.time())
                return True

            return False

    def wait_and_acquire(self, timeout: float | None = None) -> bool:
        """Wait for slot and acquire."""
        start = time.time()

        while True:
            if self.is_allowed():
                return True

            if timeout and (time.time() - start) >= timeout:
                return False

            time.sleep(0.01)

    @property
    def remaining_requests(self) -> int:
        """Get remaining requests in window."""
        with self._lock:
            self._cleanup()
            return max(0, self.max_requests - len(self._timestamps))

    @property
    def reset_time(self) -> float:
        """Get seconds until oldest request expires."""
        with self._lock:
            self._cleanup()
            if self._timestamps:
                return max(0, (self._timestamps[0] + self.window_seconds) - time.time())
            return 0.0


@dataclass
class KeyedRateLimiter:
    """
    Per-key rate limiting for multi-tenant scenarios.

    Usage:
        limiter = KeyedRateLimiter(
            max_requests=10,
            window_seconds=60,
        )

        # Limit per API key
        if limiter.is_allowed(key="user_123"):
            process_request()

        # Limit per IP
        if limiter.is_allowed(key=request.ip):
            handle_request()
    """

    max_requests: int = 10
    window_seconds: float = 60.0
    _limiters: dict[str, SlidingWindowLimiter] = field(default_factory=dict)
    _lock: Lock = field(default_factory=Lock)

    def _get_limiter(self, key: str) -> SlidingWindowLimiter:
        """Get or create limiter for key."""
        with self._lock:
            if key not in self._limiters:
                self._limiters[key] = SlidingWindowLimiter(
                    max_requests=self.max_requests,
                    window_seconds=self.window_seconds,
                )
            return self._limiters[key]

    def is_allowed(self, key: str) -> bool:
        """Check if request for key is allowed."""
        return self._get_limiter(key).is_allowed()

    def wait_and_acquire(self, key: str, timeout: float | None = None) -> bool:
        """Wait for slot and acquire for key."""
        return self._get_limiter(key).wait_and_acquire(timeout)

    def remaining(self, key: str) -> int:
        """Get remaining requests for key."""
        return self._get_limiter(key).remaining_requests

    def cleanup_expired(self) -> int:
        """Remove limiters with no recent activity."""
        with self._lock:
            to_remove = []
            for key, limiter in self._limiters.items():
                if limiter.remaining_requests == limiter.max_requests:
                    to_remove.append(key)

            for key in to_remove:
                del self._limiters[key]

            return len(to_remove)


def rate_limit(
    requests_per_second: float = 10.0,
    burst_size: int | None = None,
) -> Callable:
    """
    Decorator to rate limit function calls.

    Usage:
        @rate_limit(requests_per_second=5)
        def call_api():
            return requests.get(url)

        @rate_limit(requests_per_second=1, burst_size=5)
        def expensive_operation():
            ...
    """
    bucket = TokenBucket(
        rate=requests_per_second,
        capacity=burst_size or int(requests_per_second),
    )

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            bucket.acquire(blocking=True)
            return func(*args, **kwargs)

        return wrapper

    return decorator


class AdaptiveRateLimiter:
    """
    Adaptive rate limiter that adjusts based on response.

    Usage:
        limiter = AdaptiveRateLimiter(
            initial_rate=10,
            min_rate=1,
            max_rate=100,
        )

        for request in requests:
            limiter.wait()
            try:
                response = make_request()
                limiter.record_success()
            except RateLimitError:
                limiter.record_rate_limit()
    """

    def __init__(
        self,
        initial_rate: float = 10.0,
        min_rate: float = 1.0,
        max_rate: float = 100.0,
        increase_factor: float = 1.1,
        decrease_factor: float = 0.5,
    ):
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.increase_factor = increase_factor
        self.decrease_factor = decrease_factor

        self._current_rate = initial_rate
        self._bucket = TokenBucket(rate=initial_rate)
        self._lock = Lock()

    @property
    def current_rate(self) -> float:
        return self._current_rate

    def wait(self) -> None:
        """Wait for rate limit."""
        self._bucket.acquire(blocking=True)

    def record_success(self) -> None:
        """Record successful request, possibly increase rate."""
        with self._lock:
            new_rate = min(self._current_rate * self.increase_factor, self.max_rate)
            if new_rate != self._current_rate:
                self._current_rate = new_rate
                self._bucket = TokenBucket(rate=new_rate)

    def record_rate_limit(self) -> None:
        """Record rate limit hit, decrease rate."""
        with self._lock:
            new_rate = max(self._current_rate * self.decrease_factor, self.min_rate)
            self._current_rate = new_rate
            self._bucket = TokenBucket(rate=new_rate)

    def record_error(self) -> None:
        """Record error (non-rate-limit), slight decrease."""
        with self._lock:
            new_rate = max(self._current_rate * 0.9, self.min_rate)
            self._current_rate = new_rate
            self._bucket = TokenBucket(rate=new_rate)


def create_rate_limiter(
    requests_per_second: float = 10.0,
    burst_size: int | None = None,
) -> TokenBucket:
    """Factory to create rate limiter."""
    return TokenBucket(
        rate=requests_per_second,
        capacity=burst_size or int(requests_per_second),
    )
