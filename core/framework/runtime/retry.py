"""
Retry Utilities - Robust retry with exponential backoff.

Provides:
- Configurable retry strategies
- Exponential backoff with jitter
- Circuit breaker pattern
- Retry decorators for functions

From ROADMAP: Production reliability features
"""

import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from functools import wraps
from typing import Any, TypeVar

T = TypeVar("T")


class RetryStrategy(Enum):
    """Retry strategy types."""

    FIXED = "fixed"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""

    max_attempts: int = 3
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    base_delay: float = 1.0
    max_delay: float = 60.0
    jitter: bool = True
    retryable_exceptions: tuple = (Exception,)

    def get_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt."""
        if self.strategy == RetryStrategy.FIXED:
            delay = self.base_delay
        elif self.strategy == RetryStrategy.LINEAR:
            delay = self.base_delay * attempt
        else:  # EXPONENTIAL
            delay = self.base_delay * (2 ** (attempt - 1))

        delay = min(delay, self.max_delay)

        if self.jitter:
            delay = delay * (0.5 + random.random())

        return delay


class RetryError(Exception):
    """Error after all retries exhausted."""

    def __init__(self, message: str, last_exception: Exception | None = None):
        super().__init__(message)
        self.last_exception = last_exception


def retry(
    max_attempts: int = 3,
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter: bool = True,
    on_retry: Callable[[int, Exception], None] | None = None,
    retryable_exceptions: tuple = (Exception,),
) -> Callable:
    """
    Decorator for retrying functions with backoff.

    Usage:
        @retry(max_attempts=3, base_delay=1.0)
        def call_api():
            return requests.get(url)

        @retry(
            max_attempts=5,
            strategy=RetryStrategy.EXPONENTIAL,
            on_retry=lambda attempt, e: print(f"Retry {attempt}: {e}")
        )
        async def async_call():
            return await client.request()
    """
    config = RetryConfig(
        max_attempts=max_attempts,
        strategy=strategy,
        base_delay=base_delay,
        max_delay=max_delay,
        jitter=jitter,
        retryable_exceptions=retryable_exceptions,
    )

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None

            for attempt in range(1, config.max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except config.retryable_exceptions as e:
                    last_exception = e

                    if attempt == config.max_attempts:
                        break

                    delay = config.get_delay(attempt)

                    if on_retry:
                        on_retry(attempt, e)

                    time.sleep(delay)

            raise RetryError(
                f"Failed after {config.max_attempts} attempts",
                last_exception,
            )

        return wrapper

    return decorator


async def retry_async(
    func: Callable[..., T],
    *args,
    config: RetryConfig | None = None,
    on_retry: Callable[[int, Exception], None] | None = None,
    **kwargs,
) -> T:
    """
    Retry an async function with backoff.

    Usage:
        result = await retry_async(
            async_call,
            arg1, arg2,
            config=RetryConfig(max_attempts=3),
        )
    """
    import asyncio

    if config is None:
        config = RetryConfig()

    last_exception = None

    for attempt in range(1, config.max_attempts + 1):
        try:
            return await func(*args, **kwargs)
        except config.retryable_exceptions as e:
            last_exception = e

            if attempt == config.max_attempts:
                break

            delay = config.get_delay(attempt)

            if on_retry:
                on_retry(attempt, e)

            await asyncio.sleep(delay)

    raise RetryError(
        f"Failed after {config.max_attempts} attempts",
        last_exception,
    )


class CircuitBreaker:
    """
    Circuit breaker pattern to prevent cascading failures.

    Usage:
        breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=30,
        )

        @breaker
        def call_service():
            return service.call()

        # Or use as context manager
        with breaker:
            result = service.call()
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_requests: int = 1,
    ):
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Failures before opening circuit
            recovery_timeout: Seconds before attempting recovery
            half_open_requests: Requests to allow in half-open state
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_requests = half_open_requests

        self._failure_count = 0
        self._last_failure_time: float | None = None
        self._state = "closed"  # closed, open, half-open
        self._half_open_count = 0

    @property
    def state(self) -> str:
        """Get current state."""
        self._check_state()
        return self._state

    @property
    def is_closed(self) -> bool:
        return self.state == "closed"

    @property
    def is_open(self) -> bool:
        return self.state == "open"

    def _check_state(self) -> None:
        """Check and update state based on time."""
        if self._state == "open" and self._last_failure_time:
            if time.time() - self._last_failure_time >= self.recovery_timeout:
                self._state = "half-open"
                self._half_open_count = 0

    def _record_success(self) -> None:
        """Record a successful call."""
        if self._state == "half-open":
            self._half_open_count += 1
            if self._half_open_count >= self.half_open_requests:
                self._state = "closed"
                self._failure_count = 0
        self._failure_count = 0

    def _record_failure(self) -> None:
        """Record a failed call."""
        self._failure_count += 1
        self._last_failure_time = time.time()

        if self._state == "half-open":
            self._state = "open"
        elif self._failure_count >= self.failure_threshold:
            self._state = "open"

    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        """Use as decorator."""

        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            self._check_state()

            if self._state == "open":
                raise CircuitBreakerOpen(
                    f"Circuit breaker is open. Retry after {self.recovery_timeout}s"
                )

            try:
                result = func(*args, **kwargs)
                self._record_success()
                return result
            except Exception as e:
                self._record_failure()
                raise e

        return wrapper

    def __enter__(self) -> "CircuitBreaker":
        self._check_state()
        if self._state == "open":
            raise CircuitBreakerOpen("Circuit breaker is open")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self._record_success()
        else:
            self._record_failure()
        return False

    def reset(self) -> None:
        """Reset the circuit breaker."""
        self._failure_count = 0
        self._last_failure_time = None
        self._state = "closed"
        self._half_open_count = 0


class CircuitBreakerOpen(Exception):
    """Exception when circuit breaker is open."""

    pass


def with_timeout(
    timeout: float,
    on_timeout: Callable[[], Any] | None = None,
) -> Callable:
    """
    Decorator to add timeout to functions.

    Usage:
        @with_timeout(5.0)
        def slow_function():
            ...

        @with_timeout(10.0, on_timeout=lambda: "default")
        def api_call():
            ...
    """
    import signal

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            def timeout_handler(signum, frame):
                raise TimeoutError(f"Function timed out after {timeout}s")

            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.setitimer(signal.ITIMER_REAL, timeout)

            try:
                return func(*args, **kwargs)
            except TimeoutError:
                if on_timeout:
                    return on_timeout()
                raise
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, old_handler)

        return wrapper

    return decorator
