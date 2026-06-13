"""A small circuit breaker for LLM calls.

Complements retry + fallback: once a model/provider has failed
``failure_threshold`` times in a row, its circuit *opens* and further calls to
it are short-circuited (skipped, moving straight to the next fallback) for
``reset_timeout`` seconds. After that cooldown a single trial is allowed
(*half-open*); success closes the circuit, another failure re-opens it.

This stops a dead provider from being retried on every single invocation.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from enum import Enum


class CircuitState(str, Enum):
    """Lifecycle state of a :class:`CircuitBreaker`."""

    closed = "closed"  # normal operation, calls allowed
    open = "open"  # failing, calls skipped until the cooldown elapses
    half_open = "half_open"  # cooldown elapsed, one trial call allowed


class CircuitBreakerOpenError(RuntimeError):
    """Raised/used as the recorded error when a call is skipped by an open circuit."""

    def __init__(self, key: object, retry_after: float) -> None:
        self.key = key
        self.retry_after = retry_after
        super().__init__(f"Circuit breaker open for {key!r}; retry in {retry_after:.1f}s")


class CircuitBreaker:
    """Per-target failure tracker with open/half-open/closed states.

    Args:
        failure_threshold: Consecutive failures that trip the circuit (>= 1).
        reset_timeout: Seconds to stay open before allowing a half-open trial.
        time_func: Monotonic clock source; injectable for testing.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        reset_timeout: float = 30.0,
        time_func: Callable[[], float] = time.monotonic,
    ) -> None:
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1")
        if reset_timeout <= 0:
            raise ValueError("reset_timeout must be > 0")
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self._time = time_func
        self._failures = 0
        self._state = CircuitState.closed
        self._opened_at = 0.0

    @property
    def state(self) -> CircuitState:
        return self._state

    @property
    def failure_count(self) -> int:
        return self._failures

    def allow(self) -> bool:
        """Return True if a call may proceed, transitioning open -> half-open if due."""
        if self._state is CircuitState.open:
            if self._time() - self._opened_at >= self.reset_timeout:
                self._state = CircuitState.half_open
                return True
            return False
        return True

    def record_success(self) -> None:
        """Reset the breaker to closed after a successful call."""
        self._failures = 0
        self._state = CircuitState.closed

    def record_failure(self) -> None:
        """Register a failure, opening the circuit at/over threshold or from half-open."""
        self._failures += 1
        if self._state is CircuitState.half_open or self._failures >= self.failure_threshold:
            self._state = CircuitState.open
            self._opened_at = self._time()

    def retry_after(self) -> float:
        """Seconds remaining before an open circuit allows a half-open trial."""
        if self._state is not CircuitState.open:
            return 0.0
        return max(0.0, self.reset_timeout - (self._time() - self._opened_at))
