"""Unit tests for the CircuitBreaker used by the Agent retry/fallback loop."""

from __future__ import annotations

import pytest

from agentflow.core.graph.agent_internal.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerOpenError,
    CircuitState,
)


class _FakeClock:
    """Deterministic monotonic clock; advance() to move time forward."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _breaker(threshold: int = 3, reset: float = 30.0) -> tuple[CircuitBreaker, _FakeClock]:
    clock = _FakeClock()
    return CircuitBreaker(failure_threshold=threshold, reset_timeout=reset, time_func=clock), clock


class TestCircuitBreaker:
    def test_starts_closed_and_allows(self):
        cb, _ = _breaker()
        assert cb.state is CircuitState.closed
        assert cb.allow() is True

    def test_failures_below_threshold_stay_closed(self):
        cb, _ = _breaker(threshold=3)
        cb.record_failure()
        cb.record_failure()
        assert cb.state is CircuitState.closed
        assert cb.allow() is True

    def test_opens_at_threshold(self):
        cb, _ = _breaker(threshold=3)
        for _ in range(3):
            cb.record_failure()
        assert cb.state is CircuitState.open
        assert cb.allow() is False

    def test_success_resets_failure_count(self):
        cb, _ = _breaker(threshold=3)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        assert cb.failure_count == 0
        assert cb.state is CircuitState.closed

    def test_half_open_after_reset_timeout(self):
        cb, clock = _breaker(threshold=2, reset=30.0)
        cb.record_failure()
        cb.record_failure()
        assert cb.allow() is False  # still open before timeout
        clock.advance(30.0)
        assert cb.allow() is True  # transitions to half-open
        assert cb.state is CircuitState.half_open

    def test_half_open_success_closes(self):
        cb, clock = _breaker(threshold=1, reset=10.0)
        cb.record_failure()  # opens
        clock.advance(10.0)
        assert cb.allow() is True  # half-open
        cb.record_success()
        assert cb.state is CircuitState.closed
        assert cb.allow() is True

    def test_half_open_failure_reopens(self):
        cb, clock = _breaker(threshold=1, reset=10.0)
        cb.record_failure()  # opens
        clock.advance(10.0)
        cb.allow()  # half-open
        cb.record_failure()  # fails the trial
        assert cb.state is CircuitState.open
        assert cb.allow() is False  # reopened, cooldown restarts

    def test_retry_after_counts_down(self):
        cb, clock = _breaker(threshold=1, reset=30.0)
        cb.record_failure()
        assert cb.retry_after() == pytest.approx(30.0)
        clock.advance(10.0)
        assert cb.retry_after() == pytest.approx(20.0)
        assert _breaker()[0].retry_after() == 0.0  # closed → 0

    @pytest.mark.parametrize("bad", [0, -1])
    def test_invalid_threshold(self, bad):
        with pytest.raises(ValueError, match="failure_threshold"):
            CircuitBreaker(failure_threshold=bad)

    @pytest.mark.parametrize("bad", [0, -5.0])
    def test_invalid_reset_timeout(self, bad):
        with pytest.raises(ValueError, match="reset_timeout"):
            CircuitBreaker(reset_timeout=bad)


def test_circuit_breaker_open_error_carries_context():
    err = CircuitBreakerOpenError(("openai", "gpt-4o"), 12.5)
    assert err.key == ("openai", "gpt-4o")
    assert err.retry_after == 12.5
    assert "gpt-4o" in str(err)
