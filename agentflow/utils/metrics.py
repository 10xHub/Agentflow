"""Metrics instrumentation.

Design goals:
 - Zero dependency by default.
 - Cheap no-op when disabled.
 - Exports to OpenTelemetry when a meter is configured.

The in-process registry alone was not observable: operators got spans (if OTEL
tracing was wired) but no counters, histograms, or gauges -- nothing to alert on.
There was no exporter at all, so error rates and latencies existed only inside the
process and died with it.

`setup_otel_metrics()` bridges this facade to OpenTelemetry. Every existing
`counter(...)` / `timer(...)` call site then exports automatically, with no change
at the call site.

Usage:
    from agentflow.utils.metrics import counter, timer, setup_otel_metrics

    setup_otel_metrics()  # once, at startup, if you want OTEL export

    counter('messages_written_total').inc()
    with timer('db_write_latency_ms'):
        ...

    # With attributes (dimensions), for OTEL:
    counter('agentflow.node.executions').inc(attributes={"node": "agent"})
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any


logger = logging.getLogger("agentflow.metrics")

_LOCK = threading.RLock()
_COUNTERS: dict[str, Counter] = {}
_TIMERS: dict[str, TimerMetric] = {}

_ENABLED = True  # could be toggled by env in future

# OpenTelemetry bridge (optional).
_OTEL_METER: Any = None
_OTEL_COUNTERS: dict[str, Any] = {}
_OTEL_HISTOGRAMS: dict[str, Any] = {}


def enable_metrics(value: bool) -> None:  # simple toggle; acceptable global
    # Intentionally keeps a module-level switch—call sites cheap check.
    globals()["_ENABLED"] = value


def setup_otel_metrics(meter: Any = None) -> bool:
    """Bridge this facade to OpenTelemetry metrics.

    Call once at startup. After this, every counter/timer in the framework is
    exported through the configured MeterProvider, so error rates and latencies
    are actually alertable instead of living and dying inside the process.

    Args:
        meter: An OTEL Meter. If omitted, one is obtained from the global
            MeterProvider (which your app configures with its exporter).

    Returns:
        True if the bridge was installed; False if OpenTelemetry is not installed
        (in which case the in-process registry keeps working as before).
    """
    global _OTEL_METER  # noqa: PLW0603

    if meter is None:
        try:
            from opentelemetry import metrics as otel_metrics
        except ImportError:
            logger.info(
                "OpenTelemetry is not installed; metrics stay in-process only. "
                "Install the 'otel' extra to export them."
            )
            return False
        meter = otel_metrics.get_meter("agentflow")

    with _LOCK:
        _OTEL_METER = meter
        _OTEL_COUNTERS.clear()
        _OTEL_HISTOGRAMS.clear()

    logger.info("OpenTelemetry metrics bridge enabled")
    return True


def _otel_counter(name: str) -> Any:
    """Lazily create (and memoize) the OTEL counter mirroring `name`."""
    if _OTEL_METER is None:
        return None
    instrument = _OTEL_COUNTERS.get(name)
    if instrument is None:
        with _LOCK:
            instrument = _OTEL_COUNTERS.get(name)
            if instrument is None:
                instrument = _OTEL_METER.create_counter(name)
                _OTEL_COUNTERS[name] = instrument
    return instrument


def _otel_histogram(name: str) -> Any:
    """Lazily create (and memoize) the OTEL histogram mirroring `name`."""
    if _OTEL_METER is None:
        return None
    instrument = _OTEL_HISTOGRAMS.get(name)
    if instrument is None:
        with _LOCK:
            instrument = _OTEL_HISTOGRAMS.get(name)
            if instrument is None:
                instrument = _OTEL_METER.create_histogram(name, unit="ms")
                _OTEL_HISTOGRAMS[name] = instrument
    return instrument


@dataclass
class Counter:
    name: str
    value: int = 0

    def inc(self, amount: int = 1, attributes: dict[str, Any] | None = None) -> None:
        if not _ENABLED:
            return
        with _LOCK:
            self.value += amount

        instrument = _otel_counter(self.name)
        if instrument is not None:
            try:
                instrument.add(amount, attributes or {})
            except Exception as e:  # telemetry must never break the caller
                logger.debug("Failed to record OTEL counter %s: %s", self.name, e)


@dataclass
class TimerMetric:
    name: str
    count: int = 0
    total_ms: float = 0.0
    max_ms: float = 0.0

    def observe(self, duration_ms: float, attributes: dict[str, Any] | None = None) -> None:
        if not _ENABLED:
            return
        with _LOCK:
            self.count += 1
            self.total_ms += duration_ms
            self.max_ms = max(self.max_ms, duration_ms)

        instrument = _otel_histogram(self.name)
        if instrument is not None:
            try:
                instrument.record(duration_ms, attributes or {})
            except Exception as e:  # telemetry must never break the caller
                logger.debug("Failed to record OTEL histogram %s: %s", self.name, e)

    @property
    def avg_ms(self) -> float:
        if self.count == 0:
            return 0.0
        return self.total_ms / self.count


def counter(name: str) -> Counter:
    with _LOCK:
        c = _COUNTERS.get(name)
        if c is None:
            c = Counter(name)
            _COUNTERS[name] = c
        return c


def timer(name: str, attributes: dict[str, Any] | None = None) -> _TimerCtx:
    """Time a block. Attributes become OTEL histogram dimensions."""
    metric = _TIMERS.get(name)
    if metric is None:
        with _LOCK:
            metric = _TIMERS.get(name)
            if metric is None:
                metric = TimerMetric(name)
                _TIMERS[name] = metric
    return _TimerCtx(metric, attributes)


class _TimerCtx:
    def __init__(self, metric: TimerMetric, attributes: dict[str, Any] | None = None):
        self.metric = metric
        self.attributes = attributes
        self._start = None  # type: float | None

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._start is not None:
            elapsed_ms = (time.perf_counter() - self._start) * 1000.0
            # Tag the observation with the outcome, so latency can be split by
            # success vs failure -- a p99 that mixes them is not actionable.
            attrs = dict(self.attributes or {})
            attrs["outcome"] = "error" if exc_type is not None else "ok"
            self.metric.observe(elapsed_ms, attrs)
        # Do not suppress exceptions
        return False


def snapshot() -> dict:
    """Return a point-in-time snapshot of metrics (thread-safe copy)."""
    with _LOCK:
        return {
            "counters": {k: v.value for k, v in _COUNTERS.items()},
            "timers": {
                k: {
                    "count": t.count,
                    "avg_ms": t.avg_ms,
                    "max_ms": t.max_ms,
                }
                for k, t in _TIMERS.items()
            },
        }
