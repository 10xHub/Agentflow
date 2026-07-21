"""LogfirePublisher — OtelPublisher pre-configured for Logfire.

A dedicated publisher that calls ``logfire.configure()`` on construction so
the global TracerProvider is installed before the first span is created.
All span dispatch logic is inherited from ``OtelPublisher``.

Usage::

    from agentflow.runtime.publisher import LogfirePublisher

    publisher = LogfirePublisher(service_name="my-agent", level=ObservabilityLevel.FULL)
    graph._publisher = publisher
    compiled = graph.compile(...)

Or use the convenience helper which does the same in one call::

    from agentflow.runtime.publisher import setup_logfire

    setup_logfire(graph, service_name="my-agent")

Requires: pip install '10xscale-agentflow[logfire]'
"""

from __future__ import annotations

from typing import Any

from .exporters import _guard_logfire
from .otel_publisher import ObservabilityLevel, OtelPublisher


class LogfirePublisher(OtelPublisher):
    """OtelPublisher pre-configured to send spans to Logfire.

    Calls ``logfire.configure()`` during initialisation so the Logfire-managed
    ``TracerProvider`` is set as the global provider before any span is created.
    All event dispatch and span logic is inherited from ``OtelPublisher``.

    This publisher must be assigned to the graph **before** ``graph.compile()``.

    Args:
        token: Logfire write token. Falls back to the ``LOGFIRE_TOKEN`` env var.
        service_name: Service name shown in the Logfire UI.
        send_to_logfire: Whether to export spans to logfire.dev.  Defaults to
            ``True``.
        console: Console output option.  Pass ``False`` to suppress console
            output, a ``logfire.ConsoleOptions`` instance for fine-grained
            control, or ``None`` to rely on environment variables.
        level: How much data to emit.  ``STANDARD`` by default.  Set to
            ``FULL`` for complete prompt/response visibility (may contain PII).
        additional_span_processors: Extra ``SpanProcessor`` instances to attach
            alongside the Logfire processor (e.g. a LangSmith OTLP processor).
        **configure_kwargs: Additional keyword arguments forwarded verbatim to
            ``logfire.configure()``.

    Raises:
        ImportError: If the ``logfire`` package is not installed.

    Example::

        publisher = LogfirePublisher(
            service_name="my-agent",
            send_to_logfire=True,
            level=ObservabilityLevel.STANDARD,
        )
        graph._publisher = publisher
        compiled = graph.compile(...)
    """

    def __init__(
        self,
        *,
        token: str | None = None,
        service_name: str | None = None,
        send_to_logfire: bool = True,
        console: Any = None,
        level: ObservabilityLevel = ObservabilityLevel.STANDARD,
        additional_span_processors: list | None = None,
        **configure_kwargs: Any,
    ) -> None:
        _guard_logfire()
        import logfire

        kwargs: dict[str, Any] = {
            "send_to_logfire": send_to_logfire,
            **configure_kwargs,
        }
        if token is not None:
            kwargs["token"] = token
        if service_name is not None:
            kwargs["service_name"] = service_name
        if console is not None:
            kwargs["console"] = console
        if additional_span_processors:
            kwargs["additional_span_processors"] = additional_span_processors

        logfire.configure(**kwargs)
        # OtelPublisher picks up the global TracerProvider set by logfire.configure
        super().__init__(level=level)
