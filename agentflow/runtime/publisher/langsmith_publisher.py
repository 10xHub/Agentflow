"""LangsmithPublisher — OtelPublisher pre-configured for LangSmith.

A dedicated publisher that builds a LangSmith OTLP span processor on
construction and installs the ``TracerProvider`` before the first span is
created.  All span dispatch logic is inherited from ``OtelPublisher``.

Usage::

    from agentflow.runtime.publisher import LangsmithPublisher

    publisher = LangsmithPublisher(project="my-agent", level=ObservabilityLevel.FULL)
    graph._publisher = publisher
    compiled = graph.compile(...)

Or use the convenience helper which does the same in one call::

    from agentflow.runtime.publisher import setup_langsmith

    setup_langsmith(graph, project="my-agent")

Secrets stay in the environment: set ``LANGSMITH_API_KEY`` (or pass ``api_key=``).

Requires: pip install '10xscale-agentflow[langsmith]'
"""

from __future__ import annotations

from typing import Any

from .exporters import _build_langsmith_processor
from .otel_publisher import ObservabilityLevel, OtelPublisher


class LangsmithPublisher(OtelPublisher):
    """OtelPublisher pre-configured to send spans to LangSmith over OTLP.

    Builds an OTLP HTTP span processor pointing at LangSmith during
    initialisation and attaches it to either a supplied ``TracerProvider`` or a
    fresh one set as the global provider, so tracing is live before any span is
    created.  All event dispatch and span logic is inherited from
    ``OtelPublisher``.

    This publisher must be assigned to the graph **before** ``graph.compile()``.

    Args:
        api_key: LangSmith API key. Falls back to the ``LANGSMITH_API_KEY`` env
            var.
        project: LangSmith project name.  Sent as the ``Langsmith-Project``
            request header when provided.
        endpoint: Base OTEL endpoint URL for LangSmith.  ``/v1/traces`` is
            appended automatically.  Override for regional deployments, e.g.
            ``https://eu.api.smith.langchain.com/otel``.
        level: How much data to emit.  ``STANDARD`` by default.  Set to
            ``FULL`` for complete prompt/response visibility (may contain PII).
        tracer_provider: An existing ``TracerProvider`` to attach the processor
            to.  When ``None`` a new one is created and set as the global
            provider.

    Raises:
        ImportError: If ``opentelemetry-exporter-otlp-proto-http`` or
            ``opentelemetry-sdk`` are not installed.
        ValueError: If no API key is provided and ``LANGSMITH_API_KEY`` is not
            set.

    Example::

        publisher = LangsmithPublisher(
            project="my-agent",
            level=ObservabilityLevel.STANDARD,
        )
        graph._publisher = publisher
        compiled = graph.compile(...)
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        project: str | None = None,
        endpoint: str = "https://api.smith.langchain.com/otel",
        level: ObservabilityLevel = ObservabilityLevel.STANDARD,
        tracer_provider: Any = None,
    ) -> None:
        processor = _build_langsmith_processor(
            api_key=api_key, project=project, endpoint=endpoint
        )

        if tracer_provider is None:
            from opentelemetry import trace
            from opentelemetry.sdk.trace import TracerProvider

            provider = TracerProvider()
            provider.add_span_processor(processor)
            trace.set_tracer_provider(provider)
            # OtelPublisher picks up the global TracerProvider set above.
            super().__init__(level=level)
        else:
            tracer_provider.add_span_processor(processor)
            # Bind to the supplied provider explicitly (it may not be global).
            super().__init__(tracer=tracer_provider.get_tracer("agentflow"), level=level)
