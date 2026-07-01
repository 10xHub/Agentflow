"""Logfire and LangSmith observability helpers for Agentflow.

Both backends are OpenTelemetry-native, so neither needs a bespoke publisher.
They attach as span processors to the TracerProvider that OtelPublisher already
feeds, then `setup_tracing` is called as normal.

Usage:

    # Logfire only
    from agentflow.runtime.publisher import setup_logfire
    setup_logfire(graph, service_name="my-agent")

    # LangSmith only
    from agentflow.runtime.publisher import setup_langsmith
    setup_langsmith(graph, project="my-project")

    # Both at once (shares one TracerProvider)
    from agentflow.runtime.publisher import setup_observability
    setup_observability(graph, {
        "logfire":   {"enabled": True, "service_name": "my-agent"},
        "langsmith": {"enabled": True, "project": "my-project"},
    })

Requires extras:
  pip install '10xscale-agentflow[logfire]'       # Logfire
  pip install '10xscale-agentflow[langsmith]'     # LangSmith
  pip install '10xscale-agentflow[observability]' # both
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from .otel_publisher import ObservabilityLevel, setup_tracing


if TYPE_CHECKING:
    from agentflow.core.graph.state_graph import StateGraph


# ── Guard helpers ─────────────────────────────────────────────────────────────


def _guard_logfire() -> None:
    try:
        import logfire  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Logfire is required for logfire tracing. "
            "Install with: pip install '10xscale-agentflow[logfire]'"
        ) from exc


def _guard_otlp_http() -> None:
    try:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "opentelemetry-exporter-otlp-proto-http is required for LangSmith tracing. "
            "Install with: pip install '10xscale-agentflow[langsmith]'"
        ) from exc


def _guard_otel_sdk() -> None:
    try:
        from opentelemetry.sdk.trace import TracerProvider  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "opentelemetry-sdk is required for observability. "
            "Install with: pip install '10xscale-agentflow[otel]'"
        ) from exc


def _build_langsmith_processor(
    *,
    api_key: str | None = None,
    project: str | None = None,
    endpoint: str = "https://api.smith.langchain.com/otel",
) -> Any:
    """Build a ``BatchSpanProcessor`` exporting to LangSmith's OTLP endpoint.

    Resolves the API key from ``LANGSMITH_API_KEY`` when ``api_key`` is not
    passed, sets the ``x-api-key`` and (optionally) ``Langsmith-Project``
    headers, and appends ``/v1/traces`` to ``endpoint``.

    Args:
        api_key: LangSmith API key. Falls back to ``LANGSMITH_API_KEY``.
        project: LangSmith project name, sent as the ``Langsmith-Project`` header.
        endpoint: Base OTEL endpoint URL. ``/v1/traces`` is appended automatically.

    Returns:
        A ``BatchSpanProcessor`` wrapping the LangSmith OTLP HTTP exporter.

    Raises:
        ImportError: If ``opentelemetry-sdk`` or the OTLP HTTP exporter are missing.
        ValueError: If no API key is available.
    """
    _guard_otel_sdk()
    _guard_otlp_http()

    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    resolved_key = api_key or os.environ.get("LANGSMITH_API_KEY")
    if not resolved_key:
        raise ValueError(
            "A LangSmith API key is required. Pass api_key= or set the "
            "LANGSMITH_API_KEY environment variable."
        )

    headers: dict[str, str] = {"x-api-key": resolved_key}
    if project:
        headers["Langsmith-Project"] = project

    # The LangSmith OTLP endpoint expects /v1/traces
    traces_endpoint = endpoint.rstrip("/") + "/v1/traces"
    exporter = OTLPSpanExporter(endpoint=traces_endpoint, headers=headers)
    return BatchSpanProcessor(exporter)


# ── Public helpers ─────────────────────────────────────────────────────────────


def setup_logfire(
    graph: StateGraph | None = None,
    *,
    token: str | None = None,
    service_name: str | None = None,
    send_to_logfire: bool = True,
    console: Any = None,
    level: ObservabilityLevel = ObservabilityLevel.STANDARD,
    additional_span_processors: list | None = None,
    **configure_kwargs: Any,
) -> None:
    """Configure Logfire as the OTEL backend and instrument the graph.

    Calls ``logfire.configure(...)`` to install the global TracerProvider, then
    calls ``setup_tracing(graph, level=level)`` so OtelPublisher feeds into it.

    Must be called **before** ``graph.compile()``.

    Args:
        graph: The StateGraph to instrument.
        token: Logfire write token. Falls back to the ``LOGFIRE_TOKEN`` env var.
        service_name: Service name shown in the Logfire UI.
        send_to_logfire: Whether to export spans to logfire.dev.
        console: Console output options. Pass ``False`` to disable, or a
            ``logfire.ConsoleOptions`` instance. ``None`` uses env-var defaults.
        level: Observability level — SPANS, STANDARD (default), or FULL.
        additional_span_processors: Extra SpanProcessors to attach alongside
            the Logfire processor (e.g. a LangSmith OTLP processor).
        **configure_kwargs: Additional keyword arguments forwarded to
            ``logfire.configure()``.

    Raises:
        ImportError: If ``logfire`` is not installed.
    """
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
    # graph is None when only provider configuration is wanted (e.g. the API
    # bootstrap binds the publisher into the DI container itself).
    if graph is not None:
        setup_tracing(graph, level=level)


def setup_langsmith(
    graph: StateGraph | None = None,
    *,
    api_key: str | None = None,
    project: str | None = None,
    endpoint: str = "https://api.smith.langchain.com/otel",
    level: ObservabilityLevel = ObservabilityLevel.STANDARD,
    tracer_provider: Any = None,
) -> None:
    """Attach a LangSmith OTLP HTTP exporter and instrument the graph.

    Builds an ``OTLPSpanExporter`` pointing at LangSmith's OTEL endpoint,
    wraps it in a ``BatchSpanProcessor``, and attaches it to either the
    supplied ``tracer_provider`` or a fresh ``TracerProvider`` set as the
    global provider.  Then calls ``setup_tracing(graph, level=level)``.

    Must be called **before** ``graph.compile()``.

    Args:
        graph: The StateGraph to instrument.
        api_key: LangSmith API key. Falls back to the ``LANGSMITH_API_KEY``
            env var.
        project: LangSmith project name.  Sent as the ``Langsmith-Project``
            request header when provided.
        endpoint: Base OTEL endpoint URL for LangSmith.  The exporter appends
            ``/v1/traces`` automatically.  Override for regional deployments,
            e.g. ``https://eu.api.smith.langchain.com/otel``.
        level: Observability level — SPANS, STANDARD (default), or FULL.
        tracer_provider: An existing ``TracerProvider`` to attach the processor
            to.  When ``None`` a new one is created and set as the global
            provider.

    Raises:
        ImportError: If ``opentelemetry-exporter-otlp-proto-http`` or
            ``opentelemetry-sdk`` are not installed.
        ValueError: If no API key is provided and ``LANGSMITH_API_KEY`` is not
            set.
    """
    processor = _build_langsmith_processor(api_key=api_key, project=project, endpoint=endpoint)

    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider

    if tracer_provider is None:
        provider = TracerProvider()
        provider.add_span_processor(processor)
        trace.set_tracer_provider(provider)
    else:
        tracer_provider.add_span_processor(processor)

    # graph is None when only provider configuration is wanted (e.g. the API
    # bootstrap binds the publisher into the DI container itself).
    if graph is not None:
        setup_tracing(graph, level=level)


def setup_observability(graph: StateGraph | None, config: dict[str, Any]) -> None:
    """Unified observability entry point for declarative ``agentflow.json`` config.

    Reads the ``observability`` block and enables Logfire and/or LangSmith,
    ensuring they share one ``TracerProvider`` when both are active.

    Pass ``graph=None`` to configure the providers/exporters only, without
    binding an ``OtelPublisher`` onto a graph. The API bootstrap uses this mode
    because it binds the publisher into the DI container before the graph is
    compiled (the only reliable attach point).

    Expected ``config`` shape (mirrors ``agentflow.json``)::

        {
            "level": "standard",
            "logfire": {
                "enabled": True,
                "service_name": "my-agent",
                "send_to_logfire": True,
                "console": False,
            },
            "langsmith": {"enabled": True, "project": "my-project", "endpoint": null},
        }

    Secrets (``LOGFIRE_TOKEN``, ``LANGSMITH_API_KEY``) must be in environment
    variables — never in the config dict.

    Args:
        graph: The StateGraph to instrument.
        config: The observability config dict.

    Raises:
        ImportError: If a requested backend's package is not installed.
        ValueError: If LangSmith is enabled but no API key is available.
    """
    raw_level = config.get("level", "standard")
    try:
        level = ObservabilityLevel(raw_level)
    except ValueError:
        level = ObservabilityLevel.STANDARD

    logfire_cfg: dict[str, Any] = config.get("logfire") or {}
    langsmith_cfg: dict[str, Any] = config.get("langsmith") or {}

    logfire_on = bool(logfire_cfg.get("enabled", False))
    langsmith_on = bool(langsmith_cfg.get("enabled", False))

    if not logfire_on and not langsmith_on:
        return

    if logfire_on and langsmith_on:
        # Both enabled: build the LangSmith processor first, pass it to logfire
        # so they share the single TracerProvider that logfire installs.
        _guard_logfire()

        ls_endpoint = langsmith_cfg.get("endpoint") or "https://api.smith.langchain.com/otel"
        ls_processor = _build_langsmith_processor(
            project=langsmith_cfg.get("project"),
            endpoint=ls_endpoint,
        )

        setup_logfire(
            graph,
            service_name=logfire_cfg.get("service_name"),
            send_to_logfire=bool(logfire_cfg.get("send_to_logfire", True)),
            console=logfire_cfg.get("console"),
            level=level,
            additional_span_processors=[ls_processor],
        )
        return

    if logfire_on:
        setup_logfire(
            graph,
            service_name=logfire_cfg.get("service_name"),
            send_to_logfire=bool(logfire_cfg.get("send_to_logfire", True)),
            console=logfire_cfg.get("console"),
            level=level,
        )
        return

    # LangSmith only
    setup_langsmith(
        graph,
        project=langsmith_cfg.get("project"),
        endpoint=langsmith_cfg.get("endpoint") or "https://api.smith.langchain.com/otel",
        level=level,
    )
