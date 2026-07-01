"""Unit tests for agentflow.runtime.publisher.exporters.

All external dependencies (logfire, opentelemetry SDK / OTLP exporter) are
fully mocked so the tests run without any optional extras installed.

Coverage targets:
  - _guard_logfire         — present / missing
  - _guard_otlp_http       — present / missing
  - _guard_otel_sdk        — present / missing
  - setup_logfire          — all optional params, extra kwargs
  - setup_langsmith        — env-var key, explicit key+project, custom endpoint,
                             existing provider, no-key ValueError
  - setup_observability    — neither / logfire-only / langsmith-only / both /
                             invalid level / null sub-configs
"""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_graph():
    """Minimal graph mock that records the publisher assigned to it."""
    g = MagicMock()
    g._publisher = None
    return g


def _fake_otel_modules():
    """Return a dict of fake opentelemetry modules suitable for sys.modules patching."""
    # opentelemetry (top-level)
    otel = types.ModuleType("opentelemetry")

    # opentelemetry.trace
    trace_mod = types.ModuleType("opentelemetry.trace")
    _mock_provider = MagicMock()
    trace_mod.set_tracer_provider = MagicMock()
    trace_mod.get_tracer_provider = MagicMock(return_value=_mock_provider)
    otel.trace = trace_mod

    # opentelemetry.sdk
    sdk = types.ModuleType("opentelemetry.sdk")
    sdk_trace = types.ModuleType("opentelemetry.sdk.trace")

    FakeTracerProvider = MagicMock(name="TracerProvider")
    sdk_trace.TracerProvider = FakeTracerProvider
    sdk.trace = sdk_trace

    sdk_trace_export = types.ModuleType("opentelemetry.sdk.trace.export")
    FakeBatchSpanProcessor = MagicMock(name="BatchSpanProcessor")
    sdk_trace_export.BatchSpanProcessor = FakeBatchSpanProcessor
    sdk_trace.export = sdk_trace_export

    # opentelemetry.exporter.otlp.proto.http.trace_exporter
    exporter_root = types.ModuleType("opentelemetry.exporter")
    exporter_otlp = types.ModuleType("opentelemetry.exporter.otlp")
    exporter_proto = types.ModuleType("opentelemetry.exporter.otlp.proto")
    exporter_http = types.ModuleType("opentelemetry.exporter.otlp.proto.http")
    exporter_trace = types.ModuleType(
        "opentelemetry.exporter.otlp.proto.http.trace_exporter"
    )
    FakeOTLPExporter = MagicMock(name="OTLPSpanExporter")
    exporter_trace.OTLPSpanExporter = FakeOTLPExporter

    return {
        "opentelemetry": otel,
        "opentelemetry.trace": trace_mod,
        "opentelemetry.sdk": sdk,
        "opentelemetry.sdk.trace": sdk_trace,
        "opentelemetry.sdk.trace.export": sdk_trace_export,
        "opentelemetry.exporter": exporter_root,
        "opentelemetry.exporter.otlp": exporter_otlp,
        "opentelemetry.exporter.otlp.proto": exporter_proto,
        "opentelemetry.exporter.otlp.proto.http": exporter_http,
        "opentelemetry.exporter.otlp.proto.http.trace_exporter": exporter_trace,
        # Classes for direct import
        "_fake_TracerProvider": FakeTracerProvider,
        "_fake_BatchSpanProcessor": FakeBatchSpanProcessor,
        "_fake_OTLPSpanExporter": FakeOTLPExporter,
        "_fake_trace_mod": trace_mod,
    }


def _fake_logfire_module():
    lf = types.ModuleType("logfire")
    lf.configure = MagicMock()
    return lf


# ── Guard tests ────────────────────────────────────────────────────────────────


class TestGuardLogfire:
    def test_raises_when_logfire_missing(self):
        with patch.dict(sys.modules, {"logfire": None}):
            # Remove from sys.modules so the import attempt actually fails
            saved = sys.modules.pop("logfire", ...)
            try:
                from agentflow.runtime.publisher.exporters import _guard_logfire

                with pytest.raises(ImportError, match="logfire"):
                    _guard_logfire()
            finally:
                if saved is not ...:
                    sys.modules["logfire"] = saved

    def test_passes_when_logfire_present(self):
        fake_lf = _fake_logfire_module()
        with patch.dict(sys.modules, {"logfire": fake_lf}):
            from agentflow.runtime.publisher.exporters import _guard_logfire

            _guard_logfire()  # should not raise


class TestGuardOtlpHttp:
    def test_raises_when_package_missing(self):
        key = "opentelemetry.exporter.otlp.proto.http.trace_exporter"
        saved = sys.modules.pop(key, ...)
        try:
            sys.modules[key] = None  # type: ignore[assignment]
            from agentflow.runtime.publisher.exporters import _guard_otlp_http

            with pytest.raises(ImportError, match="opentelemetry-exporter-otlp-proto-http"):
                _guard_otlp_http()
        finally:
            if saved is not ...:
                sys.modules[key] = saved
            else:
                sys.modules.pop(key, None)

    def test_passes_when_package_present(self):
        mods = _fake_otel_modules()
        with patch.dict(sys.modules, mods):
            from agentflow.runtime.publisher.exporters import _guard_otlp_http

            _guard_otlp_http()  # should not raise


class TestGuardOtelSdk:
    def test_raises_when_sdk_missing(self):
        saved = sys.modules.pop("opentelemetry.sdk.trace", ...)
        try:
            sys.modules["opentelemetry.sdk.trace"] = None  # type: ignore[assignment]
            from agentflow.runtime.publisher.exporters import _guard_otel_sdk

            with pytest.raises(ImportError, match="opentelemetry-sdk"):
                _guard_otel_sdk()
        finally:
            if saved is not ...:
                sys.modules["opentelemetry.sdk.trace"] = saved
            else:
                sys.modules.pop("opentelemetry.sdk.trace", None)

    def test_passes_when_sdk_present(self):
        mods = _fake_otel_modules()
        with patch.dict(sys.modules, mods):
            from agentflow.runtime.publisher.exporters import _guard_otel_sdk

            _guard_otel_sdk()  # should not raise


# ── setup_logfire tests ────────────────────────────────────────────────────────


class TestSetupLogfire:
    """Tests for setup_logfire()."""

    def _run(self, graph, fake_lf, fake_otel_mods, **kwargs):
        """Patch everything and call setup_logfire."""
        all_mods = {**fake_otel_mods, "logfire": fake_lf}
        with patch.dict(sys.modules, all_mods):
            with patch(
                "agentflow.runtime.publisher.exporters.setup_tracing"
            ) as mock_setup:
                from agentflow.runtime.publisher.exporters import setup_logfire

                setup_logfire(graph, **kwargs)
                return mock_setup

    def test_minimal_call(self):
        graph = _make_graph()
        fake_lf = _fake_logfire_module()
        mods = _fake_otel_modules()

        mock_setup = self._run(graph, fake_lf, mods)

        fake_lf.configure.assert_called_once()
        call_kwargs = fake_lf.configure.call_args[1]
        assert call_kwargs["send_to_logfire"] is True
        # level defaults to STANDARD — setup_tracing called
        mock_setup.assert_called_once()

    def test_token_and_service_name_forwarded(self):
        graph = _make_graph()
        fake_lf = _fake_logfire_module()
        mods = _fake_otel_modules()

        self._run(graph, fake_lf, mods, token="tok123", service_name="svc")

        call_kwargs = fake_lf.configure.call_args[1]
        assert call_kwargs["token"] == "tok123"
        assert call_kwargs["service_name"] == "svc"

    def test_none_token_not_forwarded(self):
        """token=None should not be passed to logfire.configure."""
        graph = _make_graph()
        fake_lf = _fake_logfire_module()
        mods = _fake_otel_modules()

        self._run(graph, fake_lf, mods, token=None, service_name=None)

        call_kwargs = fake_lf.configure.call_args[1]
        assert "token" not in call_kwargs
        assert "service_name" not in call_kwargs

    def test_console_false_forwarded(self):
        graph = _make_graph()
        fake_lf = _fake_logfire_module()
        mods = _fake_otel_modules()

        self._run(graph, fake_lf, mods, console=False)

        call_kwargs = fake_lf.configure.call_args[1]
        assert call_kwargs["console"] is False

    def test_console_none_not_forwarded(self):
        graph = _make_graph()
        fake_lf = _fake_logfire_module()
        mods = _fake_otel_modules()

        self._run(graph, fake_lf, mods, console=None)

        call_kwargs = fake_lf.configure.call_args[1]
        assert "console" not in call_kwargs

    def test_additional_span_processors_forwarded(self):
        graph = _make_graph()
        fake_lf = _fake_logfire_module()
        mods = _fake_otel_modules()
        fake_proc = MagicMock()

        self._run(graph, fake_lf, mods, additional_span_processors=[fake_proc])

        call_kwargs = fake_lf.configure.call_args[1]
        assert call_kwargs["additional_span_processors"] == [fake_proc]

    def test_empty_additional_processors_not_forwarded(self):
        graph = _make_graph()
        fake_lf = _fake_logfire_module()
        mods = _fake_otel_modules()

        self._run(graph, fake_lf, mods, additional_span_processors=[])

        call_kwargs = fake_lf.configure.call_args[1]
        assert "additional_span_processors" not in call_kwargs

    def test_extra_configure_kwargs_forwarded(self):
        graph = _make_graph()
        fake_lf = _fake_logfire_module()
        mods = _fake_otel_modules()

        self._run(graph, fake_lf, mods, environment="production")

        call_kwargs = fake_lf.configure.call_args[1]
        assert call_kwargs["environment"] == "production"

    def test_send_to_logfire_false(self):
        graph = _make_graph()
        fake_lf = _fake_logfire_module()
        mods = _fake_otel_modules()

        self._run(graph, fake_lf, mods, send_to_logfire=False)

        call_kwargs = fake_lf.configure.call_args[1]
        assert call_kwargs["send_to_logfire"] is False

    def test_raises_when_logfire_missing(self):
        graph = _make_graph()
        # Remove logfire entirely
        saved = sys.modules.pop("logfire", ...)
        try:
            sys.modules.pop("logfire", None)
            from agentflow.runtime.publisher.exporters import setup_logfire

            with pytest.raises(ImportError, match="logfire"):
                setup_logfire(graph)
        finally:
            if saved is not ...:
                sys.modules["logfire"] = saved

    def test_level_passed_to_setup_tracing(self):
        from agentflow.runtime.publisher.otel_publisher import ObservabilityLevel

        graph = _make_graph()
        fake_lf = _fake_logfire_module()
        mods = _fake_otel_modules()
        all_mods = {**mods, "logfire": fake_lf}

        with patch.dict(sys.modules, all_mods):
            with patch(
                "agentflow.runtime.publisher.exporters.setup_tracing"
            ) as mock_setup:
                from agentflow.runtime.publisher.exporters import setup_logfire

                setup_logfire(graph, level=ObservabilityLevel.FULL)
                _, kwargs = mock_setup.call_args
                assert kwargs["level"] == ObservabilityLevel.FULL


# ── setup_langsmith tests ──────────────────────────────────────────────────────


class TestSetupLangsmith:
    """Tests for setup_langsmith()."""

    def _run(self, graph, mods, env=None, **kwargs):
        with patch.dict(sys.modules, mods):
            env = env or {}
            with patch.dict("os.environ", env, clear=False):
                with patch(
                    "agentflow.runtime.publisher.exporters.setup_tracing"
                ) as mock_setup:
                    from agentflow.runtime.publisher.exporters import setup_langsmith

                    setup_langsmith(graph, **kwargs)
                    return (
                        mock_setup,
                        mods["_fake_OTLPSpanExporter"],
                        mods["_fake_BatchSpanProcessor"],
                        mods["_fake_TracerProvider"],
                        mods["_fake_trace_mod"],
                    )

    def test_env_var_key_used(self):
        graph = _make_graph()
        mods = _fake_otel_modules()

        mock_setup, Exporter, Processor, Provider, trace_mod = self._run(
            graph, mods, env={"LANGSMITH_API_KEY": "env-key"}
        )

        Exporter.assert_called_once()
        init_kwargs = Exporter.call_args[1]
        assert init_kwargs["headers"]["x-api-key"] == "env-key"
        mock_setup.assert_called_once()

    def test_explicit_api_key_overrides_env(self):
        graph = _make_graph()
        mods = _fake_otel_modules()

        _, Exporter, *_ = self._run(
            graph, mods, env={"LANGSMITH_API_KEY": "env-key"}, api_key="explicit-key"
        )

        init_kwargs = Exporter.call_args[1]
        assert init_kwargs["headers"]["x-api-key"] == "explicit-key"

    def test_project_header_added(self):
        graph = _make_graph()
        mods = _fake_otel_modules()

        _, Exporter, *_ = self._run(
            graph, mods, api_key="key", project="my-project"
        )

        headers = Exporter.call_args[1]["headers"]
        assert headers["Langsmith-Project"] == "my-project"

    def test_no_project_header_when_absent(self):
        graph = _make_graph()
        mods = _fake_otel_modules()

        _, Exporter, *_ = self._run(graph, mods, api_key="key")

        headers = Exporter.call_args[1]["headers"]
        assert "Langsmith-Project" not in headers

    def test_default_endpoint_gets_traces_suffix(self):
        graph = _make_graph()
        mods = _fake_otel_modules()

        _, Exporter, *_ = self._run(graph, mods, api_key="key")

        endpoint = Exporter.call_args[1]["endpoint"]
        assert endpoint == "https://api.smith.langchain.com/otel/v1/traces"

    def test_custom_endpoint_gets_traces_suffix(self):
        graph = _make_graph()
        mods = _fake_otel_modules()

        _, Exporter, *_ = self._run(
            graph, mods, api_key="key", endpoint="https://eu.api.smith.langchain.com/otel"
        )

        endpoint = Exporter.call_args[1]["endpoint"]
        assert endpoint == "https://eu.api.smith.langchain.com/otel/v1/traces"

    def test_trailing_slash_stripped_before_appending(self):
        graph = _make_graph()
        mods = _fake_otel_modules()

        _, Exporter, *_ = self._run(
            graph, mods, api_key="key", endpoint="https://api.smith.langchain.com/otel/"
        )

        endpoint = Exporter.call_args[1]["endpoint"]
        assert endpoint == "https://api.smith.langchain.com/otel/v1/traces"

    def test_new_provider_created_and_set_global_when_none(self):
        graph = _make_graph()
        mods = _fake_otel_modules()

        _, _, Processor, Provider, trace_mod = self._run(
            graph, mods, api_key="key"
        )

        Provider.assert_called_once()
        provider_instance = Provider.return_value
        provider_instance.add_span_processor.assert_called_once()
        trace_mod.set_tracer_provider.assert_called_once_with(provider_instance)

    def test_existing_provider_reused(self):
        graph = _make_graph()
        mods = _fake_otel_modules()
        existing_provider = MagicMock()

        _, _, Processor, Provider, trace_mod = self._run(
            graph, mods, api_key="key", tracer_provider=existing_provider
        )

        # No new provider should be created
        Provider.assert_not_called()
        trace_mod.set_tracer_provider.assert_not_called()
        existing_provider.add_span_processor.assert_called_once()

    def test_raises_value_error_when_no_key(self):
        graph = _make_graph()
        mods = _fake_otel_modules()

        with patch.dict(sys.modules, mods):
            # Ensure env var is absent
            import os

            env_backup = os.environ.pop("LANGSMITH_API_KEY", None)
            try:
                from agentflow.runtime.publisher.exporters import setup_langsmith

                with pytest.raises(ValueError, match="LANGSMITH_API_KEY"):
                    setup_langsmith(graph)
            finally:
                if env_backup is not None:
                    os.environ["LANGSMITH_API_KEY"] = env_backup

    def test_raises_import_error_when_sdk_missing(self):
        graph = _make_graph()
        saved = sys.modules.pop("opentelemetry.sdk.trace", ...)
        try:
            sys.modules["opentelemetry.sdk.trace"] = None  # type: ignore[assignment]
            from agentflow.runtime.publisher.exporters import setup_langsmith

            with pytest.raises(ImportError, match="opentelemetry-sdk"):
                setup_langsmith(graph, api_key="key")
        finally:
            if saved is not ...:
                sys.modules["opentelemetry.sdk.trace"] = saved
            else:
                sys.modules.pop("opentelemetry.sdk.trace", None)


# ── setup_observability tests ──────────────────────────────────────────────────


class TestSetupObservability:
    """Tests for setup_observability()."""

    def _mock_setup_logfire(self):
        return patch("agentflow.runtime.publisher.exporters.setup_logfire")

    def _mock_setup_langsmith(self):
        return patch("agentflow.runtime.publisher.exporters.setup_langsmith")

    # ── neither enabled ──

    def test_neither_enabled_returns_early(self):
        graph = _make_graph()
        from agentflow.runtime.publisher.exporters import setup_observability

        with self._mock_setup_logfire() as lf, self._mock_setup_langsmith() as ls:
            setup_observability(graph, {})
            lf.assert_not_called()
            ls.assert_not_called()

    def test_null_sub_configs_treated_as_disabled(self):
        graph = _make_graph()
        from agentflow.runtime.publisher.exporters import setup_observability

        with self._mock_setup_logfire() as lf, self._mock_setup_langsmith() as ls:
            setup_observability(
                graph, {"logfire": None, "langsmith": None}
            )
            lf.assert_not_called()
            ls.assert_not_called()

    # ── logfire only ──

    def test_logfire_only(self):
        graph = _make_graph()
        from agentflow.runtime.publisher.exporters import setup_observability

        with self._mock_setup_logfire() as lf, self._mock_setup_langsmith() as ls:
            setup_observability(
                graph,
                {
                    "logfire": {
                        "enabled": True,
                        "service_name": "svc",
                        "send_to_logfire": False,
                        "console": False,
                    }
                },
            )
            lf.assert_called_once()
            call_kwargs = lf.call_args[1]
            assert call_kwargs["service_name"] == "svc"
            assert call_kwargs["send_to_logfire"] is False
            assert call_kwargs["console"] is False
            ls.assert_not_called()

    def test_logfire_only_default_params(self):
        graph = _make_graph()
        from agentflow.runtime.publisher.exporters import setup_observability

        with self._mock_setup_logfire() as lf, self._mock_setup_langsmith() as ls:
            setup_observability(graph, {"logfire": {"enabled": True}})
            lf.assert_called_once()
            call_kwargs = lf.call_args[1]
            # service_name not provided → None
            assert call_kwargs.get("service_name") is None
            assert call_kwargs["send_to_logfire"] is True
            ls.assert_not_called()

    # ── langsmith only ──

    def test_langsmith_only(self):
        graph = _make_graph()
        from agentflow.runtime.publisher.exporters import setup_observability

        with self._mock_setup_logfire() as lf, self._mock_setup_langsmith() as ls:
            setup_observability(
                graph,
                {
                    "langsmith": {
                        "enabled": True,
                        "project": "proj",
                        "endpoint": "https://custom.endpoint",
                    }
                },
            )
            ls.assert_called_once()
            call_kwargs = ls.call_args[1]
            assert call_kwargs["project"] == "proj"
            assert call_kwargs["endpoint"] == "https://custom.endpoint"
            lf.assert_not_called()

    def test_langsmith_only_null_endpoint_uses_default(self):
        graph = _make_graph()
        from agentflow.runtime.publisher.exporters import setup_observability

        with self._mock_setup_logfire() as lf, self._mock_setup_langsmith() as ls:
            setup_observability(
                graph,
                {"langsmith": {"enabled": True, "endpoint": None}},
            )
            call_kwargs = ls.call_args[1]
            assert call_kwargs["endpoint"] == "https://api.smith.langchain.com/otel"

    # ── both enabled ──

    def test_both_enabled_builds_langsmith_processor_and_passes_to_logfire(self):
        """When both are on the LangSmith processor must be passed to setup_logfire
        via additional_span_processors, not setup_langsmith called separately."""
        graph = _make_graph()
        mods = _fake_otel_modules()
        fake_lf = _fake_logfire_module()
        all_mods = {**mods, "logfire": fake_lf}

        with patch.dict(sys.modules, all_mods):
            with patch.dict("os.environ", {"LANGSMITH_API_KEY": "ls-key"}):
                with patch(
                    "agentflow.runtime.publisher.exporters.setup_logfire"
                ) as mock_lf:
                    from agentflow.runtime.publisher.exporters import setup_observability

                    setup_observability(
                        graph,
                        {
                            "logfire": {"enabled": True, "service_name": "svc"},
                            "langsmith": {"enabled": True, "project": "proj"},
                        },
                    )

                    mock_lf.assert_called_once()
                    call_kwargs = mock_lf.call_args[1]
                    assert "additional_span_processors" in call_kwargs
                    # The list should contain one BatchSpanProcessor instance
                    assert len(call_kwargs["additional_span_processors"]) == 1

    def test_both_enabled_raises_when_langsmith_key_missing(self):
        graph = _make_graph()
        mods = _fake_otel_modules()
        fake_lf = _fake_logfire_module()
        all_mods = {**mods, "logfire": fake_lf}

        import os

        env_backup = os.environ.pop("LANGSMITH_API_KEY", None)
        try:
            with patch.dict(sys.modules, all_mods):
                from agentflow.runtime.publisher.exporters import setup_observability

                with pytest.raises(ValueError, match="LANGSMITH_API_KEY"):
                    setup_observability(
                        graph,
                        {
                            "logfire": {"enabled": True},
                            "langsmith": {"enabled": True},
                        },
                    )
        finally:
            if env_backup is not None:
                os.environ["LANGSMITH_API_KEY"] = env_backup

    def test_both_langsmith_project_header_passed(self):
        graph = _make_graph()
        mods = _fake_otel_modules()
        fake_lf = _fake_logfire_module()
        all_mods = {**mods, "logfire": fake_lf}

        with patch.dict(sys.modules, all_mods):
            with patch.dict("os.environ", {"LANGSMITH_API_KEY": "key"}):
                with patch(
                    "agentflow.runtime.publisher.exporters.setup_logfire"
                ):
                    from agentflow.runtime.publisher.exporters import setup_observability

                    # Should not raise even when project is specified
                    setup_observability(
                        graph,
                        {
                            "logfire": {"enabled": True},
                            "langsmith": {
                                "enabled": True,
                                "project": "my-proj",
                                "endpoint": "https://api.smith.langchain.com/otel",
                            },
                        },
                    )

                    # The exporter was called with Langsmith-Project header
                    headers = mods["_fake_OTLPSpanExporter"].call_args[1]["headers"]
                    assert headers["Langsmith-Project"] == "my-proj"

    # ── level resolution ──

    def test_valid_level_string_resolved(self):
        from agentflow.runtime.publisher.otel_publisher import ObservabilityLevel

        graph = _make_graph()
        from agentflow.runtime.publisher.exporters import setup_observability

        with self._mock_setup_logfire() as lf, self._mock_setup_langsmith():
            setup_observability(
                graph, {"level": "full", "logfire": {"enabled": True}}
            )
            call_kwargs = lf.call_args[1]
            assert call_kwargs["level"] == ObservabilityLevel.FULL

    def test_invalid_level_defaults_to_standard(self):
        from agentflow.runtime.publisher.otel_publisher import ObservabilityLevel

        graph = _make_graph()
        from agentflow.runtime.publisher.exporters import setup_observability

        with self._mock_setup_logfire() as lf, self._mock_setup_langsmith():
            setup_observability(
                graph,
                {"level": "not_a_real_level", "logfire": {"enabled": True}},
            )
            call_kwargs = lf.call_args[1]
            assert call_kwargs["level"] == ObservabilityLevel.STANDARD

    def test_missing_level_defaults_to_standard(self):
        from agentflow.runtime.publisher.otel_publisher import ObservabilityLevel

        graph = _make_graph()
        from agentflow.runtime.publisher.exporters import setup_observability

        with self._mock_setup_logfire() as lf, self._mock_setup_langsmith():
            setup_observability(graph, {"logfire": {"enabled": True}})
            call_kwargs = lf.call_args[1]
            assert call_kwargs["level"] == ObservabilityLevel.STANDARD
