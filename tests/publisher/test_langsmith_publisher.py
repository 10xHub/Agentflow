"""Unit tests for agentflow.runtime.publisher.langsmith_publisher.LangsmithPublisher.

All external dependencies (opentelemetry) are fully mocked so the tests run
without any optional extras installed.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import pytest


def _fake_otel_modules():
    """Minimal opentelemetry stubs required by LangsmithPublisher.__init__."""
    provider_instance = MagicMock(name="provider_instance")

    otel = types.ModuleType("opentelemetry")
    trace_mod = types.ModuleType("opentelemetry.trace")
    trace_mod.set_tracer_provider = MagicMock()
    otel.trace = trace_mod

    sdk = types.ModuleType("opentelemetry.sdk")
    sdk_trace = types.ModuleType("opentelemetry.sdk.trace")
    sdk_trace.TracerProvider = MagicMock(name="TracerProvider", return_value=provider_instance)
    sdk.trace = sdk_trace
    sdk_trace_export = types.ModuleType("opentelemetry.sdk.trace.export")
    sdk_trace_export.BatchSpanProcessor = MagicMock(name="BatchSpanProcessor")
    sdk_trace.export = sdk_trace_export

    exporter_root = types.ModuleType("opentelemetry.exporter")
    exporter_otlp = types.ModuleType("opentelemetry.exporter.otlp")
    exporter_proto = types.ModuleType("opentelemetry.exporter.otlp.proto")
    exporter_http = types.ModuleType("opentelemetry.exporter.otlp.proto.http")
    exporter_trace = types.ModuleType("opentelemetry.exporter.otlp.proto.http.trace_exporter")
    exporter_trace.OTLPSpanExporter = MagicMock(name="OTLPSpanExporter")

    mods = {
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
    }
    return mods, provider_instance, trace_mod, exporter_trace


class TestLangsmithPublisher:
    """Tests for LangsmithPublisher."""

    def test_is_subclass_of_otel_publisher(self):
        mods, *_ = _fake_otel_modules()
        with patch.dict(sys.modules, mods), patch.dict("os.environ", {"LANGSMITH_API_KEY": "k"}):
            from agentflow.runtime.publisher.langsmith_publisher import LangsmithPublisher
            from agentflow.runtime.publisher.otel_publisher import OtelPublisher

            assert issubclass(LangsmithPublisher, OtelPublisher)

    def test_env_var_key_used(self):
        mods, _prov, _trace, exporter_trace = _fake_otel_modules()
        with patch.dict(sys.modules, mods), patch.dict("os.environ", {"LANGSMITH_API_KEY": "env-key"}):
            from agentflow.runtime.publisher.langsmith_publisher import LangsmithPublisher

            LangsmithPublisher()

        init_kwargs = exporter_trace.OTLPSpanExporter.call_args[1]
        assert init_kwargs["headers"]["x-api-key"] == "env-key"

    def test_explicit_api_key_overrides_env(self):
        mods, _prov, _trace, exporter_trace = _fake_otel_modules()
        with patch.dict(sys.modules, mods), patch.dict("os.environ", {"LANGSMITH_API_KEY": "env-key"}):
            from agentflow.runtime.publisher.langsmith_publisher import LangsmithPublisher

            LangsmithPublisher(api_key="explicit-key")

        init_kwargs = exporter_trace.OTLPSpanExporter.call_args[1]
        assert init_kwargs["headers"]["x-api-key"] == "explicit-key"

    def test_project_header_added(self):
        mods, _prov, _trace, exporter_trace = _fake_otel_modules()
        with patch.dict(sys.modules, mods), patch.dict("os.environ", {"LANGSMITH_API_KEY": "k"}):
            from agentflow.runtime.publisher.langsmith_publisher import LangsmithPublisher

            LangsmithPublisher(project="my-project")

        headers = exporter_trace.OTLPSpanExporter.call_args[1]["headers"]
        assert headers["Langsmith-Project"] == "my-project"

    def test_no_project_header_when_absent(self):
        mods, _prov, _trace, exporter_trace = _fake_otel_modules()
        with patch.dict(sys.modules, mods), patch.dict("os.environ", {"LANGSMITH_API_KEY": "k"}):
            from agentflow.runtime.publisher.langsmith_publisher import LangsmithPublisher

            LangsmithPublisher()

        headers = exporter_trace.OTLPSpanExporter.call_args[1]["headers"]
        assert "Langsmith-Project" not in headers

    def test_default_endpoint_gets_traces_suffix(self):
        mods, _prov, _trace, exporter_trace = _fake_otel_modules()
        with patch.dict(sys.modules, mods), patch.dict("os.environ", {"LANGSMITH_API_KEY": "k"}):
            from agentflow.runtime.publisher.langsmith_publisher import LangsmithPublisher

            LangsmithPublisher()

        endpoint = exporter_trace.OTLPSpanExporter.call_args[1]["endpoint"]
        assert endpoint == "https://api.smith.langchain.com/otel/v1/traces"

    def test_custom_endpoint_gets_traces_suffix(self):
        mods, _prov, _trace, exporter_trace = _fake_otel_modules()
        with patch.dict(sys.modules, mods), patch.dict("os.environ", {"LANGSMITH_API_KEY": "k"}):
            from agentflow.runtime.publisher.langsmith_publisher import LangsmithPublisher

            LangsmithPublisher(endpoint="https://eu.api.smith.langchain.com/otel/")

        endpoint = exporter_trace.OTLPSpanExporter.call_args[1]["endpoint"]
        assert endpoint == "https://eu.api.smith.langchain.com/otel/v1/traces"

    def test_new_provider_created_and_set_global_when_none(self):
        mods, provider_instance, trace_mod, _exp = _fake_otel_modules()
        with patch.dict(sys.modules, mods), patch.dict("os.environ", {"LANGSMITH_API_KEY": "k"}):
            from agentflow.runtime.publisher.langsmith_publisher import LangsmithPublisher

            LangsmithPublisher()

        provider_instance.add_span_processor.assert_called_once()
        trace_mod.set_tracer_provider.assert_called_once_with(provider_instance)

    def test_existing_provider_reused(self):
        mods, _prov, trace_mod, _exp = _fake_otel_modules()
        existing_provider = MagicMock(name="existing_provider")
        with patch.dict(sys.modules, mods), patch.dict("os.environ", {"LANGSMITH_API_KEY": "k"}):
            from agentflow.runtime.publisher.langsmith_publisher import LangsmithPublisher

            pub = LangsmithPublisher(tracer_provider=existing_provider)

        existing_provider.add_span_processor.assert_called_once()
        existing_provider.get_tracer.assert_called_once_with("agentflow")
        trace_mod.set_tracer_provider.assert_not_called()
        # The explicitly-bound tracer is used instead of the global one.
        assert pub._tracer_arg is existing_provider.get_tracer.return_value

    def test_level_stored_on_publisher(self):
        from agentflow.runtime.publisher.otel_publisher import ObservabilityLevel

        mods, *_ = _fake_otel_modules()
        with patch.dict(sys.modules, mods), patch.dict("os.environ", {"LANGSMITH_API_KEY": "k"}):
            from agentflow.runtime.publisher.langsmith_publisher import LangsmithPublisher

            pub = LangsmithPublisher(level=ObservabilityLevel.FULL)

        assert pub._level == ObservabilityLevel.FULL

    def test_default_level_is_standard(self):
        from agentflow.runtime.publisher.otel_publisher import ObservabilityLevel

        mods, *_ = _fake_otel_modules()
        with patch.dict(sys.modules, mods), patch.dict("os.environ", {"LANGSMITH_API_KEY": "k"}):
            from agentflow.runtime.publisher.langsmith_publisher import LangsmithPublisher

            pub = LangsmithPublisher()

        assert pub._level == ObservabilityLevel.STANDARD

    def test_raises_value_error_when_no_key(self):
        mods, *_ = _fake_otel_modules()
        with patch.dict(sys.modules, mods), patch.dict("os.environ", {}, clear=True):
            from agentflow.runtime.publisher.langsmith_publisher import LangsmithPublisher

            with pytest.raises(ValueError, match="LANGSMITH_API_KEY"):
                LangsmithPublisher()

    def test_raises_import_error_when_otlp_exporter_missing(self):
        # Simulate opentelemetry-sdk installed but the langsmith extra
        # (opentelemetry-exporter-otlp-proto-http) missing, so the OTLP HTTP
        # guard fires regardless of what is actually installed in the env.
        mods, *_ = _fake_otel_modules()
        mods["opentelemetry.exporter.otlp.proto.http.trace_exporter"] = None
        with patch.dict(sys.modules, mods), patch.dict("os.environ", {"LANGSMITH_API_KEY": "k"}):
            from agentflow.runtime.publisher.langsmith_publisher import LangsmithPublisher

            with pytest.raises(ImportError, match="opentelemetry-exporter-otlp-proto-http"):
                LangsmithPublisher()

    def test_raises_import_error_when_otel_sdk_missing(self):
        # Neither extra installed (the CI default): the SDK guard fires first.
        mods, *_ = _fake_otel_modules()
        mods["opentelemetry.sdk.trace"] = None
        with patch.dict(sys.modules, mods), patch.dict("os.environ", {"LANGSMITH_API_KEY": "k"}):
            from agentflow.runtime.publisher.langsmith_publisher import LangsmithPublisher

            with pytest.raises(ImportError, match="opentelemetry-sdk"):
                LangsmithPublisher()
