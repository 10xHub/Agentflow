"""Unit tests for agentflow.runtime.publisher.logfire_publisher.LogfirePublisher.

All external dependencies (logfire, opentelemetry) are fully mocked so the
tests run without any optional extras installed.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import pytest


def _fake_logfire_module():
    lf = types.ModuleType("logfire")
    lf.configure = MagicMock()
    return lf


def _fake_otel_modules():
    """Minimal opentelemetry stubs required by OtelPublisher.__init__."""
    otel = types.ModuleType("opentelemetry")
    trace_mod = types.ModuleType("opentelemetry.trace")
    trace_mod.set_tracer_provider = MagicMock()
    otel.trace = trace_mod

    sdk = types.ModuleType("opentelemetry.sdk")
    sdk_trace = types.ModuleType("opentelemetry.sdk.trace")
    sdk_trace.TracerProvider = MagicMock(name="TracerProvider")
    sdk.trace = sdk_trace
    sdk_trace_export = types.ModuleType("opentelemetry.sdk.trace.export")
    sdk_trace_export.BatchSpanProcessor = MagicMock(name="BatchSpanProcessor")
    sdk_trace.export = sdk_trace_export

    exporter_root = types.ModuleType("opentelemetry.exporter")
    exporter_otlp = types.ModuleType("opentelemetry.exporter.otlp")
    exporter_proto = types.ModuleType("opentelemetry.exporter.otlp.proto")
    exporter_http = types.ModuleType("opentelemetry.exporter.otlp.proto.http")
    exporter_trace = types.ModuleType(
        "opentelemetry.exporter.otlp.proto.http.trace_exporter"
    )
    exporter_trace.OTLPSpanExporter = MagicMock(name="OTLPSpanExporter")

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
    }


class TestLogfirePublisher:
    """Tests for LogfirePublisher."""

    def _make_publisher(self, fake_lf, otel_mods, **kwargs):
        all_mods = {**otel_mods, "logfire": fake_lf}
        with patch.dict(sys.modules, all_mods):
            from agentflow.runtime.publisher.logfire_publisher import LogfirePublisher

            return LogfirePublisher(**kwargs), fake_lf

    def test_is_subclass_of_otel_publisher(self):
        fake_lf = _fake_logfire_module()
        otel_mods = _fake_otel_modules()
        all_mods = {**otel_mods, "logfire": fake_lf}

        with patch.dict(sys.modules, all_mods):
            from agentflow.runtime.publisher.logfire_publisher import LogfirePublisher
            from agentflow.runtime.publisher.otel_publisher import OtelPublisher

            assert issubclass(LogfirePublisher, OtelPublisher)

    def test_logfire_configure_called_on_init(self):
        fake_lf = _fake_logfire_module()
        otel_mods = _fake_otel_modules()
        all_mods = {**otel_mods, "logfire": fake_lf}

        with patch.dict(sys.modules, all_mods):
            from agentflow.runtime.publisher.logfire_publisher import LogfirePublisher

            LogfirePublisher()

        fake_lf.configure.assert_called_once()

    def test_default_send_to_logfire_true(self):
        fake_lf = _fake_logfire_module()
        otel_mods = _fake_otel_modules()
        all_mods = {**otel_mods, "logfire": fake_lf}

        with patch.dict(sys.modules, all_mods):
            from agentflow.runtime.publisher.logfire_publisher import LogfirePublisher

            LogfirePublisher()

        call_kwargs = fake_lf.configure.call_args[1]
        assert call_kwargs["send_to_logfire"] is True

    def test_token_forwarded_when_provided(self):
        fake_lf = _fake_logfire_module()
        otel_mods = _fake_otel_modules()
        all_mods = {**otel_mods, "logfire": fake_lf}

        with patch.dict(sys.modules, all_mods):
            from agentflow.runtime.publisher.logfire_publisher import LogfirePublisher

            LogfirePublisher(token="mytoken")

        call_kwargs = fake_lf.configure.call_args[1]
        assert call_kwargs["token"] == "mytoken"

    def test_token_not_forwarded_when_none(self):
        fake_lf = _fake_logfire_module()
        otel_mods = _fake_otel_modules()
        all_mods = {**otel_mods, "logfire": fake_lf}

        with patch.dict(sys.modules, all_mods):
            from agentflow.runtime.publisher.logfire_publisher import LogfirePublisher

            LogfirePublisher(token=None)

        call_kwargs = fake_lf.configure.call_args[1]
        assert "token" not in call_kwargs

    def test_service_name_forwarded(self):
        fake_lf = _fake_logfire_module()
        otel_mods = _fake_otel_modules()
        all_mods = {**otel_mods, "logfire": fake_lf}

        with patch.dict(sys.modules, all_mods):
            from agentflow.runtime.publisher.logfire_publisher import LogfirePublisher

            LogfirePublisher(service_name="my-svc")

        call_kwargs = fake_lf.configure.call_args[1]
        assert call_kwargs["service_name"] == "my-svc"

    def test_service_name_not_forwarded_when_none(self):
        fake_lf = _fake_logfire_module()
        otel_mods = _fake_otel_modules()
        all_mods = {**otel_mods, "logfire": fake_lf}

        with patch.dict(sys.modules, all_mods):
            from agentflow.runtime.publisher.logfire_publisher import LogfirePublisher

            LogfirePublisher(service_name=None)

        call_kwargs = fake_lf.configure.call_args[1]
        assert "service_name" not in call_kwargs

    def test_console_false_forwarded(self):
        fake_lf = _fake_logfire_module()
        otel_mods = _fake_otel_modules()
        all_mods = {**otel_mods, "logfire": fake_lf}

        with patch.dict(sys.modules, all_mods):
            from agentflow.runtime.publisher.logfire_publisher import LogfirePublisher

            LogfirePublisher(console=False)

        call_kwargs = fake_lf.configure.call_args[1]
        assert call_kwargs["console"] is False

    def test_console_none_not_forwarded(self):
        fake_lf = _fake_logfire_module()
        otel_mods = _fake_otel_modules()
        all_mods = {**otel_mods, "logfire": fake_lf}

        with patch.dict(sys.modules, all_mods):
            from agentflow.runtime.publisher.logfire_publisher import LogfirePublisher

            LogfirePublisher(console=None)

        call_kwargs = fake_lf.configure.call_args[1]
        assert "console" not in call_kwargs

    def test_additional_span_processors_forwarded(self):
        fake_lf = _fake_logfire_module()
        otel_mods = _fake_otel_modules()
        all_mods = {**otel_mods, "logfire": fake_lf}
        fake_proc = MagicMock()

        with patch.dict(sys.modules, all_mods):
            from agentflow.runtime.publisher.logfire_publisher import LogfirePublisher

            LogfirePublisher(additional_span_processors=[fake_proc])

        call_kwargs = fake_lf.configure.call_args[1]
        assert call_kwargs["additional_span_processors"] == [fake_proc]

    def test_empty_processors_not_forwarded(self):
        fake_lf = _fake_logfire_module()
        otel_mods = _fake_otel_modules()
        all_mods = {**otel_mods, "logfire": fake_lf}

        with patch.dict(sys.modules, all_mods):
            from agentflow.runtime.publisher.logfire_publisher import LogfirePublisher

            LogfirePublisher(additional_span_processors=[])

        call_kwargs = fake_lf.configure.call_args[1]
        assert "additional_span_processors" not in call_kwargs

    def test_extra_kwargs_forwarded_to_configure(self):
        fake_lf = _fake_logfire_module()
        otel_mods = _fake_otel_modules()
        all_mods = {**otel_mods, "logfire": fake_lf}

        with patch.dict(sys.modules, all_mods):
            from agentflow.runtime.publisher.logfire_publisher import LogfirePublisher

            LogfirePublisher(environment="staging")

        call_kwargs = fake_lf.configure.call_args[1]
        assert call_kwargs["environment"] == "staging"

    def test_level_stored_on_publisher(self):
        from agentflow.runtime.publisher.otel_publisher import ObservabilityLevel

        fake_lf = _fake_logfire_module()
        otel_mods = _fake_otel_modules()
        all_mods = {**otel_mods, "logfire": fake_lf}

        with patch.dict(sys.modules, all_mods):
            from agentflow.runtime.publisher.logfire_publisher import LogfirePublisher

            pub = LogfirePublisher(level=ObservabilityLevel.FULL)

        assert pub._level == ObservabilityLevel.FULL

    def test_default_level_is_standard(self):
        from agentflow.runtime.publisher.otel_publisher import ObservabilityLevel

        fake_lf = _fake_logfire_module()
        otel_mods = _fake_otel_modules()
        all_mods = {**otel_mods, "logfire": fake_lf}

        with patch.dict(sys.modules, all_mods):
            from agentflow.runtime.publisher.logfire_publisher import LogfirePublisher

            pub = LogfirePublisher()

        assert pub._level == ObservabilityLevel.STANDARD

    def test_raises_import_error_when_logfire_missing(self):
        saved = sys.modules.pop("logfire", ...)
        try:
            sys.modules.pop("logfire", None)
            from agentflow.runtime.publisher.logfire_publisher import LogfirePublisher

            with pytest.raises(ImportError, match="logfire"):
                LogfirePublisher()
        finally:
            if saved is not ...:
                sys.modules["logfire"] = saved

    def test_send_to_logfire_false(self):
        fake_lf = _fake_logfire_module()
        otel_mods = _fake_otel_modules()
        all_mods = {**otel_mods, "logfire": fake_lf}

        with patch.dict(sys.modules, all_mods):
            from agentflow.runtime.publisher.logfire_publisher import LogfirePublisher

            LogfirePublisher(send_to_logfire=False)

        call_kwargs = fake_lf.configure.call_args[1]
        assert call_kwargs["send_to_logfire"] is False
