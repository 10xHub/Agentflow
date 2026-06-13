"""Tests for secret redaction in logging utilities."""

import logging

import pytest

from agentflow.utils.logging import (
    SecretRedactionFilter,
    install_secret_redaction,
    mask_secrets,
)


class TestMaskSecrets:
    """Unit tests for the mask_secrets() pure function."""

    @pytest.mark.parametrize(
        "secret",
        [
            "sk-abcdefghijklmnopqrstuvwxyz0123",
            "sk-proj-abcdEFGH1234ijklMNOP5678qrst",
            "AIzaSyA1234567890abcdefghijklmnopqrstuvw",
            "ghp_0123456789abcdefghijklmnopqrstuvwx",
            "xoxb-1234567890-abcdEFGHijkl",
            "AKIAIOSFODNN7EXAMPLE",
        ],
    )
    def test_known_key_formats_are_redacted(self, secret):
        out = mask_secrets(f"using credential {secret} now")
        assert secret not in out
        assert "***REDACTED***" in out

    def test_bearer_token_redacted_keeps_scheme(self):
        out = mask_secrets("Authorization: Bearer abc.def.ghi-123")
        assert "abc.def.ghi-123" not in out
        assert "Bearer ***REDACTED***" in out

    def test_key_value_redacts_value_keeps_key(self):
        out = mask_secrets('{"api_key": "super-secret-value-123"}')
        assert "super-secret-value-123" not in out
        assert "api_key" in out
        assert "***REDACTED***" in out

    def test_signed_url_query_token_redacted(self):
        url = "https://storage.example.com/blob/abc?X-Amz-Signature=deadbeefcafe&expires=99"
        out = mask_secrets(url)
        assert "deadbeefcafe" not in out
        assert "***REDACTED***" in out
        # Non-secret query params are preserved
        assert "expires=99" in out

    def test_plain_text_is_unchanged(self):
        text = "Switching to fallback model gemini-2.5-flash (provider=google)"
        assert mask_secrets(text) == text

    def test_empty_string(self):
        assert mask_secrets("") == ""


class TestSecretRedactionFilter:
    """The logging.Filter integration."""

    def test_filter_redacts_record_message(self):
        record = logging.LogRecord(
            name="agentflow.test",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="key sk-abcdefghijklmnopqrstuvwxyz0123 leaked",
            args=(),
            exc_info=None,
        )
        assert SecretRedactionFilter().filter(record) is True
        assert "sk-abcdefghijklmnopqrstuvwxyz0123" not in record.getMessage()
        assert "***REDACTED***" in record.getMessage()

    def test_filter_redacts_args_interpolated_message(self):
        record = logging.LogRecord(
            name="agentflow.test",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="token=%s",
            args=("Bearer secret-token-value",),
            exc_info=None,
        )
        SecretRedactionFilter().filter(record)
        assert "secret-token-value" not in record.getMessage()

    def test_install_attaches_filter_to_handlers(self):
        log = logging.getLogger("agentflow.test.install")
        handler = logging.StreamHandler()
        log.addHandler(handler)
        try:
            redactor = install_secret_redaction("agentflow.test.install")
            assert redactor in log.filters
            assert redactor in handler.filters
        finally:
            log.removeHandler(handler)
            log.filters.clear()
