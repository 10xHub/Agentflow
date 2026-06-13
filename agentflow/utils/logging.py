"""
Logging utilities for Agentflow.

This module provides logging support for the Agentflow library following Python
logging best practices for library code.

By default, Agentflow uses a NullHandler to prevent "No handlers could be found"
warnings. Users can configure logging by getting the logger and adding their own
handlers.

Library Usage (within agentflow modules):
    Each module should create its own logger:

    >>> import logging
    >>> logger = logging.getLogger(__name__)
    >>> logger.info("This is an info message")

User Configuration Example:
    Users of the Agentflow library can configure logging like this::

        import logging

        # Configure the agentflow logger
        logger = logging.getLogger("agentflow")
        logger.setLevel(logging.DEBUG)

        # Add a handler
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s"))
        logger.addHandler(handler)

Best Practices:
    - Library code should NEVER configure the root logger
    - Library code should NEVER add handlers except NullHandler
    - Library code should use module-level loggers (logging.getLogger(__name__))
    - Users control logging configuration in their applications

References:
    https://docs.python.org/3/howto/logging.html#configuring-logging-for-a-library
"""

import logging
import re
from collections.abc import Callable


# Create the main agentflow logger
logger = logging.getLogger("agentflow")

# Add NullHandler by default to prevent "No handlers found" warnings
# Users can configure their own handlers as needed
logger.addHandler(logging.NullHandler())


# ── Secret redaction ─────────────────────────────────────────────────────────
#
# Best-effort masking of credentials that may otherwise surface in debug logs
# (e.g. signed URLs with query-string tokens, Authorization headers, provider
# API keys). This is defence-in-depth, not a guarantee: prefer never logging
# secrets in the first place.

_REDACTED = "***REDACTED***"

_Replacement = str | Callable[[re.Match[str]], str]

# (pattern, replacement) pairs. Replacement is either the placeholder string
# (full match redacted) or a callable that preserves the key name and redacts
# only the value. ``Bearer`` is handled before the generic key=value rule so an
# Authorization header keeps its scheme instead of being double-redacted.
_SECRET_SUBS: list[tuple[re.Pattern[str], _Replacement]] = [
    # OpenAI-style secret keys: sk-... and sk-proj-...
    (re.compile(r"sk-(?:proj-)?[A-Za-z0-9_-]{16,}"), _REDACTED),
    # Google API keys
    (re.compile(r"AIza[0-9A-Za-z_-]{35}"), _REDACTED),
    # GitHub tokens (ghp_, gho_, ghu_, ghs_, ghr_)
    (re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"), _REDACTED),
    # Slack tokens
    (re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}"), _REDACTED),
    # AWS access key id
    (re.compile(r"AKIA[0-9A-Z]{16}"), _REDACTED),
    # Bearer tokens (e.g. in Authorization headers) — keep the scheme
    (re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._\-]+"), "Bearer " + _REDACTED),
    # key/secret/token/password = value  (JSON or key=value form)
    (
        re.compile(
            r"(?i)(api[_-]?key|access[_-]?token|secret|password)"
            r"""(["']?\s*[:=]\s*["']?)"""
            r"""([^\s"',&}]{4,})"""
        ),
        lambda m: f"{m.group(1)}{m.group(2)}{_REDACTED}",
    ),
    # Signed-URL credential query params (?token=…, &sig=…, &X-Amz-Signature=…)
    (
        re.compile(
            r"(?i)([?&](?:token|sig|signature|x-amz-signature|"
            r"x-goog-signature|key|password)=)([^&\s]+)"
        ),
        lambda m: f"{m.group(1)}{_REDACTED}",
    ),
]


def mask_secrets(text: str) -> str:
    """Redact common credential formats from a string.

    Masks OpenAI/Google/GitHub/Slack/AWS keys, ``Bearer`` tokens,
    ``key=value`` secrets, and signed-URL credential query parameters. Returns
    the input unchanged when it contains nothing that looks like a secret.

    This is a heuristic. It will not catch every possible secret and may
    occasionally over-redact; treat it as a safety net, not a guarantee.
    """
    if not text:
        return text
    for pattern, repl in _SECRET_SUBS:
        text = pattern.sub(repl, text)
    return text


class SecretRedactionFilter(logging.Filter):
    """Logging filter that redacts secrets from a record's formatted message.

    Add it to a *handler* to cover every logger that propagates to that handler::

        handler.addFilter(SecretRedactionFilter())

    Adding it to a logger only redacts records emitted directly on that logger,
    not its children (Python applies logger-level filters only at the originating
    logger, while handler-level filters run for propagated records too).
    """

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:  # pragma: no cover - never block logging on redaction
            return True
        redacted = mask_secrets(message)
        if redacted != message:
            record.msg = redacted
            record.args = ()
        return True


def install_secret_redaction(logger_name: str = "agentflow") -> SecretRedactionFilter:
    """Attach a :class:`SecretRedactionFilter` to ``logger_name`` and its handlers.

    Call this *after* configuring your logging handlers so the filter covers the
    records they emit. For complete coverage of child loggers, prefer adding the
    filter to your handler(s) directly. Returns the installed filter.
    """
    target = logging.getLogger(logger_name)
    redactor = SecretRedactionFilter()
    target.addFilter(redactor)
    for handler in target.handlers:
        handler.addFilter(redactor)
    return redactor


__all__ = [
    "SecretRedactionFilter",
    "install_secret_redaction",
    "logger",
    "mask_secrets",
]
