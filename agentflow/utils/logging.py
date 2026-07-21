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

import contextlib
import contextvars
import json
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


# ── Run correlation ──────────────────────────────────────────────────────────
#
# Logs carried no run_id or thread_id, so on a busy server you could not pull out
# the lines belonging to ONE run: an operator debugging a single failing thread had
# to eyeball interleaved output from every concurrent execution. These context
# variables are set once when a run starts and are attached to every record emitted
# underneath it -- including from library code that knows nothing about them.
#
# contextvars (not thread-locals) because runs are asyncio tasks: a thread-local
# would bleed between concurrently interleaved runs on the same event loop.

_run_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "agentflow_run_id", default=None
)
_thread_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "agentflow_thread_id", default=None
)
_user_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "agentflow_user_id", default=None
)
_node_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "agentflow_node", default=None
)

_CORRELATION_FIELDS = ("run_id", "thread_id", "user_id", "node")


def get_log_context() -> dict[str, str]:
    """Return the correlation fields currently in scope."""
    values = {
        "run_id": _run_id_var.get(),
        "thread_id": _thread_id_var.get(),
        "user_id": _user_id_var.get(),
        "node": _node_var.get(),
    }
    return {k: v for k, v in values.items() if v is not None}


def set_log_context(
    run_id: str | None = None,
    thread_id: str | None = None,
    user_id: str | None = None,
    node: str | None = None,
) -> None:
    """Bind correlation fields for the current async context.

    Called by the execution loop at the start of a run (and when entering a node),
    so every log line emitted during it can be filtered by run_id/thread_id.
    """
    if run_id is not None:
        _run_id_var.set(str(run_id))
    if thread_id is not None:
        _thread_id_var.set(str(thread_id))
    if user_id is not None:
        _user_id_var.set(str(user_id))
    if node is not None:
        _node_var.set(str(node))


def bind_log_context_from_config(config: dict) -> None:
    """Bind correlation fields from a run config."""
    set_log_context(
        run_id=config.get("run_id"),
        thread_id=config.get("thread_id"),
        user_id=config.get("user_id"),
    )


class CorrelationFilter(logging.Filter):
    """Attach the current run's correlation fields to every log record.

    A filter rather than a formatter, so the fields land on the record itself and
    are available to ANY formatter or handler downstream (including a user's own
    JSON handler, or an APM agent), not just ours.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        context = get_log_context()
        for field in _CORRELATION_FIELDS:
            if not hasattr(record, field):
                setattr(record, field, context.get(field))
        return True


class JsonFormatter(logging.Formatter):
    """Emit one JSON object per line, with the correlation fields promoted.

    Structured output is what makes logs queryable ("show me every ERROR for
    thread X"), which plain text and f-string interpolation cannot support.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        for field in _CORRELATION_FIELDS:
            value = getattr(record, field, None)
            if value is not None:
                payload[field] = value

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        # Anything the caller passed via `extra=`.
        for key, value in record.__dict__.items():
            if key not in _RESERVED_LOG_RECORD_KEYS and key not in payload:
                with contextlib.suppress(TypeError, ValueError):
                    json.dumps(value)  # only include what is serializable
                    payload[key] = value

        return json.dumps(payload, default=str)


# Attributes present on every LogRecord; anything else came from `extra=`.
_RESERVED_LOG_RECORD_KEYS = frozenset(
    {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "module",
        "msecs",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "taskName",
        "thread",
        "threadName",
        *_CORRELATION_FIELDS,
    }
)


def setup_structured_logging(
    level: int = logging.INFO,
    json_format: bool = True,
    redact_secrets: bool = True,
    logger_name: str = "agentflow",
) -> logging.Handler:
    """Configure Agentflow logging for production: correlated and queryable.

    Adds the correlation filter (so every record carries run_id/thread_id), a JSON
    formatter, and secret redaction. Returns the installed handler.

    This is opt-in: library code still never configures logging by itself.
    """
    target = logging.getLogger(logger_name)
    target.setLevel(level)

    handler = logging.StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(
        JsonFormatter()
        if json_format
        else logging.Formatter(
            "[%(asctime)s] %(levelname)s %(name)s "
            "[run=%(run_id)s thread=%(thread_id)s node=%(node)s]: %(message)s"
        )
    )
    handler.addFilter(CorrelationFilter())
    if redact_secrets:
        handler.addFilter(SecretRedactionFilter())

    target.addHandler(handler)
    return handler


__all__ = [
    "CorrelationFilter",
    "JsonFormatter",
    "SecretRedactionFilter",
    "bind_log_context_from_config",
    "get_log_context",
    "install_secret_redaction",
    "logger",
    "mask_secrets",
    "set_log_context",
    "setup_structured_logging",
]
