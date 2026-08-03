"""Shared helper functions and constants for tool node executors."""

from __future__ import annotations

import datetime as dt
import decimal
import enum
import json
import pathlib
import typing as t
import uuid


_STATUS_OK: set[str] = {"completed", "success", "ok", "done", "true", "1"}
_STATUS_FAIL: set[str] = {"failed", "failure", "error", "false", "0"}
_ERROR_TRUE: set[str] = {"true", "1", "yes", "error", "failed", "failure"}


def _stable_members(obj: set | frozenset) -> list:
    """Order a set so the same result serializes identically on every run.

    Set iteration order is not stable across processes, which would make tool output
    flap between runs and tests flaky.
    """
    try:
        return sorted(obj)
    except TypeError:
        # Mixed or uncomparable members: order is unavoidably arbitrary here, but
        # losing the values entirely would be worse.
        return list(obj)


# Rendering rules for the scalar types a tool can return but JSON cannot hold. Ordered:
# the first matching entry wins. Decimal renders as str, not float, so a money value
# does not lose precision on the way to the model.
_JSON_ENCODERS: tuple[tuple[t.Any, t.Callable[[t.Any], t.Any]], ...] = (
    ((dt.datetime, dt.date, dt.time), lambda o: o.isoformat()),
    ((uuid.UUID, pathlib.PurePath), str),
    (decimal.Decimal, str),
    (enum.Enum, lambda o: o.value),
    ((set, frozenset), _stable_members),
    ((bytes, bytearray), lambda o: o.decode("utf-8", errors="replace")),
)


def _json_default(obj: t.Any) -> t.Any:
    """Render the scalar types a tool can return but JSON cannot hold.

    Mirrors the scalars ``schema.py`` accepts on the way in, so a value the model sent
    as ``"2026-01-15T09:30:00"`` comes back in that same textual form rather than as a
    Python repr.

    Raises:
        TypeError: For anything else, so ``json.dumps`` propagates and the caller falls
            through to its existing repr fallback. Unknown objects must not be silently
            stringified here, or a genuinely unserializable payload would look clean.
    """
    for types_, encode in _JSON_ENCODERS:
        if isinstance(obj, types_):
            return encode(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _safe_serialize(obj: t.Any) -> dict[str, t.Any]:
    try:
        json.dumps(obj)
        return obj if isinstance(obj, dict) else {"content": obj}
    except (TypeError, OverflowError):
        if hasattr(obj, "model_dump"):
            dumped = obj.model_dump()  # type: ignore
            if isinstance(dumped, dict) and dumped.get("type") == "resource":
                resource = dumped.get("resource", {})
                if isinstance(resource, dict) and "uri" in resource:
                    resource["uri"] = str(resource["uri"])
                    dumped["resource"] = resource
            return dumped

        # Retry with the scalar renderer so a container keeps its shape and only the
        # offending leaves become text. Without this a single datetime collapses the
        # whole return value into one repr string.
        try:
            normalized = json.loads(json.dumps(obj, default=_json_default))
        except (TypeError, OverflowError, ValueError):
            return {"content": str(obj), "type": "fallback"}
        return normalized if isinstance(normalized, dict) else {"content": normalized}


def _as_bool(val: t.Any, truthy_set: set[str]) -> bool:
    if isinstance(val, bool):
        return val
    return str(val).lower() in truthy_set


def _extract_block_meta(
    data: dict[str, t.Any],
) -> tuple[bool, dict[str, t.Any]]:
    """Normalize arbitrary status/error keys; return (is_error, cleaned_data)."""
    data = dict(data)

    raw_status = data.pop("status", None)
    raw_is_error = data.pop("is_error", data.pop("error", None))
    raw_success = data.pop("success", None)

    if raw_is_error is not None:
        is_error = _as_bool(raw_is_error, _ERROR_TRUE)
    elif raw_success is not None:
        is_error = not _as_bool(raw_success, _STATUS_OK)
    else:
        is_error = False

    if raw_status is not None:
        s = str(raw_status).lower()
        if s in _STATUS_FAIL:
            is_error = True

    return is_error, data
