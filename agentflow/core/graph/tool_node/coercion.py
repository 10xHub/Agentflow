"""Coercion of raw LLM tool arguments into the types a tool function declares.

A provider hands back plain JSON, so a parameter annotated with a pydantic model, a
dataclass, or an enum arrives as a ``dict`` / ``str``. Passing that through untouched
means the tool body receives a ``dict`` where it expects a model instance, or the raw
string ``"active"`` where it expects ``Status.ACTIVE``. Both fail silently.

Validation goes through pydantic in both cases. Notably, a dataclass must **not** be
built with ``Cls(**value)``: a dataclass constructor performs no validation and no
conversion, so enum fields stay strings, nested dataclass fields stay dicts, and int
fields keep whatever the model sent, all without raising. ``TypeAdapter`` handles the
whole tree correctly and reports a precise error when the payload is wrong.

The same applies to the scalar stdlib types. JSON has no date or UUID, so ``schema.py``
advertises ``datetime`` as ``{"type": "string", "format": "date-time"}`` and the model
can only answer with a string. Leaving it as one means a body that does ``when.year``
raises ``AttributeError`` on a ``str``, so every type ``schema.py`` lists in ``_SCALARS``
must be coerced back here. That list is imported rather than restated: a type advertised
as a formatted string in one module and unknown to the other is exactly the asymmetry
this module exists to prevent.

Coercion is applied only when the annotation actually contains a structured or scalar
type. Plain primitives (``int``, ``str``, ``float``, ``bool``), ``dict``, and
``list[str]`` are passed through unchanged so existing tools see no behaviour change.
"""

from __future__ import annotations

import dataclasses
import enum
import inspect
import json
import typing as t
from functools import lru_cache

from pydantic import BaseModel, TypeAdapter, ValidationError

from .schema import _SCALARS


_EMPTY = inspect._empty
_MAX_DEPTH = 8

# Types that reach the tool as a JSON string (or number) and must be parsed back into
# the annotated type. Sourced from the schema builder so the two cannot drift.
_SCALAR_TYPES: tuple[type, ...] = tuple(k for k in _SCALARS if isinstance(k, type))


def _is_structured(annotation: t.Any) -> bool:
    """Return True for the types that need pydantic validation to be usable."""
    if not isinstance(annotation, type):
        return False
    return (
        issubclass(annotation, BaseModel)
        or issubclass(annotation, enum.Enum)
        or dataclasses.is_dataclass(annotation)
        or annotation in _SCALAR_TYPES
    )


def _needs_coercion(annotation: t.Any, depth: int = 0) -> bool:
    """Return True when the annotation contains a structured type anywhere inside it."""
    if depth > _MAX_DEPTH:
        return False
    if _is_structured(annotation):
        return True
    return any(_needs_coercion(arg, depth + 1) for arg in t.get_args(annotation))


@lru_cache(maxsize=512)
def _cached_needs_coercion(annotation: t.Any) -> bool:
    return _needs_coercion(annotation)


@lru_cache(maxsize=512)
def _adapter(annotation: t.Any) -> TypeAdapter:
    """Build and cache a TypeAdapter. Construction is expensive; this runs per call."""
    return TypeAdapter(annotation)


def _maybe_json(value: t.Any) -> t.Any:
    """Decode a JSON object/array string.

    Weaker models, and any model whose prompt history was built against the old
    ``{"type": "string"}`` schema, hand-serialize nested objects into a single string.
    Only strings that clearly look like a JSON object or array are decoded, so an
    enum member such as ``"5"`` is never turned into an int.
    """
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if stripped[:1] not in ("{", "["):
        return value
    try:
        return json.loads(stripped)
    except ValueError:
        return value


def coerce_tool_argument(
    value: t.Any,
    annotation: t.Any,
    *,
    tool_name: str,
    param_name: str,
) -> t.Any:
    """Coerce one raw tool argument into its declared type.

    Args:
        value: The raw argument value as provided by the model.
        annotation: The resolved annotation of the target parameter.
        tool_name: Tool name, used only for error messages.
        param_name: Parameter name, used only for error messages.

    Returns:
        The coerced value, or the original value when no coercion applies.

    Raises:
        TypeError: If the value does not satisfy the annotation. The message carries
            the pydantic field errors so it can be handed back to the model as a
            retryable tool error.
    """
    if annotation is _EMPTY or annotation is t.Any or annotation is None:
        return value

    try:
        needs = _cached_needs_coercion(annotation)
    except TypeError:
        # Unhashable annotation; fall back to the uncached walk.
        needs = _needs_coercion(annotation)

    if not needs:
        return value

    candidate = _maybe_json(value)

    try:
        return _adapter(annotation).validate_python(candidate)
    except TypeError:
        # Unhashable annotation could not be cached; build the adapter directly.
        return TypeAdapter(annotation).validate_python(candidate)
    except ValidationError as exc:
        details = "; ".join(
            f"{'.'.join(str(p) for p in err['loc']) or param_name}: {err['msg']}"
            for err in exc.errors()
        )
        raise TypeError(
            f"Invalid argument {param_name!r} for tool {tool_name!r}: {details}"
        ) from exc
