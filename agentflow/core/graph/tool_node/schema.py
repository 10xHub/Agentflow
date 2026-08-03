"""Schema utilities and local tool description building for ToolNode.

This module provides the SchemaMixin class which handles automatic schema generation
for local Python functions, converting their type annotations and signatures into
OpenAI-compatible function schemas.

Design constraints
------------------
The generated ``parameters`` dict is forwarded **verbatim** to the provider: Google
receives it as ``FunctionDeclaration(parameters_json_schema=...)`` and OpenAI-style
providers receive it as ``tools[].function.parameters``. Many OpenAI-compatible
endpoints ship weak JSON Schema parsers, so the emitted schema is deliberately
restricted to a portable subset:

* no ``$ref`` / ``$defs`` (nested models are inlined)
* no ``anyOf`` (``Optional[X]`` collapses to ``X``)
* no ``allOf`` / ``title`` / ``discriminator`` / ``const``

That is why nested objects are walked by hand rather than delegating to
``BaseModel.model_json_schema()``, which emits all of the above. The rule is:
**pydantic for validation, hand-rolled for schema.**

Supported parameter annotations
-------------------------------
``str``, ``int``, ``float``, ``bool``, a few scalar stdlib types (``datetime``,
``date``, ``time``, ``UUID``, ``Path``, ``Decimal``, ``bytes``), ``Optional[X]``,
``list[X]``, ``dict`` / ``dict[str, X]``, ``Literal[...]``, ``enum.Enum``
subclasses, pydantic ``BaseModel`` subclasses, and dataclasses, plus any nesting
of those.

Anything else raises :class:`UnsupportedToolParameterError` at schema-build time
rather than silently degrading to ``{"type": "string"}``, which is what caused
malformed function-call arguments in the past. Use ``@tool(parameters=...)`` to
supply a hand-written schema when a parameter cannot be expressed here.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import decimal
import enum
import inspect
import logging
import pathlib
import types
import typing as t
import uuid

from pydantic import BaseModel

from .constants import is_injected_param


logger = logging.getLogger("agentflow.graph.tool_node")

_EMPTY = inspect._empty

# Exact-match primitives. Deliberately identity-ish lookups, not issubclass, so that
# `class Status(str, Enum)` is routed to the enum branch instead of being seen as a str.
_PRIMITIVES: dict[t.Any, dict] = {
    str: {"type": "string"},
    int: {"type": "integer"},
    float: {"type": "number"},
    bool: {"type": "boolean"},
}

# Scalar stdlib types a model can only produce as a JSON string. Only the four
# standard JSON Schema formats are emitted; anything exotic is left off.
_SCALARS: dict[t.Any, dict] = {
    dt.datetime: {"type": "string", "format": "date-time"},
    dt.date: {"type": "string", "format": "date"},
    dt.time: {"type": "string", "format": "time"},
    uuid.UUID: {"type": "string", "format": "uuid"},
    pathlib.Path: {"type": "string"},
    pathlib.PurePath: {"type": "string"},
    bytes: {"type": "string"},
    decimal.Decimal: {"type": "number"},
}

# Cut-off for pathological nesting. Cycles are caught by the `seen` chain; this is a
# backstop for deeply generic types.
_MAX_DEPTH = 8

# dict[K, V] has exactly two type args; anything else is treated as an untyped object.
_DICT_ARG_COUNT = 2

_HELP = (
    "Supported: str, int, float, bool, datetime/date/time/UUID/Path/Decimal/bytes, "
    "Optional[X], list[X], dict, dict[str, X], Literal[...], enum.Enum, "
    "pydantic BaseModel, or a dataclass. "
    "To bypass automatic generation entirely, pass an explicit schema: "
    "@tool(parameters={...})."
)


class UnsupportedToolParameterError(TypeError):
    """A tool parameter annotation cannot be expressed as a portable JSON Schema.

    Subclasses :class:`TypeError` so that existing ``except TypeError`` handlers
    around tool registration keep working.
    """


def _safe_type_hints(obj: t.Any) -> dict[str, t.Any]:
    """Resolve PEP 563 string annotations, degrading gracefully.

    ``inspect.signature`` returns raw strings for any module using
    ``from __future__ import annotations``, which previously made every parameter
    (including plain ``int``) fall through to ``{"type": "string"}``.

    ``typing.get_type_hints`` resolves the whole object at once and therefore fails
    outright if *any* annotation is unresolvable, for example a ``TYPE_CHECKING``-only
    import on an injected parameter. In that case each annotation is resolved
    individually so one bad annotation cannot poison the rest.
    """
    try:
        return t.get_type_hints(obj)
    except Exception as exc:
        logger.debug("Bulk annotation resolution failed for %r: %s", obj, exc)

    module = inspect.getmodule(obj)
    globalns = getattr(module, "__dict__", {})
    resolved: dict[str, t.Any] = {}
    for name, raw in getattr(obj, "__annotations__", {}).items():
        if not isinstance(raw, str):
            resolved[name] = raw
            continue
        try:
            resolved[name] = eval(raw, globalns, None)  # noqa: S307 # nosec B307
        except Exception as exc:
            logger.debug("Could not resolve annotation %r for %r: %s", raw, name, exc)
            continue
    return resolved


class SchemaMixin:
    """Mixin providing schema generation and local tool description building.

    This mixin provides functionality to automatically generate JSON Schema definitions
    from Python function signatures. It handles type annotation conversion, parameter
    analysis, and OpenAI-compatible function schema generation for local tools.

    Attributes:
        _funcs: Dictionary mapping function names to callable functions. This
            attribute is expected to be provided by the mixing class.
    """

    _funcs: dict[str, t.Callable]

    # ---------------------------------------------------------------- primitives

    @staticmethod
    def _enum_schema(values: list[t.Any]) -> dict:
        """Build an enum schema, typing it from the member values.

        Shared by ``Literal[...]`` and ``enum.Enum`` so the two stay consistent.
        An ``IntEnum`` must not be advertised as a string.
        """
        if values and all(isinstance(v, str) for v in values):
            return {"type": "string", "enum": list(values)}
        if values and all(isinstance(v, int) and not isinstance(v, bool) for v in values):
            return {"type": "integer", "enum": list(values)}
        return {"enum": list(values)}

    @staticmethod
    def _handle_optional_annotation(annotation: t.Any, default: t.Any) -> dict | None:
        """Handle ``Optional[T]`` annotations by generating schema for ``T``.

        Kept for backwards compatibility with callers that used this directly.

        Args:
            annotation: The type annotation to process, potentially an Optional type.
            default: The default value for the parameter, used for schema generation.

        Returns:
            Schema for the non-None member if the annotation is Optional, else None.
        """
        args = getattr(annotation, "__args__", None)
        if args and any(a is type(None) for a in args):
            non_none = [a for a in args if a is not type(None)]
            if non_none:
                return SchemaMixin._annotation_to_schema(non_none[0], default)
        return None

    @staticmethod
    def _handle_complex_annotation(annotation: t.Any) -> dict:
        """Handle generic annotations (list, Literal, dict).

        Retained for backwards compatibility; delegates to the main resolver.

        Args:
            annotation: The complex type annotation to process.

        Returns:
            The JSON schema for the annotation.

        Raises:
            UnsupportedToolParameterError: If the annotation is not supported.
        """
        return SchemaMixin._schema_for(annotation, (), 0)

    # ------------------------------------------------------------ object walking

    @staticmethod
    def _model_fields(model: type[BaseModel]) -> list[tuple]:
        """Extract ``(key, annotation, required, default, description)`` from a model."""
        fields = []
        for name, info in model.model_fields.items():
            required = info.is_required()
            # A default_factory value must not be called at schema-build time; the
            # field is simply not required and carries no advertised default.
            skip_default = required or info.default_factory is not None
            default = _EMPTY if skip_default else info.default
            fields.append(
                (info.alias or name, info.annotation, required, default, info.description)
            )
        return fields

    @staticmethod
    def _dataclass_fields(owner: type) -> list[tuple]:
        """Extract ``(key, annotation, required, default, description)`` from a dataclass.

        ``dataclasses.fields(...).type`` is a raw string whenever the defining module
        uses ``from __future__ import annotations``, so hints are resolved separately.
        """
        hints = _safe_type_hints(owner)
        fields = []
        for f in dataclasses.fields(owner):
            if not f.init:
                continue
            has_default = f.default is not dataclasses.MISSING
            has_factory = f.default_factory is not dataclasses.MISSING
            fields.append(
                (
                    f.name,
                    hints.get(f.name, f.type),
                    not (has_default or has_factory),
                    f.default if has_default else _EMPTY,
                    None,
                )
            )
        return fields

    @staticmethod
    def _object_schema(owner: type, fields: list[tuple], seen: tuple, depth: int) -> dict:
        """Build an inlined object schema from extracted field descriptors."""
        if owner in seen:
            # Self-referential model. Emit an untyped object at the cut point rather
            # than recursing forever; there is no $ref to fall back on.
            return {"type": "object"}

        seen = (*seen, owner)
        properties: dict[str, dict] = {}
        required: list[str] = []
        for key, annotation, is_required, default, description in fields:
            sub = SchemaMixin._schema_for(annotation, seen, depth + 1)
            if default is not _EMPTY and default is not None:
                sub = {**sub, "default": default}
            if description:
                sub = {**sub, "description": description}
            properties[key] = sub
            if is_required:
                required.append(key)

        schema: dict = {"type": "object", "properties": properties}
        if required:
            schema["required"] = required
        return schema

    # ----------------------------------------------------------- main dispatcher

    @staticmethod
    def _schema_for(annotation: t.Any, seen: tuple, depth: int) -> dict:
        """Resolve a single annotation to a portable JSON Schema fragment.

        Args:
            annotation: The annotation to convert.
            seen: Chain of object types currently being expanded, for cycle detection.
            depth: Current nesting depth, bounded by ``_MAX_DEPTH``.

        Returns:
            A JSON schema fragment containing only portable keywords.

        Raises:
            UnsupportedToolParameterError: If the annotation is not supported.
        """
        if depth > _MAX_DEPTH:
            return {"type": "object"}

        # Unannotated or explicitly unconstrained: no schema constraint to express.
        if annotation is _EMPTY or annotation is t.Any or annotation is None:
            return {"type": "string"}

        # An annotation that is still a string here means resolution failed upstream.
        if isinstance(annotation, str | t.ForwardRef):
            raise UnsupportedToolParameterError(
                f"annotation {annotation!r} is still a string and could not be resolved "
                f"to a real type. Under `from __future__ import annotations` a name is "
                f"only resolvable if it exists at module scope at runtime, so this "
                f"happens when the type is imported under `if TYPE_CHECKING:` or is "
                f"defined inside a function body. Move it to module scope, import it "
                f"normally, or bypass generation with @tool(parameters={{...}})."
            )

        generic = SchemaMixin._schema_for_generic(annotation, seen, depth)
        if generic is not None:
            return generic

        if t.is_typeddict(annotation):
            raise UnsupportedToolParameterError(
                f"TypedDict {getattr(annotation, '__name__', annotation)!r} is not "
                f"supported; use a pydantic BaseModel or a dataclass so arguments can "
                f"also be validated at call time. {_HELP}"
            )

        if isinstance(annotation, type):
            concrete = SchemaMixin._schema_for_class(annotation, seen, depth)
            if concrete is not None:
                return concrete

        raise UnsupportedToolParameterError(f"annotation {annotation!r} is not supported. {_HELP}")

    @staticmethod
    def _schema_for_generic(annotation: t.Any, seen: tuple, depth: int) -> dict | None:
        """Resolve unions and generic containers. Returns None if not one of those.

        Raises:
            UnsupportedToolParameterError: For containers with no portable schema.
        """
        origin = t.get_origin(annotation)
        args = t.get_args(annotation)

        # Optional[X] / X | None collapse to X. A union of several real types has no
        # portable representation, so it is rejected instead of silently picking one.
        if origin is t.Union or origin is types.UnionType:
            non_none = [a for a in args if a is not type(None)]
            if not non_none:
                return {"type": "string"}
            if len(non_none) == 1:
                return SchemaMixin._schema_for(non_none[0], seen, depth)
            raise UnsupportedToolParameterError(
                f"union of multiple types {annotation!r} is not supported; a portable "
                f"schema cannot express it without anyOf. Use a single type, or "
                f"@tool(parameters={{...}})."
            )

        if origin is t.Literal:
            return SchemaMixin._enum_schema(list(args))

        if annotation is list:
            return {"type": "array", "items": {"type": "string"}}
        if origin is list:
            item = args[0] if args else str
            # Note: no default is threaded into `items`; a nested item schema must
            # never carry a `"default": null`.
            return {"type": "array", "items": SchemaMixin._schema_for(item, seen, depth + 1)}

        if annotation is dict:
            return {"type": "object"}
        if origin is dict:
            value_type = args[1] if len(args) == _DICT_ARG_COUNT else t.Any
            if value_type is t.Any:
                return {"type": "object"}
            return {
                "type": "object",
                "additionalProperties": SchemaMixin._schema_for(value_type, seen, depth + 1),
            }

        if origin in (set, frozenset, tuple):
            raise UnsupportedToolParameterError(
                f"{annotation!r} is not supported; use list[...] instead. {_HELP}"
            )

        return None

    @staticmethod
    def _schema_for_class(annotation: type, seen: tuple, depth: int) -> dict | None:
        """Resolve a concrete class. Returns None if the class is not supported."""
        if annotation in _PRIMITIVES:
            return dict(_PRIMITIVES[annotation])
        if annotation in _SCALARS:
            return dict(_SCALARS[annotation])
        if issubclass(annotation, enum.Enum):
            return SchemaMixin._enum_schema([m.value for m in annotation])
        if issubclass(annotation, BaseModel):
            return SchemaMixin._object_schema(
                annotation, SchemaMixin._model_fields(annotation), seen, depth
            )
        if dataclasses.is_dataclass(annotation):
            return SchemaMixin._object_schema(
                annotation, SchemaMixin._dataclass_fields(annotation), seen, depth
            )
        return None

    @staticmethod
    def _annotation_to_schema(annotation: t.Any, default: t.Any) -> dict:
        """Convert a Python type annotation to portable JSON Schema.

        Args:
            annotation: The Python type annotation to convert.
            default: The default value for the parameter. Included in the schema
                unless it is ``inspect._empty``.

        Returns:
            The JSON schema representation of the annotation.

        Raises:
            UnsupportedToolParameterError: If the annotation is not supported.

        Example:
            str -> {"type": "string"}
            list[int] -> {"type": "array", "items": {"type": "integer"}}
            SomeModel -> {"type": "object", "properties": {...}, "required": [...]}
        """
        schema = SchemaMixin._schema_for(annotation, (), 0)
        # A null default is dropped: it tells the model nothing that absence from
        # `required` does not already say, and a `"default": null` on a typed field is
        # rejected by some strict schema parsers.
        if default is not _EMPTY and default is not None:
            schema = {**schema, "default": default}
        return schema

    # -------------------------------------------------------------- tool listing

    def get_local_tool(
        self,
        tags: set[str] | None = None,
    ) -> list[dict]:
        """Generate OpenAI-compatible tool definitions for all registered local functions.

        Inspects all registered functions in _funcs and automatically generates
        tool schemas by analyzing function signatures, type annotations, and docstrings.
        Excludes injectable parameters that are provided by the framework.

        Returns:
            List of tool definitions in OpenAI function calling format.

        Raises:
            UnsupportedToolParameterError: If a tool declares a parameter whose
                annotation cannot be expressed as a portable JSON Schema. Attach an
                explicit schema with ``@tool(parameters={...})`` to bypass generation.

        Example:
            For a function:
            ```python
            def calculate(a: int, b: int, operation: str = "add") -> int:
                '''Perform arithmetic calculation.'''
                return a + b if operation == "add" else a - b
            ```

            Returns:
            ```python
            [
                {
                    "type": "function",
                    "function": {
                        "name": "calculate",
                        "description": "Perform arithmetic calculation.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "a": {"type": "integer"},
                                "b": {"type": "integer"},
                                "operation": {"type": "string", "default": "add"},
                            },
                            "required": ["a", "b"],
                        },
                    },
                }
            ]
            ```

        Note:
            Parameters listed in INJECTABLE_PARAMS (like 'state', 'config',
            'tool_call_id') are automatically excluded from the generated schema
            as they are provided by the framework during execution.
        """
        tools: list[dict] = []
        for name, fn in self._funcs.items():
            # Use decorator metadata if available, otherwise fall back to defaults
            tool_name = getattr(fn, "_py_tool_name", name)
            description = getattr(fn, "_py_tool_description", None)
            if description is None:
                description = inspect.getdoc(fn) or "No description provided."

            fun_tags = getattr(fn, "_py_tool_tags", None)
            capabilities = getattr(fn, "_py_tool_capabilities", None)
            if tags and fun_tags and tags.isdisjoint(fun_tags):
                continue

            override = getattr(fn, "_py_tool_parameters", None)
            if override is not None:
                params_schema = override
            else:
                params_schema = self._build_params_schema(fn, tool_name)

            entry = {
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": description,
                    "parameters": params_schema,
                },
            }
            if capabilities is not None:
                entry["function"]["x-function-capabilities"] = capabilities

            tools.append(entry)

        return tools

    @staticmethod
    def _build_params_schema(fn: t.Callable, tool_name: str) -> dict:
        """Build the ``parameters`` object for one tool function."""
        sig = inspect.signature(fn)
        hints = _safe_type_hints(fn)
        params_schema: dict = {"type": "object", "properties": {}, "required": []}

        for p_name, p in sig.parameters.items():
            if p.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue

            if is_injected_param(p_name, p):
                continue

            annotation = hints.get(p_name, p.annotation)
            if annotation is _EMPTY:
                annotation = str

            try:
                prop = SchemaMixin._annotation_to_schema(annotation, p.default)
            except UnsupportedToolParameterError as exc:
                raise UnsupportedToolParameterError(
                    f"tool {tool_name!r}, parameter {p_name!r}: {exc}"
                ) from exc

            params_schema["properties"][p_name] = prop

            if p.default is _EMPTY:
                params_schema["required"].append(p_name)

        if not params_schema["required"]:
            params_schema.pop("required")

        return params_schema
