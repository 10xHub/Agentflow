"""Tests for nested-object tool parameter schemas and argument coercion.

Covers the regression where any structured parameter (pydantic model, dataclass,
dict, enum) was advertised to the model as ``{"type": "string"}``, forcing it to
hand-serialize JSON into a string argument that frequently came back malformed.
"""

import dataclasses
import enum
import json
import typing as t
from datetime import date, datetime, time
from decimal import Decimal
from pathlib import Path
from uuid import UUID

import pytest
from pydantic import BaseModel, Field

from agentflow.core.graph import ToolNode
from agentflow.core.graph.tool_node import UnsupportedToolParameterError
from agentflow.utils import tool


class Priority(str, enum.Enum):
    LOW = "low"
    HIGH = "high"


class Level(enum.IntEnum):
    ONE = 1
    TWO = 2


class Address(BaseModel):
    city: str
    zip_code: str = Field(description="postal code")


class Person(BaseModel):
    name: str
    address: Address
    priority: Priority
    nickname: str | None = None
    tags: list[str] = Field(default_factory=list)


class TreeNode(BaseModel):
    """Self-referential model; inlining must terminate since there is no $ref."""

    value: str
    children: list["TreeNode"] = Field(default_factory=list)


@dataclasses.dataclass
class DcInner:
    note: str


@dataclasses.dataclass
class DcOuter:
    city: str
    priority: Priority
    inner: DcInner
    count: int = 0


def props_for(fn) -> dict:
    """Generated properties for a single tool function."""
    return ToolNode([fn]).get_local_tool()[0]["function"]["parameters"]["properties"]


class TestObjectSchemas:
    def test_pydantic_model_becomes_object_schema(self):
        def save(address: Address) -> str:
            """Save."""
            return ""

        assert props_for(save)["address"] == {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "zip_code": {"type": "string", "description": "postal code"},
            },
            "required": ["city", "zip_code"],
        }

    def test_nested_model_is_inlined_recursively(self):
        def save(person: Person) -> str:
            """Save."""
            return ""

        schema = props_for(save)["person"]
        assert schema["properties"]["address"]["type"] == "object"
        assert schema["properties"]["address"]["properties"]["city"] == {"type": "string"}
        assert schema["properties"]["priority"] == {
            "type": "string",
            "enum": ["low", "high"],
        }
        assert schema["properties"]["tags"] == {"type": "array", "items": {"type": "string"}}
        # Optional and default-factory fields are not required.
        assert schema["required"] == ["name", "address", "priority"]

    def test_dataclass_becomes_object_schema(self):
        def save(outer: DcOuter) -> str:
            """Save."""
            return ""

        schema = props_for(save)["outer"]
        assert schema["properties"]["inner"] == {
            "type": "object",
            "properties": {"note": {"type": "string"}},
            "required": ["note"],
        }
        assert schema["properties"]["count"] == {"type": "integer", "default": 0}
        assert schema["required"] == ["city", "priority", "inner"]

    def test_list_of_models(self):
        def save(people: list[Address]) -> str:
            """Save."""
            return ""

        schema = props_for(save)["people"]
        assert schema["type"] == "array"
        assert schema["items"]["type"] == "object"
        assert "city" in schema["items"]["properties"]

    def test_dict_variants(self):
        def save(meta: dict, typed: dict[str, int], anyval: dict[str, t.Any]) -> str:
            """Save."""
            return ""

        schema = props_for(save)
        assert schema["meta"] == {"type": "object"}
        assert schema["typed"] == {
            "type": "object",
            "additionalProperties": {"type": "integer"},
        }
        assert schema["anyval"] == {"type": "object"}

    def test_enum_typing_follows_member_values(self):
        def save(priority: Priority, level: Level) -> str:
            """Save."""
            return ""

        schema = props_for(save)
        assert schema["priority"] == {"type": "string", "enum": ["low", "high"]}
        assert schema["level"] == {"type": "integer", "enum": [1, 2]}

    def test_scalar_stdlib_types(self):
        def save(when: datetime, ident: UUID) -> str:
            """Save."""
            return ""

        schema = props_for(save)
        assert schema["when"] == {"type": "string", "format": "date-time"}
        assert schema["ident"] == {"type": "string", "format": "uuid"}

    def test_optional_collapses_and_drops_null_default(self):
        def save(note: str | None = None) -> str:
            """Save."""
            return ""

        assert props_for(save)["note"] == {"type": "string"}

    def test_no_unportable_keywords_anywhere(self):
        """No $ref/$defs/anyOf/allOf/title reaches the provider."""

        def save(person: Person, outer: DcOuter) -> str:
            """Save."""
            return ""

        raw = json.dumps(ToolNode([save]).get_local_tool())
        for keyword in ("$defs", "$ref", "$schema", "allOf", "anyOf", "title", "discriminator"):
            assert f'"{keyword}"' not in raw

    def test_self_referential_model_terminates(self):
        def save(node: TreeNode) -> str:
            """Save."""
            return ""

        schema = props_for(save)["node"]
        assert schema["properties"]["value"] == {"type": "string"}
        assert schema["properties"]["children"]["items"] == {"type": "object"}

    def test_stringized_annotations_are_resolved(self):
        """PEP 563 turns annotations into strings; they must still resolve."""

        def save(count: "int", address: "Address") -> str:
            """Save."""
            return ""

        schema = props_for(save)
        assert schema["count"] == {"type": "integer"}
        assert schema["address"]["type"] == "object"


class TestUnsupportedParameters:
    def test_typeddict_is_rejected(self):
        class Plan(t.TypedDict):
            title: str

        def save(plan: Plan) -> str:
            """Save."""
            return ""

        with pytest.raises(UnsupportedToolParameterError, match="TypedDict"):
            ToolNode([save]).get_local_tool()

    def test_multi_type_union_is_rejected(self):
        def save(value: int | str) -> str:
            """Save."""
            return ""

        with pytest.raises(UnsupportedToolParameterError, match="union of multiple types"):
            ToolNode([save]).get_local_tool()

    def test_tuple_is_rejected_with_list_hint(self):
        def save(pair: tuple[int, int]) -> str:
            """Save."""
            return ""

        with pytest.raises(UnsupportedToolParameterError, match=r"use list\[\.\.\.\]"):
            ToolNode([save]).get_local_tool()

    def test_arbitrary_class_is_rejected(self):
        class Widget:
            pass

        def save(widget: Widget) -> str:
            """Save."""
            return ""

        with pytest.raises(UnsupportedToolParameterError, match="not supported"):
            ToolNode([save]).get_local_tool()

    def test_unresolvable_string_annotation_is_rejected(self):
        def save(thing: "NoSuchTypeAnywhere") -> str:  # noqa: F821
            """Save."""
            return ""

        with pytest.raises(UnsupportedToolParameterError, match="still a string"):
            ToolNode([save]).get_local_tool()

    def test_error_names_the_tool_and_parameter(self):
        def save(pair: tuple[int, int]) -> str:
            """Save."""
            return ""

        with pytest.raises(UnsupportedToolParameterError) as exc:
            ToolNode([save]).get_local_tool()
        assert "'save'" in str(exc.value)
        assert "'pair'" in str(exc.value)

    def test_unsupported_error_is_a_type_error(self):
        """Existing `except TypeError` handlers around registration keep working."""
        assert issubclass(UnsupportedToolParameterError, TypeError)


class TestParametersOverride:
    def test_explicit_schema_bypasses_generation(self):
        @tool(parameters={"type": "object", "properties": {"raw": {"type": "string"}}})
        def save(payload: dict[str, t.Any]) -> str:
            """Save."""
            return ""

        params = ToolNode([save]).get_local_tool()[0]["function"]["parameters"]
        assert params == {"type": "object", "properties": {"raw": {"type": "string"}}}

    def test_explicit_schema_rescues_an_unsupported_type(self):
        class Plan(t.TypedDict):
            title: str

        @tool(parameters={"type": "object", "properties": {"plan": {"type": "object"}}})
        def save(plan: Plan) -> str:
            """Save."""
            return ""

        params = ToolNode([save]).get_local_tool()[0]["function"]["parameters"]
        assert params["properties"]["plan"] == {"type": "object"}

    def test_non_dict_override_is_rejected(self):
        with pytest.raises(ValueError, match="must be a JSON Schema dict"):

            @tool(parameters="not a dict")  # type: ignore[arg-type]
            def save(x: int) -> str:
                """Save."""
                return ""


class TestArgumentCoercion:
    def _prepare(self, fn, args: dict) -> dict:
        return ToolNode([fn])._prepare_input_data_tool(fn, fn.__name__, args, {})

    def test_dict_becomes_model_instance(self):
        def save(address: Address) -> str:
            """Save."""
            return ""

        out = self._prepare(save, {"address": {"city": "NYC", "zip_code": "10001"}})
        assert isinstance(out["address"], Address)
        assert out["address"].city == "NYC"

    def test_dataclass_fields_are_fully_converted(self):
        """A dataclass constructor would leave enum/nested/int fields as raw JSON."""

        def save(outer: DcOuter) -> str:
            """Save."""
            return ""

        out = self._prepare(
            save,
            {
                "outer": {
                    "city": "NYC",
                    "priority": "high",
                    "inner": {"note": "x"},
                    "count": "5",
                }
            },
        )
        outer = out["outer"]
        assert isinstance(outer, DcOuter)
        assert outer.priority is Priority.HIGH
        assert isinstance(outer.inner, DcInner)
        assert outer.count == 5

    def test_enum_argument_becomes_enum_member(self):
        def save(priority: Priority) -> str:
            """Save."""
            return ""

        assert self._prepare(save, {"priority": "low"})["priority"] is Priority.LOW

    def test_list_of_models_is_coerced(self):
        def save(people: list[Address]) -> str:
            """Save."""
            return ""

        out = self._prepare(save, {"people": [{"city": "NYC", "zip_code": "1"}]})
        assert all(isinstance(p, Address) for p in out["people"])

    def test_hand_serialized_json_string_still_works(self):
        """Backward compatibility with models trained by the old string schema."""

        def save(address: Address) -> str:
            """Save."""
            return ""

        payload = json.dumps({"city": "NYC", "zip_code": "10001"})
        out = self._prepare(save, {"address": payload})
        assert isinstance(out["address"], Address)

    def test_already_correct_instance_passes_through(self):
        def save(address: Address) -> str:
            """Save."""
            return ""

        original = Address(city="NYC", zip_code="10001")
        assert self._prepare(save, {"address": original})["address"] == original

    def test_primitives_are_untouched(self):
        def save(a: int, b: str, c: dict) -> str:
            """Save."""
            return ""

        out = self._prepare(save, {"a": 1, "b": "x", "c": {"k": "v"}})
        assert out == {"a": 1, "b": "x", "c": {"k": "v"}}

    def test_invalid_payload_raises_a_readable_type_error(self):
        def save(address: Address) -> str:
            """Save."""
            return ""

        with pytest.raises(TypeError) as exc:
            self._prepare(save, {"address": {"city": "NYC"}})
        message = str(exc.value)
        assert "'address'" in message
        assert "'save'" in message
        assert "zip_code" in message

    def test_optional_model_accepts_none(self):
        def save(address: Address | None = None) -> str:
            """Save."""
            return ""

        assert self._prepare(save, {"address": None})["address"] is None


class TestScalarCoercion:
    """A scalar the schema advertises as a formatted string must arrive as its real type.

    JSON has no date/uuid/decimal type, so these go on the wire as strings with a
    ``format`` hint. Whatever the schema promises the model, the tool body must receive
    the annotated type -- otherwise ``when.year`` raises ``AttributeError`` on a ``str``.
    """

    def _prepare(self, fn, args: dict) -> dict:
        return ToolNode([fn])._prepare_input_data_tool(fn, fn.__name__, args, {})

    def test_top_level_datetime_becomes_datetime(self):
        def schedule(when: datetime) -> str:
            """Schedule."""
            return ""

        out = self._prepare(schedule, {"when": "2026-01-15T09:30:00"})
        assert out["when"] == datetime(2026, 1, 15, 9, 30)

    def test_every_advertised_scalar_round_trips(self):
        """Each type in schema._SCALARS must be coercible; the two lists must not drift."""

        def take(
            when: datetime,
            day: date,
            at: time,
            ref: UUID,
            path: Path,
            amount: Decimal,
            blob: bytes,
        ) -> str:
            """Take."""
            return ""

        out = self._prepare(
            take,
            {
                "when": "2026-01-15T09:30:00",
                "day": "2026-01-15",
                "at": "09:30:00",
                "ref": "12345678-1234-5678-1234-567812345678",
                "path": "/data/x",
                "amount": "10.50",
                "blob": "hello",
            },
        )
        assert out["when"] == datetime(2026, 1, 15, 9, 30)
        assert out["day"] == date(2026, 1, 15)
        assert out["at"] == time(9, 30)
        assert out["ref"] == UUID("12345678-1234-5678-1234-567812345678")
        assert out["path"] == Path("/data/x")
        assert out["amount"] == Decimal("10.50")
        assert out["blob"] == b"hello"

    def test_optional_and_list_scalars_are_coerced(self):
        def take(when: datetime | None = None, days: list[date] | None = None) -> str:
            """Take."""
            return ""

        out = self._prepare(take, {"when": "2026-01-15T09:30:00", "days": ["2026-01-15"]})
        assert out["when"] == datetime(2026, 1, 15, 9, 30)
        assert out["days"] == [date(2026, 1, 15)]

    def test_scalar_inside_a_model_still_works(self):
        """The nested path already worked; guard it against regressions."""

        class Booking(BaseModel):
            label: str
            when: datetime

        def save(booking: Booking) -> str:
            """Save."""
            return ""

        out = self._prepare(save, {"booking": {"label": "x", "when": "2026-01-15T09:30:00"}})
        assert out["booking"].when == datetime(2026, 1, 15, 9, 30)

    def test_already_correct_type_passes_through(self):
        def schedule(when: datetime) -> str:
            """Schedule."""
            return ""

        original = datetime(2026, 1, 15, 9, 30)
        assert self._prepare(schedule, {"when": original})["when"] == original

    def test_optional_scalar_accepts_none(self):
        def schedule(when: datetime | None = None) -> str:
            """Schedule."""
            return ""

        assert self._prepare(schedule, {"when": None})["when"] is None

    def test_unparseable_scalar_raises_a_readable_type_error(self):
        def schedule(when: datetime) -> str:
            """Schedule."""
            return ""

        with pytest.raises(TypeError) as exc:
            self._prepare(schedule, {"when": "not a date"})
        message = str(exc.value)
        assert "'when'" in message
        assert "'schedule'" in message

    def test_plain_primitives_remain_untouched(self):
        """Widening coercion to scalars must not start coercing int/float/str/bool."""

        def save(a: int, b: str, c: float, d: bool) -> str:
            """Save."""
            return ""

        out = self._prepare(save, {"a": "5", "b": "x", "c": "1.5", "d": "yes"})
        assert out == {"a": "5", "b": "x", "c": "1.5", "d": "yes"}
