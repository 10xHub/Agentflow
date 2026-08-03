import enum
from datetime import date, datetime, time
from decimal import Decimal
from pathlib import Path
from uuid import UUID

from agentflow.core.graph.tool_node._helpers import (
    _as_bool,
    _extract_block_meta,
    _safe_serialize,
)


class _Priority(str, enum.Enum):
    HIGH = "high"


class _ResourceDump:
    def model_dump(self):
        return {
            "type": "resource",
            "resource": {"uri": Path("/tmp/data.json"), "mime": "application/json"},
        }


class _NoJsonNoDump:
    def __repr__(self):
        return "<non-json>"


class _NonSerializableNoDump:
    def __init__(self):
        self.value = {1, 2, 3}


def test_safe_serialize_handles_dict_and_scalar_values():
    assert _safe_serialize({"a": 1}) == {"a": 1}
    assert _safe_serialize("text") == {"content": "text"}


def test_safe_serialize_uses_model_dump_and_normalizes_resource_uri():
    result = _safe_serialize(_ResourceDump())
    assert result["type"] == "resource"
    assert result["resource"]["uri"] == "/tmp/data.json"


def test_safe_serialize_falls_back_to_string_when_not_serializable():
    result = _safe_serialize(_NonSerializableNoDump())
    assert result["type"] == "fallback"
    assert "content" in result


class TestSafeSerializeScalarFormatting:
    """A tool returning a datetime must reach the model as a readable date.

    JSON cannot hold a datetime, so the whole return value used to collapse into a
    Python repr string: ``{"content": "{'echo': datetime.datetime(2026, 1, 15, 9, 30)}"}``.
    The structure must survive, with only the scalar leaves rendered as text.
    """

    def test_datetime_in_dict_keeps_structure_and_is_iso(self):
        out = _safe_serialize({"echo": datetime(2026, 1, 15, 9, 30), "ok": True})
        assert out == {"echo": "2026-01-15T09:30:00", "ok": True}

    def test_bare_datetime_is_wrapped_as_content(self):
        assert _safe_serialize(datetime(2026, 1, 15, 9, 30)) == {
            "content": "2026-01-15T09:30:00"
        }

    def test_all_scalar_types_render_as_text(self):
        out = _safe_serialize(
            {
                "when": datetime(2026, 1, 15, 9, 30),
                "day": date(2026, 1, 15),
                "at": time(9, 30),
                "ref": UUID("12345678-1234-5678-1234-567812345678"),
                "path": Path("/data/x"),
                "amount": Decimal("10.50"),
                "blob": b"hello",
                "tag": _Priority.HIGH,
            }
        )
        assert out == {
            "when": "2026-01-15T09:30:00",
            "day": "2026-01-15",
            "at": "09:30:00",
            "ref": "12345678-1234-5678-1234-567812345678",
            "path": "/data/x",
            "amount": "10.50",
            "blob": "hello",
            "tag": "high",
        }

    def test_nested_containers_are_walked(self):
        out = _safe_serialize({"rows": [{"when": datetime(2026, 1, 15, 9, 30)}]})
        assert out == {"rows": [{"when": "2026-01-15T09:30:00"}]}

    def test_sets_become_sorted_lists(self):
        """Set iteration order is not stable, so unordered output would flap per run."""
        out = _safe_serialize({"tags": {"c", "a", "b"}})
        assert out == {"tags": ["a", "b", "c"]}

    def test_set_of_uncomparable_values_still_serializes(self):
        """Sorting is best-effort; mixed types must not crash the whole result."""
        out = _safe_serialize({"tags": {1, "a"}})
        assert sorted(map(str, out["tags"])) == ["1", "a"]

    def test_plain_json_values_are_untouched(self):
        payload = {"a": 1, "b": "x", "c": [1, 2], "d": None, "e": True}
        assert _safe_serialize(payload) == payload

    def test_unknown_object_inside_a_dict_still_falls_back(self):
        """The formatter must not swallow genuinely unserializable payloads."""
        out = _safe_serialize({"obj": _NoJsonNoDump()})
        assert out["type"] == "fallback"
        assert "<non-json>" in out["content"]


def test_as_bool_handles_native_and_string_values():
    truthy = {"true", "1", "yes"}
    assert _as_bool(True, truthy) is True
    assert _as_bool(False, truthy) is False
    assert _as_bool("YES", truthy) is True
    assert _as_bool("no", truthy) is False


def test_extract_block_meta_prefers_explicit_error_and_strips_meta_keys():
    is_error, cleaned = _extract_block_meta(
        {"is_error": "yes", "status": "ok", "payload": 1, "success": True}
    )
    assert is_error is True
    assert cleaned == {"payload": 1}


def test_extract_block_meta_uses_success_flag_when_error_not_present():
    is_error, cleaned = _extract_block_meta({"success": "false", "x": 1})
    assert is_error is True
    assert cleaned == {"x": 1}


def test_extract_block_meta_marks_failure_status_as_error():
    is_error, cleaned = _extract_block_meta({"status": "failed", "value": "v"})
    assert is_error is True
    assert cleaned == {"value": "v"}


def test_extract_block_meta_defaults_to_non_error_without_status_fields():
    is_error, cleaned = _extract_block_meta({"answer": 42})
    assert is_error is False
    assert cleaned == {"answer": 42}
