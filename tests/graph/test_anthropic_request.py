"""Tests for pure Anthropic request translation.

Everything here runs without a client, a network call, or an AgentState: the
module under test is deliberately side-effect free so the message and tool
mapping (the hard part) can be covered exhaustively.
"""

from __future__ import annotations

import base64

import pytest

from agentflow.core.graph.agent_internal.anthropic_request import (
    convert_assistant_tool_calls,
    convert_content_parts,
    convert_tools,
    drop_trailing_assistant,
    merge_tool_results,
    split_system,
)


class TestSplitSystem:
    def test_no_system_prompt_returns_none(self):
        messages = [{"role": "user", "content": "hi"}]
        system, remainder = split_system(messages)
        assert system is None
        assert remainder == messages

    def test_single_system_prompt_extracted(self):
        messages = [
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "hi"},
        ]
        system, remainder = split_system(messages)
        assert system == [{"type": "text", "text": "be terse"}]
        assert remainder == [{"role": "user", "content": "hi"}]

    def test_many_system_prompts_become_block_list(self):
        """Kept as separate blocks: that list is the cache_control attach point."""
        messages = [
            {"role": "system", "content": "first"},
            {"role": "system", "content": "second"},
            {"role": "user", "content": "hi"},
        ]
        system, remainder = split_system(messages)
        assert system == [
            {"type": "text", "text": "first"},
            {"type": "text", "text": "second"},
        ]
        assert len(remainder) == 1

    def test_structured_system_content(self):
        messages = [
            {"role": "system", "content": [{"type": "text", "text": "block form"}]},
        ]
        system, _ = split_system(messages)
        assert system == [{"type": "text", "text": "block form"}]

    def test_empty_system_content_is_dropped(self):
        messages = [{"role": "system", "content": ""}, {"role": "user", "content": "hi"}]
        system, remainder = split_system(messages)
        assert system is None
        assert len(remainder) == 1


class TestMergeToolResults:
    def test_single_tool_result_becomes_user_message(self):
        messages = [
            {"role": "tool", "tool_call_id": "call_1", "content": "42"},
        ]
        merged = merge_tool_results(messages)
        assert merged == [
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "call_1", "content": "42"}
                ],
            }
        ]

    def test_parallel_tool_results_collapse_into_one_turn(self):
        """The whole point: splitting these degrades parallel tool calling."""
        messages = [
            {"role": "tool", "tool_call_id": "call_1", "content": "a"},
            {"role": "tool", "tool_call_id": "call_2", "content": "b"},
            {"role": "tool", "tool_call_id": "call_3", "content": "c"},
        ]
        merged = merge_tool_results(messages)
        assert len(merged) == 1
        assert merged[0]["role"] == "user"
        assert [b["tool_use_id"] for b in merged[0]["content"]] == [
            "call_1",
            "call_2",
            "call_3",
        ]

    def test_failed_call_marked_is_error(self):
        messages = [
            {"role": "tool", "tool_call_id": "c1", "content": "boom", "is_error": True},
            {"role": "tool", "tool_call_id": "c2", "content": "ok"},
        ]
        merged = merge_tool_results(messages)
        blocks = merged[0]["content"]
        assert blocks[0]["is_error"] is True
        assert "is_error" not in blocks[1]

    def test_interleaved_assistant_turns_split_the_groups(self):
        messages = [
            {"role": "tool", "tool_call_id": "c1", "content": "a"},
            {"role": "assistant", "content": "thinking"},
            {"role": "tool", "tool_call_id": "c2", "content": "b"},
        ]
        merged = merge_tool_results(messages)
        assert [m["role"] for m in merged] == ["user", "assistant", "user"]
        assert merged[0]["content"][0]["tool_use_id"] == "c1"
        assert merged[2]["content"][0]["tool_use_id"] == "c2"

    def test_ordering_is_preserved(self):
        messages = [
            {"role": "user", "content": "q"},
            {"role": "tool", "tool_call_id": "c1", "content": "a"},
        ]
        merged = merge_tool_results(messages)
        assert [m["role"] for m in merged] == ["user", "user"]

    def test_assistant_tool_calls_become_tool_use_blocks(self):
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {"name": "search", "arguments": '{"q": "x"}'},
                    }
                ],
            }
        ]
        merged = merge_tool_results(messages)
        block = merged[0]["content"][0]
        assert block["type"] == "tool_use"
        assert block["id"] == "call_1"
        assert block["name"] == "search"
        assert block["input"] == {"q": "x"}

    def test_empty_input_is_unchanged(self):
        assert merge_tool_results([]) == []


class TestConvertAssistantToolCalls:
    def test_arguments_are_parsed_not_string_matched(self):
        """Claude varies JSON escaping; arguments must go through a real parser."""
        message = {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "c1",
                    "function": {
                        "name": "f",
                        "arguments": '{"path": "a\\/b", "n": 1}',
                    },
                }
            ],
        }
        result = convert_assistant_tool_calls(message)
        assert result["content"][0]["input"] == {"path": "a/b", "n": 1}

    def test_malformed_arguments_degrade_to_empty_dict(self):
        message = {
            "role": "assistant",
            "tool_calls": [{"id": "c1", "function": {"name": "f", "arguments": "not json"}}],
        }
        result = convert_assistant_tool_calls(message)
        assert result["content"][0]["input"] == {}

    def test_text_content_precedes_tool_use_blocks(self):
        message = {
            "role": "assistant",
            "content": "let me check",
            "tool_calls": [{"id": "c1", "function": {"name": "f", "arguments": "{}"}}],
        }
        result = convert_assistant_tool_calls(message)
        assert result["content"][0]["type"] == "text"
        assert result["content"][1]["type"] == "tool_use"

    def test_dict_arguments_pass_through(self):
        message = {
            "role": "assistant",
            "tool_calls": [{"id": "c1", "function": {"name": "f", "arguments": {"a": 1}}}],
        }
        result = convert_assistant_tool_calls(message)
        assert result["content"][0]["input"] == {"a": 1}


class TestConvertTools:
    def test_openai_shape_is_unwrapped(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Look up weather",
                    "parameters": {"type": "object", "properties": {"city": {}}},
                },
            }
        ]
        assert convert_tools(tools) == [
            {
                "name": "get_weather",
                "description": "Look up weather",
                "input_schema": {"type": "object", "properties": {"city": {}}},
            }
        ]

    def test_already_anthropic_shaped_tool_passes_through(self):
        """Server tools must not be mangled."""
        tools = [{"type": "web_search_20260209", "name": "web_search", "max_uses": 3}]
        assert convert_tools(tools) == tools

    def test_strict_is_lifted_to_top_level(self):
        tools = [
            {
                "type": "function",
                "function": {"name": "f", "description": "", "strict": True},
            }
        ]
        assert convert_tools(tools)[0]["strict"] is True

    def test_missing_parameters_gets_empty_object_schema(self):
        tools = [{"type": "function", "function": {"name": "f", "description": "d"}}]
        assert convert_tools(tools)[0]["input_schema"] == {
            "type": "object",
            "properties": {},
        }

    @pytest.mark.parametrize("empty", [None, []])
    def test_empty_tools_returns_none(self, empty):
        assert convert_tools(empty) is None


class TestConvertContentParts:
    def test_plain_string_passes_through(self):
        assert convert_content_parts("hello") == "hello"

    def test_text_part(self):
        assert convert_content_parts([{"type": "text", "text": "hi"}]) == [
            {"type": "text", "text": "hi"}
        ]

    def test_remote_image_url_becomes_url_source(self):
        parts = [{"type": "image_url", "image_url": {"url": "https://x.test/a.png"}}]
        assert convert_content_parts(parts) == [
            {"type": "image", "source": {"type": "url", "url": "https://x.test/a.png"}}
        ]

    def test_data_uri_image_becomes_base64_source(self):
        payload = base64.standard_b64encode(b"png-bytes").decode()
        parts = [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{payload}"},
            }
        ]
        assert convert_content_parts(parts) == [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": payload,
                },
            }
        ]

    def test_document_with_extracted_text_degrades_to_text(self):
        parts = [{"type": "document", "document": {"text": "already extracted"}}]
        assert convert_content_parts(parts) == [
            {"type": "text", "text": "already extracted"}
        ]

    def test_pdf_url_becomes_document_block(self):
        parts = [{"type": "document", "document": {"url": "https://x.test/a.pdf"}}]
        assert convert_content_parts(parts) == [
            {"type": "document", "source": {"type": "url", "url": "https://x.test/a.pdf"}}
        ]

    @pytest.mark.parametrize("ptype", ["input_audio", "video"])
    def test_unsupported_media_is_dropped(self, ptype):
        """Anthropic has no equivalent; degrade rather than invent a failure."""
        parts = [{"type": "text", "text": "keep"}, {"type": ptype, ptype: {"url": "u"}}]
        assert convert_content_parts(parts) == [{"type": "text", "text": "keep"}]

    def test_image_with_no_url_is_dropped(self):
        parts = [{"type": "image_url", "image_url": {}}]
        assert convert_content_parts(parts) == []

    def test_unknown_part_passes_through(self):
        parts = [{"type": "thinking", "thinking": "..."}]
        assert convert_content_parts(parts) == parts


class TestDropTrailingAssistant:
    def test_trailing_assistant_is_dropped(self):
        """Prefill returns a 400 on every current Claude model."""
        messages = [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "summary"},
        ]
        assert drop_trailing_assistant(messages) == [{"role": "user", "content": "q"}]

    def test_trailing_user_is_kept(self):
        messages = [{"role": "user", "content": "q"}]
        assert drop_trailing_assistant(messages) == messages

    def test_empty_list_is_safe(self):
        assert drop_trailing_assistant([]) == []

    def test_only_the_last_assistant_is_dropped(self):
        messages = [
            {"role": "assistant", "content": "earlier"},
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "trailing"},
        ]
        result = drop_trailing_assistant(messages)
        assert len(result) == 2
        assert result[0]["content"] == "earlier"
