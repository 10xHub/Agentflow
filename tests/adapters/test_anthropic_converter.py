"""Tests for the Anthropic response converter.

Responses and stream events are built as lightweight stand-ins rather than real
SDK objects: the converter reads them purely by attribute, so this keeps the
tests independent of the installed SDK version.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agentflow.runtime.adapters.llm.anthropic_converter import AnthropicConverter


def _usage(**overrides):
    base = {
        "input_tokens": 10,
        "output_tokens": 5,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _response(content, *, stop_reason="end_turn", **overrides):
    base = {
        "id": "msg_01",
        "role": "assistant",
        "model": "claude-opus-5",
        "content": content,
        "stop_reason": stop_reason,
        "stop_details": None,
        "usage": _usage(),
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _text_block(text):
    return SimpleNamespace(type="text", text=text)


def _tool_block(block_id, name, payload):
    return SimpleNamespace(type="tool_use", id=block_id, name=name, input=payload)


class TestConvertResponse:
    @pytest.mark.asyncio
    async def test_text_response(self):
        message = await AnthropicConverter().convert_response(
            _response([_text_block("hello")])
        )
        assert message.role == "assistant"
        assert message.content[0].text == "hello"
        assert message.metadata["provider"] == "anthropic"
        assert message.metadata["finish_reason"] == "end_turn"

    @pytest.mark.asyncio
    async def test_usage_is_mapped(self):
        response = _response(
            [_text_block("hi")],
            usage=_usage(
                input_tokens=100,
                output_tokens=20,
                cache_read_input_tokens=80,
                cache_creation_input_tokens=15,
            ),
        )
        message = await AnthropicConverter().convert_response(response)
        assert message.usages.prompt_tokens == 100
        assert message.usages.completion_tokens == 20
        assert message.usages.total_tokens == 120
        assert message.usages.cache_read_input_tokens == 80
        assert message.usages.cache_creation_input_tokens == 15

    @pytest.mark.asyncio
    async def test_missing_usage_does_not_raise(self):
        message = await AnthropicConverter().convert_response(
            _response([_text_block("hi")], usage=None)
        )
        assert message.usages.total_tokens == 0

    @pytest.mark.asyncio
    async def test_tool_use_becomes_tool_call(self):
        response = _response(
            [_tool_block("toolu_1", "get_weather", {"city": "Dhaka"})],
            stop_reason="tool_use",
        )
        message = await AnthropicConverter().convert_response(response)
        assert message.tools_calls[0]["id"] == "toolu_1"
        assert message.tools_calls[0]["function"]["name"] == "get_weather"

    @pytest.mark.asyncio
    async def test_parallel_tool_calls_are_all_captured(self):
        response = _response(
            [
                _text_block("looking those up"),
                _tool_block("toolu_1", "a", {"x": 1}),
                _tool_block("toolu_2", "b", {"y": 2}),
            ],
            stop_reason="tool_use",
        )
        message = await AnthropicConverter().convert_response(response)
        assert len(message.tools_calls) == 2
        assert [c["id"] for c in message.tools_calls] == ["toolu_1", "toolu_2"]

    @pytest.mark.asyncio
    async def test_thinking_block_becomes_reasoning(self):
        response = _response(
            [SimpleNamespace(type="thinking", thinking="step one"), _text_block("answer")]
        )
        message = await AnthropicConverter().convert_response(response)
        assert message.reasoning == "step one"

    @pytest.mark.asyncio
    async def test_refusal_with_empty_content_does_not_raise(self):
        """A refusal is a 200 whose content may be empty; never index blindly."""
        response = _response(
            [],
            stop_reason="refusal",
            stop_details=SimpleNamespace(
                type="refusal", category="cyber", explanation="declined"
            ),
        )
        message = await AnthropicConverter().convert_response(response)
        assert message.metadata["refusal"] is True
        assert message.metadata["refusal_category"] == "cyber"
        assert message.content[0].text == "declined"

    @pytest.mark.asyncio
    async def test_non_refusal_does_not_set_refusal_metadata(self):
        message = await AnthropicConverter().convert_response(
            _response([_text_block("fine")])
        )
        assert "refusal" not in message.metadata


class _FakeStream:
    """Async-iterable stand-in for the SDK stream context manager."""

    def __init__(self, events):
        self._events = events

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return False

    def __aiter__(self):
        async def gen():
            for event in self._events:
                yield event

        return gen()


def _delta(index, dtype, **fields):
    return SimpleNamespace(
        type="content_block_delta",
        index=index,
        delta=SimpleNamespace(type=dtype, **fields),
    )


async def _collect(events, converter=None):
    converter = converter or AnthropicConverter()
    return [
        message
        async for message in converter.convert_streaming_response(
            {"thread_id": "t1"}, "agent", _FakeStream(events)
        )
    ]


class TestStreaming:
    @pytest.mark.asyncio
    async def test_text_deltas_accumulate_into_final_message(self):
        events = [
            SimpleNamespace(
                type="message_start",
                message=SimpleNamespace(id="msg_1", model="claude-opus-5", usage=_usage()),
            ),
            _delta(0, "text_delta", text="Hel"),
            _delta(0, "text_delta", text="lo"),
            SimpleNamespace(type="message_stop"),
        ]
        messages = await _collect(events)
        assert [m.content[0].text for m in messages[:-1]] == ["Hel", "lo"]
        final = messages[-1]
        assert final.delta is False
        assert final.content[0].text == "Hello"

    @pytest.mark.asyncio
    async def test_chunks_are_marked_as_deltas(self):
        events = [_delta(0, "text_delta", text="x")]
        messages = await _collect(events)
        assert messages[0].delta is True
        assert messages[-1].delta is False

    @pytest.mark.asyncio
    async def test_fragmented_tool_json_is_parsed_only_at_block_stop(self):
        """A partial input_json_delta is not valid JSON on its own."""
        events = [
            SimpleNamespace(
                type="content_block_start",
                index=0,
                content_block=SimpleNamespace(type="tool_use", id="toolu_1", name="search"),
            ),
            _delta(0, "input_json_delta", partial_json='{"qu'),
            _delta(0, "input_json_delta", partial_json='ery": "a'),
            _delta(0, "input_json_delta", partial_json='gentflow"}'),
            SimpleNamespace(type="content_block_stop", index=0),
        ]
        messages = await _collect(events)
        final = messages[-1]
        assert len(final.tools_calls) == 1
        assert final.tools_calls[0]["function"]["arguments"] == '{"query": "agentflow"}'
        # No tool call was emitted mid-stream from a partial fragment.
        assert all(m.tools_calls is None for m in messages[:-1])

    @pytest.mark.asyncio
    async def test_parallel_tool_blocks_are_kept_separate_by_index(self):
        events = [
            SimpleNamespace(
                type="content_block_start",
                index=0,
                content_block=SimpleNamespace(type="tool_use", id="t1", name="a"),
            ),
            SimpleNamespace(
                type="content_block_start",
                index=1,
                content_block=SimpleNamespace(type="tool_use", id="t2", name="b"),
            ),
            _delta(0, "input_json_delta", partial_json='{"x":1}'),
            _delta(1, "input_json_delta", partial_json='{"y":2}'),
            SimpleNamespace(type="content_block_stop", index=0),
            SimpleNamespace(type="content_block_stop", index=1),
        ]
        final = (await _collect(events))[-1]
        assert {c["id"] for c in final.tools_calls} == {"t1", "t2"}
        arguments = {c["id"]: c["function"]["arguments"] for c in final.tools_calls}
        assert arguments["t1"] == '{"x":1}'
        assert arguments["t2"] == '{"y":2}'

    @pytest.mark.asyncio
    async def test_malformed_tool_json_degrades_to_empty_object(self):
        events = [
            SimpleNamespace(
                type="content_block_start",
                index=0,
                content_block=SimpleNamespace(type="tool_use", id="t1", name="a"),
            ),
            _delta(0, "input_json_delta", partial_json="not json"),
            SimpleNamespace(type="content_block_stop", index=0),
        ]
        final = (await _collect(events))[-1]
        assert final.tools_calls[0]["function"]["arguments"] == "{}"

    @pytest.mark.asyncio
    async def test_thinking_deltas_accumulate_into_reasoning(self):
        events = [
            _delta(0, "thinking_delta", thinking="first "),
            _delta(0, "thinking_delta", thinking="second"),
        ]
        final = (await _collect(events))[-1]
        assert final.reasoning == "first second"

    @pytest.mark.asyncio
    async def test_message_delta_carries_stop_reason_and_usage(self):
        events = [
            _delta(0, "text_delta", text="hi"),
            SimpleNamespace(
                type="message_delta",
                delta=SimpleNamespace(stop_reason="max_tokens", stop_details=None),
                usage=_usage(input_tokens=7, output_tokens=3),
            ),
        ]
        final = (await _collect(events))[-1]
        assert final.metadata["finish_reason"] == "max_tokens"
        assert final.usages.prompt_tokens == 7

    @pytest.mark.asyncio
    async def test_mid_stream_refusal_surfaces_in_metadata(self):
        events = [
            SimpleNamespace(
                type="message_delta",
                delta=SimpleNamespace(
                    stop_reason="refusal",
                    stop_details=SimpleNamespace(category="bio", explanation="no"),
                ),
                usage=_usage(),
            ),
        ]
        final = (await _collect(events))[-1]
        assert final.metadata["refusal"] is True
        assert final.metadata["refusal_category"] == "bio"

    @pytest.mark.asyncio
    async def test_empty_stream_still_yields_a_final_message(self):
        messages = await _collect([])
        assert len(messages) == 1
        assert messages[0].delta is False


class TestServerToolBlocks:
    @pytest.mark.asyncio
    async def test_server_tool_use_is_not_added_to_tools_calls(self):
        """Anthropic already ran it; the graph must not run it again."""
        response = _response(
            [
                SimpleNamespace(
                    type="server_tool_use", id="srv_1", name="web_search", input={"query": "x"}
                ),
                _text_block("here is what I found"),
            ]
        )
        message = await AnthropicConverter().convert_response(response)
        assert message.tools_calls is None
        assert any(
            getattr(b, "tool_type", None) == "server_tool" for b in message.content
        )

    @pytest.mark.asyncio
    async def test_web_search_results_list_is_captured(self):
        block = SimpleNamespace(
            type="web_search_tool_result",
            tool_use_id="srv_1",
            content=[{"title": "a"}, {"title": "b"}],
        )
        message = await AnthropicConverter().convert_response(_response([block]))
        results = message.metadata["server_tool_results"]
        assert results[0]["results"] == [{"title": "a"}, {"title": "b"}]

    @pytest.mark.asyncio
    async def test_web_search_error_object_is_flagged(self):
        """Server-tool errors arrive as a 200 with an error object, not a raise."""
        block = SimpleNamespace(
            type="web_search_tool_result",
            tool_use_id="srv_1",
            content=SimpleNamespace(error_code="max_uses_exceeded"),
        )
        message = await AnthropicConverter().convert_response(_response([block]))
        assert message.metadata["server_tool_results"][0]["error_code"] == "max_uses_exceeded"

    @pytest.mark.asyncio
    async def test_code_execution_result_is_captured(self):
        block = SimpleNamespace(
            type="bash_code_execution_tool_result",
            tool_use_id="srv_2",
            content=SimpleNamespace(error_code=None),
        )
        message = await AnthropicConverter().convert_response(_response([block]))
        assert message.metadata["server_tool_results"][0]["type"] == (
            "bash_code_execution_tool_result"
        )

    @pytest.mark.asyncio
    async def test_pause_turn_is_flagged_as_resumable(self):
        response = _response([_text_block("partial")], stop_reason="pause_turn")
        message = await AnthropicConverter().convert_response(response)
        assert message.metadata["pause_turn"] is True
        assert message.metadata["finish_reason"] == "pause_turn"

    @pytest.mark.asyncio
    async def test_no_server_tools_leaves_metadata_clean(self):
        message = await AnthropicConverter().convert_response(_response([_text_block("hi")]))
        assert "server_tool_results" not in message.metadata
        assert "pause_turn" not in message.metadata
