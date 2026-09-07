"""Tests for prompt caching, count_tokens, server tools, and batches."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from agentflow.core.graph.agent_internal.anthropic import AgentAnthropicMixin
from agentflow.core.graph.agent_internal.anthropic_request import (
    apply_cache_control,
    strip_bedrock_prefix,
)
from agentflow.core.graph.agent_internal.anthropic_server_tools import (
    CODE_EXECUTION,
    WEB_FETCH_BASIC,
    WEB_FETCH_DYNAMIC,
    WEB_SEARCH_BASIC,
    WEB_SEARCH_DYNAMIC,
    code_execution_tool,
    is_server_tool,
    web_fetch_tool,
    web_search_tool,
)
from agentflow.core.llm.anthropic_batch import AnthropicBatch, BatchResult


class TestStripBedrockPrefix:
    def test_prefix_removed_for_lookup(self):
        assert strip_bedrock_prefix("anthropic.claude-opus-5") == "claude-opus-5"

    def test_bare_id_unchanged(self):
        assert strip_bedrock_prefix("claude-opus-5") == "claude-opus-5"


class TestCacheControl:
    def test_disabled_by_default(self):
        system = [{"type": "text", "text": "sys"}]
        apply_cache_control(None, system, None)
        assert "cache_control" not in system[0]

    def test_false_is_a_noop(self):
        system = [{"type": "text", "text": "sys"}]
        apply_cache_control(False, system, None)
        assert "cache_control" not in system[0]

    def test_true_marks_last_system_block(self):
        system = [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]
        apply_cache_control(True, system, None)
        assert "cache_control" not in system[0]
        assert system[1]["cache_control"] == {"type": "ephemeral"}

    def test_dict_allows_custom_ttl(self):
        system = [{"type": "text", "text": "sys"}]
        apply_cache_control({"type": "ephemeral", "ttl": "1h"}, system, None)
        assert system[0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}

    def test_tools_get_a_breakpoint_too(self):
        """Tools render before system, so the tool list is part of the prefix."""
        tools = [{"name": "a"}, {"name": "b"}]
        apply_cache_control(True, None, tools)
        assert "cache_control" not in tools[0]
        assert tools[1]["cache_control"] == {"type": "ephemeral"}

    def test_no_system_and_no_tools_is_safe(self):
        apply_cache_control(True, None, None)


class _StubAgent(AgentAnthropicMixin):
    def __init__(self, **overrides):
        self.model = overrides.get("model", "claude-opus-5")
        self.llm_kwargs = overrides.get("llm_kwargs", {})
        self.reasoning_config = overrides.get("reasoning_config")
        self.output_schema = overrides.get("output_schema")
        self.client = MagicMock()
        self.client.messages.create = AsyncMock(
            return_value=SimpleNamespace(usage=None, stop_reason="end_turn")
        )
        self.client.messages.count_tokens = AsyncMock(
            return_value=SimpleNamespace(input_tokens=1234)
        )


class TestCachingInRequest:
    @pytest.mark.asyncio
    async def test_cache_flag_reaches_the_system_block(self):
        agent = _StubAgent(llm_kwargs={"anthropic_cache": True})
        await agent._call_anthropic(
            [{"role": "system", "content": "big prompt"}, {"role": "user", "content": "hi"}]
        )
        system = agent.client.messages.create.call_args.kwargs["system"]
        assert system[-1]["cache_control"] == {"type": "ephemeral"}

    @pytest.mark.asyncio
    async def test_anthropic_cache_is_not_sent_as_a_request_param(self):
        agent = _StubAgent(llm_kwargs={"anthropic_cache": True})
        await agent._call_anthropic([{"role": "user", "content": "hi"}])
        assert "anthropic_cache" not in agent.client.messages.create.call_args.kwargs

    @pytest.mark.asyncio
    async def test_explicit_cache_control_is_not_doubled(self):
        """A caller placing their own breakpoints owns the strategy."""
        agent = _StubAgent(
            llm_kwargs={"anthropic_cache": True, "cache_control": {"type": "ephemeral"}}
        )
        await agent._call_anthropic(
            [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
        )
        system = agent.client.messages.create.call_args.kwargs["system"]
        assert "cache_control" not in system[-1]


class TestCountTokens:
    @pytest.mark.asyncio
    async def test_returns_input_tokens(self):
        agent = _StubAgent()
        assert await agent._count_tokens_anthropic([{"role": "user", "content": "hi"}]) == 1234

    @pytest.mark.asyncio
    async def test_generation_params_are_not_sent(self):
        """count_tokens prices the input; max_tokens et al. are rejected."""
        agent = _StubAgent(llm_kwargs={"max_tokens": 500}, reasoning_config={"effort": "high"})
        await agent._count_tokens_anthropic([{"role": "user", "content": "hi"}])
        kwargs = agent.client.messages.count_tokens.call_args.kwargs
        for rejected in ("max_tokens", "stream", "output_config", "temperature"):
            assert rejected not in kwargs

    @pytest.mark.asyncio
    async def test_counts_against_the_real_payload(self):
        """System prompt and tools are part of the counted input."""
        agent = _StubAgent()
        await agent._count_tokens_anthropic(
            [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}],
            tools=[{"type": "function", "function": {"name": "f", "description": "d"}}],
        )
        kwargs = agent.client.messages.count_tokens.call_args.kwargs
        assert kwargs["system"] == [{"type": "text", "text": "sys"}]
        assert kwargs["tools"][0]["name"] == "f"

    @pytest.mark.asyncio
    async def test_missing_input_tokens_returns_zero(self):
        agent = _StubAgent()
        agent.client.messages.count_tokens = AsyncMock(return_value=SimpleNamespace())
        assert await agent._count_tokens_anthropic([]) == 0


class TestServerTools:
    @pytest.mark.parametrize(
        ("model", "expected"),
        [
            ("claude-opus-5", WEB_SEARCH_DYNAMIC),
            ("claude-sonnet-5", WEB_SEARCH_DYNAMIC),
            ("claude-haiku-4-5", WEB_SEARCH_BASIC),
            ("anthropic.claude-opus-5", WEB_SEARCH_DYNAMIC),
        ],
    )
    def test_web_search_variant_is_model_gated(self, model, expected):
        assert web_search_tool(model)["type"] == expected

    @pytest.mark.parametrize(
        ("model", "expected"),
        [
            ("claude-opus-5", WEB_FETCH_DYNAMIC),
            ("claude-haiku-4-5", WEB_FETCH_BASIC),
        ],
    )
    def test_web_fetch_variant_is_model_gated(self, model, expected):
        assert web_fetch_tool(model)["type"] == expected

    def test_optional_params_only_included_when_set(self):
        tool = web_search_tool("claude-opus-5")
        assert set(tool) == {"type", "name"}

    def test_domain_filters_are_mutually_exclusive(self):
        with pytest.raises(ValueError, match="never both"):
            web_search_tool("claude-opus-5", allowed_domains=["a.test"], blocked_domains=["b.test"])

    def test_web_fetch_domain_filters_are_mutually_exclusive(self):
        with pytest.raises(ValueError, match="never both"):
            web_fetch_tool("claude-opus-5", allowed_domains=["a.test"], blocked_domains=["b.test"])

    def test_code_execution_tool_shape(self):
        assert code_execution_tool() == {"type": CODE_EXECUTION, "name": "code_execution"}

    @pytest.mark.parametrize(
        ("tool", "expected"),
        [
            ({"type": WEB_SEARCH_DYNAMIC, "name": "web_search"}, True),
            ({"type": CODE_EXECUTION, "name": "code_execution"}, True),
            ({"type": "function", "function": {"name": "f"}}, False),
            ("not a dict", False),
        ],
    )
    def test_is_server_tool(self, tool, expected):
        assert is_server_tool(tool) is expected

    @pytest.mark.asyncio
    async def test_server_tools_pass_through_conversion_unmangled(self):
        agent = _StubAgent()
        search = web_search_tool("claude-opus-5", max_uses=3)
        await agent._call_anthropic([{"role": "user", "content": "hi"}], tools=[search])
        assert agent.client.messages.create.call_args.kwargs["tools"] == [search]


class TestBatch:
    def _batch(self):
        return AnthropicBatch(model="claude-haiku-4-5", client=MagicMock())

    def test_add_translates_system_prompt(self):
        batch = self._batch()
        batch.add(
            "row-1",
            [{"role": "system", "content": "sys"}, {"role": "user", "content": "q"}],
        )
        body = batch.requests[0]["params"]
        assert body["system"] == [{"type": "text", "text": "sys"}]
        assert body["messages"] == [{"role": "user", "content": "q"}]

    def test_max_tokens_is_defaulted(self):
        batch = self._batch()
        batch.add("row-1", [{"role": "user", "content": "q"}])
        assert batch.requests[0]["params"]["max_tokens"] == batch.max_tokens

    def test_duplicate_custom_id_is_rejected(self):
        """custom_id is the only way results are matched back."""
        batch = self._batch()
        batch.add("dup", [{"role": "user", "content": "a"}])
        with pytest.raises(ValueError, match="Duplicate custom_id"):
            batch.add("dup", [{"role": "user", "content": "b"}])

    def test_add_is_chainable(self):
        batch = self._batch()
        assert batch.add("a", [{"role": "user", "content": "1"}]) is batch

    @pytest.mark.asyncio
    async def test_submit_rejects_an_empty_batch(self):
        with pytest.raises(ValueError, match="empty batch"):
            await self._batch().submit()

    @pytest.mark.asyncio
    async def test_results_are_keyed_by_custom_id_not_position(self):
        """Batch results come back in any order."""
        batch = self._batch()

        def entry(custom_id, text):
            return SimpleNamespace(
                custom_id=custom_id,
                result=SimpleNamespace(
                    type="succeeded",
                    message=SimpleNamespace(
                        content=[SimpleNamespace(type="text", text=text)],
                        stop_reason="end_turn",
                        usage=SimpleNamespace(input_tokens=5, output_tokens=2),
                    ),
                ),
            )

        async def gen():
            # Deliberately out of submission order.
            for item in (entry("row-2", "second"), entry("row-1", "first")):
                yield item

        batch.client.messages.batches.results = AsyncMock(return_value=gen())

        results = await batch.results("batch_1")
        assert results["row-1"].text == "first"
        assert results["row-2"].text == "second"
        assert results["row-1"].ok

    @pytest.mark.asyncio
    async def test_errored_entry_is_captured_not_dropped(self):
        batch = self._batch()

        async def gen():
            yield SimpleNamespace(
                custom_id="row-1",
                result=SimpleNamespace(type="errored", error={"type": "invalid_request"}),
            )

        batch.client.messages.batches.results = AsyncMock(return_value=gen())

        results = await batch.results("batch_1")
        assert results["row-1"].status == "errored"
        assert results["row-1"].ok is False
        assert results["row-1"].error == {"type": "invalid_request"}

    @pytest.mark.asyncio
    async def test_wait_times_out_when_batch_never_ends(self):
        batch = self._batch()
        batch.client.messages.batches.retrieve = AsyncMock(
            return_value=SimpleNamespace(processing_status="in_progress")
        )
        with pytest.raises(TimeoutError, match="in_progress"):
            await batch.wait("batch_1", poll_interval=0.01, timeout=0.02)

    def test_batch_result_ok_property(self):
        assert BatchResult("a", "succeeded").ok is True
        assert BatchResult("a", "expired").ok is False
