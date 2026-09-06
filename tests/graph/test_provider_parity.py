"""Tests for cross-provider parity: count_tokens and the batch helpers.

The point of these is that the *surface* is the same across providers even
where the underlying APIs differ, so switching providers does not mean
relearning the interface.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from agentflow.core.graph.agent_internal.google import AgentGoogleMixin
from agentflow.core.llm import AnthropicBatch, BatchResult, OpenAIBatch
from agentflow.core.llm.openai_batch import normalise_status


class _GoogleStub(AgentGoogleMixin):
    def __init__(self):
        self.model = "gemini-2.5-flash"
        self.llm_kwargs = {}
        self.client = MagicMock()
        self.client.aio.models.count_tokens = AsyncMock(
            return_value=SimpleNamespace(total_tokens=321)
        )


class TestGoogleCountTokens:
    @pytest.mark.asyncio
    async def test_returns_total_tokens(self):
        agent = _GoogleStub()
        assert await agent._count_tokens_google([{"role": "user", "content": "hi"}]) == 321

    @pytest.mark.asyncio
    async def test_counts_against_converted_contents(self):
        """Uses the same conversion the real request does, not the raw dicts."""
        agent = _GoogleStub()
        await agent._count_tokens_google([{"role": "user", "content": "hi"}])
        kwargs = agent.client.aio.models.count_tokens.call_args.kwargs
        assert kwargs["model"] == "gemini-2.5-flash"
        assert kwargs["contents"]  # converted Content objects, not plain dicts

    @pytest.mark.asyncio
    async def test_missing_total_returns_zero(self):
        agent = _GoogleStub()
        agent.client.aio.models.count_tokens = AsyncMock(return_value=SimpleNamespace())
        assert await agent._count_tokens_google([]) == 0


class TestBatchSurfaceParity:
    """Both helpers expose the same methods and return the same result type."""

    @pytest.mark.parametrize("method", ["add", "submit", "status", "wait", "results"])
    def test_same_methods_on_both(self, method):
        assert hasattr(AnthropicBatch, method)
        assert hasattr(OpenAIBatch, method)

    def test_both_reject_duplicate_custom_ids(self):
        for batch in (
            AnthropicBatch(model="claude-haiku-4-5", client=MagicMock()),
            OpenAIBatch(model="gpt-4o-mini", client=MagicMock()),
        ):
            batch.add("dup", [{"role": "user", "content": "a"}])
            with pytest.raises(ValueError, match="Duplicate custom_id"):
                batch.add("dup", [{"role": "user", "content": "b"}])

    @pytest.mark.asyncio
    async def test_both_reject_an_empty_batch(self):
        for batch in (
            AnthropicBatch(model="claude-haiku-4-5", client=MagicMock()),
            OpenAIBatch(model="gpt-4o-mini", client=MagicMock()),
        ):
            with pytest.raises(ValueError, match="empty batch"):
                await batch.submit()

    def test_both_chain_from_add(self):
        for batch in (
            AnthropicBatch(model="claude-haiku-4-5", client=MagicMock()),
            OpenAIBatch(model="gpt-4o-mini", client=MagicMock()),
        ):
            assert batch.add("a", [{"role": "user", "content": "1"}]) is batch


def _openai_batch():
    return OpenAIBatch(model="gpt-4o-mini", client=MagicMock())


class TestOpenAIBatch:
    def test_request_uses_openai_jsonl_envelope(self):
        batch = _openai_batch()
        batch.add("row-1", [{"role": "user", "content": "q"}])
        entry = batch.requests[0]
        assert entry["method"] == "POST"
        assert entry["url"] == "/v1/chat/completions"
        assert entry["body"]["model"] == "gpt-4o-mini"

    def test_messages_need_no_translation(self):
        """agentflow's internal dialect is already OpenAI-shaped."""
        messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "q"}]
        batch = _openai_batch()
        batch.add("row-1", messages)
        assert batch.requests[0]["body"]["messages"] == messages

    def test_jsonl_is_one_object_per_line(self):
        batch = _openai_batch()
        batch.add("a", [{"role": "user", "content": "1"}])
        batch.add("b", [{"role": "user", "content": "2"}])
        lines = batch._to_jsonl().decode().splitlines()
        assert len(lines) == 2
        assert [json.loads(line)["custom_id"] for line in lines] == ["a", "b"]

    @pytest.mark.asyncio
    async def test_submit_uploads_then_creates(self):
        batch = _openai_batch()
        batch.add("row-1", [{"role": "user", "content": "q"}])
        batch.client.files.create = AsyncMock(return_value=SimpleNamespace(id="file_1"))
        batch.client.batches.create = AsyncMock(return_value=SimpleNamespace(id="batch_1"))

        assert await batch.submit() == "batch_1"
        assert batch.client.batches.create.call_args.kwargs["input_file_id"] == "file_1"

    @pytest.mark.asyncio
    async def test_results_are_keyed_by_custom_id_not_position(self):
        batch = _openai_batch()

        def line(custom_id, text):
            return json.dumps(
                {
                    "custom_id": custom_id,
                    "response": {
                        "status_code": 200,
                        "body": {
                            "choices": [
                                {"message": {"content": text}, "finish_reason": "stop"}
                            ],
                            "usage": {"prompt_tokens": 5, "completion_tokens": 2},
                        },
                    },
                }
            )

        batch.client.batches.retrieve = AsyncMock(
            return_value=SimpleNamespace(status="completed", output_file_id="out_1")
        )
        # Deliberately out of submission order.
        batch.client.files.content = AsyncMock(
            return_value=SimpleNamespace(text=f"{line('row-2', 'second')}\n{line('row-1', 'first')}")
        )

        results = await batch.results("batch_1")
        assert results["row-1"].text == "first"
        assert results["row-2"].text == "second"
        assert results["row-1"].input_tokens == 5
        assert results["row-1"].ok

    @pytest.mark.asyncio
    async def test_per_entry_error_is_captured_not_dropped(self):
        batch = _openai_batch()
        batch.client.batches.retrieve = AsyncMock(
            return_value=SimpleNamespace(status="completed", output_file_id="out_1")
        )
        batch.client.files.content = AsyncMock(
            return_value=SimpleNamespace(
                text=json.dumps(
                    {"custom_id": "row-1", "error": {"message": "bad request"}, "response": None}
                )
            )
        )

        results = await batch.results("batch_1")
        assert results["row-1"].status == "errored"
        assert results["row-1"].ok is False

    @pytest.mark.asyncio
    async def test_http_error_status_marks_the_entry_failed(self):
        batch = _openai_batch()
        batch.client.batches.retrieve = AsyncMock(
            return_value=SimpleNamespace(status="completed", output_file_id="out_1")
        )
        batch.client.files.content = AsyncMock(
            return_value=SimpleNamespace(
                text=json.dumps(
                    {
                        "custom_id": "row-1",
                        "response": {"status_code": 429, "body": {"error": "rate limited"}},
                    }
                )
            )
        )

        results = await batch.results("batch_1")
        assert results["row-1"].status == "errored"

    @pytest.mark.asyncio
    async def test_no_output_file_returns_empty_rather_than_raising(self):
        """A failed or expired batch has no output file to download."""
        batch = _openai_batch()
        batch.client.batches.retrieve = AsyncMock(
            return_value=SimpleNamespace(status="failed", output_file_id=None)
        )
        assert await batch.results("batch_1") == {}

    @pytest.mark.asyncio
    async def test_wait_stops_on_every_terminal_status(self):
        """Not just 'completed' — a failed batch must not poll forever."""
        batch = _openai_batch()
        batch.client.batches.retrieve = AsyncMock(
            return_value=SimpleNamespace(status="failed", output_file_id=None)
        )
        assert await batch.wait("batch_1", poll_interval=0.01, timeout=1) == {}

    @pytest.mark.asyncio
    async def test_wait_times_out_while_in_progress(self):
        batch = _openai_batch()
        batch.client.batches.retrieve = AsyncMock(
            return_value=SimpleNamespace(status="in_progress")
        )
        with pytest.raises(TimeoutError, match="in_progress"):
            await batch.wait("batch_1", poll_interval=0.01, timeout=0.02)

    @pytest.mark.parametrize(
        ("openai_status", "expected"),
        [("cancelled", "canceled"), ("failed", "errored"), ("completed", "completed")],
    )
    def test_status_vocabulary_is_normalised(self, openai_status, expected):
        assert normalise_status(openai_status) == expected


class TestSharedBatchResult:
    def test_both_helpers_return_the_same_type(self):
        assert BatchResult("a", "succeeded").ok is True
        assert BatchResult("a", "errored").ok is False
        assert BatchResult("a", "expired").ok is False
