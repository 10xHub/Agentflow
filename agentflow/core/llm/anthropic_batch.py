"""Anthropic Message Batches helper.

Batches run asynchronously at reduced cost and are a poor fit for the graph
execution model (a graph run is interactive and stateful; a batch is neither),
so this is a standalone utility rather than an ``Agent`` method.

    from agentflow.core.llm.anthropic_batch import AnthropicBatch

    batch = AnthropicBatch(model="claude-haiku-4-5")
    batch.add("row-1", [{"role": "user", "content": "Summarise: ..."}])
    batch.add("row-2", [{"role": "user", "content": "Summarise: ..."}])

    batch_id = await batch.submit()
    results = await batch.wait(batch_id)      # keyed by custom_id
    print(results["row-1"].text)

Results arrive in **any order**, so they are keyed by ``custom_id`` throughout.
Indexing by position is the classic way to silently mismatch a batch.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

from agentflow.core.llm.client_factory import create_llm_client


# ``core.graph`` imports this package, so these are resolved lazily inside
# ``add()`` rather than at module import time, which would be a cycle.
DEFAULT_MAX_TOKENS = 16000


logger = logging.getLogger("agentflow.llm.anthropic_batch")

_TERMINAL_STATUS = "ended"


@dataclass
class BatchResult:
    """One entry from a completed batch."""

    custom_id: str
    status: str
    """``succeeded``, ``errored``, ``canceled``, or ``expired``."""
    text: str = ""
    stop_reason: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    error: Any = None
    raw: Any = None

    @property
    def ok(self) -> bool:
        return self.status == "succeeded"


@dataclass
class AnthropicBatch:
    """Build, submit, and collect an Anthropic message batch."""

    model: str
    max_tokens: int = DEFAULT_MAX_TOKENS
    anthropic_backend: str | None = None
    client: Any = None
    requests: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.client is None:
            self.client = create_llm_client("anthropic", anthropic_backend=self.anthropic_backend)

    def add(
        self,
        custom_id: str,
        messages: list[dict[str, Any]],
        *,
        tools: list[Any] | None = None,
        **params: Any,
    ) -> AnthropicBatch:
        """Queue one request. ``custom_id`` is how its result is found later.

        Messages use agentflow's internal dialect and go through the same
        translation as a live call, so system prompts and tool results are
        shaped correctly.
        """
        from agentflow.core.graph.agent_internal.anthropic_request import (
            convert_tools,
            merge_tool_results,
            split_system,
        )

        if any(entry["custom_id"] == custom_id for entry in self.requests):
            raise ValueError(f"Duplicate custom_id in batch: {custom_id!r}")

        system, remainder = split_system(messages)
        remainder = merge_tool_results(remainder)

        body: dict[str, Any] = {
            "model": self.model,
            "max_tokens": params.pop("max_tokens", self.max_tokens),
            "messages": remainder,
            **params,
        }
        if system:
            body["system"] = system
        anthropic_tools = convert_tools(tools)
        if anthropic_tools:
            body["tools"] = anthropic_tools

        self.requests.append({"custom_id": custom_id, "params": body})
        return self

    async def submit(self) -> str:
        """Create the batch and return its id."""
        if not self.requests:
            raise ValueError("Cannot submit an empty batch; call add() first.")

        batch = await self.client.messages.batches.create(requests=self.requests)
        batch_id = getattr(batch, "id", "")
        logger.info("Submitted Anthropic batch %s with %d requests", batch_id, len(self.requests))
        return batch_id

    async def status(self, batch_id: str) -> str:
        """Return the batch's current ``processing_status``."""
        batch = await self.client.messages.batches.retrieve(batch_id)
        return getattr(batch, "processing_status", "")

    async def wait(
        self,
        batch_id: str,
        *,
        poll_interval: float = 30.0,
        timeout: float | None = None,
    ) -> dict[str, BatchResult]:
        """Poll until the batch ends, then return results keyed by ``custom_id``.

        Args:
            batch_id: The id returned by :meth:`submit`.
            poll_interval: Seconds between polls. Batches are not latency
                sensitive; polling hard just burns rate limit.
            timeout: Give up after this many seconds. ``None`` waits forever.

        Raises:
            TimeoutError: If *timeout* elapses before the batch ends.
        """
        waited = 0.0
        while True:
            status = await self.status(batch_id)
            if status == _TERMINAL_STATUS:
                break

            if timeout is not None and waited >= timeout:
                raise TimeoutError(f"Batch {batch_id} still '{status}' after {waited:.0f}s.")

            await asyncio.sleep(poll_interval)
            waited += poll_interval

        return await self.results(batch_id)

    async def results(self, batch_id: str) -> dict[str, BatchResult]:
        """Collect a finished batch's results, keyed by ``custom_id``."""
        collected: dict[str, BatchResult] = {}

        async for entry in await self.client.messages.batches.results(batch_id):
            custom_id = getattr(entry, "custom_id", "")
            result = getattr(entry, "result", None)
            status = getattr(result, "type", "errored")

            if status != "succeeded":
                collected[custom_id] = BatchResult(
                    custom_id=custom_id,
                    status=status,
                    error=getattr(result, "error", None),
                    raw=entry,
                )
                continue

            message = getattr(result, "message", None)
            text = "".join(
                getattr(block, "text", "") or ""
                for block in getattr(message, "content", None) or []
                if getattr(block, "type", None) == "text"
            )
            usage = getattr(message, "usage", None)
            collected[custom_id] = BatchResult(
                custom_id=custom_id,
                status=status,
                text=text,
                stop_reason=getattr(message, "stop_reason", None),
                input_tokens=getattr(usage, "input_tokens", 0) or 0,
                output_tokens=getattr(usage, "output_tokens", 0) or 0,
                raw=entry,
            )

        logger.info("Collected %d results for batch %s", len(collected), batch_id)
        return collected
