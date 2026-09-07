"""OpenAI Message Batches helper.

Same surface as :class:`~agentflow.core.llm.anthropic_batch.AnthropicBatch`, so
switching providers does not mean relearning the interface:

    from agentflow.core.llm import OpenAIBatch

    batch = OpenAIBatch(model="gpt-4o-mini")
    batch.add("row-1", [{"role": "user", "content": "Summarise: ..."}])
    batch.add("row-2", [{"role": "user", "content": "Summarise: ..."}])

    batch_id = await batch.submit()
    results = await batch.wait(batch_id)      # keyed by custom_id
    print(results["row-1"].text)

The mechanics underneath differ from Anthropic's: OpenAI takes a JSONL **file**
of requests rather than an inline list, so ``submit()`` uploads one first and
``results()`` downloads and parses the output file. That is deliberately hidden.

Results are keyed by ``custom_id`` throughout, because batch results arrive in
any order.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
from dataclasses import dataclass, field
from typing import Any

from agentflow.core.llm.batch_common import BatchResult
from agentflow.core.llm.client_factory import create_llm_client


logger = logging.getLogger("agentflow.llm.openai_batch")

# OpenAI batch statuses that mean "no further progress will happen".
_TERMINAL_STATUSES = frozenset({"completed", "failed", "expired", "cancelled"})

# Maps OpenAI's terminal batch statuses onto the shared result vocabulary.
_STATUS_ALIASES = {"cancelled": "canceled", "failed": "errored"}

DEFAULT_COMPLETION_WINDOW = "24h"
CHAT_COMPLETIONS_ENDPOINT = "/v1/chat/completions"

# Per-request HTTP status at or above which the entry counts as failed.
_HTTP_ERROR_FLOOR = 400


@dataclass
class OpenAIBatch:
    """Build, submit, and collect an OpenAI batch."""

    model: str
    completion_window: str = DEFAULT_COMPLETION_WINDOW
    endpoint: str = CHAT_COMPLETIONS_ENDPOINT
    client: Any = None
    requests: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.client is None:
            self.client = create_llm_client("openai")

    def add(
        self,
        custom_id: str,
        messages: list[dict[str, Any]],
        *,
        tools: list[Any] | None = None,
        **params: Any,
    ) -> OpenAIBatch:
        """Queue one request. ``custom_id`` is how its result is found later.

        Messages are already in agentflow's internal (OpenAI-shaped) dialect, so
        unlike the Anthropic helper no translation is needed here.
        """
        if any(entry["custom_id"] == custom_id for entry in self.requests):
            raise ValueError(f"Duplicate custom_id in batch: {custom_id!r}")

        body: dict[str, Any] = {"model": self.model, "messages": messages, **params}
        if tools:
            body["tools"] = tools

        self.requests.append(
            {
                "custom_id": custom_id,
                "method": "POST",
                "url": self.endpoint,
                "body": body,
            }
        )
        return self

    def _to_jsonl(self) -> bytes:
        """Render the queued requests as the JSONL payload OpenAI expects."""
        return "\n".join(json.dumps(entry) for entry in self.requests).encode("utf-8")

    async def submit(self) -> str:
        """Upload the request file, create the batch, and return its id."""
        if not self.requests:
            raise ValueError("Cannot submit an empty batch; call add() first.")

        upload = io.BytesIO(self._to_jsonl())
        upload.name = "batch_requests.jsonl"

        uploaded = await self.client.files.create(file=upload, purpose="batch")
        batch = await self.client.batches.create(
            input_file_id=getattr(uploaded, "id", ""),
            endpoint=self.endpoint,
            completion_window=self.completion_window,
        )
        batch_id = getattr(batch, "id", "")
        logger.info("Submitted OpenAI batch %s with %d requests", batch_id, len(self.requests))
        return batch_id

    async def status(self, batch_id: str) -> str:
        """Return the batch's current status."""
        batch = await self.client.batches.retrieve(batch_id)
        return getattr(batch, "status", "")

    async def wait(
        self,
        batch_id: str,
        *,
        poll_interval: float = 30.0,
        timeout: float | None = None,
    ) -> dict[str, BatchResult]:
        """Poll until the batch reaches a terminal status, then return results.

        Args:
            batch_id: The id returned by :meth:`submit`.
            poll_interval: Seconds between polls. Batches are not latency
                sensitive; polling hard just burns rate limit.
            timeout: Give up after this many seconds. ``None`` waits forever.

        Raises:
            TimeoutError: If *timeout* elapses before the batch finishes.
        """
        waited = 0.0
        while True:
            status = await self.status(batch_id)
            if status in _TERMINAL_STATUSES:
                break

            if timeout is not None and waited >= timeout:
                raise TimeoutError(f"Batch {batch_id} still '{status}' after {waited:.0f}s.")

            await asyncio.sleep(poll_interval)
            waited += poll_interval

        return await self.results(batch_id)

    async def results(self, batch_id: str) -> dict[str, BatchResult]:
        """Download and parse a finished batch's output, keyed by ``custom_id``."""
        batch = await self.client.batches.retrieve(batch_id)
        output_file_id = getattr(batch, "output_file_id", None)

        if not output_file_id:
            status = getattr(batch, "status", "unknown")
            logger.warning(
                "OpenAI batch %s has no output file (status=%s); nothing to collect.",
                batch_id,
                status,
            )
            return {}

        content = await self.client.files.content(output_file_id)
        raw = getattr(content, "text", None)
        if raw is None:
            body = await content.aread() if hasattr(content, "aread") else content.read()
            raw = body.decode("utf-8") if isinstance(body, bytes) else str(body)

        collected: dict[str, BatchResult] = {}
        for line in raw.splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            result = _parse_line(entry)
            collected[result.custom_id] = result

        logger.info("Collected %d results for batch %s", len(collected), batch_id)
        return collected


def _parse_line(entry: dict[str, Any]) -> BatchResult:
    """Turn one output-file JSONL line into a :class:`BatchResult`."""
    custom_id = entry.get("custom_id", "")
    error = entry.get("error")
    response = entry.get("response") or {}
    status_code = response.get("status_code")

    if error or (status_code is not None and status_code >= _HTTP_ERROR_FLOOR):
        return BatchResult(
            custom_id=custom_id,
            status="errored",
            error=error or response.get("body"),
            raw=entry,
        )

    body = response.get("body") or {}
    choices = body.get("choices") or []
    message = choices[0].get("message", {}) if choices else {}
    usage = body.get("usage") or {}

    return BatchResult(
        custom_id=custom_id,
        status="succeeded",
        text=message.get("content") or "",
        stop_reason=choices[0].get("finish_reason") if choices else None,
        input_tokens=usage.get("prompt_tokens", 0) or 0,
        output_tokens=usage.get("completion_tokens", 0) or 0,
        raw=entry,
    )


def normalise_status(status: str) -> str:
    """Map an OpenAI terminal status onto the shared result vocabulary."""
    return _STATUS_ALIASES.get(status, status)
