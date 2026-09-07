"""Convert Anthropic Messages API responses into agentflow messages."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncGenerator
from datetime import datetime
from typing import Any

from agentflow.core.state.message import (
    Message,
    TokenUsages,
    generate_id,
)
from agentflow.core.state.message_block import (
    ReasoningBlock,
    TextBlock,
    ToolCallBlock,
)

from .base_converter import BaseConverter


logger = logging.getLogger("agentflow.adapters.anthropic")

# A refusal is a successful HTTP 200 whose content may be empty or partial.
_REFUSAL = "refusal"

# Result block types the server tools emit. Defined here rather than imported
# from the request-side builders in ``core.graph``: this package is imported
# *by* ``core.graph``, so reaching back into it would be a circular import.
# Response block types are a converter concern anyway.
SERVER_TOOL_RESULT_TYPES = frozenset(
    {
        "web_search_tool_result",
        "web_fetch_tool_result",
        "bash_code_execution_tool_result",
        "code_execution_tool_result",
        "tool_search_tool_result",
    }
)


def _server_result_payload(block: Any) -> dict[str, Any]:
    """Normalise a server-tool result block into a plain dict.

    Web search returns a *list* of results on success but a single error
    *object* on failure, and neither raises: a server-tool error arrives as a
    normal HTTP 200. Branch on that shape before treating it as results.
    """
    content = getattr(block, "content", None)
    payload: dict[str, Any] = {
        "type": getattr(block, "type", ""),
        "tool_use_id": getattr(block, "tool_use_id", ""),
    }

    if isinstance(content, list):
        payload["results"] = [
            item.model_dump() if hasattr(item, "model_dump") else item for item in content
        ]
    elif content is not None:
        payload["result"] = content.model_dump() if hasattr(content, "model_dump") else content
        error_code = getattr(content, "error_code", None)
        if error_code:
            payload["error_code"] = error_code
            logger.warning(
                "Anthropic server tool %s failed: %s",
                payload["type"],
                error_code,
            )

    return payload


class AnthropicConverter(BaseConverter):
    """Convert Anthropic responses and stream events into agentflow messages."""

    async def convert_response(self, response: Any) -> Message:
        """Convert a non-streaming Anthropic Message into an agentflow Message.

        ``stop_reason`` is checked before ``content`` is read: on a refusal the
        request succeeded at the HTTP layer but ``content`` may be empty, so
        indexing into it would raise on an otherwise valid response.
        """
        usages = self._extract_usage(response)
        stop_reason = getattr(response, "stop_reason", None)
        model = getattr(response, "model", "")
        response_id = getattr(response, "id", None)

        metadata: dict[str, Any] = {
            "provider": "anthropic",
            "model": model,
            "finish_reason": stop_reason or "UNKNOWN",
        }

        if stop_reason == _REFUSAL:
            return self._build_refusal_message(response, usages, metadata)

        blocks: list[Any] = []
        tool_calls: list[dict[str, Any]] = []
        server_results: list[dict[str, Any]] = []
        reasoning_text = ""

        for block in getattr(response, "content", None) or []:
            btype = getattr(block, "type", None)

            if btype == "text":
                blocks.append(TextBlock(text=getattr(block, "text", "")))

            elif btype == "thinking":
                thinking = getattr(block, "thinking", "") or ""
                if thinking:
                    reasoning_text += thinking
                    blocks.append(ReasoningBlock(summary=thinking))

            elif btype == "tool_use":
                args = getattr(block, "input", None) or {}
                if not isinstance(args, dict):
                    args = {}
                call_id = getattr(block, "id", "")
                name = getattr(block, "name", "")
                blocks.append(ToolCallBlock(id=call_id, name=name, args=args))
                tool_calls.append(
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": name, "arguments": json.dumps(args)},
                    }
                )

            elif btype == "server_tool_use":
                # Anthropic ran this itself; it is a record of what happened, not
                # a call for agentflow's ToolNode to execute. Never added to
                # tools_calls, or the graph would try to run it again.
                args = getattr(block, "input", None) or {}
                blocks.append(
                    ToolCallBlock(
                        id=getattr(block, "id", ""),
                        name=getattr(block, "name", ""),
                        args=args if isinstance(args, dict) else {},
                        tool_type="server_tool",
                    )
                )

            elif btype in SERVER_TOOL_RESULT_TYPES:
                server_results.append(_server_result_payload(block))

        if server_results:
            metadata["server_tool_results"] = server_results

        if stop_reason == "pause_turn":
            # Server-tool iteration limit. The turn is resumable by re-sending
            # the conversation with this response appended.
            metadata["pause_turn"] = True
            logger.info(
                "Anthropic paused the turn at a server-tool iteration limit; "
                "re-send the conversation including this response to resume."
            )

        logger.debug("Creating message from Anthropic response with id: %s", response_id)

        return Message(
            message_id=generate_id(response_id),
            role=getattr(response, "role", "assistant") or "assistant",
            content=blocks,
            reasoning=reasoning_text or None,
            timestamp=datetime.now().timestamp(),
            metadata=metadata,
            usages=usages,
            raw=response.model_dump() if hasattr(response, "model_dump") else {},
            tools_calls=tool_calls or None,
        )

    @staticmethod
    def _build_refusal_message(
        response: Any,
        usages: TokenUsages,
        metadata: dict[str, Any],
    ) -> Message:
        """Build a message for a policy refusal.

        ``stop_details`` is populated only when ``stop_reason == "refusal"``, so
        it is read only here.
        """
        details = getattr(response, "stop_details", None)
        category = getattr(details, "category", None)
        explanation = getattr(details, "explanation", None)

        metadata["refusal"] = True
        metadata["refusal_category"] = category
        metadata["refusal_explanation"] = explanation

        logger.warning(
            "Anthropic refused the request (category=%s): %s",
            category,
            explanation,
        )

        text = explanation or "The model declined to respond to this request."
        return Message(
            message_id=generate_id(getattr(response, "id", None)),
            role="assistant",
            content=[TextBlock(text=text)],
            timestamp=datetime.now().timestamp(),
            metadata=metadata,
            usages=usages,
            raw=response.model_dump() if hasattr(response, "model_dump") else {},
        )

    @staticmethod
    def _extract_usage(response: Any) -> TokenUsages:
        """Map Anthropic usage onto agentflow's token fields."""
        usage = getattr(response, "usage", None)
        if usage is None:
            return TokenUsages(
                completion_tokens=0,
                prompt_tokens=0,
                total_tokens=0,
                reasoning_tokens=0,
                cache_creation_input_tokens=0,
                cache_read_input_tokens=0,
            )

        prompt = getattr(usage, "input_tokens", 0) or 0
        completion = getattr(usage, "output_tokens", 0) or 0
        return TokenUsages(
            completion_tokens=completion,
            prompt_tokens=prompt,
            total_tokens=prompt + completion,
            reasoning_tokens=0,
            cache_creation_input_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
            cache_read_input_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
        )

    async def convert_streaming_response(  # type: ignore[override]
        self,
        config: dict,
        node_name: str,
        response: Any,
        meta: dict | None = None,
    ) -> AsyncGenerator[Message]:
        """Convert an Anthropic stream into incremental and final Messages."""
        async for message in self._handle_stream(config or {}, node_name or "", response, meta):
            yield message

    async def _handle_stream(  # noqa: PLR0912, PLR0915
        self,
        config: dict,
        node_name: str,
        stream: Any,
        meta: dict | None = None,
    ) -> AsyncGenerator[Message]:
        """Consume the SDK event stream and emit chunk plus final messages.

        ``input_json_delta`` fragments are accumulated per content-block index and
        only parsed at ``content_block_stop``: a partial fragment is not valid
        JSON, so emitting a tool call mid-block would produce garbage arguments.
        """
        accumulated_text = ""
        accumulated_reasoning = ""
        # block index -> {"id", "name", "json"}
        tool_blocks: dict[int, dict[str, Any]] = {}
        tool_calls: list[dict[str, Any]] = []
        stop_reason: str | None = None
        refusal_details: Any = None
        model = ""
        response_id: str | None = None
        usage_obj: Any = None

        metadata: dict[str, Any] = dict(meta or {})
        metadata["provider"] = "anthropic"
        metadata["node_name"] = node_name
        metadata["thread_id"] = config.get("thread_id")

        # ``messages.stream()`` returns a context manager; ``client.messages.create
        # (stream=True)`` returns a bare async iterator. Support both.
        manager = stream
        if hasattr(stream, "__aenter__"):
            stream = await stream.__aenter__()

        try:
            async for event in stream:
                etype = getattr(event, "type", None)

                if etype == "message_start":
                    message_obj = getattr(event, "message", None)
                    response_id = getattr(message_obj, "id", None)
                    model = getattr(message_obj, "model", "") or ""
                    usage_obj = getattr(message_obj, "usage", None)

                elif etype == "content_block_start":
                    block = getattr(event, "content_block", None)
                    if getattr(block, "type", None) == "tool_use":
                        tool_blocks[getattr(event, "index", 0)] = {
                            "id": getattr(block, "id", ""),
                            "name": getattr(block, "name", ""),
                            "json": "",
                        }

                elif etype == "content_block_delta":
                    delta = getattr(event, "delta", None)
                    dtype = getattr(delta, "type", None)

                    if dtype == "text_delta":
                        piece = getattr(delta, "text", "") or ""
                        accumulated_text += piece
                        if piece:
                            yield self._chunk_message(
                                [TextBlock(text=piece)], metadata, response_id
                            )

                    elif dtype == "thinking_delta":
                        piece = getattr(delta, "thinking", "") or ""
                        accumulated_reasoning += piece
                        if piece:
                            yield self._chunk_message(
                                [ReasoningBlock(summary=piece)], metadata, response_id
                            )

                    elif dtype == "input_json_delta":
                        index = getattr(event, "index", 0)
                        if index in tool_blocks:
                            tool_blocks[index]["json"] += getattr(delta, "partial_json", "") or ""

                elif etype == "content_block_stop":
                    index = getattr(event, "index", 0)
                    entry = tool_blocks.pop(index, None)
                    if entry is not None:
                        tool_calls.append(self._finalise_tool_call(entry))

                elif etype == "message_delta":
                    delta = getattr(event, "delta", None)
                    stop_reason = getattr(delta, "stop_reason", None) or stop_reason
                    refusal_details = getattr(delta, "stop_details", None) or refusal_details
                    usage_obj = getattr(event, "usage", None) or usage_obj

        except Exception as exc:
            logger.error("Anthropic stream error: %s", exc)
            raise
        finally:
            if hasattr(manager, "__aexit__"):
                await manager.__aexit__(None, None, None)

        metadata["model"] = model
        metadata["finish_reason"] = stop_reason or "UNKNOWN"

        if stop_reason == _REFUSAL:
            metadata["refusal"] = True
            metadata["refusal_category"] = getattr(refusal_details, "category", None)
            metadata["refusal_explanation"] = getattr(refusal_details, "explanation", None)
            logger.warning(
                "Anthropic refused mid-stream (category=%s)",
                metadata["refusal_category"],
            )

        blocks: list[Any] = []
        if accumulated_text:
            blocks.append(TextBlock(text=accumulated_text))
        if accumulated_reasoning:
            blocks.append(ReasoningBlock(summary=accumulated_reasoning))
        for call in tool_calls:
            function = call["function"]
            try:
                args = json.loads(function["arguments"])
            except (TypeError, ValueError):
                args = {}
            blocks.append(ToolCallBlock(id=call["id"], name=function["name"], args=args))

        logger.debug(
            "Stream complete - text=%d chars, reasoning=%d chars, tool_calls=%d",
            len(accumulated_text),
            len(accumulated_reasoning),
            len(tool_calls),
        )

        yield Message(
            message_id=generate_id(response_id),
            role="assistant",
            content=blocks,
            delta=False,
            reasoning=accumulated_reasoning or None,
            timestamp=datetime.now().timestamp(),
            metadata=metadata,
            usages=self._usage_from_obj(usage_obj),
            tools_calls=tool_calls or None,
        )

    @staticmethod
    def _finalise_tool_call(entry: dict[str, Any]) -> dict[str, Any]:
        """Turn an accumulated tool_use block into an OpenAI-shaped tool call."""
        raw_json = entry["json"] or "{}"
        try:
            json.loads(raw_json)
        except (TypeError, ValueError):
            logger.warning(
                "Accumulated tool input for %s was not valid JSON; using {}.",
                entry.get("name", ""),
            )
            raw_json = "{}"

        return {
            "id": entry["id"],
            "type": "function",
            "function": {"name": entry["name"], "arguments": raw_json},
        }

    @staticmethod
    def _chunk_message(
        blocks: list[Any],
        metadata: dict[str, Any],
        response_id: str | None,
    ) -> Message:
        """Build an incremental (delta) message for a stream chunk."""
        return Message(
            message_id=generate_id(response_id),
            role="assistant",
            content=blocks,
            delta=True,
            metadata=metadata,
        )

    @staticmethod
    def _usage_from_obj(usage: Any) -> TokenUsages:
        """Build TokenUsages from a raw usage object (or zeros when absent)."""
        if usage is None:
            return TokenUsages(
                completion_tokens=0,
                prompt_tokens=0,
                total_tokens=0,
                reasoning_tokens=0,
                cache_creation_input_tokens=0,
                cache_read_input_tokens=0,
            )
        prompt = getattr(usage, "input_tokens", 0) or 0
        completion = getattr(usage, "output_tokens", 0) or 0
        return TokenUsages(
            completion_tokens=completion,
            prompt_tokens=prompt,
            total_tokens=prompt + completion,
            reasoning_tokens=0,
            cache_creation_input_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
            cache_read_input_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
        )
