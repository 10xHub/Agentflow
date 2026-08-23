"""Anthropic request helpers for Agent."""

from __future__ import annotations

import logging
from typing import Any

from .anthropic_request import (
    apply_cache_control,
    convert_tools,
    drop_trailing_assistant,
    merge_tool_results,
    split_system,
    strip_bedrock_prefix,
)
from .constants import (
    ANTHROPIC_DEFAULT_MAX_TOKENS,
    ANTHROPIC_DEFAULT_MAX_TOKENS_STREAMING,
    ANTHROPIC_NO_SAMPLING_MODELS,
    CALL_EXCLUDED_KWARGS,
)


logger = logging.getLogger("agentflow.agent")

# Rejected with a 400 by the models in ANTHROPIC_NO_SAMPLING_MODELS.
_SAMPLING_KEYS = ("temperature", "top_p", "top_k")

# Anthropic-specific keys that agentflow threads through llm_kwargs but which are
# not request parameters.
_ANTHROPIC_EXCLUDED_KWARGS = CALL_EXCLUDED_KWARGS | frozenset(
    {
        "anthropic_backend",
        "anthropic_cache",
        "reasoning_effort",
        "auth_token",
        "middleware",
    }
)


def strip_rejected_sampling_params(model: str, call_kwargs: dict[str, Any]) -> None:
    """Drop temperature/top_p/top_k for models that 400 on them, in place."""
    if strip_bedrock_prefix(model) not in ANTHROPIC_NO_SAMPLING_MODELS:
        return

    dropped = [key for key in _SAMPLING_KEYS if key in call_kwargs]
    if not dropped:
        return

    for key in dropped:
        call_kwargs.pop(key)
    logger.warning(
        "Model %s rejects sampling parameters; dropped %s from the request. "
        "Use reasoning_config effort to control depth instead.",
        model,
        ", ".join(dropped),
    )


def apply_reasoning_config(
    reasoning_config: dict[str, Any] | None,
    call_kwargs: dict[str, Any],
) -> None:
    """Map agentflow's reasoning_config onto Anthropic thinking/effort, in place.

    ``budget_tokens`` is never emitted: it is removed on every current Claude
    model and returns a 400. Depth is controlled by ``output_config.effort``.
    """
    if not reasoning_config:
        return

    call_kwargs.setdefault("thinking", {"type": "adaptive"})

    effort = reasoning_config.get("effort")
    if effort:
        output_config = dict(call_kwargs.get("output_config") or {})
        output_config.setdefault("effort", effort)
        call_kwargs["output_config"] = output_config

    if reasoning_config.get("thinking_budget") or reasoning_config.get("budget_tokens"):
        logger.warning(
            "budget_tokens/thinking_budget is not supported by current Claude "
            "models and would return a 400; ignoring it. Use "
            "reasoning_config={'effort': ...} instead."
        )


class AgentAnthropicMixin:
    """Anthropic Messages API request helpers."""

    def _build_anthropic_request(
        self,
        messages: list[dict[str, Any]],
        tools: list | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Assemble the ``(messages, kwargs)`` pair for a Messages API call.

        Shared by the request path and ``count_tokens`` so a token count is
        taken against exactly the payload that would be sent, not an
        approximation of it.
        """
        call_kwargs = {
            key: value
            for key, value in {**self.llm_kwargs, **kwargs}.items()
            if key not in _ANTHROPIC_EXCLUDED_KWARGS
        }

        system, remainder = split_system(messages)
        remainder = merge_tool_results(remainder)
        remainder = drop_trailing_assistant(remainder)

        # max_tokens is required by the API. Streaming gets a larger default
        # because a big max_tokens on a non-streaming request risks a timeout.
        call_kwargs.setdefault(
            "max_tokens",
            ANTHROPIC_DEFAULT_MAX_TOKENS_STREAMING if stream else ANTHROPIC_DEFAULT_MAX_TOKENS,
        )

        strip_rejected_sampling_params(self.model, call_kwargs)
        apply_reasoning_config(getattr(self, "reasoning_config", None), call_kwargs)

        anthropic_tools = convert_tools(tools)

        # Prefix-stable caching. Skipped when the caller placed their own
        # cache_control breakpoints, so an explicit strategy is never doubled.
        if "cache_control" not in call_kwargs:
            apply_cache_control(self.llm_kwargs.get("anthropic_cache"), system, anthropic_tools)

        if anthropic_tools:
            call_kwargs["tools"] = anthropic_tools
        if system:
            call_kwargs["system"] = system

        output_schema = getattr(self, "output_schema", None)
        if output_schema is not None:
            call_kwargs.setdefault("output_config", {})
            call_kwargs["output_config"] = {
                **call_kwargs["output_config"],
                "format": _as_json_schema_format(output_schema),
            }

        return remainder, call_kwargs

    async def _call_anthropic(
        self,
        messages: list[dict[str, Any]],
        tools: list | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Call the Anthropic Messages API."""
        remainder, call_kwargs = self._build_anthropic_request(messages, tools, stream, **kwargs)

        if stream:
            logger.debug("Calling Anthropic messages.stream with model=%s", self.model)
            return self.client.messages.stream(
                model=self.model,
                messages=remainder,
                **call_kwargs,
            )

        logger.debug("Calling Anthropic messages.create with model=%s", self.model)
        response = await self.client.messages.create(
            model=self.model,
            messages=remainder,
            **call_kwargs,
        )

        usage = getattr(response, "usage", None)
        cached = getattr(usage, "cache_read_input_tokens", 0) or 0
        if cached:
            logger.debug("Cache hit: %d cached tokens (Anthropic messages)", cached)

        if getattr(response, "stop_reason", None) == "pause_turn":
            logger.info(
                "Anthropic returned stop_reason='pause_turn' (server-tool "
                "iteration limit). Re-send the conversation including this "
                "response to resume."
            )
        return response

    async def _count_tokens_anthropic(
        self,
        messages: list[dict[str, Any]],
        tools: list | None = None,
        **kwargs: Any,
    ) -> int:
        """Count input tokens for a request via ``messages.count_tokens``.

        Counted against the exact payload ``_call_anthropic`` would send, so the
        number reflects the real system prompt, tool schemas, and merged tool
        results rather than an estimate.
        """
        remainder, call_kwargs = self._build_anthropic_request(
            messages, tools, stream=False, **kwargs
        )

        # count_tokens prices the input; generation-side parameters are not part
        # of its request shape and are rejected.
        for key in ("max_tokens", "stream", "output_config", "temperature", "top_p", "top_k"):
            call_kwargs.pop(key, None)

        response = await self.client.messages.count_tokens(
            model=self.model,
            messages=remainder,
            **call_kwargs,
        )
        return getattr(response, "input_tokens", 0) or 0


def _as_json_schema_format(output_schema: Any) -> dict[str, Any]:
    """Render an output schema into Anthropic's ``output_config.format`` shape."""
    if isinstance(output_schema, dict):
        return output_schema

    # Pydantic model
    schema_fn = getattr(output_schema, "model_json_schema", None)
    if callable(schema_fn):
        return {
            "type": "json_schema",
            "schema": schema_fn(),
            "name": getattr(output_schema, "__name__", "output"),
        }

    return {"type": "json_schema", "schema": {}}
