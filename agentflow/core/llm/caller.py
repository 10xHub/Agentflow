"""Shared single-turn LLM call utility.

Centralises the Google / OpenAI dispatch that was duplicated across
SummaryContextManager, LLMCallerMixin (eval judge), and UserSimulator.

Returns a plain 4-tuple so callers can choose how much they consume:

    text, *_ = await call_llm(...)              # only text
    text, inp, out, cache = await call_llm(...)  # text + token counts

OpenAI supports two API styles:
- ``"responses"`` (default) — ``client.responses.create()``, the current
  recommended API with ``input`` / ``instructions`` / ``max_output_tokens``.
- ``"chat"`` — ``client.chat.completions.create()``, the legacy style
  required by older or third-party-hosted models (e.g. some Chinese models).

The Agent class keeps its own execution path (streaming, tools, retry, etc.)
and is unaffected by this module.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from agentflow.core.llm.client_factory import create_llm_client, resolve_provider_and_model


logger = logging.getLogger("agentflow.llm.caller")


async def call_llm(
    model: str,
    prompt: str,
    *,
    system_prompt: str | None = None,
    max_tokens: int = 1024,
    temperature: float = 0.3,
    json_mode: bool = False,
    use_vertex_ai: bool = False,
    api_style: Literal["responses", "chat"] = "responses",
    **llm_kwargs: Any,
) -> tuple[str, int, int, int]:
    """Single-turn LLM call with provider auto-detection.

    Args:
        model: Model identifier (e.g. ``"gemini-2.0-flash"``, ``"gpt-4o-mini"``).
            Provider is inferred from the name via ``detect_provider``.
        prompt: The user-turn content to send.
        system_prompt: Optional system instruction prepended to the request.
        max_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.
        json_mode: When ``True``, instructs the provider to return valid JSON.
        use_vertex_ai: When ``True``, force Google Vertex AI client.
        api_style: OpenAI only. ``"responses"`` (default) uses the current
            Responses API (``client.responses.create``). Use ``"chat"`` for
            models that only support the legacy Chat Completions endpoint
            (e.g. older or self-hosted Chinese models).
        **llm_kwargs: Provider-specific parameters forwarded directly to the
            underlying API call. Examples:

            - Google: ``cached_content="cachedContents/abc123"`` — attaches an
              explicit Gemini context cache created via the Google SDK.
            - OpenAI: ``prompt_cache_key="my-agent-v1"`` — improves cache hit
              rates across requests sharing the same long system-prompt prefix.
            - OpenAI: ``prompt_cache_retention="24h"`` — extends cache retention
              for gpt-5.5+ models (default is in-memory, ~5-10 min).

    Returns:
        ``(text, input_tokens, output_tokens, cache_read_tokens)`` — plain tuple.
        Token counts are 0 when the provider does not report them.
    """
    # resolve_provider_and_model (not detect_provider) so a recognised
    # ``provider/`` prefix is stripped before the name reaches the SDK. Sending
    # "gemini/gemini-2.5-flash" or "anthropic/claude-opus-5" verbatim fails.
    provider, model = resolve_provider_and_model(model, use_vertex_ai=use_vertex_ai)
    client = create_llm_client(
        provider,
        use_vertex_ai=use_vertex_ai,
        anthropic_backend=llm_kwargs.pop("anthropic_backend", None),
    )

    if provider == "anthropic":
        return await _call_anthropic(
            client,
            model,
            prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            json_mode=json_mode,
            **llm_kwargs,
        )

    if provider == "google":
        return await _call_google(
            client,
            model,
            prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            json_mode=json_mode,
            **llm_kwargs,
        )

    if api_style == "chat":
        return await _call_openai_chat(
            client,
            model,
            prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            json_mode=json_mode,
            **llm_kwargs,
        )
    return await _call_openai_responses(
        client,
        model,
        prompt,
        system_prompt=system_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        json_mode=json_mode,
        **llm_kwargs,
    )


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------


async def _call_anthropic(
    client: Any,
    model: str,
    prompt: str,
    *,
    system_prompt: str | None,
    max_tokens: int,
    temperature: float,
    json_mode: bool,
    **llm_kwargs: Any,
) -> tuple[str, int, int, int]:
    from agentflow.core.graph.agent_internal.anthropic_request import strip_bedrock_prefix
    from agentflow.core.graph.agent_internal.constants import ANTHROPIC_NO_SAMPLING_MODELS

    call_kwargs: dict[str, Any] = dict(llm_kwargs)
    call_kwargs["max_tokens"] = max_tokens

    # Current Claude models reject temperature with a 400. call_llm defaults it
    # to 0.3 for every caller, so it has to be dropped rather than forwarded.
    if strip_bedrock_prefix(model) not in ANTHROPIC_NO_SAMPLING_MODELS:
        call_kwargs["temperature"] = temperature

    if system_prompt:
        call_kwargs["system"] = system_prompt

    if json_mode:
        # Anthropic has no response_mime_type; steer with an instruction, which
        # is what the Messages API supports without a full JSON schema.
        suffix = "\n\nRespond with valid JSON only, no prose and no code fences."
        call_kwargs["system"] = (call_kwargs.get("system") or "") + suffix

    response = await client.messages.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        **call_kwargs,
    )

    if getattr(response, "stop_reason", None) == "refusal":
        details = getattr(response, "stop_details", None)
        logger.warning(
            "Anthropic refused the request (category=%s)",
            getattr(details, "category", None),
        )

    text = ""
    for block in getattr(response, "content", None) or []:
        if getattr(block, "type", None) == "text":
            text += getattr(block, "text", "") or ""

    usage = getattr(response, "usage", None)
    inp = getattr(usage, "input_tokens", 0) or 0
    out = getattr(usage, "output_tokens", 0) or 0
    cache = getattr(usage, "cache_read_input_tokens", 0) or 0

    if cache:
        logger.debug("Cache hit: %d cached tokens (Anthropic)", cache)

    return text.strip(), inp, out, cache


# ---------------------------------------------------------------------------
# Google
# ---------------------------------------------------------------------------


async def _call_google(
    client: Any,
    model: str,
    prompt: str,
    *,
    system_prompt: str | None,
    max_tokens: int,
    temperature: float,
    json_mode: bool,
    **llm_kwargs: Any,
) -> tuple[str, int, int, int]:
    from google.genai import types

    config_kwargs: dict[str, Any] = {
        "max_output_tokens": max_tokens,
        "temperature": temperature,
    }
    if json_mode:
        config_kwargs["response_mime_type"] = "application/json"

    cached_content = llm_kwargs.pop("cached_content", None)
    if cached_content:
        # system_prompt is already inside the cache — don't resend it
        config_kwargs["cached_content"] = cached_content
    elif system_prompt:
        config_kwargs["system_instruction"] = system_prompt

    response = await client.aio.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(**config_kwargs),
    )

    text = (response.text or "").strip()
    inp = out = cache = 0
    meta = getattr(response, "usage_metadata", None)
    if meta is not None:
        inp = getattr(meta, "prompt_token_count", 0) or 0
        out = getattr(meta, "candidates_token_count", 0) or 0
        cache = getattr(meta, "cached_content_token_count", 0) or 0

    if cache:
        logger.debug("Cache hit: %d cached tokens (Google)", cache)

    return text, inp, out, cache


# ---------------------------------------------------------------------------
# OpenAI — Responses API (default)
# ---------------------------------------------------------------------------


async def _call_openai_responses(
    client: Any,
    model: str,
    prompt: str,
    *,
    system_prompt: str | None,
    max_tokens: int,
    temperature: float,
    json_mode: bool,
    **llm_kwargs: Any,
) -> tuple[str, int, int, int]:
    """Call the OpenAI Responses API (client.responses.create)."""
    kwargs: dict[str, Any] = {
        "model": model,
        "input": prompt,
        "max_output_tokens": max_tokens,
        "temperature": temperature,
    }
    if system_prompt:
        kwargs["instructions"] = system_prompt
    if json_mode:
        kwargs["text"] = {"format": {"type": "json_object"}}
    kwargs.update(llm_kwargs)

    response = await client.responses.create(**kwargs)

    text = _extract_responses_text(response)
    inp = out = cache = 0
    usage = getattr(response, "usage", None)
    if usage is not None:
        inp = getattr(usage, "input_tokens", 0) or 0
        out = getattr(usage, "output_tokens", 0) or 0
        details = getattr(usage, "input_tokens_details", None)
        if details is not None:
            cache = getattr(details, "cached_tokens", 0) or 0

    if cache:
        logger.debug("Cache hit: %d cached tokens (OpenAI responses API)", cache)

    return text, inp, out, cache


def _extract_responses_text(response: Any) -> str:
    """Extract the assistant text from an OpenAI Responses API response object."""
    # SDK convenience property available in openai >= 1.61
    output_text = getattr(response, "output_text", None)
    if output_text is not None:
        return str(output_text).strip()

    # Manual fallback: iterate output items
    for item in getattr(response, "output", []):
        item_type = getattr(item, "type", None)
        if item_type == "message":
            for part in getattr(item, "content", []):
                if getattr(part, "type", None) == "output_text":
                    return (getattr(part, "text", "") or "").strip()

    return ""


# ---------------------------------------------------------------------------
# OpenAI — Chat Completions (legacy / compat)
# ---------------------------------------------------------------------------


async def _call_openai_chat(
    client: Any,
    model: str,
    prompt: str,
    *,
    system_prompt: str | None,
    max_tokens: int,
    temperature: float,
    json_mode: bool,
    **llm_kwargs: Any,
) -> tuple[str, int, int, int]:
    """Call the OpenAI Chat Completions API (client.chat.completions.create)."""
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    kwargs.update(llm_kwargs)

    response = await client.chat.completions.create(**kwargs)

    text = (response.choices[0].message.content or "").strip()
    inp = out = cache = 0
    usage = getattr(response, "usage", None)
    if usage is not None:
        inp = getattr(usage, "prompt_tokens", 0) or 0
        out = getattr(usage, "completion_tokens", 0) or 0
        details = getattr(usage, "prompt_tokens_details", None)
        if details is not None:
            cache = getattr(details, "cached_tokens", 0) or 0

    if cache:
        logger.debug("Cache hit: %d cached tokens (OpenAI chat completions)", cache)

    return text, inp, out, cache
