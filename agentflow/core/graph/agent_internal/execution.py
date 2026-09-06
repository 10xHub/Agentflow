"""Execution helpers for Agent."""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

from injectq import Inject, InjectQ
from injectq.utils.exceptions import DependencyNotFoundError

from agentflow.core.graph.tool_node import ToolNode
from agentflow.core.state import AgentState
from agentflow.core.state.base_context import BaseContextManager
from agentflow.runtime.adapters.llm.model_response_converter import ModelResponseConverter
from agentflow.utils.converter import (
    convert_messages,
    strip_media_blocks,
)

from .circuit_breaker import CircuitBreaker, CircuitBreakerOpenError
from .constants import RetryConfig


logger = logging.getLogger("agentflow.agent")

# Valid HTTP status range, used to sanity-check anything parsed out of a message.
HTTP_STATUS_MIN = 100
HTTP_STATUS_MAX = 599

# Matches a status code only when it is actually presented as one, either
# labelled ("status code: 503", "HTTP 429", "error code 500") or bracketed, which
# is how several SDKs render it ("Overloaded [529]").
#
# A bare "500" in prose -- a token count, a dimension, "max_tokens <= 500" -- must
# NOT be read as a status. That was the M3 bug: the old check asked only whether
# the digits appeared anywhere in the message, so permanent errors were retried.
_STATUS_CODE_PATTERNS = (
    # Labelled: "status code: 503", "status_code=503", "HTTP 429", "Error code: 500"
    re.compile(
        r"(?:status(?:[\s_-]*code)?|error[\s_-]*code|\bhttp\b|\bcode\b)[\s:=\-]*(\d{3})\b",
        re.IGNORECASE,
    ),
    # Bracketed, as several SDKs render it: "Overloaded [529]"
    re.compile(r"[\[\(](\d{3})[\]\)]"),
    # Leading, as most HTTP errors render it: "503 Service Unavailable. ..."
    re.compile(r"^\s*(\d{3})\b"),
    # Followed by a standard reason phrase: "Rate limited: 429 Too Many Requests"
    re.compile(
        r"\b(\d{3})\s+(?:"
        r"service unavailable|too many requests|internal server error|"
        r"bad gateway|gateway time-?out|request time-?out|overloaded"
        r")",
        re.IGNORECASE,
    ),
)


def _extract_input_tokens(response: Any) -> int:
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0
    return getattr(usage, "prompt_tokens", None) or getattr(usage, "input_tokens", None) or 0


def _extract_output_tokens(response: Any) -> int:
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0
    return getattr(usage, "completion_tokens", None) or getattr(usage, "output_tokens", None) or 0


def _extract_cache_read_tokens(response: Any) -> int:
    """Tokens served from provider cache (OpenAI cached prompt tokens, Anthropic cache read)."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0
    # Anthropic: usage.cache_read_input_tokens
    v = getattr(usage, "cache_read_input_tokens", None)
    if v:
        return int(v)
    # OpenAI chat completions: usage.prompt_tokens_details.cached_tokens
    details = getattr(usage, "prompt_tokens_details", None)
    if details:
        v = getattr(details, "cached_tokens", None)
        if v:
            return int(v)
    # OpenAI responses API: usage.input_tokens_details.cached_tokens
    details = getattr(usage, "input_tokens_details", None)
    if details:
        v = getattr(details, "cached_tokens", None)
        if v:
            return int(v)
    # Google: usage_metadata.cached_content_token_count
    usage_meta = getattr(response, "usage_metadata", None)
    if usage_meta:
        v = getattr(usage_meta, "cached_content_token_count", None)
        if v:
            return int(v)
    return 0


def _extract_cache_creation_tokens(response: Any) -> int:
    """Tokens written to provider cache (Anthropic cache_creation only;
    OpenAI has no equivalent)."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0
    v = getattr(usage, "cache_creation_input_tokens", None)
    return int(v) if v else 0


def _extract_reasoning_tokens(response: Any) -> int:
    """Reasoning/thinking tokens (OpenAI o-series completion_tokens_details, Google thoughts)."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0
    # OpenAI: usage.completion_tokens_details.reasoning_tokens
    details = getattr(usage, "completion_tokens_details", None)
    if details:
        v = getattr(details, "reasoning_tokens", None)
        if v:
            return int(v)
    # OpenAI responses API: usage.output_tokens_details.reasoning_tokens
    details = getattr(usage, "output_tokens_details", None)
    if details:
        v = getattr(details, "reasoning_tokens", None)
        if v:
            return int(v)
    # Google thinking models: usage_metadata.thoughts_token_count
    usage_meta = getattr(response, "usage_metadata", None)
    if usage_meta:
        v = getattr(usage_meta, "thoughts_token_count", None)
        if v:
            return int(v)
    return 0


def _extract_finish_reason(response: Any) -> str:
    """Return a single finish reason string from the raw provider response."""
    # OpenAI chat completions: response.choices[0].finish_reason
    choices = getattr(response, "choices", None)
    if choices:
        reason = getattr(choices[0], "finish_reason", None)
        if reason:
            return str(reason)
    # OpenAI responses API: response.status ("completed", "incomplete", etc.)
    status = getattr(response, "status", None)
    if status and isinstance(status, str) and status not in ("", "in_progress"):
        return status
    # Anthropic / generic: response.stop_reason
    stop_reason = getattr(response, "stop_reason", None)
    if stop_reason:
        return str(stop_reason)
    # Google: response.candidates[0].finish_reason (may be an enum)
    candidates = getattr(response, "candidates", None)
    if candidates:
        fr = getattr(candidates[0], "finish_reason", None)
        if fr is not None:
            name = getattr(fr, "name", None)
            return str(name) if name else str(fr)
    return ""


def _extract_response_id(response: Any) -> str:
    """Return the unique completion ID from the provider response."""
    v = getattr(response, "id", None)
    return str(v) if v else ""


def _extract_response_model(response: Any) -> str:
    """Return the model name the provider actually used (may differ from requested)."""
    v = getattr(response, "model", None)
    return str(v) if v else ""


def _extract_response_text(response: Any) -> str:
    text = getattr(response, "text", None)
    if text:
        return str(text).strip()
    output_text = getattr(response, "output_text", None)
    if output_text:
        return str(output_text).strip()
    choices = getattr(response, "choices", None)
    if choices:
        return str(getattr(getattr(choices[0], "message", None), "content", "") or "").strip()
    return ""


class AgentExecutionMixin:
    """Execution flow, tool resolution, and provider dispatch helpers."""

    # Set by ``Agent.__init__``; declared here so the mixin can read it when
    # lazily building fallback clients (the mixin never assigns it itself).
    use_vertex_ai: bool

    def _setup_tools(self) -> ToolNode | None:
        """Normalize the tool_node input and wire internal state.

        - ``ToolNode`` → stored as ``self._tool_node``; ``tool_node_name`` remains ``None``.
        - ``str``     → stored as ``self.tool_node_name`` for lazy lookup via the DI
                        container at execution time; returns ``None``.
        - ``None``    → no tools; both attributes remain ``None``.
        """
        tn = self.tool_node  # str | ToolNode | None
        if tn is None:
            logger.debug("No tool_node provided")
            return None

        if isinstance(tn, str):
            logger.debug("tool_node is a named graph-node reference: '%s'", tn)
            self.tool_node_name = tn
            # Lazy resolution: the named node is looked up at execution time
            # via _resolve_tools(), once the graph has been compiled and the
            # "get_node" factory is registered in the DI container.
            return None

        # If name is passed then we need to load the ToolNode
        logger.debug("tool_node is a ToolNode instance")
        return tn

    def get_tool_node(self) -> ToolNode | None:
        """Return the agent-owned ``ToolNode`` when one is configured."""
        return getattr(self, "_tool_node", None)

    async def _trim_context(
        self,
        state: AgentState,
        context_manager: BaseContextManager | None = Inject[BaseContextManager],
    ) -> AgentState:
        """Trim state context when a context manager is configured."""
        if not self.trim_context:
            logger.debug("Context trimming not enabled")
            return state

        if context_manager is None:
            logger.warning("trim_context is enabled but no context manager is available")
            return state

        try:
            new_state = await context_manager.atrim_context(state)
            logger.debug("Context trimmed using context manager")
            return new_state
        except AttributeError:
            logger.warning(
                "trim_context is enabled but no BaseContextManager is registered. "
                "Skipping context trimming."
            )
            return state

    # ------------------------------------------------------------------
    # Retry / fallback helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_status_code(exc: Exception) -> int | None:
        """Best-effort extraction of an HTTP status code from an SDK exception.

        Structured attributes are trusted. The string fallback is deliberately
        strict: the previous version asked whether "500" appeared *anywhere* in
        the message, so an error mentioning a token count ("max_tokens: 500") or
        a model name was classified as a retryable server error and the call was
        retried -- burning the retry budget on an error that would never succeed.
        The number must now actually be presented as a status code.
        """
        # OpenAI SDK: openai.APIStatusError has .status_code
        status = getattr(exc, "status_code", None)
        if status is not None:
            try:
                return int(status)
            except (TypeError, ValueError):
                pass

        # httpx-style: exc.response.status_code
        response = getattr(exc, "response", None)
        if response is not None:
            status = getattr(response, "status_code", None)
            if status is not None:
                try:
                    return int(status)
                except (TypeError, ValueError):
                    pass

        # Google GenAI and generic HTTP errors often embed a code attribute
        code = getattr(exc, "code", None)
        if code is not None:
            try:
                return int(code)
            except (TypeError, ValueError):
                pass

        # String fallback: only when the number is *labelled* as a status/error
        # code, never a bare digit sequence found somewhere in the message.
        exc_str = str(exc)
        for pattern in _STATUS_CODE_PATTERNS:
            match = pattern.search(exc_str)
            if match:
                parsed = int(match.group(1))
                if HTTP_STATUS_MIN <= parsed <= HTTP_STATUS_MAX:
                    return parsed
        return None

    def _is_retryable_error(self, exc: Exception, retry_cfg: RetryConfig) -> bool:
        """Determine whether *exc* is a transient error worth retrying."""
        status = self._extract_status_code(exc)
        if status is not None and status in retry_cfg.retryable_status_codes:
            return True
        # Connection-level / transport errors are always retryable
        if isinstance(exc, ConnectionError | TimeoutError | OSError):
            return True
        exc_name = type(exc).__name__.lower()
        return any(
            keyword in exc_name
            for keyword in ("timeout", "connection", "unavailable", "serviceunav")
        )

    def _get_circuit_breaker(
        self,
        provider: str,
        model: str,
        retry_cfg: RetryConfig | None,
    ) -> CircuitBreaker | None:
        """Return the circuit breaker for ``(provider, model)``, or None if disabled.

        Breakers are created lazily and cached on the instance so their state
        persists across calls (the whole point: stop hammering a dead provider on
        every invocation).
        """
        if retry_cfg is None or not retry_cfg.circuit_breaker_enabled:
            return None
        registry: dict[tuple[str, str], CircuitBreaker] | None = self.__dict__.get(
            "_circuit_breakers"
        )
        if registry is None:
            registry = {}
            self._circuit_breakers = registry
        key = (provider, model)
        breaker = registry.get(key)
        if breaker is None:
            breaker = CircuitBreaker(
                failure_threshold=retry_cfg.circuit_breaker_threshold,
                reset_timeout=retry_cfg.circuit_breaker_reset_timeout,
            )
            registry[key] = breaker
        return breaker

    async def _call_llm_with_retry(  # noqa: PLR0912, PLR0915
        self,
        messages: list[dict[str, Any]],
        tools: list | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Wrap ``_call_llm`` with retry + exponential back-off + fallback models.

        Execution order:
        1. Try the primary model up to ``retry_config.max_retries`` times.
        2. For each fallback model, try up to ``retry_config.max_retries`` times.
        3. If everything fails, raise the last exception.
        """
        retry_cfg: RetryConfig | None = getattr(self, "retry_config", None)
        fallback_models: list[tuple[str, str | None]] = getattr(self, "fallback_models", [])

        # Fast-path: no retry config at all → single attempt, no catch
        if retry_cfg is None and not fallback_models:
            return await self._call_llm(messages, tools, stream, **kwargs)

        max_retries = retry_cfg.max_retries if retry_cfg else 0

        # Build the ordered attempt list: primary + fallbacks
        attempts: list[tuple[str, str, Any, str | None]] = [
            (self.model, self.provider, self.client, self.base_url),
        ]
        for fb_model, fb_provider in fallback_models:
            attempts.append((fb_model, fb_provider or self.provider, None, self.base_url))

        last_exc: Exception | None = None

        for attempt_idx, (model, provider, fallback_client, base_url) in enumerate(attempts):
            is_fallback = attempt_idx > 0

            if is_fallback:
                logger.info(
                    "Switching to fallback model %s (provider=%s)",
                    model,
                    provider,
                )

            breaker = self._get_circuit_breaker(provider, model, retry_cfg)
            if breaker is not None and not breaker.allow():
                retry_after = breaker.retry_after()
                logger.warning(
                    "Circuit open for %s (provider=%s); skipping (retry in %.1fs)",
                    model,
                    provider,
                    retry_after,
                )
                last_exc = CircuitBreakerOpenError((provider, model), retry_after)
                continue

            for retry in range(max_retries + 1):  # 0 .. max_retries
                try:
                    if is_fallback:
                        # Temporarily swap model/provider/client for the call
                        orig_model, orig_provider, orig_client, orig_base_url = (
                            self.model,
                            self.provider,
                            self.client,
                            self.base_url,
                        )
                        self.model = model
                        self.provider = provider
                        self.base_url = base_url
                        active_client = fallback_client
                        if active_client is None:
                            # Lazily build the fallback client, honouring the
                            # agent's Vertex AI selection (only affects google).
                            active_client = self._create_client(
                                provider,
                                base_url,
                                self.use_vertex_ai,
                            )
                        self.client = active_client
                        try:
                            result = await self._call_llm(messages, tools, stream, **kwargs)
                        finally:
                            # Restore originals regardless of outcome
                            self.model = orig_model
                            self.provider = orig_provider
                            self.client = orig_client
                            self.base_url = orig_base_url
                    else:
                        result = await self._call_llm(messages, tools, stream, **kwargs)

                    if breaker is not None:
                        breaker.record_success()
                    if is_fallback or retry > 0:
                        logger.info(
                            "LLM call succeeded on %s (attempt %d/%d, model_index=%d)",
                            model,
                            retry + 1,
                            max_retries + 1,
                            attempt_idx,
                        )
                    return result

                except Exception as exc:
                    last_exc = exc

                    if retry_cfg is None or not self._is_retryable_error(exc, retry_cfg):
                        logger.warning(
                            "Non-retryable error from %s: %s",
                            model,
                            exc,
                        )
                        # Non-retryable → skip remaining retries, try next fallback
                        break

                    if retry < max_retries:
                        delay = min(
                            retry_cfg.initial_delay * (retry_cfg.backoff_factor**retry),
                            retry_cfg.max_delay,
                        )
                        logger.warning(
                            "Retryable error from %s (attempt %d/%d): %s. Retrying in %.1fs …",
                            model,
                            retry + 1,
                            max_retries + 1,
                            exc,
                            delay,
                        )
                        await asyncio.sleep(delay)
                    else:
                        logger.warning(
                            "All %d retries exhausted for model %s.",
                            max_retries + 1,
                            model,
                        )

            # This model's attempt cycle failed (retries exhausted or
            # non-retryable). Record one failure against its circuit.
            if breaker is not None:
                breaker.record_failure()

        # Every model exhausted → re-raise the last exception
        assert last_exc is not None  # noqa: S101
        raise last_exc

    async def _call_llm(  # noqa: PLR0911
        self,
        messages: list[dict[str, Any]],
        tools: list | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Route requests to the active provider and API style."""
        logger.debug(
            "Calling LLM: provider=%s, output_type=%s, model=%s, stream=%s",
            self.provider,
            self.output_type,
            self.model,
            stream,
        )

        if self.provider == "openai":
            if self.api_style == "responses":
                if getattr(self, "output_schema", None):
                    logger.debug(
                        "output_schema is set; using chat completions parse path "
                        "instead of Responses API"
                    )
                    self._effective_api_style = "chat"
                    if self.reasoning_config and self.reasoning_config.get("effort"):
                        kwargs.setdefault("reasoning_effort", self.reasoning_config["effort"])
                    if self.base_url and self.reasoning_config:
                        existing_extra = kwargs.get("extra_body", {})
                        existing_extra["reasoning"] = self.reasoning_config
                        kwargs["extra_body"] = existing_extra
                    return await self._call_openai(messages, tools, stream, **kwargs)

                if self.base_url:
                    try:
                        result = await self._call_openai_responses(
                            messages, tools, stream, **kwargs
                        )
                        self._effective_api_style = "responses"
                        return result
                    except Exception as exc:
                        logger.warning(
                            "Responses API not supported at %s (%s). "
                            "Falling back to chat.completions.create().",
                            self.base_url,
                            exc,
                        )
                        self._effective_api_style = "chat"
                        if self.reasoning_config and self.reasoning_config.get("effort"):
                            kwargs.setdefault("reasoning_effort", self.reasoning_config["effort"])
                        return await self._call_openai(messages, tools, stream, **kwargs)

                self._effective_api_style = "responses"
                return await self._call_openai_responses(messages, tools, stream, **kwargs)

            self._effective_api_style = "chat"
            if self.reasoning_config and self.reasoning_config.get("effort"):
                kwargs.setdefault("reasoning_effort", self.reasoning_config["effort"])
            if self.base_url and self.reasoning_config:
                existing_extra = kwargs.get("extra_body", {})
                existing_extra["reasoning"] = self.reasoning_config
                kwargs["extra_body"] = existing_extra
            return await self._call_openai(messages, tools, stream, **kwargs)

        if self.provider == "google":
            return await self._call_google(messages, tools, stream, **kwargs)

        if self.provider == "anthropic":
            return await self._call_anthropic(messages, tools, stream, **kwargs)

        raise ValueError(f"Unsupported provider: {self.provider}")

    async def count_tokens(
        self,
        messages: list[dict[str, Any]],
        tools: list | None = None,
        **kwargs: Any,
    ) -> int:
        """Count the input tokens a request would consume, before sending it.

        Uses the provider's own counting endpoint, so the number accounts for
        the system prompt, tool schemas, and content blocks exactly as the real
        request would encode them. Useful for budgeting a call, deciding whether
        to trim context, or estimating cost up front.

        Args:
            messages: Messages in agentflow's internal (OpenAI-shaped) dialect.
            tools: Optional tool definitions, which are part of the counted input.
            **kwargs: Extra request parameters, matching ``_call_llm``.

        Returns:
            The input token count.

        Note:
            Coverage differs by provider. Anthropic's endpoint counts messages,
            the system prompt, and tool schemas. Google's counts ``contents``
            only, since its endpoint takes no system instruction or tools, so
            the total is lower than what the request will actually bill.

        Raises:
            NotImplementedError: If the active provider exposes no server-side
                token counting endpoint.
        """
        if self.provider == "anthropic":
            return await self._count_tokens_anthropic(messages, tools, **kwargs)

        if self.provider == "google":
            return await self._count_tokens_google(messages, tools, **kwargs)

        raise NotImplementedError(
            f"Provider '{self.provider}' has no server-side token counting "
            "endpoint, so an exact count is not available. 'anthropic' and "
            "'google' support Agent.count_tokens(); OpenAI does not expose one. "
            "Read usage from the response after the call instead, or count "
            "client-side with the tokenizer matching your model."
        )

    async def execute(
        self,
        state: AgentState,
        config: dict[str, Any],
    ) -> ModelResponseConverter:
        """Execute the Agent node against the current graph state."""
        container = InjectQ.get_instance()

        state = await self._trim_context(state)

        # Build effective system prompts (with trigger table if skills configured)
        effective_system_prompt = list(self.system_prompt)

        if hasattr(self, "_build_skill_prompts") and callable(self._build_skill_prompts):
            effective_system_prompt = self._build_skill_prompts(state, self.system_prompt)

        if hasattr(self, "_build_memory_prompts") and callable(self._build_memory_prompts):
            effective_system_prompt.extend(await self._build_memory_prompts(state, config))

        messages = convert_messages(
            system_prompts=effective_system_prompt,
            state=state,
            extra_messages=self.extra_messages or [],
        )

        # Resolve internal media refs (agentflow://media/...) before provider call.
        # This converts internal refs to signed URLs or inline base64, using the
        # capability-aware path when provider+model are known.
        messages = await self._resolve_media_in_messages(messages)

        # Multi-agent safety: strip media blocks for text-only agents.
        # When an agent has no multimodal_config, it cannot process images,
        # audio, video, or documents — so we remove them.  This handles the
        # case where a multimodal agent earlier in the graph added media to
        # the shared state and a downstream text-only agent inherits it.
        multimodal_config = getattr(self, "multimodal_config", None)
        if multimodal_config is None:
            messages = strip_media_blocks(messages)

        is_stream = config.get("is_stream", False)

        # Always resolve tools - even after tool results, the model may want to call
        # additional tools (e.g., Gemini 2.5+ with sequential tool calls)
        tools = await self._resolve_tools(container)

        from agentflow.runtime.publisher.events import (
            ContentType,
            Event,
            EventModel,
            EventType,
        )
        from agentflow.runtime.publisher.publish import publish_event

        node_name = config.get("_node_name", "unknown")

        # Collect LLM request parameters from llm_kwargs (temperature, max_tokens, etc.)
        _REQUEST_PARAM_KEYS = (
            "temperature",
            "max_tokens",
            "top_p",
            "top_k",
            "frequency_penalty",
            "presence_penalty",
            "stop",
            "seed",
        )
        request_params: dict[str, Any] = {
            k: v for k, v in self.llm_kwargs.items() if k in _REQUEST_PARAM_KEYS and v is not None
        }

        llm_event = EventModel.default(
            config,
            data={
                "model": self.model,
                "provider": getattr(self, "provider", ""),
                "input_messages": messages,
                "system_prompt": effective_system_prompt,
                "request_params": request_params,
                "tool_count": len(tools) if tools else 0,
            },
            event=Event.LLM_CALL,
            event_type=EventType.START,
            content_type=[ContentType.STATE],
            node_name=node_name,
        )
        publish_event(llm_event)

        response = await self._call_llm_with_retry(
            messages=messages,
            tools=tools if tools else None,
            stream=is_stream,
        )

        if not is_stream:
            llm_event.event_type = EventType.END  # type: ignore[assignment]
            llm_event.data["output_response"] = _extract_response_text(response)
            llm_event.data["input_tokens"] = _extract_input_tokens(response)
            llm_event.data["output_tokens"] = _extract_output_tokens(response)
            llm_event.data["cache_read_tokens"] = _extract_cache_read_tokens(response)
            llm_event.data["cache_creation_tokens"] = _extract_cache_creation_tokens(response)
            llm_event.data["reasoning_tokens"] = _extract_reasoning_tokens(response)
            llm_event.data["finish_reason"] = _extract_finish_reason(response)
            llm_event.data["response_id"] = _extract_response_id(response)
            llm_event.data["response_model"] = _extract_response_model(response)
            publish_event(llm_event)

        converter_key = self._get_converter_key()
        return ModelResponseConverter(response, converter=converter_key)

    async def _resolve_media_in_messages(  # noqa: PLR0915, PLR0912
        self,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Resolve internal media refs in converted message dicts.

        For messages that contain ``agentflow://media/`` URLs in their
        content parts, this method resolves them to signed URLs or inline
        base64 using the capability-aware resolver.

        Returns the messages list with resolved URLs in place.
        """
        media_store = getattr(self, "media_store", None)
        if media_store is None:
            # Try to get from container
            container = InjectQ.get_instance()
            media_store = container.try_get("media_store") or container.try_get("BaseMediaStore")

        if media_store is None:
            return messages

        from agentflow.storage.media.resolver import MediaRefResolver

        resolver = MediaRefResolver(media_store=media_store)
        provider = getattr(self, "provider", None)
        model = getattr(self, "model", None)
        # Strip provider prefix from model (e.g. "openai/gpt-4o" -> "gpt-4o")
        if model and "/" in model:
            model = model.split("/", 1)[1]

        for msg in messages:
            content = msg.get("content")
            if not isinstance(content, list):
                continue

            for i, part in enumerate(content):
                if not isinstance(part, dict):
                    continue

                part_type = part.get("type", "")
                if part_type == "image_url":
                    url = part.get("image_url", {}).get("url", "")
                    if url.startswith("agentflow://media/"):
                        ref_media = type(
                            "MediaRef",
                            (),
                            {
                                "kind": "url",
                                "url": url,
                                "mime_type": part.get("image_url", {}).get("mime_type"),
                                "data_base64": None,
                                "file_id": None,
                            },
                        )()
                        try:
                            if provider == "google":
                                resolved = await resolver.resolve_for_google(
                                    ref_media,
                                    model=model,
                                )
                                if hasattr(resolved, "inline_data"):
                                    # Google Part with inline data — convert back to OpenAI format
                                    inline = resolved.inline_data
                                    import base64

                                    b64 = base64.b64encode(inline.data).decode()
                                    content[i] = {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:{inline.mime_type};base64,{b64}",
                                        },
                                    }
                                elif hasattr(resolved, "file_data"):
                                    content[i] = {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": resolved.file_data.file_uri,
                                        },
                                    }
                            else:
                                resolved = await resolver.resolve_for_openai(
                                    ref_media,
                                    model=model,
                                )
                                content[i] = resolved
                        except Exception:
                            logger.warning(
                                "Failed to resolve media ref %s for %s/%s",
                                url,
                                provider,
                                model,
                            )

                elif part_type in ("document", "video"):
                    media_info = part.get(part_type, {})
                    url = media_info.get("url", "")
                    if url and url.startswith("agentflow://media/"):
                        ref_media = type(
                            "MediaRef",
                            (),
                            {
                                "kind": "url",
                                "url": url,
                                "mime_type": media_info.get("mime_type"),
                                "data_base64": media_info.get("data"),
                                "file_id": None,
                            },
                        )()
                        try:
                            if provider == "google":
                                resolved = await resolver.resolve_for_google(
                                    ref_media,
                                    model=model,
                                )
                                if hasattr(resolved, "inline_data"):
                                    import base64

                                    b64 = base64.b64encode(resolved.inline_data.data).decode()
                                    content[i] = {
                                        "type": part_type,
                                        part_type: {
                                            "data": b64,
                                            "mime_type": resolved.inline_data.mime_type,
                                        },
                                    }
                            else:
                                resolved = await resolver.resolve_for_openai(
                                    ref_media,
                                    model=model,
                                )
                                if "image_url" in resolved:
                                    content[i] = {
                                        "type": part_type,
                                        part_type: {
                                            "url": resolved["image_url"]["url"],
                                            "mime_type": media_info.get("mime_type"),
                                        },
                                    }
                        except Exception:
                            logger.warning(
                                "Failed to resolve media ref %s for %s/%s",
                                url,
                                provider,
                                model,
                            )

        return messages

    async def _resolve_tools(self, container: InjectQ) -> list[dict[str, Any]]:
        """Resolve tool definitions from inline tools and named ToolNodes.

        On the first call when ``tool_node_name`` is set, the named graph node is
        resolved via InjectQ, cached as ``self._tool_node``, and any tools that
        were queued in ``self._extra_tools`` before the graph was compiled are
        registered into the ToolNode.  Subsequent calls use the cached instance
        directly — no repeated DI lookup, no duplicates.
        """
        # Resolve a named ToolNode on first call, then cache it
        if self.tool_node_name:
            try:
                node = container.call_factory("get_node", self.tool_node_name)
            except (KeyError, DependencyNotFoundError) as exc:
                raise RuntimeError(
                    f"ToolNode named '{self.tool_node_name}' was not found in the compiled graph. "
                    "Register the named ToolNode in the graph before executing the Agent."
                ) from exc

            if node is None:
                raise RuntimeError(
                    f"ToolNode named '{self.tool_node_name}' was not found in the compiled graph. "
                    "Register the named ToolNode in the graph before executing the Agent."
                )

            if not isinstance(node.func, ToolNode):
                raise RuntimeError(
                    f"Graph node '{self.tool_node_name}' is not a ToolNode. "
                    "Pass a ToolNode instance or register the proper graph node."
                )

            self._tool_node = node.func
            self.tool_node_name = None  # prevent duplicate resolution on future calls

            # Drain any tools that were queued before the graph was compiled
            for fn in getattr(self, "_extra_tools", []):
                self._tool_node.add_tool(fn)
            self._extra_tools = []

        if not self._tool_node:
            return []

        return await self._tool_node.all_tools(tags=self.tools_tags)

    def _extract_prompt(self, messages: list[dict[Any, Any]]) -> str:
        """Extract the last user message as a plain string for non-chat generation endpoints.

        Used by both OpenAI (image / audio) and Google (image / video / audio) providers.
        """
        for message in reversed(messages):
            if message.get("role") == "user":
                content = message.get("content", "")
                return str(content) if content else ""
        return ""

    def _get_converter_key(self) -> str:
        """Return the correct response converter key for the active provider."""
        effective = getattr(self, "_effective_api_style", self.api_style)
        if self.provider == "openai" and effective == "responses":
            return "openai_responses"
        return self.provider
