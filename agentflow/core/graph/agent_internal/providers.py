"""Provider detection, validation, and client creation helpers for Agent."""

from __future__ import annotations

import logging
from typing import Any, Protocol

from agentflow.core.llm.client_factory import (
    create_llm_client,
    detect_provider,
    resolve_provider_and_model,
)

from .constants import (
    ANTHROPIC_OUTPUT_TYPES,
    CLIENT_CONSTRUCTOR_KWARGS,
    GOOGLE_OUTPUT_TYPES,
    OPENAI_OUTPUT_TYPES,
    VALID_OUTPUT_TYPES,
)


logger = logging.getLogger("agentflow.agent")


class _ProviderAgentLike(Protocol):
    output_type: str
    provider: str
    llm_kwargs: dict[str, Any]


class AgentProviderMixin:
    """Provider-specific validation and client creation helpers."""

    def _validate_output_type(self: _ProviderAgentLike) -> None:
        """Validate that the selected provider supports the requested output type."""
        if self.output_type not in VALID_OUTPUT_TYPES:
            raise ValueError(
                "Invalid output_type "
                f"'{self.output_type}'. Supported types: {list(VALID_OUTPUT_TYPES)}"
            )

        if self.provider == "google" and self.output_type not in GOOGLE_OUTPUT_TYPES:
            raise ValueError(
                f"Google provider doesn't support output_type='{self.output_type}'. "
                f"Supported: {list(GOOGLE_OUTPUT_TYPES)}"
            )

        if self.provider == "openai" and self.output_type not in OPENAI_OUTPUT_TYPES:
            raise ValueError(
                f"OpenAI provider doesn't support output_type='{self.output_type}'. "
                f"Supported: {list(OPENAI_OUTPUT_TYPES)}"
            )

        if self.provider == "anthropic" and self.output_type not in ANTHROPIC_OUTPUT_TYPES:
            raise ValueError(
                f"Anthropic provider doesn't support output_type='{self.output_type}'. "
                f"Supported: {list(ANTHROPIC_OUTPUT_TYPES)}. Anthropic's Messages API "
                "generates text and tool calls only; there is no image/audio/video "
                "generation endpoint."
            )

        logger.debug("%s provider supports output_type='%s'", self.provider, self.output_type)

    def _detect_provider_from_model(self, model: str, use_vertex_ai: bool = False) -> str:
        """Infer the provider from the model name when not explicitly supplied."""
        return detect_provider(model, use_vertex_ai=use_vertex_ai)

    def _resolve_provider_and_model(
        self, model: str, use_vertex_ai: bool = False
    ) -> tuple[str, str]:
        """Resolve a model string into a ``(provider, model)`` pair.

        Recognised ``provider/`` prefixes (``gemini``, ``google``, ``openai``,
        ``gpt``) select the provider and are stripped from the model name.
        Unknown prefixes are kept intact and resolve to the ``openai`` provider
        so OpenAI-compatible / self-hosted models work out of the box.
        """
        return resolve_provider_and_model(model, use_vertex_ai=use_vertex_ai)

    def _create_google_vertex_ai_client(self) -> Any:
        return create_llm_client("google", use_vertex_ai=True)

    def _create_client(
        self: _ProviderAgentLike,
        provider: str,
        base_url: str | None = None,
        use_vertex_ai: bool = False,
    ) -> Any:
        """Create a native SDK client for the selected provider."""
        api_key = self.llm_kwargs.get("api_key")
        if provider == "anthropic":
            # Anthropic's constructors reject the OpenAI-shaped allowlist, so the
            # factory does its own per-backend filtering; pass llm_kwargs through.
            # use_vertex_ai is honoured as a backend selector so the flag means
            # the same thing for Claude as it does for Gemini.
            backend = self.llm_kwargs.get("anthropic_backend")
            if backend is None and use_vertex_ai:
                backend = "vertex"
            return create_llm_client(
                provider,
                base_url=base_url,
                api_key=api_key,
                anthropic_backend=backend,
                **{
                    k: v
                    for k, v in self.llm_kwargs.items()
                    if k not in ("api_key", "anthropic_backend", "base_url")
                },
            )

        extra = {k: v for k, v in self.llm_kwargs.items() if k in CLIENT_CONSTRUCTOR_KWARGS}
        return create_llm_client(
            provider,
            use_vertex_ai=use_vertex_ai,
            base_url=base_url,
            api_key=api_key,
            **extra,
        )
