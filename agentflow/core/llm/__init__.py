"""LLM client creation utilities shared across agents and evaluators."""

from .anthropic_batch import AnthropicBatch, BatchResult
from .caller import call_llm
from .client_factory import (
    DEFAULT_LLM_TIMEOUT_SECONDS,
    create_llm_client,
    detect_provider,
    get_default_llm_timeout,
    resolve_provider_and_model,
    set_default_llm_timeout,
)


__all__ = [
    "DEFAULT_LLM_TIMEOUT_SECONDS",
    "AnthropicBatch",
    "BatchResult",
    "call_llm",
    "create_llm_client",
    "detect_provider",
    "get_default_llm_timeout",
    "resolve_provider_and_model",
    "set_default_llm_timeout",
]
