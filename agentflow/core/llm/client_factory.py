"""
Shared LLM client factory.

Single place where provider detection and client construction live.
Used by both the Agent class and the evaluation judge.
"""

from __future__ import annotations

import logging
import os
from typing import Any


logger = logging.getLogger("agentflow.llm")

# Default timeout (in seconds) applied to LLM client construction when the
# caller does not pass an explicit ``timeout``. Bounds every request so a stalled
# provider connection cannot hang a graph run indefinitely. Override globally via
# the ``AGENTFLOW_LLM_TIMEOUT`` environment variable (seconds) or programmatically
# via :func:`set_default_llm_timeout`.
DEFAULT_LLM_TIMEOUT_SECONDS = 600.0

# Single-element holder so the override can be mutated without a ``global``
# statement (which ruff's PLW0603 flags).
_default_timeout_override: dict[str, float | None] = {"value": None}


def _env_timeout() -> float | None:
    """Read ``AGENTFLOW_LLM_TIMEOUT`` (seconds), or None if unset/invalid."""
    raw = os.getenv("AGENTFLOW_LLM_TIMEOUT")
    if raw is None or not raw.strip():
        return None
    try:
        value = float(raw)
    except ValueError:
        logger.warning(
            "Invalid AGENTFLOW_LLM_TIMEOUT=%r; expected a number of seconds. Ignoring.",
            raw,
        )
        return None
    if value <= 0:
        logger.warning("AGENTFLOW_LLM_TIMEOUT=%s must be positive. Ignoring.", value)
        return None
    return value


def get_default_llm_timeout() -> float:
    """Return the default LLM request timeout in seconds.

    Resolution order (first match wins):

    1. A programmatic override set via :func:`set_default_llm_timeout`.
    2. The ``AGENTFLOW_LLM_TIMEOUT`` environment variable (seconds).
    3. :data:`DEFAULT_LLM_TIMEOUT_SECONDS`.
    """
    override = _default_timeout_override["value"]
    if override is not None:
        return override
    env = _env_timeout()
    if env is not None:
        return env
    return DEFAULT_LLM_TIMEOUT_SECONDS


def set_default_llm_timeout(seconds: float | None) -> None:
    """Globally override the default LLM request timeout, in seconds.

    Pass ``None`` to clear the override and fall back to the
    ``AGENTFLOW_LLM_TIMEOUT`` environment variable / built-in default.

    Raises:
        ValueError: If ``seconds`` is not a positive number.
    """
    if seconds is not None and seconds <= 0:
        raise ValueError("LLM timeout must be a positive number of seconds.")
    _default_timeout_override["value"] = seconds


# Recognised ``provider/`` prefixes mapped to the concrete provider the client
# factory can build. Anything not listed here is an unknown prefix and resolves
# to ``"openai"`` (the OpenAI SDK is used for OpenAI-compatible endpoints).
_PROVIDER_PREFIXES = {
    "gemini": "google",
    "google": "google",
    "openai": "openai",
    "gpt": "openai",
    "anthropic": "anthropic",
    "claude": "anthropic",
}

# Keys allowed in the AsyncOpenAI constructor but NOT in per-request calls.
_CLIENT_CONSTRUCTOR_KWARGS = frozenset(
    {
        "organization",
        "project",
        "timeout",
        "max_retries",
        "default_headers",
        "default_query",
        "http_client",
    }
)

# Anthropic's constructors do not accept ``organization`` / ``project``, so the
# OpenAI-shaped allowlist above cannot be reused. ``http_client`` here must be an
# httpx2 client: the anthropic SDK moved to httpx2 in 1.0, while the OpenAI SDK
# still uses httpx. Passing the wrong one fails at request time.
_ANTHROPIC_CONSTRUCTOR_KWARGS = frozenset(
    {
        "timeout",
        "max_retries",
        "default_headers",
        "default_query",
        "http_client",
        "auth_token",
        "middleware",
    }
)
_ANTHROPIC_VERTEX_CONSTRUCTOR_KWARGS = _ANTHROPIC_CONSTRUCTOR_KWARGS | frozenset(
    {
        "region",
        "project_id",
        "access_token",
        "credentials",
    }
)
_ANTHROPIC_BEDROCK_CONSTRUCTOR_KWARGS = _ANTHROPIC_CONSTRUCTOR_KWARGS | frozenset(
    {
        "aws_access_key",
        "aws_secret_key",
        "aws_session_token",
        "aws_region",
        "aws_profile",
        "api_key",
    }
)


def detect_provider(model: str, use_vertex_ai: bool = False) -> str:
    """Infer the provider from a model name.

    Args:
        model: Model identifier, optionally prefixed with ``"provider/"``
            (e.g. ``"gemini/gemini-2.5-flash"``, ``"openai/gpt-4o"``).
        use_vertex_ai: When True, selects ``"google"`` unless the model is
            clearly a Claude model. Claude runs on Vertex too, so the flag acts
            as a backend selector there rather than a provider override.

    Returns:
        One of ``"google"``, ``"openai"``, or ``"anthropic"``.
    """
    if "/" in model:
        prefix = model.split("/", 1)[0].lower()
        if prefix in _PROVIDER_PREFIXES:
            return _PROVIDER_PREFIXES[prefix]
        # Unknown prefix — fall through to name-based detection using the suffix
        model = model.split("/", 1)[1]

    # Checked before the use_vertex_ai short-circuit: Claude runs on Vertex too,
    # and use_vertex_ai is the flag a caller naturally reaches for. Letting the
    # flag force "google" here would build a Google client for a Claude model and
    # fail at request time with a confusing error.
    if model.lower().startswith(("claude-", "anthropic.")):
        return "anthropic"

    if use_vertex_ai:
        return "google"

    lower = model.lower()
    if lower.startswith(("gemini-", "imagen-", "veo-", "chirp")):
        return "google"
    if lower.startswith(("gpt-", "o1-", "o3-", "o4-")):
        return "openai"

    logger.info(
        "Could not auto-detect provider for model '%s'. Defaulting to 'openai'.",
        model,
    )
    return "openai"


def resolve_provider_and_model(model: str, use_vertex_ai: bool = False) -> tuple[str, str]:
    """Resolve a model string into a concrete ``(provider, model)`` pair.

    Unlike :func:`detect_provider`, this also returns the model name that should
    be sent to the provider. A *recognised* ``provider/`` prefix (e.g.
    ``"gemini/..."``, ``"openai/..."``) is stripped, since the provider is
    selected from the prefix. An *unrecognised* prefix is kept intact: it may be
    an OpenAI-compatible / HuggingFace-style identifier (e.g.
    ``"meta-llama/Llama-3-70b"``) where the slash is part of the real model name.
    Such models always resolve to the ``"openai"`` provider.

    Args:
        model: Model identifier, optionally prefixed with ``"provider/"``.
        use_vertex_ai: When True, selects ``"google"`` unless the model is
            clearly a Claude model (see :func:`detect_provider`).

    Returns:
        A ``(provider, model)`` tuple where provider is ``"google"``,
        ``"openai"``, or ``"anthropic"``.
    """
    if "/" in model:
        prefix, rest = model.split("/", 1)
        if prefix.lower() in _PROVIDER_PREFIXES:
            return detect_provider(model, use_vertex_ai=use_vertex_ai), rest

    return detect_provider(model, use_vertex_ai=use_vertex_ai), model


def create_llm_client(
    provider: str,
    *,
    use_vertex_ai: bool = False,
    base_url: str | None = None,
    api_key: str | None = None,
    anthropic_backend: str | None = None,
    **extra_kwargs: Any,
) -> Any:
    """Create a native async SDK client for the given provider.

    Args:
        provider: ``"google"``, ``"openai"``, or ``"anthropic"``.
        use_vertex_ai: When True and provider is ``"google"``, creates a
            Vertex AI client using ``GOOGLE_CLOUD_PROJECT`` / ``GOOGLE_CLOUD_LOCATION``.
            Google-specific; use ``anthropic_backend`` for Anthropic on Vertex.
        base_url: Custom base URL for OpenAI-compatible APIs (ollama, vllm, …).
        api_key: Explicit API key. Falls back to env vars when omitted.
        anthropic_backend: Backend selector for the Anthropic provider.
            ``None`` (direct Claude API), ``"vertex"``, or ``"bedrock"``.
        **extra_kwargs: Forwarded to the AsyncOpenAI constructor
            (only recognised constructor keys are passed through).

    Returns:
        An async-capable SDK client instance.

    Raises:
        ImportError: If the required SDK is not installed.
        ValueError: If required configuration (project, api_key) is missing.
    """
    if provider == "google":
        return _create_google_client(use_vertex_ai=use_vertex_ai)

    if provider == "openai":
        return _create_openai_client(
            base_url=base_url,
            api_key=api_key,
            **extra_kwargs,
        )

    if provider == "anthropic":
        return _create_anthropic_client(
            base_url=base_url,
            api_key=api_key,
            anthropic_backend=anthropic_backend,
            **extra_kwargs,
        )

    raise ValueError(
        f"Unsupported provider: '{provider}'. Supported: 'google', 'openai', 'anthropic'."
    )


def _create_google_client(*, use_vertex_ai: bool) -> Any:
    try:
        from google import genai
        from google.genai.types import HttpOptions
    except ImportError as exc:
        raise ImportError(
            "google-genai SDK is required for the Google provider. "
            "Install it with: pip install 10xscale-agentflow[google-genai]"
        ) from exc

    # google-genai expresses the request timeout in milliseconds.
    http_options = HttpOptions(timeout=int(get_default_llm_timeout() * 1000))

    if use_vertex_ai:
        project = os.getenv("GOOGLE_CLOUD_PROJECT")
        location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
        if not project:
            raise ValueError("GOOGLE_CLOUD_PROJECT environment variable must be set for Vertex AI.")
        logger.info(
            "Creating Google GenAI client (Vertex AI, project=%s, location=%s)",
            project,
            location,
        )
        return genai.Client(
            vertexai=True, project=project, location=location, http_options=http_options
        )

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY or GOOGLE_API_KEY environment variable must be set "
            "for the Google provider."
        )
    logger.info("Creating Google GenAI client (API key)")
    # Pass vertexai=False explicitly: google-genai's Client() otherwise reads the
    # GOOGLE_GENAI_USE_VERTEXAI env var and silently switches to Vertex mode,
    # which rejects API keys (401 UNAUTHENTICATED). The caller asked for the
    # Developer API (use_vertex_ai=False), so honour that over the env.
    return genai.Client(vertexai=False, api_key=api_key, http_options=http_options)


def _create_openai_client(
    *,
    base_url: str | None,
    api_key: str | None,
    **extra_kwargs: Any,
) -> Any:
    try:
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise ImportError(
            "openai SDK is required for the OpenAI provider. "
            "Install it with: pip install 10xscale-agentflow[openai]"
        ) from exc

    resolved_key = api_key or os.getenv("OPENAI_API_KEY")
    if not resolved_key:
        logger.warning(
            "OPENAI_API_KEY not set. API calls may fail unless using a "
            "custom base_url that doesn't require authentication."
        )

    client_kwargs = {k: v for k, v in extra_kwargs.items() if k in _CLIENT_CONSTRUCTOR_KWARGS}
    # Bound the request unless the caller opted into their own timeout.
    client_kwargs.setdefault("timeout", get_default_llm_timeout())
    if base_url:
        logger.info("Creating OpenAI client with custom base_url: %s", base_url)
        return AsyncOpenAI(api_key=resolved_key, base_url=base_url, **client_kwargs)

    return AsyncOpenAI(api_key=resolved_key, **client_kwargs)


def _create_anthropic_client(
    *,
    base_url: str | None,
    api_key: str | None,
    anthropic_backend: str | None = None,
    **extra_kwargs: Any,
) -> Any:
    """Create an async Anthropic client for the selected backend.

    Anthropic reaches three distinct backends, so unlike Google's boolean
    ``use_vertex_ai`` flag the selector is a string:

    * ``None``      -> ``AsyncAnthropic``              (direct Claude API)
    * ``"vertex"``  -> ``AsyncAnthropicVertex``        (Google Cloud Vertex AI)
    * ``"bedrock"`` -> ``AsyncAnthropicBedrockMantle`` (Amazon Bedrock)

    ``AsyncAnthropicBedrockMantle`` is the Messages-API Bedrock endpoint. The
    plain ``AsyncAnthropicBedrock`` client is the legacy ``InvokeModel`` path and
    is deliberately not used here.

    Region, project, and AWS credentials are resolved by the SDK from the
    environment unless passed explicitly through ``llm_kwargs``.
    """
    backend = (anthropic_backend or "").lower() or None

    if backend not in (None, "vertex", "bedrock"):
        raise ValueError(
            f"Unsupported anthropic_backend: '{anthropic_backend}'. "
            "Supported: None (direct API), 'vertex', 'bedrock'."
        )

    if backend == "vertex":
        try:
            from anthropic import AsyncAnthropicVertex
        except ImportError as exc:
            raise ImportError(
                "anthropic SDK with Vertex support is required for "
                "anthropic_backend='vertex'. Install it with: "
                'pip install "10xscale-agentflow[anthropic-vertex]"'
            ) from exc

        client_kwargs = {
            k: v for k, v in extra_kwargs.items() if k in _ANTHROPIC_VERTEX_CONSTRUCTOR_KWARGS
        }
        client_kwargs.setdefault("timeout", get_default_llm_timeout())
        if base_url:
            client_kwargs["base_url"] = base_url
        logger.info("Creating Anthropic client (Vertex AI)")
        return AsyncAnthropicVertex(**client_kwargs)

    if backend == "bedrock":
        try:
            from anthropic import AsyncAnthropicBedrockMantle
        except ImportError as exc:
            raise ImportError(
                "anthropic SDK with Bedrock support is required for "
                "anthropic_backend='bedrock'. Install it with: "
                'pip install "10xscale-agentflow[anthropic-bedrock]"'
            ) from exc

        client_kwargs = {
            k: v for k, v in extra_kwargs.items() if k in _ANTHROPIC_BEDROCK_CONSTRUCTOR_KWARGS
        }
        client_kwargs.setdefault("timeout", get_default_llm_timeout())
        if api_key and "api_key" not in client_kwargs:
            client_kwargs["api_key"] = api_key
        if base_url:
            client_kwargs["base_url"] = base_url
        logger.info("Creating Anthropic client (Bedrock Mantle)")
        return AsyncAnthropicBedrockMantle(**client_kwargs)

    try:
        from anthropic import AsyncAnthropic
    except ImportError as exc:
        raise ImportError(
            "anthropic SDK is required for the Anthropic provider. "
            'Install it with: pip install "10xscale-agentflow[anthropic]"'
        ) from exc

    resolved_key = api_key or os.getenv("ANTHROPIC_API_KEY")
    if not resolved_key:
        # Not fatal: the SDK also resolves ANTHROPIC_AUTH_TOKEN, an `ant auth
        # login` profile, or Workload Identity Federation. An unset API key does
        # not mean there are no credentials.
        logger.info(
            "ANTHROPIC_API_KEY not set. Falling back to the SDK's own credential "
            "resolution (ANTHROPIC_AUTH_TOKEN, auth profile, or federation)."
        )

    client_kwargs = {k: v for k, v in extra_kwargs.items() if k in _ANTHROPIC_CONSTRUCTOR_KWARGS}
    client_kwargs.setdefault("timeout", get_default_llm_timeout())
    if resolved_key:
        client_kwargs["api_key"] = resolved_key
    if base_url:
        logger.info("Creating Anthropic client with custom base_url: %s", base_url)
        client_kwargs["base_url"] = base_url

    return AsyncAnthropic(**client_kwargs)
