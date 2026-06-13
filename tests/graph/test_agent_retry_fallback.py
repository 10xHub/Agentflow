"""Tests for Agent retry + exponential back-off + fallback model functionality.

Covers:
- RetryConfig defaults and custom values
- Agent.__init__ with retry_config / fallback_models
- _is_retryable_error detection (status codes, connection errors, string matching)
- _extract_status_code from various exception types
- _call_llm_with_retry: retry loop, exponential backoff delays, fallback model switching
- Fast-path (no retry config)
- Non-retryable errors skip retry loop
- Cross-provider fallback (e.g. google → openai)
- All retries and fallbacks exhausted raises last exception
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentflow.core.graph.agent import Agent
from agentflow.core.graph.agent_internal.circuit_breaker import CircuitBreakerOpenError
from agentflow.core.graph.agent_internal.constants import DEFAULT_RETRY_CONFIG, RetryConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chat_response(content: str = "Hello") -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(role="assistant", content=content, tool_calls=None),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=5,
            completion_tokens=10,
            total_tokens=15,
            completion_tokens_details=None,
        ),
        model="gpt-4o",
        id="chatcmpl-1",
        model_dump=MagicMock(return_value={"id": "chatcmpl-1"}),
    )


def _mock_openai_client() -> MagicMock:
    client = MagicMock()
    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=_chat_response())
    client.responses = MagicMock()
    client.responses.create = AsyncMock(return_value=MagicMock())
    client.images = MagicMock()
    client.images.generate = AsyncMock(return_value=MagicMock())
    client.audio = MagicMock()
    client.audio.speech = MagicMock()
    client.audio.speech.create = AsyncMock(return_value=MagicMock())
    return client


def _make_agent(
    model: str = "gpt-4o",
    provider: str = "openai",
    retry_config: RetryConfig | bool | None = True,
    fallback_models: list | None = None,
    **kwargs,
) -> Agent:
    """Create an Agent with mocked client and the specified retry/fallback config."""
    mock_client = _mock_openai_client()
    kwargs.setdefault("reasoning_config", None)
    with patch.object(Agent, "_create_client", return_value=mock_client):
        agent = Agent(
            model=model,
            provider=provider,
            retry_config=retry_config,
            fallback_models=fallback_models,
            **kwargs,
        )
    agent.client = mock_client
    return agent


class _FakeAPIStatusError(Exception):
    """Mimics openai.APIStatusError with a status_code attribute."""

    def __init__(self, status_code: int, message: str = "error"):
        self.status_code = status_code
        super().__init__(message)


class _FakeGoogleError(Exception):
    """Mimics a Google GenAI error with a code attribute."""

    def __init__(self, code: int, message: str = "error"):
        self.code = code
        super().__init__(message)


class _FakeServiceUnavailableError(Exception):
    """Error whose string contains 503."""

    def __init__(self):
        super().__init__(
            "503 Service Unavailable. This model is currently experiencing high demand."
        )


# ═════════════════════════════════════════════════════════════════════════════
# RetryConfig
# ═════════════════════════════════════════════════════════════════════════════


class TestRetryConfig:
    def test_defaults(self):
        cfg = RetryConfig()
        assert cfg.max_retries == 3
        assert cfg.initial_delay == 1.0
        assert cfg.max_delay == 30.0
        assert cfg.backoff_factor == 2.0
        assert 503 in cfg.retryable_status_codes
        assert 429 in cfg.retryable_status_codes

    def test_custom_values(self):
        cfg = RetryConfig(max_retries=5, initial_delay=0.5, max_delay=60.0, backoff_factor=3.0)
        assert cfg.max_retries == 5
        assert cfg.initial_delay == 0.5
        assert cfg.max_delay == 60.0
        assert cfg.backoff_factor == 3.0

    def test_custom_status_codes(self):
        cfg = RetryConfig(retryable_status_codes=frozenset({408, 502}))
        assert 408 in cfg.retryable_status_codes
        assert 503 not in cfg.retryable_status_codes

    def test_frozen(self):
        cfg = RetryConfig()
        with pytest.raises(AttributeError):
            cfg.max_retries = 10  # type: ignore[misc]

    def test_default_retry_config_constant(self):
        assert DEFAULT_RETRY_CONFIG.max_retries == 3


# ═════════════════════════════════════════════════════════════════════════════
# Agent.__init__ retry/fallback config
# ═════════════════════════════════════════════════════════════════════════════


class TestAgentRetryInit:
    def test_default_retry_enabled(self):
        agent = _make_agent()
        assert agent.retry_config is not None
        assert agent.retry_config.max_retries == 3

    def test_retry_true_uses_defaults(self):
        agent = _make_agent(retry_config=True)
        assert agent.retry_config == DEFAULT_RETRY_CONFIG

    def test_retry_false_disables(self):
        agent = _make_agent(retry_config=False)
        assert agent.retry_config is None

    def test_retry_none_disables(self):
        agent = _make_agent(retry_config=None)
        assert agent.retry_config is None

    def test_custom_retry_config(self):
        cfg = RetryConfig(max_retries=7)
        agent = _make_agent(retry_config=cfg)
        assert agent.retry_config is cfg
        assert agent.retry_config.max_retries == 7

    def test_no_fallback_models_default(self):
        agent = _make_agent()
        assert agent.fallback_models == []

    def test_fallback_models_string_entries(self):
        agent = _make_agent(fallback_models=["gpt-4o-mini", "gpt-3.5-turbo"])
        assert agent.fallback_models == [
            ("gpt-4o-mini", None),
            ("gpt-3.5-turbo", None),
        ]

    def test_fallback_models_tuple_entries(self):
        agent = _make_agent(fallback_models=[("gemini-2.0-flash", "google")])
        assert agent.fallback_models == [("gemini-2.0-flash", "google")]

    def test_fallback_models_mixed(self):
        agent = _make_agent(fallback_models=["gpt-4o-mini", ("gemini-2.0-flash", "google")])
        assert agent.fallback_models == [
            ("gpt-4o-mini", None),
            ("gemini-2.0-flash", "google"),
        ]


# ═════════════════════════════════════════════════════════════════════════════
# _extract_status_code
# ═════════════════════════════════════════════════════════════════════════════


class TestExtractStatusCode:
    def test_openai_api_status_error(self):
        agent = _make_agent()
        exc = _FakeAPIStatusError(503)
        assert agent._extract_status_code(exc) == 503

    def test_google_error_code(self):
        agent = _make_agent()
        exc = _FakeGoogleError(429)
        assert agent._extract_status_code(exc) == 429

    def test_string_fallback_503(self):
        agent = _make_agent()
        exc = _FakeServiceUnavailableError()
        assert agent._extract_status_code(exc) == 503

    def test_generic_error_no_code(self):
        agent = _make_agent()
        exc = ValueError("something went wrong")
        assert agent._extract_status_code(exc) is None

    def test_string_fallback_429(self):
        agent = _make_agent()
        exc = Exception("Rate limited: 429 Too Many Requests")
        assert agent._extract_status_code(exc) == 429

    def test_non_integer_code_returns_none(self):
        agent = _make_agent()
        exc = Exception("some error")
        exc.code = "UNKNOWN"
        assert agent._extract_status_code(exc) is None

    def test_none_code_returns_none(self):
        agent = _make_agent()
        exc = Exception("some error")
        exc.code = None
        assert agent._extract_status_code(exc) is None

    def test_string_with_no_code_returns_none(self):
        agent = _make_agent()
        exc = Exception("Something went wrong without a status code")
        assert agent._extract_status_code(exc) is None


# ═════════════════════════════════════════════════════════════════════════════
# _is_retryable_error
# ═════════════════════════════════════════════════════════════════════════════


class TestIsRetryableError:
    def setup_method(self):
        self.agent = _make_agent()
        self.cfg = DEFAULT_RETRY_CONFIG

    def test_503_status_code_is_retryable(self):
        exc = _FakeAPIStatusError(503)
        assert self.agent._is_retryable_error(exc, self.cfg) is True

    def test_429_status_code_is_retryable(self):
        exc = _FakeAPIStatusError(429)
        assert self.agent._is_retryable_error(exc, self.cfg) is True

    def test_500_status_code_is_retryable(self):
        exc = _FakeAPIStatusError(500)
        assert self.agent._is_retryable_error(exc, self.cfg) is True

    def test_400_is_not_retryable(self):
        exc = _FakeAPIStatusError(400)
        assert self.agent._is_retryable_error(exc, self.cfg) is False

    def test_401_is_not_retryable(self):
        exc = _FakeAPIStatusError(401)
        assert self.agent._is_retryable_error(exc, self.cfg) is False

    def test_connection_error_is_retryable(self):
        exc = ConnectionError("Connection refused")
        assert self.agent._is_retryable_error(exc, self.cfg) is True

    def test_timeout_error_is_retryable(self):
        exc = TimeoutError("Request timed out")
        assert self.agent._is_retryable_error(exc, self.cfg) is True

    def test_os_error_is_retryable(self):
        exc = OSError("Network unreachable")
        assert self.agent._is_retryable_error(exc, self.cfg) is True

    def test_string_503_in_message(self):
        exc = _FakeServiceUnavailableError()
        assert self.agent._is_retryable_error(exc, self.cfg) is True

    def test_regular_value_error_not_retryable(self):
        exc = ValueError("invalid argument")
        assert self.agent._is_retryable_error(exc, self.cfg) is False

    def test_custom_status_codes(self):
        cfg = RetryConfig(retryable_status_codes=frozenset({408}))
        exc = _FakeAPIStatusError(408)
        assert self.agent._is_retryable_error(exc, cfg) is True
        exc2 = _FakeAPIStatusError(503)
        assert self.agent._is_retryable_error(exc2, cfg) is False


# ═════════════════════════════════════════════════════════════════════════════
# _call_llm_with_retry
# ═════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
class TestCallLLMWithRetryNoRetryConfig:
    """When retry_config is None and no fallbacks, should be a straight pass-through."""

    async def test_fast_path_no_retry(self):
        agent = _make_agent(retry_config=None, fallback_models=None)
        response = _chat_response("fast")
        agent._call_llm = AsyncMock(return_value=response)

        result = await agent._call_llm_with_retry(
            [{"role": "user", "content": "Hi"}],
        )
        assert result is response
        agent._call_llm.assert_awaited_once()

    async def test_fast_path_raises_without_catching(self):
        agent = _make_agent(retry_config=None, fallback_models=None)
        agent._call_llm = AsyncMock(side_effect=_FakeAPIStatusError(503))

        with pytest.raises(_FakeAPIStatusError):
            await agent._call_llm_with_retry(
                [{"role": "user", "content": "Hi"}],
            )
        agent._call_llm.assert_awaited_once()


@pytest.mark.asyncio
class TestCallLLMWithRetrySuccess:
    async def test_succeeds_on_first_attempt(self):
        agent = _make_agent(retry_config=RetryConfig(max_retries=3))
        response = _chat_response("ok")
        agent._call_llm = AsyncMock(return_value=response)

        result = await agent._call_llm_with_retry(
            [{"role": "user", "content": "Hi"}],
        )
        assert result is response
        assert agent._call_llm.await_count == 1

    async def test_succeeds_after_retries(self):
        agent = _make_agent(
            retry_config=RetryConfig(max_retries=3, initial_delay=0.01),
        )
        response = _chat_response("recovered")
        agent._call_llm = AsyncMock(
            side_effect=[
                _FakeAPIStatusError(503, "unavailable"),
                _FakeAPIStatusError(503, "unavailable"),
                response,
            ]
        )

        with patch("agentflow.core.graph.agent_internal.execution.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            result = await agent._call_llm_with_retry(
                [{"role": "user", "content": "Hi"}],
            )

        assert result is response
        assert agent._call_llm.await_count == 3
        assert mock_sleep.await_count == 2


@pytest.mark.asyncio
class TestCallLLMWithRetryExponentialBackoff:
    async def test_backoff_delays_are_correct(self):
        """Verify that delays follow initial_delay * backoff_factor^retry pattern."""
        cfg = RetryConfig(
            max_retries=4,
            initial_delay=1.0,
            backoff_factor=2.0,
            max_delay=100.0,
        )
        agent = _make_agent(retry_config=cfg)
        agent._call_llm = AsyncMock(
            side_effect=[
                _FakeAPIStatusError(503),
                _FakeAPIStatusError(503),
                _FakeAPIStatusError(503),
                _FakeAPIStatusError(503),
                _chat_response("finally"),
            ]
        )

        delays: list[float] = []

        async def capture_delay(d):
            delays.append(d)

        with patch("agentflow.core.graph.agent_internal.execution.asyncio.sleep", side_effect=capture_delay):
            result = await agent._call_llm_with_retry(
                [{"role": "user", "content": "Hi"}],
            )

        assert result.choices[0].message.content == "finally"
        # Expected delays: 1*2^0=1, 1*2^1=2, 1*2^2=4, 1*2^3=8
        assert delays == [1.0, 2.0, 4.0, 8.0]

    async def test_backoff_capped_by_max_delay(self):
        cfg = RetryConfig(
            max_retries=5,
            initial_delay=10.0,
            backoff_factor=3.0,
            max_delay=50.0,
        )
        agent = _make_agent(retry_config=cfg)
        agent._call_llm = AsyncMock(
            side_effect=[
                _FakeAPIStatusError(503),
                _FakeAPIStatusError(503),
                _FakeAPIStatusError(503),
                _FakeAPIStatusError(503),
                _FakeAPIStatusError(503),
                _chat_response("ok"),
            ]
        )

        delays: list[float] = []

        async def capture_delay(d):
            delays.append(d)

        with patch("agentflow.core.graph.agent_internal.execution.asyncio.sleep", side_effect=capture_delay):
            await agent._call_llm_with_retry(
                [{"role": "user", "content": "Hi"}],
            )

        # 10*3^0=10, 10*3^1=30, 10*3^2=90→50, 10*3^3=270→50, 10*3^4=2430→50
        assert delays == [10.0, 30.0, 50.0, 50.0, 50.0]


@pytest.mark.asyncio
class TestCallLLMWithRetryExhausted:
    async def test_raises_last_error_when_all_retries_exhausted(self):
        cfg = RetryConfig(max_retries=2, initial_delay=0.01)
        agent = _make_agent(retry_config=cfg, fallback_models=None)
        agent._call_llm = AsyncMock(
            side_effect=_FakeAPIStatusError(503, "always_unavailable"),
        )

        with patch("agentflow.core.graph.agent_internal.execution.asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(_FakeAPIStatusError, match="always_unavailable"):
                await agent._call_llm_with_retry(
                    [{"role": "user", "content": "Hi"}],
                )

        # 1 initial + 2 retries = 3
        assert agent._call_llm.await_count == 3


@pytest.mark.asyncio
class TestCallLLMWithRetryNonRetryable:
    async def test_non_retryable_error_does_not_retry(self):
        cfg = RetryConfig(max_retries=3, initial_delay=0.01)
        agent = _make_agent(retry_config=cfg, fallback_models=None)
        agent._call_llm = AsyncMock(
            side_effect=_FakeAPIStatusError(400, "bad_request"),
        )

        with pytest.raises(_FakeAPIStatusError, match="bad_request"):
            await agent._call_llm_with_retry(
                [{"role": "user", "content": "Hi"}],
            )

        # Should NOT retry — only 1 attempt
        assert agent._call_llm.await_count == 1


# ═════════════════════════════════════════════════════════════════════════════
# Fallback models
# ═════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
class TestCallLLMFallbackModels:
    async def test_fallback_model_succeeds_after_primary_exhausted(self):
        cfg = RetryConfig(max_retries=1, initial_delay=0.01)
        agent = _make_agent(
            retry_config=cfg,
            fallback_models=["gpt-4o-mini"],
        )

        primary_error = _FakeAPIStatusError(503, "primary_down")
        fallback_response = _chat_response("from_fallback")

        call_count = 0
        original_model = agent.model

        async def mock_call_llm(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            # First 2 calls are primary (1 + 1 retry)
            if agent.model == original_model:
                raise primary_error
            # Fallback model should succeed
            return fallback_response

        agent._call_llm = AsyncMock(side_effect=mock_call_llm)

        with patch("agentflow.core.graph.agent_internal.execution.asyncio.sleep", new_callable=AsyncMock):
            with patch.object(Agent, "_create_client", return_value=_mock_openai_client()):
                result = await agent._call_llm_with_retry(
                    [{"role": "user", "content": "Hi"}],
                )

        assert result is fallback_response
        # 2 primary attempts + 1 fallback attempt = 3
        assert agent._call_llm.await_count == 3
        # Verify model/provider is restored after fallback
        assert agent.model == original_model

    async def test_cross_provider_fallback(self):
        """Test fallback from OpenAI to Google provider."""
        cfg = RetryConfig(max_retries=0, initial_delay=0.01)
        agent = _make_agent(
            model="gpt-4o",
            provider="openai",
            retry_config=cfg,
            fallback_models=[("gemini-2.0-flash", "google")],
        )

        fallback_response = MagicMock(name="google_response")

        call_count = 0

        async def mock_call_llm(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if agent.provider == "openai":
                raise _FakeAPIStatusError(503, "openai_down")
            return fallback_response

        agent._call_llm = AsyncMock(side_effect=mock_call_llm)

        with patch.object(Agent, "_create_client", return_value=MagicMock()):
            result = await agent._call_llm_with_retry(
                [{"role": "user", "content": "Hi"}],
            )

        assert result is fallback_response
        # 1 primary + 1 fallback
        assert agent._call_llm.await_count == 2
        # Originals restored
        assert agent.model == "gpt-4o"
        assert agent.provider == "openai"

    async def test_all_fallbacks_exhausted_raises(self):
        cfg = RetryConfig(max_retries=1, initial_delay=0.01)
        agent = _make_agent(
            retry_config=cfg,
            fallback_models=["gpt-4o-mini"],
        )

        agent._call_llm = AsyncMock(
            side_effect=_FakeAPIStatusError(503, "all_down"),
        )

        with patch("agentflow.core.graph.agent_internal.execution.asyncio.sleep", new_callable=AsyncMock):
            with patch.object(Agent, "_create_client", return_value=_mock_openai_client()):
                with pytest.raises(_FakeAPIStatusError, match="all_down"):
                    await agent._call_llm_with_retry(
                        [{"role": "user", "content": "Hi"}],
                    )

        # primary: 2 attempts + fallback: 2 attempts = 4
        assert agent._call_llm.await_count == 4

    async def test_model_and_provider_restored_on_fallback_failure(self):
        cfg = RetryConfig(max_retries=0, initial_delay=0.01)
        agent = _make_agent(
            model="gpt-4o",
            provider="openai",
            retry_config=cfg,
            fallback_models=[("gemini-2.0-flash", "google")],
        )

        agent._call_llm = AsyncMock(
            side_effect=_FakeAPIStatusError(503, "down"),
        )

        with patch.object(Agent, "_create_client", return_value=MagicMock()):
            with pytest.raises(_FakeAPIStatusError):
                await agent._call_llm_with_retry(
                    [{"role": "user", "content": "Hi"}],
                )

        # Verify originals are restored even after failure
        assert agent.model == "gpt-4o"
        assert agent.provider == "openai"


@pytest.mark.asyncio
class TestCallLLMFallbackChain:
    async def test_multiple_fallbacks_tried_in_order(self):
        """When primary and first fallback fail, second fallback should succeed."""
        cfg = RetryConfig(max_retries=0, initial_delay=0.01)
        agent = _make_agent(
            model="primary-model",
            provider="openai",
            retry_config=cfg,
            fallback_models=["fallback-1", "fallback-2"],
        )

        models_tried: list[str] = []
        response = _chat_response("from_fallback_2")

        async def mock_call_llm(*args, **kwargs):
            models_tried.append(agent.model)
            if agent.model in ("primary-model", "fallback-1"):
                raise _FakeAPIStatusError(503, "down")
            return response

        agent._call_llm = AsyncMock(side_effect=mock_call_llm)

        with patch.object(Agent, "_create_client", return_value=_mock_openai_client()):
            result = await agent._call_llm_with_retry(
                [{"role": "user", "content": "Hi"}],
            )

        assert result is response
        assert models_tried == ["primary-model", "fallback-1", "fallback-2"]

    async def test_non_retryable_on_primary_skips_to_fallback(self):
        """A 400 error on primary should skip retries but still try fallbacks."""
        cfg = RetryConfig(max_retries=3, initial_delay=0.01)
        agent = _make_agent(
            model="primary",
            provider="openai",
            retry_config=cfg,
            fallback_models=["fallback-ok"],
        )

        response = _chat_response("fallback_ok")

        async def mock_call_llm(*args, **kwargs):
            if agent.model == "primary":
                raise _FakeAPIStatusError(400, "bad_request")
            return response

        agent._call_llm = AsyncMock(side_effect=mock_call_llm)

        with patch.object(Agent, "_create_client", return_value=_mock_openai_client()):
            result = await agent._call_llm_with_retry(
                [{"role": "user", "content": "Hi"}],
            )

        assert result is response
        # 1 primary (no retries for 400) + 1 fallback = 2
        assert agent._call_llm.await_count == 2


# ═════════════════════════════════════════════════════════════════════════════
# Fallback-only (no retry_config)
# ═════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
class TestFallbackOnlyNoRetry:
    async def test_fallback_with_retry_none_still_works(self):
        """Fallback should work even when retry_config=None (0 retries per model)."""
        agent = _make_agent(
            retry_config=None,
            fallback_models=["fallback-model"],
        )

        response = _chat_response("fallback_no_retry")

        async def mock_call_llm(*args, **kwargs):
            if agent.model == "gpt-4o":
                raise _FakeAPIStatusError(503, "primary_down")
            return response

        agent._call_llm = AsyncMock(side_effect=mock_call_llm)

        with patch.object(Agent, "_create_client", return_value=_mock_openai_client()):
            result = await agent._call_llm_with_retry(
                [{"role": "user", "content": "Hi"}],
            )

        assert result is response
        # 1 primary + 1 fallback = 2 (no retries since retry_config is None)
        assert agent._call_llm.await_count == 2


# ═════════════════════════════════════════════════════════════════════════════
# Connection and timeout errors
# ═════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
class TestConnectionErrors:
    async def test_connection_error_retried(self):
        cfg = RetryConfig(max_retries=2, initial_delay=0.01)
        agent = _make_agent(retry_config=cfg)
        response = _chat_response("recovered")
        agent._call_llm = AsyncMock(
            side_effect=[
                ConnectionError("Connection refused"),
                response,
            ]
        )

        with patch("agentflow.core.graph.agent_internal.execution.asyncio.sleep", new_callable=AsyncMock):
            result = await agent._call_llm_with_retry(
                [{"role": "user", "content": "Hi"}],
            )

        assert result is response
        assert agent._call_llm.await_count == 2

    async def test_timeout_error_retried(self):
        cfg = RetryConfig(max_retries=2, initial_delay=0.01)
        agent = _make_agent(retry_config=cfg)
        response = _chat_response("recovered")
        agent._call_llm = AsyncMock(
            side_effect=[
                TimeoutError("timed out"),
                response,
            ]
        )

        with patch("agentflow.core.graph.agent_internal.execution.asyncio.sleep", new_callable=AsyncMock):
            result = await agent._call_llm_with_retry(
                [{"role": "user", "content": "Hi"}],
            )

        assert result is response


# ═════════════════════════════════════════════════════════════════════════════
# Google-specific retryable error
# ═════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
class TestGoogleRetryable:
    async def test_google_503_retried(self):
        cfg = RetryConfig(max_retries=2, initial_delay=0.01)
        agent = _make_agent(retry_config=cfg)
        response = _chat_response("google_ok")

        google_error = _FakeGoogleError(503, "high demand")
        agent._call_llm = AsyncMock(
            side_effect=[google_error, response],
        )

        with patch("agentflow.core.graph.agent_internal.execution.asyncio.sleep", new_callable=AsyncMock):
            result = await agent._call_llm_with_retry(
                [{"role": "user", "content": "Hi"}],
            )

        assert result is response
        assert agent._call_llm.await_count == 2

    async def test_google_string_503_retried(self):
        """The exact error from the issue description should be retried."""
        cfg = RetryConfig(max_retries=2, initial_delay=0.01)
        agent = _make_agent(retry_config=cfg)
        response = _chat_response("recovered")

        error = _FakeServiceUnavailableError()
        agent._call_llm = AsyncMock(
            side_effect=[error, response],
        )

        with patch("agentflow.core.graph.agent_internal.execution.asyncio.sleep", new_callable=AsyncMock):
            result = await agent._call_llm_with_retry(
                [{"role": "user", "content": "Hi"}],
            )

        assert result is response


# ═════════════════════════════════════════════════════════════════════════════
# Circuit breaker (opt-in via RetryConfig)
# ═════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
class TestCircuitBreakerIntegration:
    async def test_disabled_by_default_primary_retried_every_call(self):
        """Without circuit_breaker_enabled, a dead primary is retried every call."""
        cfg = RetryConfig(max_retries=0, initial_delay=0.01)
        agent = _make_agent(retry_config=cfg, fallback_models=["gpt-4o-mini"])
        original_model = agent.model
        primary_calls = 0

        async def mock_call_llm(*args, **kwargs):
            nonlocal primary_calls
            if agent.model == original_model:
                primary_calls += 1
                raise _FakeAPIStatusError(503, "primary_down")
            return _chat_response("fallback")

        agent._call_llm = AsyncMock(side_effect=mock_call_llm)
        with patch("agentflow.core.graph.agent_internal.execution.asyncio.sleep", new_callable=AsyncMock):
            for _ in range(4):
                await agent._call_llm_with_retry([{"role": "user", "content": "Hi"}])

        # Primary is attempted on every one of the 4 invocations.
        assert primary_calls == 4

    async def test_open_circuit_skips_dead_primary(self):
        """After threshold failures the primary is skipped, going straight to fallback."""
        cfg = RetryConfig(
            max_retries=0,
            initial_delay=0.01,
            circuit_breaker_enabled=True,
            circuit_breaker_threshold=2,
        )
        agent = _make_agent(retry_config=cfg, fallback_models=["gpt-4o-mini"])
        original_model = agent.model
        primary_calls = 0
        fallback_response = _chat_response("fallback")

        async def mock_call_llm(*args, **kwargs):
            nonlocal primary_calls
            if agent.model == original_model:
                primary_calls += 1
                raise _FakeAPIStatusError(503, "primary_down")
            return fallback_response

        agent._call_llm = AsyncMock(side_effect=mock_call_llm)
        with patch("agentflow.core.graph.agent_internal.execution.asyncio.sleep", new_callable=AsyncMock):
            results = [
                await agent._call_llm_with_retry([{"role": "user", "content": "Hi"}])
                for _ in range(4)
            ]

        # Primary fails on invocations 1 and 2 (opening the circuit at 2), then is
        # skipped on 3 and 4. Every call still succeeds via the fallback.
        assert primary_calls == 2
        assert all(r is fallback_response for r in results)

    async def test_all_circuits_open_raises_circuit_breaker_error(self):
        """If every model's circuit is open, the recorded CircuitBreakerOpenError is raised."""
        cfg = RetryConfig(
            max_retries=0,
            initial_delay=0.01,
            circuit_breaker_enabled=True,
            circuit_breaker_threshold=1,
        )
        agent = _make_agent(retry_config=cfg, fallback_models=None)
        agent._call_llm = AsyncMock(side_effect=_FakeAPIStatusError(503, "down"))

        with patch("agentflow.core.graph.agent_internal.execution.asyncio.sleep", new_callable=AsyncMock):
            # First call fails normally and opens the circuit (threshold=1).
            with pytest.raises(_FakeAPIStatusError):
                await agent._call_llm_with_retry([{"role": "user", "content": "Hi"}])
            # Second call is short-circuited.
            with pytest.raises(CircuitBreakerOpenError):
                await agent._call_llm_with_retry([{"role": "user", "content": "Hi"}])
