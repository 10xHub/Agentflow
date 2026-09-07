"""Tests for Anthropic provider detection, client construction, and request assembly."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from agentflow.core.graph.agent_internal.anthropic import (
    AgentAnthropicMixin,
    apply_reasoning_config,
    strip_rejected_sampling_params,
)
from agentflow.core.graph.agent_internal.constants import (
    ANTHROPIC_DEFAULT_MAX_TOKENS,
    ANTHROPIC_DEFAULT_MAX_TOKENS_STREAMING,
)
from agentflow.core.llm.client_factory import (
    create_llm_client,
    detect_provider,
    resolve_provider_and_model,
)


class TestProviderDetection:
    @pytest.mark.parametrize(
        "model",
        [
            "claude-opus-5",
            "claude-sonnet-5",
            "claude-haiku-4-5",
            "claude-fable-5",
            "anthropic/claude-opus-5",
            "claude/claude-opus-5",
        ],
    )
    def test_claude_models_resolve_to_anthropic(self, model):
        assert detect_provider(model) == "anthropic"

    def test_bedrock_style_id_resolves_to_anthropic(self):
        assert detect_provider("anthropic.claude-opus-5") == "anthropic"

    def test_bedrock_prefix_is_preserved_in_model_string(self):
        """The ``anthropic.`` prefix is part of the Bedrock model id."""
        provider, model = resolve_provider_and_model("anthropic.claude-opus-5")
        assert provider == "anthropic"
        assert model == "anthropic.claude-opus-5"

    def test_recognised_prefix_is_stripped(self):
        provider, model = resolve_provider_and_model("anthropic/claude-opus-5")
        assert provider == "anthropic"
        assert model == "claude-opus-5"

    @pytest.mark.parametrize(
        ("model", "expected"),
        [
            ("gpt-4o", "openai"),
            ("gemini-2.5-flash", "google"),
            ("openai/gpt-4o", "openai"),
            ("meta-llama/Llama-3-70b", "openai"),
        ],
    )
    def test_other_providers_are_unaffected(self, model, expected):
        assert detect_provider(model) == expected

    def test_vertex_flag_does_not_hijack_claude_models(self):
        """Claude runs on Vertex too. Letting use_vertex_ai force "google" would
        build a Google client for a Claude model and fail at request time."""
        assert detect_provider("claude-opus-5", use_vertex_ai=True) == "anthropic"

    @pytest.mark.parametrize("model", ["gemini-2.5-flash", "gpt-4o", "some-unknown-model"])
    def test_vertex_flag_still_forces_google_for_everything_else(self, model):
        assert detect_provider(model, use_vertex_ai=True) == "google"

    def test_vertex_flag_selects_the_anthropic_vertex_backend(self, monkeypatch):
        """The flag means the same thing for Claude as it does for Gemini."""
        import sys
        from agentflow.core.graph.agent_internal.providers import AgentProviderMixin

        module = SimpleNamespace(
            AsyncAnthropic=MagicMock(),
            AsyncAnthropicVertex=MagicMock(),
            AsyncAnthropicBedrockMantle=MagicMock(),
        )
        monkeypatch.setitem(sys.modules, "anthropic", module)

        agent = AgentProviderMixin()
        agent.llm_kwargs = {}
        agent._create_client("anthropic", None, True)

        assert module.AsyncAnthropicVertex.called
        assert not module.AsyncAnthropic.called

    def test_explicit_backend_beats_the_vertex_flag(self, monkeypatch):
        import sys
        from agentflow.core.graph.agent_internal.providers import AgentProviderMixin

        module = SimpleNamespace(
            AsyncAnthropic=MagicMock(),
            AsyncAnthropicVertex=MagicMock(),
            AsyncAnthropicBedrockMantle=MagicMock(),
        )
        monkeypatch.setitem(sys.modules, "anthropic", module)

        agent = AgentProviderMixin()
        agent.llm_kwargs = {"anthropic_backend": "bedrock"}
        agent._create_client("anthropic", None, True)

        assert module.AsyncAnthropicBedrockMantle.called
        assert not module.AsyncAnthropicVertex.called


class TestClientConstruction:
    """Client selection per backend, with a stubbed anthropic module."""

    @pytest.fixture
    def fake_anthropic(self, monkeypatch):
        module = SimpleNamespace(
            AsyncAnthropic=MagicMock(name="AsyncAnthropic"),
            AsyncAnthropicVertex=MagicMock(name="AsyncAnthropicVertex"),
            AsyncAnthropicBedrockMantle=MagicMock(name="AsyncAnthropicBedrockMantle"),
        )
        monkeypatch.setitem(sys.modules, "anthropic", module)
        return module

    def test_default_backend_builds_direct_client(self, fake_anthropic, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        create_llm_client("anthropic")
        assert fake_anthropic.AsyncAnthropic.called
        assert fake_anthropic.AsyncAnthropic.call_args.kwargs["api_key"] == "sk-test"

    def test_vertex_backend_builds_vertex_client(self, fake_anthropic):
        create_llm_client("anthropic", anthropic_backend="vertex")
        assert fake_anthropic.AsyncAnthropicVertex.called
        assert not fake_anthropic.AsyncAnthropic.called

    def test_bedrock_backend_builds_mantle_client(self, fake_anthropic):
        """Mantle is the Messages-API endpoint; the plain client is legacy."""
        create_llm_client("anthropic", anthropic_backend="bedrock")
        assert fake_anthropic.AsyncAnthropicBedrockMantle.called

    def test_unknown_backend_raises(self, fake_anthropic):
        with pytest.raises(ValueError, match="Unsupported anthropic_backend"):
            create_llm_client("anthropic", anthropic_backend="azure")

    def test_openai_constructor_keys_are_not_forwarded(self, fake_anthropic, monkeypatch):
        """``organization``/``project`` are not accepted by AsyncAnthropic."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        create_llm_client("anthropic", organization="org", project="proj", max_retries=4)
        kwargs = fake_anthropic.AsyncAnthropic.call_args.kwargs
        assert "organization" not in kwargs
        assert "project" not in kwargs
        assert kwargs["max_retries"] == 4

    def test_vertex_accepts_region_and_project(self, fake_anthropic):
        create_llm_client(
            "anthropic", anthropic_backend="vertex", region="us-east5", project_id="p1"
        )
        kwargs = fake_anthropic.AsyncAnthropicVertex.call_args.kwargs
        assert kwargs["region"] == "us-east5"
        assert kwargs["project_id"] == "p1"

    def test_timeout_default_is_applied(self, fake_anthropic, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        create_llm_client("anthropic")
        assert "timeout" in fake_anthropic.AsyncAnthropic.call_args.kwargs

    def test_missing_sdk_raises_with_install_command(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "anthropic", None)
        with pytest.raises(ImportError, match=r"10xscale-agentflow\[anthropic\]"):
            create_llm_client("anthropic")


class TestSamplingParamStripping:
    @pytest.mark.parametrize(
        "model",
        ["claude-opus-5", "claude-sonnet-5", "claude-opus-4-8", "claude-fable-5"],
    )
    def test_rejected_models_have_sampling_stripped(self, model):
        kwargs = {"temperature": 0.7, "top_p": 0.9, "top_k": 40, "max_tokens": 100}
        strip_rejected_sampling_params(model, kwargs)
        assert kwargs == {"max_tokens": 100}

    def test_older_models_keep_sampling(self):
        """Not a blanket strip: older models still accept these."""
        kwargs = {"temperature": 0.7, "max_tokens": 100}
        strip_rejected_sampling_params("claude-haiku-4-5", kwargs)
        assert kwargs["temperature"] == 0.7

    def test_bedrock_prefixed_id_is_matched(self):
        kwargs = {"temperature": 0.5}
        strip_rejected_sampling_params("anthropic.claude-opus-5", kwargs)
        assert "temperature" not in kwargs


class TestReasoningConfig:
    def test_effort_maps_to_output_config(self):
        kwargs: dict = {}
        apply_reasoning_config({"effort": "high"}, kwargs)
        assert kwargs["thinking"] == {"type": "adaptive"}
        assert kwargs["output_config"] == {"effort": "high"}

    def test_no_reasoning_config_is_a_noop(self):
        kwargs: dict = {}
        apply_reasoning_config(None, kwargs)
        assert kwargs == {}

    def test_budget_tokens_is_never_emitted(self):
        """budget_tokens returns a 400 on every current Claude model."""
        kwargs: dict = {}
        apply_reasoning_config({"thinking_budget": 5000}, kwargs)
        assert "budget_tokens" not in kwargs
        assert "thinking_budget" not in kwargs
        assert kwargs["thinking"] == {"type": "adaptive"}


class _StubAgent(AgentAnthropicMixin):
    """Minimal Agent stand-in exposing only what _call_anthropic touches."""

    def __init__(self, **overrides):
        self.model = overrides.get("model", "claude-opus-5")
        self.llm_kwargs = overrides.get("llm_kwargs", {})
        self.reasoning_config = overrides.get("reasoning_config")
        self.output_schema = overrides.get("output_schema")
        self.client = MagicMock()
        self.client.messages.create = AsyncMock(
            return_value=SimpleNamespace(usage=SimpleNamespace(cache_read_input_tokens=0))
        )
        self.client.messages.stream = MagicMock(return_value="stream-handle")


class TestRequestAssembly:
    @pytest.mark.asyncio
    async def test_max_tokens_is_injected(self):
        """Anthropic requires max_tokens on every request."""
        agent = _StubAgent()
        await agent._call_anthropic([{"role": "user", "content": "hi"}])
        kwargs = agent.client.messages.create.call_args.kwargs
        assert kwargs["max_tokens"] == ANTHROPIC_DEFAULT_MAX_TOKENS

    @pytest.mark.asyncio
    async def test_streaming_gets_a_larger_max_tokens_default(self):
        agent = _StubAgent()
        await agent._call_anthropic([{"role": "user", "content": "hi"}], stream=True)
        kwargs = agent.client.messages.stream.call_args.kwargs
        assert kwargs["max_tokens"] == ANTHROPIC_DEFAULT_MAX_TOKENS_STREAMING

    @pytest.mark.asyncio
    async def test_explicit_max_tokens_wins(self):
        agent = _StubAgent(llm_kwargs={"max_tokens": 512})
        await agent._call_anthropic([{"role": "user", "content": "hi"}])
        assert agent.client.messages.create.call_args.kwargs["max_tokens"] == 512

    @pytest.mark.asyncio
    async def test_system_prompt_is_lifted_out_of_messages(self):
        agent = _StubAgent()
        await agent._call_anthropic(
            [
                {"role": "system", "content": "be terse"},
                {"role": "user", "content": "hi"},
            ]
        )
        kwargs = agent.client.messages.create.call_args.kwargs
        assert kwargs["system"] == [{"type": "text", "text": "be terse"}]
        assert all(m["role"] != "system" for m in kwargs["messages"])

    @pytest.mark.asyncio
    async def test_trailing_assistant_turn_is_dropped(self):
        """Guards the injected context_summary, which 400s as a prefill."""
        agent = _StubAgent()
        await agent._call_anthropic(
            [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "summary"},
            ]
        )
        messages = agent.client.messages.create.call_args.kwargs["messages"]
        assert messages[-1]["role"] == "user"

    @pytest.mark.asyncio
    async def test_sampling_params_stripped_for_rejecting_model(self):
        agent = _StubAgent(model="claude-opus-5", llm_kwargs={"temperature": 0.7})
        await agent._call_anthropic([{"role": "user", "content": "hi"}])
        assert "temperature" not in agent.client.messages.create.call_args.kwargs

    @pytest.mark.asyncio
    async def test_tools_are_converted_to_anthropic_shape(self):
        agent = _StubAgent()
        await agent._call_anthropic(
            [{"role": "user", "content": "hi"}],
            tools=[
                {
                    "type": "function",
                    "function": {"name": "f", "description": "d", "parameters": {}},
                }
            ],
        )
        tools = agent.client.messages.create.call_args.kwargs["tools"]
        assert tools[0]["name"] == "f"
        assert "input_schema" in tools[0]

    @pytest.mark.asyncio
    async def test_excluded_kwargs_are_not_sent_as_request_params(self):
        agent = _StubAgent(
            llm_kwargs={"api_key": "sk-x", "anthropic_backend": "vertex", "timeout": 5}
        )
        await agent._call_anthropic([{"role": "user", "content": "hi"}])
        kwargs = agent.client.messages.create.call_args.kwargs
        for leaked in ("api_key", "anthropic_backend", "timeout"):
            assert leaked not in kwargs

    @pytest.mark.asyncio
    async def test_reasoning_config_reaches_the_request(self):
        agent = _StubAgent(reasoning_config={"effort": "xhigh"})
        await agent._call_anthropic([{"role": "user", "content": "hi"}])
        kwargs = agent.client.messages.create.call_args.kwargs
        assert kwargs["thinking"] == {"type": "adaptive"}
        assert kwargs["output_config"]["effort"] == "xhigh"
