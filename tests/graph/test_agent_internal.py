"""Comprehensive tests for agent_internal modules.

Covers:
- AgentProviderMixin : provider detection, output-type validation, client creation
- AgentOpenAIMixin   : _call_openai (text / image / audio), _call_openai_responses
- AgentGoogleMixin   : message conversion, tool conversion, config building, API dispatch
- AgentExecutionMixin: _extract_prompt, _setup_tools, _trim_context, _resolve_tools,
                       _get_converter_key
- Agent.__init__     : model-prefix splitting, reasoning-config normalisation,
                       api_style validation, output_type lowercasing
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentflow.core.graph.agent import Agent
from agentflow.core.graph.tool_node import ToolNode
from agentflow.core.state import AgentState, Message
from agentflow.storage.store.base_store import BaseStore
from agentflow.storage.store.memory_config import (
    AgentMemoryConfig,
    MemoryConfig,
    UserMemoryConfig,
)
from agentflow.storage.store.store_schema import MemorySearchResult, MemoryType


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────


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


def _responses_response() -> SimpleNamespace:
    return SimpleNamespace(
        id="resp_1",
        model="o4-mini",
        status="completed",
        created_at=1_700_000_000,
        output=[],
        usage=SimpleNamespace(
            input_tokens=5,
            output_tokens=10,
            total_tokens=15,
            output_tokens_details=SimpleNamespace(reasoning_tokens=0),
            input_tokens_details=SimpleNamespace(cached_tokens=0),
        ),
        model_dump=MagicMock(return_value={"id": "resp_1"}),
    )


def _mock_openai_client() -> MagicMock:
    client = MagicMock()
    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=_chat_response())
    client.responses = MagicMock()
    client.responses.create = AsyncMock(return_value=_responses_response())
    client.images = MagicMock()
    client.images.generate = AsyncMock(return_value=MagicMock())
    client.audio = MagicMock()
    client.audio.speech = MagicMock()
    client.audio.speech.create = AsyncMock(return_value=MagicMock())
    return client


def _make_openai_agent(model: str = "gpt-4o", provider: str = "openai", **kwargs) -> Agent:
    """Create an Agent with the OpenAI client fully mocked out."""
    mock_client = _mock_openai_client()
    # Default to reasoning_config=None to keep tests self-contained, but allow
    # callers to override it by passing reasoning_config explicitly.
    kwargs.setdefault("reasoning_config", None)
    with patch.object(Agent, "_create_client", return_value=mock_client):
        agent = Agent(model=model, provider=provider, **kwargs)
    agent.client = mock_client
    return agent


def _make_google_client_mock() -> MagicMock:
    client = MagicMock()
    client.aio = MagicMock()
    client.aio.models = MagicMock()
    client.aio.models.generate_content = AsyncMock(return_value=MagicMock())
    client.aio.models.generate_content_stream = AsyncMock(return_value=MagicMock())
    client.aio.models.generate_images = AsyncMock(return_value=MagicMock())
    client.aio.models.generate_videos = AsyncMock(return_value=MagicMock())
    client.aio.models.generate_audio = AsyncMock(return_value=MagicMock())
    return client


def _make_google_agent(model: str = "gemini-2.0-flash") -> Agent:
    mock_client = _make_google_client_mock()
    with patch.object(Agent, "_create_client", return_value=mock_client):
        agent = Agent(model=model, provider="google", reasoning_config=None)
    agent.client = mock_client
    return agent


class _FakeMemoryStore(BaseStore):
    def __init__(self, results: list[MemorySearchResult]):
        self.asearch_mock = AsyncMock(return_value=results)

    async def astore(self, *args, **kwargs):
        return "stored"

    async def asearch(self, *args, **kwargs):
        return await self.asearch_mock(*args, **kwargs)

    async def aget(self, *args, **kwargs):
        return None

    async def aget_all(self, *args, **kwargs):
        return []

    async def aupdate(self, *args, **kwargs):
        return None

    async def adelete(self, *args, **kwargs):
        return None

    async def aforget_memory(self, *args, **kwargs):
        return None


# ---------------------------------------------------------------------------
# Minimal fake google.genai.types objects used in conversion tests
# ---------------------------------------------------------------------------


def _build_google_types_mock() -> MagicMock:
    """Return a MagicMock namespace that mimics the google.genai.types API
    closely enough for conversion/config tests."""

    types = MagicMock(name="google.genai.types")

    class _Content:
        def __init__(self, role, parts):
            self.role = role
            self.parts = parts

    class _Part:
        def __init__(self, text=None, function_call=None):
            self.text = text
            self.function_call = function_call

        @staticmethod
        def from_function_response(name, response):
            p = _Part()
            p._fn_name = name
            p._fn_response = response
            return p

    class _FunctionCall:
        def __init__(self, name, args):
            self.name = name
            self.args = args

    class _FunctionDeclaration:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    class _Tool:
        def __init__(self, function_declarations):
            self.function_declarations = function_declarations

    class _ThinkingConfig:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    class _GenerateContentConfig:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    types.Content = _Content
    types.Part = _Part
    types.FunctionCall = _FunctionCall
    types.FunctionDeclaration = _FunctionDeclaration
    types.Tool = _Tool
    types.ThinkingConfig = _ThinkingConfig
    types.GenerateContentConfig = _GenerateContentConfig
    return types


def _google_types_patch(types_mock: MagicMock):
    """Return a context manager that installs *types_mock* where the code expects
    ``from google.genai import types`` to succeed."""
    google_mod = MagicMock(name="google")
    google_genai_mod = MagicMock(name="google.genai")
    google_genai_mod.types = types_mock
    google_mod.genai = google_genai_mod
    return patch.dict(
        sys.modules,
        {
            "google": google_mod,
            "google.genai": google_genai_mod,
            "google.genai.types": types_mock,
        },
    )


# ═════════════════════════════════════════════════════════════════════════════
# AgentProviderMixin
# ═════════════════════════════════════════════════════════════════════════════


class TestDetectProviderFromModel:
    def test_gpt_prefix_returns_openai(self):
        agent = _make_openai_agent()
        assert agent._detect_provider_from_model("gpt-4o") == "openai"

    def test_o1_prefix_returns_openai(self):
        agent = _make_openai_agent()
        assert agent._detect_provider_from_model("o1-mini") == "openai"

    def test_o3_prefix_returns_openai(self):
        agent = _make_openai_agent()
        assert agent._detect_provider_from_model("o3") == "openai"

    def test_o4_prefix_returns_openai(self):
        agent = _make_openai_agent()
        assert agent._detect_provider_from_model("o4-mini") == "openai"

    def test_gemini_prefix_returns_google(self):
        agent = _make_openai_agent()
        assert agent._detect_provider_from_model("gemini-2.0-flash") == "google"

    def test_imagen_prefix_returns_google(self):
        agent = _make_openai_agent()
        assert agent._detect_provider_from_model("imagen-3.0-generate-001") == "google"

    def test_veo_prefix_returns_google(self):
        agent = _make_openai_agent()
        assert agent._detect_provider_from_model("veo-2.0-generate-001") == "google"

    def test_chirp_prefix_returns_google(self):
        agent = _make_openai_agent()
        assert agent._detect_provider_from_model("chirp-3") == "google"

    def test_unknown_model_defaults_to_openai(self):
        agent = _make_openai_agent()
        assert agent._detect_provider_from_model("llama3:70b") == "openai"

    def test_deepseek_defaults_to_openai(self):
        agent = _make_openai_agent()
        assert agent._detect_provider_from_model("deepseek-chat") == "openai"


class TestValidateOutputType:
    def test_valid_text_type_does_not_raise(self):
        agent = _make_openai_agent(output_type="text")
        # Should not raise
        agent._validate_output_type()

    def test_invalid_type_raises_value_error(self):
        agent = _make_openai_agent()
        agent.output_type = "document"  # force an invalid value
        with pytest.raises(ValueError, match="Invalid output_type"):
            agent._validate_output_type()

    def test_video_unsupported_by_openai_raises_on_construction(self):
        with patch.object(Agent, "_create_client", return_value=MagicMock()):
            with pytest.raises(ValueError, match="doesn't support output_type"):
                Agent(model="gpt-4o", provider="openai", output_type="video", reasoning_config=None)

    def test_all_output_types_valid_for_google(self):
        for otype in ("text", "image", "video", "audio"):
            agent = _make_google_agent()
            agent.output_type = otype
            # Should not raise
            agent._validate_output_type()

    def test_text_valid_for_both_providers(self):
        for provider in ("openai", "google"):
            with patch.object(Agent, "_create_client", return_value=MagicMock()):
                agent = Agent(model="gpt-4o" if provider == "openai" else "gemini-2.0-flash",
                              provider=provider, output_type="text", reasoning_config=None)
            agent._validate_output_type()  # no raise


class TestCreateClient:
    def test_unsupported_provider_raises_value_error(self):
        agent = _make_openai_agent()
        with pytest.raises(ValueError, match="Unsupported provider"):
            agent._create_client("unsupported_provider")

    def test_openai_client_created_with_base_url(self):
        mock_cls = MagicMock(return_value=MagicMock())
        agent = _make_openai_agent()
        with patch("openai.AsyncOpenAI", mock_cls):
            agent._create_client("openai", base_url="http://localhost:11434/v1")
        kwargs = mock_cls.call_args[1]
        assert kwargs.get("base_url") == "http://localhost:11434/v1"

    def test_openai_client_created_without_base_url(self):
        mock_cls = MagicMock(return_value=MagicMock())
        agent = _make_openai_agent()
        with patch("openai.AsyncOpenAI", mock_cls):
            agent._create_client("openai")
        kwargs = mock_cls.call_args[1]
        assert "base_url" not in kwargs or kwargs.get("base_url") is None


# ═════════════════════════════════════════════════════════════════════════════
# AgentExecutionMixin – _extract_prompt
# ═════════════════════════════════════════════════════════════════════════════


class TestExtractPrompt:
    def test_returns_last_user_message(self):
        agent = _make_openai_agent()
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "First"},
            {"role": "assistant", "content": "Response"},
            {"role": "user", "content": "Last user message"},
        ]
        assert agent._extract_prompt(messages) == "Last user message"

    def test_returns_empty_when_no_user_message(self):
        agent = _make_openai_agent()
        assert agent._extract_prompt([{"role": "assistant", "content": "hi"}]) == ""

    def test_handles_empty_message_list(self):
        agent = _make_openai_agent()
        assert agent._extract_prompt([]) == ""

    def test_returns_last_not_first_user_turn(self):
        agent = _make_openai_agent()
        messages = [
            {"role": "user", "content": "earlier"},
            {"role": "user", "content": "later"},
        ]
        assert agent._extract_prompt(messages) == "later"

    def test_returns_empty_string_for_empty_content(self):
        agent = _make_openai_agent()
        messages = [{"role": "user", "content": ""}]
        assert agent._extract_prompt(messages) == ""


# ═════════════════════════════════════════════════════════════════════════════
# AgentOpenAIMixin – _call_openai
# ═════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
class TestCallOpenAI:
    async def test_text_calls_chat_completions_create(self):
        agent = _make_openai_agent(output_type="text")
        await agent._call_openai([{"role": "user", "content": "Hi"}], tools=None, stream=False)
        agent.client.chat.completions.create.assert_called_once()

    async def test_text_with_tools_passes_tools_kwarg(self):
        agent = _make_openai_agent(output_type="text")
        tools = [{"type": "function", "function": {"name": "test"}}]
        await agent._call_openai([{"role": "user", "content": "Hi"}], tools=tools, stream=False)
        call_kwargs = agent.client.chat.completions.create.call_args[1]
        assert "tools" in call_kwargs

    async def test_text_without_tools_omits_tools_kwarg(self):
        agent = _make_openai_agent(output_type="text")
        await agent._call_openai([{"role": "user", "content": "Hi"}], tools=None, stream=False)
        call_kwargs = agent.client.chat.completions.create.call_args[1]
        assert "tools" not in call_kwargs

    async def test_image_calls_images_generate(self):
        agent = _make_openai_agent(output_type="image", model="dall-e-3")
        await agent._call_openai([{"role": "user", "content": "a cat"}], tools=None, stream=False)
        agent.client.images.generate.assert_called_once()

    async def test_audio_calls_speech_create(self):
        agent = _make_openai_agent(output_type="audio", model="tts-1")
        await agent._call_openai(
            [{"role": "user", "content": "Hello world"}], tools=None, stream=False
        )
        agent.client.audio.speech.create.assert_called_once()

    async def test_unsupported_output_type_raises(self):
        agent = _make_openai_agent()
        agent.output_type = "video"  # force unsupported value
        with pytest.raises(ValueError, match="Unsupported output_type"):
            await agent._call_openai([{"role": "user", "content": "hi"}])

    async def test_excluded_kwargs_not_forwarded_to_completions(self):
        agent = _make_openai_agent()
        agent.llm_kwargs = {"api_key": "sk-secret", "temperature": 0.7}
        await agent._call_openai([{"role": "user", "content": "hi"}], tools=None, stream=False)
        call_kwargs = agent.client.chat.completions.create.call_args[1]
        assert "api_key" not in call_kwargs
        assert call_kwargs.get("temperature") == 0.7

    async def test_stream_flag_passed_through(self):
        agent = _make_openai_agent(output_type="text")
        await agent._call_openai([{"role": "user", "content": "hi"}], tools=None, stream=True)
        call_kwargs = agent.client.chat.completions.create.call_args[1]
        assert call_kwargs["stream"] is True


# ═════════════════════════════════════════════════════════════════════════════
# AgentOpenAIMixin – _call_openai_responses
# ═════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
class TestCallOpenAIResponses:
    async def test_system_messages_become_instructions(self):
        agent = _make_openai_agent(api_style="responses", model="o4-mini")
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hi"},
        ]
        await agent._call_openai_responses(messages, tools=None, stream=False)
        call_kwargs = agent.client.responses.create.call_args[1]
        assert call_kwargs["instructions"] == "You are helpful"

    async def test_multiple_system_messages_concatenated(self):
        agent = _make_openai_agent(api_style="responses", model="o4-mini")
        messages = [
            {"role": "system", "content": "First"},
            {"role": "system", "content": "Second"},
            {"role": "user", "content": "Hi"},
        ]
        await agent._call_openai_responses(messages, tools=None, stream=False)
        call_kwargs = agent.client.responses.create.call_args[1]
        assert "First" in call_kwargs["instructions"]
        assert "Second" in call_kwargs["instructions"]

    async def test_no_system_message_omits_instructions(self):
        agent = _make_openai_agent(api_style="responses", model="o4-mini")
        messages = [{"role": "user", "content": "Hi"}]
        await agent._call_openai_responses(messages, tools=None, stream=False)
        call_kwargs = agent.client.responses.create.call_args[1]
        assert "instructions" not in call_kwargs

    async def test_tool_result_converted_to_function_call_output(self):
        agent = _make_openai_agent(api_style="responses", model="o4-mini")
        messages = [
            {"role": "tool", "tool_call_id": "call_abc", "content": '{"result": 42}'},
        ]
        await agent._call_openai_responses(messages, tools=None, stream=False)
        call_kwargs = agent.client.responses.create.call_args[1]
        input_items = call_kwargs["input"]
        assert any(item.get("type") == "function_call_output" for item in input_items)

    async def test_assistant_tool_calls_converted_to_function_call(self):
        agent = _make_openai_agent(api_style="responses", model="o4-mini")
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "call_1", "function": {"name": "my_tool", "arguments": '{"x": 1}'}}
                ],
            }
        ]
        await agent._call_openai_responses(messages, tools=None, stream=False)
        call_kwargs = agent.client.responses.create.call_args[1]
        input_items = call_kwargs["input"]
        assert any(item.get("type") == "function_call" for item in input_items)

    async def test_openai_format_tools_converted(self):
        agent = _make_openai_agent(api_style="responses", model="o4-mini")
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "my_fn",
                    "description": "desc",
                    "parameters": {"type": "object"},
                },
            }
        ]
        await agent._call_openai_responses(
            [{"role": "user", "content": "hi"}], tools=tools, stream=False
        )
        call_kwargs = agent.client.responses.create.call_args[1]
        assert "tools" in call_kwargs
        assert call_kwargs["tools"][0]["name"] == "my_fn"

    async def test_reasoning_config_included_when_set(self):
        agent = _make_openai_agent(
            api_style="responses", model="o4-mini", reasoning_config={"effort": "high"}
        )
        await agent._call_openai_responses(
            [{"role": "user", "content": "hi"}], tools=None, stream=False
        )
        call_kwargs = agent.client.responses.create.call_args[1]
        assert call_kwargs.get("reasoning") == {"effort": "high"}

    async def test_reasoning_effort_kwarg_stripped_before_create(self):
        """reasoning_effort must not be forwarded to the Responses API."""
        agent = _make_openai_agent(api_style="responses", model="o4-mini")
        await agent._call_openai_responses(
            [{"role": "user", "content": "hi"}],
            tools=None,
            stream=False,
            reasoning_effort="medium",
        )
        call_kwargs = agent.client.responses.create.call_args[1]
        assert "reasoning_effort" not in call_kwargs


# ═════════════════════════════════════════════════════════════════════════════
# AgentGoogleMixin – _convert_to_google_format
# ═════════════════════════════════════════════════════════════════════════════


class TestConvertToGoogleFormat:
    def test_system_message_extracted_as_instruction(self):
        agent = _make_google_agent()
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
        ]
        types_mock = _build_google_types_mock()
        with _google_types_patch(types_mock):
            sys_instr, _ = agent._convert_to_google_format(messages)
        assert sys_instr == "You are helpful"

    def test_multiple_system_messages_concatenated(self):
        agent = _make_google_agent()
        messages = [
            {"role": "system", "content": "First"},
            {"role": "system", "content": "Second"},
        ]
        types_mock = _build_google_types_mock()
        with _google_types_patch(types_mock):
            sys_instr, _ = agent._convert_to_google_format(messages)
        assert sys_instr == "First\nSecond"

    def test_no_system_message_returns_none(self):
        agent = _make_google_agent()
        messages = [{"role": "user", "content": "Hello"}]
        types_mock = _build_google_types_mock()
        with _google_types_patch(types_mock):
            sys_instr, _ = agent._convert_to_google_format(messages)
        assert sys_instr is None

    def test_user_message_mapped_to_user_role(self):
        agent = _make_google_agent()
        messages = [{"role": "user", "content": "Hello"}]
        types_mock = _build_google_types_mock()
        with _google_types_patch(types_mock):
            _, contents = agent._convert_to_google_format(messages)
        assert len(contents) == 1
        assert contents[0].role == "user"

    def test_assistant_message_mapped_to_model_role(self):
        agent = _make_google_agent()
        messages = [{"role": "assistant", "content": "Hi there"}]
        types_mock = _build_google_types_mock()
        with _google_types_patch(types_mock):
            _, contents = agent._convert_to_google_format(messages)
        assert len(contents) == 1
        assert contents[0].role == "model"

    def test_assistant_with_tool_calls_creates_function_call_parts(self):
        agent = _make_google_agent()
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "call_1", "function": {"name": "search", "arguments": '{"q": "hi"}'}}
                ],
            }
        ]
        types_mock = _build_google_types_mock()
        with _google_types_patch(types_mock):
            _, contents = agent._convert_to_google_format(messages)
        assert len(contents) == 1
        assert contents[0].role == "model"
        fn_call_parts = [
            p for p in contents[0].parts if isinstance(p.function_call, MagicMock.__class__)
        ]
        # There should be at least one part with function_call
        assert any(hasattr(p, "function_call") for p in contents[0].parts)

    def test_tool_result_message_mapped_to_user_role(self):
        agent = _make_google_agent()
        # first set up a prior assistant message so call_id_to_name is populated
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "call_1", "function": {"name": "my_fn", "arguments": "{}"}}
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "result_data"},
        ]
        types_mock = _build_google_types_mock()
        with _google_types_patch(types_mock):
            _, contents = agent._convert_to_google_format(messages)
        # The tool result should be the second Content item with role == "user"
        tool_content = contents[-1]
        assert tool_content.role == "user"

    def test_parallel_tool_results_merged_into_single_content(self):
        # Gemini requires the number of function_response parts to equal the
        # number of function_call parts in the preceding model turn. Parallel
        # tool calls must therefore collapse into one user Content.
        agent = _make_google_agent()
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "call_1", "function": {"name": "fn_a", "arguments": "{}"}},
                    {"id": "call_2", "function": {"name": "fn_b", "arguments": "{}"}},
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "res_a"},
            {"role": "tool", "tool_call_id": "call_2", "content": "res_b"},
        ]
        types_mock = _build_google_types_mock()
        with _google_types_patch(types_mock):
            _, contents = agent._convert_to_google_format(messages)
        # One model Content (2 function_call parts) + one merged user Content
        assert len(contents) == 2
        model_content, tool_content = contents
        fn_call_parts = [p for p in model_content.parts if p.function_call is not None]
        assert len(fn_call_parts) == 2
        assert tool_content.role == "user"
        assert len(tool_content.parts) == 2


# ═════════════════════════════════════════════════════════════════════════════
# AgentGoogleMixin – _convert_tools_to_google_format
# ═════════════════════════════════════════════════════════════════════════════


class TestConvertToolsToGoogleFormat:
    def test_converts_openai_style_tool(self):
        agent = _make_google_agent()
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "my_tool",
                    "description": "A tool",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        types_mock = _build_google_types_mock()
        with _google_types_patch(types_mock):
            result = agent._convert_tools_to_google_format(tools)
        assert len(result) == 1
        assert result[0].name == "my_tool"

    def test_skips_non_function_tools(self):
        agent = _make_google_agent()
        tools = [{"type": "builtin_code_execution"}]
        types_mock = _build_google_types_mock()
        with _google_types_patch(types_mock):
            result = agent._convert_tools_to_google_format(tools)
        assert result == []

    def test_tool_without_parameters_omits_schema(self):
        agent = _make_google_agent()
        tools = [{"type": "function", "function": {"name": "simple", "description": "no params"}}]
        captured: dict = {}

        types_mock = _build_google_types_mock()
        original_fn_decl = types_mock.FunctionDeclaration

        def capturing_fn_decl(**kw):
            captured.update(kw)
            return original_fn_decl(**kw)

        types_mock.FunctionDeclaration = capturing_fn_decl
        with _google_types_patch(types_mock):
            agent._convert_tools_to_google_format(tools)
        assert "parameters_json_schema" not in captured

    def test_tool_with_parameters_includes_schema(self):
        agent = _make_google_agent()
        schema = {"type": "object", "properties": {"q": {"type": "string"}}}
        tools = [
            {
                "type": "function",
                "function": {"name": "search", "description": "search", "parameters": schema},
            }
        ]
        captured: dict = {}

        types_mock = _build_google_types_mock()
        original_fn_decl = types_mock.FunctionDeclaration

        def capturing_fn_decl(**kw):
            captured.update(kw)
            return original_fn_decl(**kw)

        types_mock.FunctionDeclaration = capturing_fn_decl
        with _google_types_patch(types_mock):
            agent._convert_tools_to_google_format(tools)
        assert captured.get("parameters_json_schema") == schema


# ═════════════════════════════════════════════════════════════════════════════
# AgentGoogleMixin – _build_google_config
# ═════════════════════════════════════════════════════════════════════════════


class TestBuildGoogleConfig:
    def test_returns_none_for_empty_config(self):
        agent = _make_google_agent()
        types_mock = _build_google_types_mock()
        with _google_types_patch(types_mock):
            result = agent._build_google_config(None, None, {})
        assert result is None

    def test_system_instruction_included(self):
        agent = _make_google_agent()
        captured: dict = {}
        types_mock = _build_google_types_mock()
        types_mock.GenerateContentConfig = lambda **kw: (captured.update(kw) or MagicMock())
        with _google_types_patch(types_mock):
            agent._build_google_config("You are a bot", None, {})
        assert captured.get("system_instruction") == "You are a bot"

    def test_temperature_extracted_from_call_kwargs(self):
        agent = _make_google_agent()
        captured: dict = {}
        types_mock = _build_google_types_mock()
        types_mock.GenerateContentConfig = lambda **kw: (captured.update(kw) or MagicMock())
        call_kwargs = {"temperature": 0.5}
        with _google_types_patch(types_mock):
            agent._build_google_config(None, None, call_kwargs)
        assert captured.get("temperature") == 0.5
        # Must be popped from call_kwargs to avoid double-passing downstream
        assert "temperature" not in call_kwargs

    def test_max_tokens_alias_to_max_output_tokens(self):
        agent = _make_google_agent()
        captured: dict = {}
        types_mock = _build_google_types_mock()
        types_mock.GenerateContentConfig = lambda **kw: (captured.update(kw) or MagicMock())
        call_kwargs = {"max_tokens": 256}
        with _google_types_patch(types_mock):
            agent._build_google_config(None, None, call_kwargs)
        assert captured.get("max_output_tokens") == 256

    def test_reasoning_budget_passed_to_thinking_config(self):
        agent = _make_google_agent()
        agent.reasoning_config = {"thinking_budget": 2048}
        agent.output_type = "text"

        thinking_kwargs: dict = {}
        types_mock = _build_google_types_mock()
        types_mock.ThinkingConfig = lambda **kw: (thinking_kwargs.update(kw) or MagicMock())
        types_mock.GenerateContentConfig = lambda **kw: MagicMock()
        with _google_types_patch(types_mock):
            agent._build_google_config("system", None, {})
        assert thinking_kwargs.get("thinking_budget") == 2048

    def test_reasoning_effort_low_translates_to_budget_512(self):
        agent = _make_google_agent()
        agent.reasoning_config = {"effort": "low"}
        agent.output_type = "text"

        thinking_kwargs: dict = {}
        types_mock = _build_google_types_mock()
        types_mock.ThinkingConfig = lambda **kw: (thinking_kwargs.update(kw) or MagicMock())
        types_mock.GenerateContentConfig = lambda **kw: MagicMock()
        with _google_types_patch(types_mock):
            agent._build_google_config("system", None, {})
        assert thinking_kwargs.get("thinking_budget") == 512

    def test_reasoning_effort_medium_translates_to_budget_8192(self):
        agent = _make_google_agent()
        agent.reasoning_config = {"effort": "medium"}
        agent.output_type = "text"

        thinking_kwargs: dict = {}
        types_mock = _build_google_types_mock()
        types_mock.ThinkingConfig = lambda **kw: (thinking_kwargs.update(kw) or MagicMock())
        types_mock.GenerateContentConfig = lambda **kw: MagicMock()
        with _google_types_patch(types_mock):
            agent._build_google_config("system", None, {})
        assert thinking_kwargs.get("thinking_budget") == 8192

    def test_reasoning_effort_high_translates_to_budget_24576(self):
        agent = _make_google_agent()
        agent.reasoning_config = {"effort": "high"}
        agent.output_type = "text"

        thinking_kwargs: dict = {}
        types_mock = _build_google_types_mock()
        types_mock.ThinkingConfig = lambda **kw: (thinking_kwargs.update(kw) or MagicMock())
        types_mock.GenerateContentConfig = lambda **kw: MagicMock()
        with _google_types_patch(types_mock):
            agent._build_google_config("system", None, {})
        assert thinking_kwargs.get("thinking_budget") == 24576

    def test_reasoning_not_applied_for_non_text_output_type(self):
        agent = _make_google_agent()
        agent.reasoning_config = {"effort": "high"}
        agent.output_type = "image"

        types_mock = _build_google_types_mock()
        thinking_called = []
        types_mock.ThinkingConfig = lambda **kw: thinking_called.append(kw) or MagicMock()
        types_mock.GenerateContentConfig = lambda **kw: MagicMock()
        with _google_types_patch(types_mock):
            agent._build_google_config("system", None, {})
        assert len(thinking_called) == 0

    def test_reasoning_thinking_level_passed_directly(self):
        """thinking_level key in reasoning_config is forwarded directly to ThinkingConfig."""
        agent = _make_google_agent()
        agent.reasoning_config = {"thinking_level": "high"}
        agent.output_type = "text"

        thinking_kwargs: dict = {}
        types_mock = _build_google_types_mock()
        types_mock.ThinkingConfig = lambda **kw: (thinking_kwargs.update(kw) or MagicMock())
        types_mock.GenerateContentConfig = lambda **kw: MagicMock()
        with _google_types_patch(types_mock):
            agent._build_google_config("system", None, {})
        assert thinking_kwargs.get("thinking_level") == "high"
        assert "thinking_budget" not in thinking_kwargs

    def test_reasoning_thinking_level_takes_precedence_over_effort(self):
        """thinking_level beats effort when both are present in reasoning_config."""
        agent = _make_google_agent()
        agent.reasoning_config = {"thinking_level": "minimal", "effort": "high"}
        agent.output_type = "text"

        thinking_kwargs: dict = {}
        types_mock = _build_google_types_mock()
        types_mock.ThinkingConfig = lambda **kw: (thinking_kwargs.update(kw) or MagicMock())
        types_mock.GenerateContentConfig = lambda **kw: MagicMock()
        with _google_types_patch(types_mock):
            agent._build_google_config("system", None, {})
        assert thinking_kwargs.get("thinking_level") == "minimal"
        assert "thinking_budget" not in thinking_kwargs


class TestConvertToGoogleFormatThoughtSignature:
    """Tests for thought_signature handling in _convert_to_google_format."""

    def test_stored_thought_signature_used_on_first_fc(self):
        """Real base64 signature is decoded and set on the first function call part."""
        import base64

        agent = _make_google_agent()
        sig_bytes = b"real_signature_bytes"
        sig_b64 = base64.b64encode(sig_bytes).decode()

        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {"name": "fn", "arguments": "{}"},
                        "type": "function",
                        "thought_signature": sig_b64,
                    }
                ],
            }
        ]

        assigned_sigs: list = []

        types_mock = _build_google_types_mock()
        original_part = types_mock.Part

        class CapturingSigPart(original_part.__class__):
            def __init__(self, **kw):
                super().__init__(**kw)
                self._sig = None

            @property  # type: ignore[override]
            def thought_signature(self):
                return self._sig

            @thought_signature.setter
            def thought_signature(self, v):
                self._sig = v
                assigned_sigs.append(v)

        # Use a simpler capture approach: wrap Part to record attribute sets
        captured_parts: list = []
        _OrigPart = types_mock.Part

        class TrackingPart(_OrigPart):
            def __setattr__(self, name, value):
                if name == "thought_signature":
                    captured_parts.append(value)
                super().__setattr__(name, value)

        types_mock.Part = TrackingPart

        with _google_types_patch(types_mock):
            agent._convert_to_google_format(messages)

        assert len(captured_parts) == 1
        assert captured_parts[0] == sig_bytes

    def test_bypass_string_used_when_no_stored_signature(self):
        """Bypass string is set on the first FC when no thought_signature is stored."""
        agent = _make_google_agent()
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "call_1", "function": {"name": "fn", "arguments": "{}"}, "type": "function"}
                ],
            }
        ]

        captured_parts: list = []

        types_mock = _build_google_types_mock()
        _OrigPart = types_mock.Part

        class TrackingPart(_OrigPart):
            def __setattr__(self, name, value):
                if name == "thought_signature":
                    captured_parts.append(value)
                super().__setattr__(name, value)

        types_mock.Part = TrackingPart

        with _google_types_patch(types_mock):
            agent._convert_to_google_format(messages)

        assert len(captured_parts) == 1
        assert captured_parts[0] == b"skip_thought_signature_validator"


# ═════════════════════════════════════════════════════════════════════════════
# AgentGoogleMixin – _call_google dispatch
# ═════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
class TestCallGoogle:
    def _agent(self) -> Agent:
        return _make_google_agent()

    async def _call(self, agent: Agent, output_type: str = "text", messages=None, stream=False):
        agent.output_type = output_type
        types_mock = _build_google_types_mock()
        # Patch conversion helpers so the test focuses on dispatch
        with patch.object(agent, "_convert_to_google_format", return_value=(None, [])), \
             patch.object(agent, "_build_google_config", return_value=None), \
             _google_types_patch(types_mock):
            await agent._call_google(messages or [{"role": "user", "content": "hi"}], stream=stream)

    async def test_text_calls_generate_content(self):
        agent = self._agent()
        await self._call(agent, output_type="text", stream=False)
        agent.client.aio.models.generate_content.assert_called_once()

    async def test_text_stream_calls_generate_content_stream(self):
        agent = self._agent()
        await self._call(agent, output_type="text", stream=True)
        agent.client.aio.models.generate_content_stream.assert_called_once()

    async def test_image_calls_generate_images(self):
        agent = self._agent()
        await self._call(agent, output_type="image")
        agent.client.aio.models.generate_images.assert_called_once()

    async def test_video_calls_generate_videos(self):
        agent = self._agent()
        await self._call(agent, output_type="video")
        agent.client.aio.models.generate_videos.assert_called_once()

    async def test_audio_calls_generate_audio(self):
        agent = self._agent()
        await self._call(agent, output_type="audio")
        agent.client.aio.models.generate_audio.assert_called_once()

    async def test_unsupported_output_type_raises(self):
        agent = self._agent()
        agent.output_type = "document"
        types_mock = _build_google_types_mock()
        with patch.object(agent, "_convert_to_google_format", return_value=(None, [])), \
             patch.object(agent, "_build_google_config", return_value=None), \
             _google_types_patch(types_mock):
            with pytest.raises(ValueError, match="Unsupported output_type"):
                await agent._call_google([{"role": "user", "content": "hi"}])


# ═════════════════════════════════════════════════════════════════════════════
# AgentExecutionMixin – _setup_tools
# ═════════════════════════════════════════════════════════════════════════════


class TestSetupTools:
    def test_none_returns_none(self):
        agent = _make_openai_agent()
        agent.tool_node = None
        assert agent._setup_tools() is None

    def test_tool_node_instance_returned_as_is(self):
        agent = _make_openai_agent()
        tn = ToolNode([lambda x: x])
        agent.tool_node = tn
        assert agent._setup_tools() is tn

    def test_str_sets_tool_node_name_returns_none(self):
        agent = _make_openai_agent()
        agent.tool_node = "TOOL"
        result = agent._setup_tools()
        assert result is None
        assert agent.tool_node_name == "TOOL"


# ═════════════════════════════════════════════════════════════════════════════
# AgentExecutionMixin – _trim_context
# ═════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
class TestTrimContext:
    async def test_disabled_returns_original_state(self):
        agent = _make_openai_agent()
        agent.trim_context = False
        state = AgentState()
        result = await agent._trim_context(state)
        assert result is state

    async def test_enabled_no_manager_returns_original_state(self):
        agent = _make_openai_agent()
        agent.trim_context = True
        state = AgentState()
        result = await agent._trim_context(state, context_manager=None)
        assert result is state

    async def test_enabled_with_manager_returns_trimmed_state(self):
        agent = _make_openai_agent()
        agent.trim_context = True
        original_state = AgentState()
        trimmed_state = AgentState()
        mock_manager = MagicMock()
        mock_manager.atrim_context = AsyncMock(return_value=trimmed_state)
        result = await agent._trim_context(original_state, context_manager=mock_manager)
        assert result is trimmed_state
        mock_manager.atrim_context.assert_awaited_once_with(original_state)

    async def test_enabled_manager_attribute_error_returns_state(self):
        """If the manager doesn't implement atrim_context, return state unchanged."""
        agent = _make_openai_agent()
        agent.trim_context = True
        state = AgentState()
        bad_manager = MagicMock(spec=[])  # no atrim_context attribute
        result = await agent._trim_context(state, context_manager=bad_manager)
        assert result is state


# ═════════════════════════════════════════════════════════════════════════════
# AgentExecutionMixin – _get_converter_key
# ═════════════════════════════════════════════════════════════════════════════


class TestGetConverterKey:
    def test_openai_chat_style_returns_openai(self):
        agent = _make_openai_agent(api_style="chat")
        assert agent._get_converter_key() == "openai"

    def test_openai_responses_style_returns_openai_responses(self):
        agent = _make_openai_agent(api_style="responses", model="o4-mini")
        agent._effective_api_style = "responses"
        assert agent._get_converter_key() == "openai_responses"

    def test_google_provider_returns_google(self):
        agent = _make_google_agent()
        assert agent._get_converter_key() == "google"

    def test_after_responses_to_chat_fallback_returns_openai(self):
        agent = _make_openai_agent(api_style="responses", base_url="http://localhost:8000/v1")
        agent._effective_api_style = "chat"
        assert agent._get_converter_key() == "openai"


# ═════════════════════════════════════════════════════════════════════════════
# AgentExecutionMixin – _resolve_tools
# ═════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
class TestResolveTools:
    async def test_no_tools_no_named_node_returns_empty(self):
        agent = _make_openai_agent()
        agent._tool_node = None
        agent.tool_node_name = None
        from injectq import InjectQ
        result = await agent._resolve_tools(InjectQ.get_instance())
        assert result == []

    async def test_inline_tool_node_tools_resolved(self):
        def my_tool(x: str) -> str:
            """A test tool."""
            return x

        agent = _make_openai_agent(tool_node=ToolNode([my_tool]))
        from injectq import InjectQ
        result = await agent._resolve_tools(InjectQ.get_instance())
        assert isinstance(result, list)
        assert len(result) > 0

    async def test_named_node_not_found_raises(self):
        agent = _make_openai_agent(tool_node="nonexistent_node")
        from injectq import InjectQ
        with pytest.raises(RuntimeError, match="ToolNode named 'nonexistent_node' was not found"):
            await agent._resolve_tools(InjectQ.get_instance())

    async def test_named_node_resolves_and_registers_pending_tools(self):
        def pending_tool(value: str) -> str:
            return value

        agent = _make_openai_agent(tool_node="TOOL")
        agent._tool_node = None
        agent.tool_node_name = "TOOL"
        agent._extra_tools = [pending_tool]

        fake_tool_node = ToolNode([])
        fake_node = MagicMock()
        fake_node.func = fake_tool_node

        container = MagicMock()
        container.call_factory.return_value = fake_node

        result = await agent._resolve_tools(container)

        assert agent._tool_node is fake_tool_node
        # tool_node_name cleared to prevent duplicate tool resolution on subsequent calls
        assert agent.tool_node_name is None
        assert getattr(agent, "_extra_tools", []) == []
        assert any(tool["function"]["name"] == "pending_tool" for tool in result)

    async def test_named_node_no_duplicate_tools_on_second_call(self):
        """Resolving a named ToolNode and calling _resolve_tools again must not duplicate tools."""
        def my_tool(x: str) -> str:
            """A test tool."""
            return x

        agent = _make_openai_agent(tool_node="TOOL")
        agent._tool_node = None
        agent.tool_node_name = "TOOL"

        fake_tool_node = ToolNode([my_tool])
        fake_node = MagicMock()
        fake_node.func = fake_tool_node

        container = MagicMock()
        container.call_factory.return_value = fake_node

        first = await agent._resolve_tools(container)
        second = await agent._resolve_tools(container)

        names_first = [t["function"]["name"] for t in first]
        names_second = [t["function"]["name"] for t in second]
        assert names_first.count("my_tool") == 1
        assert names_second.count("my_tool") == 1


# ═════════════════════════════════════════════════════════════════════════════
# Agent.__init__ – construction edge cases
# ═════════════════════════════════════════════════════════════════════════════


class TestAgentInit:
    # ── model prefix splitting ──────────────────────────────────────────────

    def test_openai_slash_prefix_splits_provider_and_model(self):
        with patch.object(Agent, "_create_client", return_value=MagicMock()):
            agent = Agent(model="openai/gpt-4o", reasoning_config=None)
        assert agent.provider == "openai"
        assert agent.model == "gpt-4o"

    def test_google_slash_prefix_splits_provider_and_model(self):
        with patch.object(Agent, "_create_client", return_value=MagicMock()):
            agent = Agent(model="google/gemini-2.0-flash", reasoning_config=None)
        assert agent.provider == "google"
        assert agent.model == "gemini-2.0-flash"

    def test_explicit_provider_prevents_prefix_split(self):
        """When provider is given explicitly, the model string is not munged."""
        with patch.object(Agent, "_create_client", return_value=MagicMock()):
            agent = Agent(model="gemini-2.0-flash", provider="google", reasoning_config=None)
        assert agent.provider == "google"
        assert agent.model == "gemini-2.0-flash"

    def test_unknown_model_without_provider_auto_detects_openai(self):
        with patch.object(Agent, "_create_client", return_value=MagicMock()):
            agent = Agent(model="llama3:70b", reasoning_config=None)
        assert agent.provider == "openai"

    # ── reasoning config normalization ────────────────────────────────────

    def test_default_sentinel_produces_medium_effort(self):
        with patch.object(Agent, "_create_client", return_value=MagicMock()):
            agent = Agent(model="gpt-4o", provider="openai")
        assert agent.reasoning_config == {"effort": "medium"}

    def test_reasoning_config_none_stored_as_none(self):
        agent = _make_openai_agent(reasoning_config=None)
        assert agent.reasoning_config is None

    def test_reasoning_config_false_stored_as_none(self):
        with patch.object(Agent, "_create_client", return_value=MagicMock()):
            agent = Agent(model="gpt-4o", provider="openai", reasoning_config=False)
        assert agent.reasoning_config is None

    def test_explicit_reasoning_config_stored_unchanged(self):
        cfg = {"effort": "high", "summary": "auto"}
        agent = _make_openai_agent(reasoning_config=cfg)
        assert agent.reasoning_config == cfg

    # ── api_style validation ───────────────────────────────────────────────

    def test_invalid_api_style_raises_value_error(self):
        with patch.object(Agent, "_create_client", return_value=MagicMock()):
            with pytest.raises(ValueError, match="Invalid api_style"):
                Agent(model="gpt-4o", provider="openai", api_style="sse", reasoning_config=None)

    def test_valid_api_style_chat_accepted(self):
        agent = _make_openai_agent(api_style="chat")
        assert agent.api_style == "chat"

    def test_valid_api_style_responses_accepted(self):
        agent = _make_openai_agent(api_style="responses", model="o4-mini")
        assert agent.api_style == "responses"

    # ── output type ────────────────────────────────────────────────────────

    def test_output_type_stored_lowercase(self):
        agent = _make_openai_agent(output_type="TEXT")
        assert agent.output_type == "text"

    def test_default_output_type_is_text(self):
        agent = _make_openai_agent()
        assert agent.output_type == "text"

    # ── misc attributes ────────────────────────────────────────────────────

    def test_tool_node_str_sets_tool_node_name(self):
        agent = _make_openai_agent(tool_node="my_tools")
        assert agent.tool_node_name == "my_tools"
        assert agent._tool_node is None

    def test_extra_messages_stored(self):
        msg = Message.text_message("hi", role="user")
        agent = _make_openai_agent(extra_messages=[msg])
        assert agent.extra_messages == [msg]

    def test_trim_context_stored(self):
        agent = _make_openai_agent(trim_context=True)
        assert agent.trim_context is True

    def test_tool_node_instance_stored(self):
        tn = ToolNode([lambda x: x])
        agent = _make_openai_agent(tool_node=tn)
        assert agent._tool_node is tn
        assert agent.tool_node_name is None

    def test_tool_node_none_gives_none_internal_tool_node(self):
        agent = _make_openai_agent(tool_node=None)
        assert agent._tool_node is None
        assert agent.tool_node_name is None

    def test_memory_config_requires_existing_tool_node_for_postload(self):
        memory = MemoryConfig(user_memory=UserMemoryConfig(user_id="u1"))
        with pytest.raises(RuntimeError, match="Memory requires an existing ToolNode"):
            _make_openai_agent(tool_node=None, memory=memory)

    def test_memory_config_defers_to_named_tool_node(self):
        memory = MemoryConfig(user_memory=UserMemoryConfig(user_id="u1"))
        agent = _make_openai_agent(tool_node="my_tools", memory=memory)

        assert agent.tool_node_name == "my_tools"
        assert getattr(agent, "_extra_tools", None) is not None
        assert len(agent._extra_tools) > 0

    def test_agent_init_defers_skills_to_named_tool_node(self, tmp_path: Path):
        from agentflow.core.skills.models import SkillConfig

        skill_dir = tmp_path / "alpha"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: alpha\ndescription: Test skill\n---\n# Alpha skill\n", encoding="utf-8"
        )

        with patch.object(Agent, "_create_client", return_value=MagicMock()):
            agent = Agent(
                model="gpt-4o",
                provider="openai",
                tool_node="TOOL",
                skills=SkillConfig(skills_dir=str(tmp_path), inject_trigger_table=False),
                reasoning_config=None,
            )

        assert agent.tool_node_name == "TOOL"
        assert getattr(agent, "_extra_tools", None) is not None
        assert len(agent._extra_tools) == 1

    def test_memory_config_adds_tools_to_existing_tool_node(self):
        memory = MemoryConfig(user_memory=UserMemoryConfig(user_id="u1"))
        tool_node = ToolNode([])
        agent = _make_openai_agent(tool_node=tool_node, memory=memory)

        assert agent._tool_node is tool_node
        assert any("user_memory_tool" in p["content"] for p in agent.system_prompt)
        assert "user_memory_tool" in agent._tool_node._funcs
        assert agent.get_tool_node() is agent._tool_node

    def test_memory_config_preserves_existing_tools(self):
        def my_tool(x: str) -> str:
            return x

        memory = MemoryConfig(
            user_memory=UserMemoryConfig(),
            agent_memory=AgentMemoryConfig(enabled=True, agent_id="agent-1"),
        )
        agent = _make_openai_agent(tool_node=ToolNode([my_tool]), memory=memory)

        assert agent._tool_node is not None
        assert set(agent._tool_node._funcs) == {
            "my_tool",
            "user_memory_tool",
            "agent_memory_tool",
        }

    @pytest.mark.asyncio
    async def test_preload_memory_does_not_register_tools_and_builds_prompt(self):
        store = _FakeMemoryStore(
            [
                MemorySearchResult(
                    id="m1",
                    content="User prefers concise answers",
                    score=0.91,
                    memory_type=MemoryType.SEMANTIC,
                )
            ]
        )
        memory = MemoryConfig(
            retrieval_mode="preload",
            user_memory=UserMemoryConfig(store=store, user_id="u1", memory_type="semantic"),
        )
        agent = _make_openai_agent(tool_node=None, memory=memory)

        assert agent._tool_node is None
        assert not any("user_memory_tool" in p["content"] for p in agent.system_prompt)

        prompts = await agent._build_memory_prompts(
            AgentState(context=[Message.text_message("What do I like?", role="user")]),
            {"user_id": "runtime-user", "thread_id": "runtime-thread"},
        )

        assert len(prompts) == 1
        assert prompts[0]["role"] == "system"
        assert "Long-term Memory Context" in prompts[0]["content"]
        assert "User prefers concise answers" in prompts[0]["content"]
        store.asearch_mock.assert_awaited_once()
        assert store.asearch_mock.call_args.args[0] == {"user_id": "u1"}