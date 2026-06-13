"""Tests for agentflow.core.llm.caller.call_llm."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentflow.core.llm.caller import (
    _extract_responses_text,
    call_llm,
    _call_google,
    _call_openai_responses,
    _call_openai_chat,
)


# ---------------------------------------------------------------------------
# Provider dispatch
# ---------------------------------------------------------------------------

_DETECT = "agentflow.core.llm.caller.detect_provider"
_CREATE = "agentflow.core.llm.caller.create_llm_client"
_CALL_GOOGLE = "agentflow.core.llm.caller._call_google"
_CALL_RESP = "agentflow.core.llm.caller._call_openai_responses"
_CALL_CHAT = "agentflow.core.llm.caller._call_openai_chat"

_DUMMY = ("text", 10, 5, 0)


@pytest.mark.anyio
async def test_google_model_dispatches_to_google():
    with (
        patch(_DETECT, return_value="google"),
        patch(_CREATE, return_value=MagicMock()),
        patch(_CALL_GOOGLE, new=AsyncMock(return_value=_DUMMY)) as mock,
    ):
        result = await call_llm("gemini-2.0-flash", "hello")

    mock.assert_called_once()
    assert result == _DUMMY


@pytest.mark.anyio
async def test_openai_default_dispatches_to_responses():
    with (
        patch(_DETECT, return_value="openai"),
        patch(_CREATE, return_value=MagicMock()),
        patch(_CALL_RESP, new=AsyncMock(return_value=_DUMMY)) as mock,
    ):
        result = await call_llm("gpt-4o-mini", "hello")

    mock.assert_called_once()
    assert result == _DUMMY


@pytest.mark.anyio
async def test_openai_chat_style_dispatches_to_chat():
    with (
        patch(_DETECT, return_value="openai"),
        patch(_CREATE, return_value=MagicMock()),
        patch(_CALL_CHAT, new=AsyncMock(return_value=_DUMMY)) as mock,
    ):
        result = await call_llm("gpt-4o-mini", "hello", api_style="chat")

    mock.assert_called_once()
    assert result == _DUMMY


@pytest.mark.anyio
async def test_openai_responses_style_explicit():
    """Explicitly passing api_style='responses' still hits the Responses path."""
    with (
        patch(_DETECT, return_value="openai"),
        patch(_CREATE, return_value=MagicMock()),
        patch(_CALL_RESP, new=AsyncMock(return_value=_DUMMY)) as mock_resp,
        patch(_CALL_CHAT, new=AsyncMock(return_value=_DUMMY)) as mock_chat,
    ):
        await call_llm("gpt-4o-mini", "hello", api_style="responses")

    mock_resp.assert_called_once()
    mock_chat.assert_not_called()


@pytest.mark.anyio
async def test_api_style_irrelevant_for_google():
    """api_style has no effect when the provider is Google."""
    with (
        patch(_DETECT, return_value="google"),
        patch(_CREATE, return_value=MagicMock()),
        patch(_CALL_GOOGLE, new=AsyncMock(return_value=_DUMMY)) as mock_google,
        patch(_CALL_CHAT, new=AsyncMock(return_value=_DUMMY)) as mock_chat,
    ):
        await call_llm("gemini-2.0-flash", "hello", api_style="chat")

    mock_google.assert_called_once()
    mock_chat.assert_not_called()


# ---------------------------------------------------------------------------
# Parameters forwarded correctly
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_system_prompt_forwarded_to_responses():
    with (
        patch(_DETECT, return_value="openai"),
        patch(_CREATE, return_value=MagicMock()),
        patch(_CALL_RESP, new=AsyncMock(return_value=_DUMMY)) as mock,
    ):
        await call_llm("gpt-4o-mini", "hi", system_prompt="Be brief.", json_mode=True)

    _, kwargs = mock.call_args
    assert kwargs["system_prompt"] == "Be brief."
    assert kwargs["json_mode"] is True


@pytest.mark.anyio
async def test_system_prompt_forwarded_to_chat():
    with (
        patch(_DETECT, return_value="openai"),
        patch(_CREATE, return_value=MagicMock()),
        patch(_CALL_CHAT, new=AsyncMock(return_value=_DUMMY)) as mock,
    ):
        await call_llm("gpt-4o-mini", "hi", system_prompt="Be brief.", api_style="chat")

    _, kwargs = mock.call_args
    assert kwargs["system_prompt"] == "Be brief."


# ---------------------------------------------------------------------------
# _extract_responses_text
# ---------------------------------------------------------------------------

def _make_response(output_text=None, output=None):
    r = MagicMock()
    r.output_text = output_text
    r.output = output or []
    return r


def test_extract_uses_output_text_property():
    r = _make_response(output_text="  hello  ")
    assert _extract_responses_text(r) == "hello"


def test_extract_falls_back_to_output_items():
    part = MagicMock()
    part.type = "output_text"
    part.text = "fallback"

    item = MagicMock()
    item.type = "message"
    item.content = [part]

    r = _make_response(output_text=None, output=[item])
    assert _extract_responses_text(r) == "fallback"


def test_extract_returns_empty_when_no_text():
    r = _make_response(output_text=None, output=[])
    assert _extract_responses_text(r) == ""


# ---------------------------------------------------------------------------
# Direct private call implementations
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_call_google_implementation():
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "  hello google  "
    mock_response.usage_metadata.prompt_token_count = 10
    mock_response.usage_metadata.candidates_token_count = 5
    mock_response.usage_metadata.cached_content_token_count = 3
    
    mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)
    
    # 1. With system prompt, temperature, json mode
    res = await _call_google(
        mock_client,
        "gemini-2.0-flash",
        "user prompt",
        system_prompt="sys prompt",
        max_tokens=100,
        temperature=0.5,
        json_mode=True
    )
    
    assert res == ("hello google", 10, 5, 3)
    mock_client.aio.models.generate_content.assert_called_once()
    _, kwargs = mock_client.aio.models.generate_content.call_args
    assert kwargs["model"] == "gemini-2.0-flash"
    assert kwargs["contents"] == "user prompt"
    config = kwargs["config"]
    assert config.max_output_tokens == 100
    assert config.temperature == 0.5
    assert config.response_mime_type == "application/json"
    assert config.system_instruction == "sys prompt"
    
    # 2. With cached content (system prompt should be ignored)
    mock_client.aio.models.generate_content.reset_mock()
    res = await _call_google(
        mock_client,
        "gemini-2.0-flash",
        "user prompt",
        system_prompt="sys prompt",
        max_tokens=100,
        temperature=0.5,
        json_mode=False,
        cached_content="cachedContents/abc123"
    )
    
    _, kwargs = mock_client.aio.models.generate_content.call_args
    config = kwargs["config"]
    assert config.cached_content == "cachedContents/abc123"
    assert getattr(config, "system_instruction", None) is None


@pytest.mark.anyio
async def test_call_openai_responses_implementation():
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.output_text = "  hello responses  "
    
    mock_response.usage.input_tokens = 20
    mock_response.usage.output_tokens = 15
    mock_response.usage.input_tokens_details.cached_tokens = 7
    
    mock_client.responses.create = AsyncMock(return_value=mock_response)
    
    res = await _call_openai_responses(
        mock_client,
        "gpt-4o-mini",
        "user prompt",
        system_prompt="sys prompt",
        max_tokens=200,
        temperature=0.7,
        json_mode=True,
        extra_param="extra_val"
    )
    
    assert res == ("hello responses", 20, 15, 7)
    
    mock_client.responses.create.assert_called_once()
    _, kwargs = mock_client.responses.create.call_args
    assert kwargs["model"] == "gpt-4o-mini"
    assert kwargs["input"] == "user prompt"
    assert kwargs["max_output_tokens"] == 200
    assert kwargs["temperature"] == 0.7
    assert kwargs["instructions"] == "sys prompt"
    assert kwargs["text"] == {"format": {"type": "json_object"}}
    assert kwargs["extra_param"] == "extra_val"


@pytest.mark.anyio
async def test_call_openai_chat_implementation():
    mock_client = MagicMock()
    mock_response = MagicMock()
    
    choice = MagicMock()
    choice.message.content = "  hello chat  "
    mock_response.choices = [choice]
    
    mock_response.usage.prompt_tokens = 30
    mock_response.usage.completion_tokens = 25
    mock_response.usage.prompt_tokens_details.cached_tokens = 12
    
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
    
    res = await _call_openai_chat(
        mock_client,
        "gpt-4o-mini",
        "user prompt",
        system_prompt="sys prompt",
        max_tokens=300,
        temperature=0.8,
        json_mode=True,
        another_param="another_val"
    )
    
    assert res == ("hello chat", 30, 25, 12)
    
    mock_client.chat.completions.create.assert_called_once()
    _, kwargs = mock_client.chat.completions.create.call_args
    assert kwargs["model"] == "gpt-4o-mini"
    assert kwargs["messages"] == [
        {"role": "system", "content": "sys prompt"},
        {"role": "user", "content": "user prompt"}
    ]
    assert kwargs["max_tokens"] == 300
    assert kwargs["temperature"] == 0.8
    assert kwargs["response_format"] == {"type": "json_object"}
    assert kwargs["another_param"] == "another_val"
