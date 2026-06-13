"""
Basic tests for OpenAI converter functionality.
"""

import json
from unittest.mock import Mock, patch
from typing import Any

import pytest

from agentflow.runtime.adapters.llm.openai_converter import OpenAIConverter
from agentflow.core.state.message import Message, TokenUsages
from agentflow.core.state.message_block import (
    AudioBlock,
    ImageBlock,
    ReasoningBlock,
    TextBlock,
    ToolCallBlock,
)


class MockModelResponse:
    """Mock ChatCompletion response for testing."""

    def __init__(self, data):
        self.id = data.get("id", "test_id")
        self.model = data.get("model", "gpt-4o")
        self.created = data.get("created", 1234567890)
        
        # Mock usage
        usage_data = data.get("usage", {})

        # Build nested detail objects when provided
        completion_details_data = usage_data.get("completion_tokens_details", None)
        prompt_details_data = usage_data.get("prompt_tokens_details", None)
        completion_details = (
            type("CompletionDetails", (), completion_details_data)
            if isinstance(completion_details_data, dict) else completion_details_data
        )
        prompt_details = (
            type("PromptDetails", (), prompt_details_data)
            if isinstance(prompt_details_data, dict) else prompt_details_data
        )

        self.usage = type('Usage', (), {
            'prompt_tokens': usage_data.get('prompt_tokens', 10),
            'completion_tokens': usage_data.get('completion_tokens', 20),
            'total_tokens': usage_data.get('total_tokens', 30),
            'completion_tokens_details': completion_details,
            'prompt_tokens_details': prompt_details,
        })
        
        # Mock choices
        choices_data = data.get("choices", [{}])
        self.choices = []
        for choice_data in choices_data:
            message_data = choice_data.get("message", {})
            message = type('Message', (), {
                'role': message_data.get('role', 'assistant'),
                'content': message_data.get('content'),
                'audio': message_data.get('audio'),
                'images': message_data.get('images'),
                'reasoning_content': message_data.get('reasoning_content'),
                'tool_calls': message_data.get('tool_calls')
            })
            choice = type('Choice', (), {
                'message': message,
                'finish_reason': choice_data.get('finish_reason', 'stop')
            })
            self.choices.append(choice)


class TestOpenAIConverter:
    """Test class for OpenAI converter."""

    @pytest.fixture
    def converter(self):
        """Provide OpenAIConverter instance."""
        return OpenAIConverter()

    @patch("agentflow.runtime.adapters.llm.openai_converter.HAS_OPENAI", True)
    @pytest.mark.asyncio
    async def test_basic_text_conversion(self, converter):
        """Test basic text message conversion."""
        response_data = {
            "id": "chatcmpl-123",
            "model": "gpt-4o",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Hello, world!",
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        }

        response = MockModelResponse(response_data)
        message = await converter.convert_response(response)

        assert isinstance(message, Message)
        assert message.role == "assistant"
        assert len(message.content) == 1
        assert isinstance(message.content[0], TextBlock)
        assert message.content[0].text == "Hello, world!"

    @patch("agentflow.runtime.adapters.llm.openai_converter.HAS_OPENAI", True)
    @pytest.mark.asyncio
    async def test_audio_conversion(self, converter):
        """Test audio content conversion."""
        response_data = {
            "id": "chatcmpl-audio",
            "choices": [
                {
                    "message": {
                        "content": "Audio response",
                        "audio": {
                            "id": "audio_123",
                            "data": "base64encodeddata",
                            "transcript": "Hello from audio",
                        }
                    }
                }
            ],
            "usage": {}
        }

        response = MockModelResponse(response_data)
        message = await converter.convert_response(response)

        assert len(message.content) == 2
        assert isinstance(message.content[0], TextBlock)
        assert message.content[0].text == "Audio response"
        assert isinstance(message.content[1], AudioBlock)
        assert message.content[1].transcript == "Hello from audio"
        assert message.content[1].media.data_base64 == "base64encodeddata"

    @patch("agentflow.runtime.adapters.llm.openai_converter.HAS_OPENAI", True)
    @pytest.mark.asyncio
    async def test_images_conversion(self, converter):
        """Test image content conversion."""
        response_data = {
            "id": "chatcmpl-images",
            "choices": [
                {
                    "message": {
                        "content": "Here are images",
                        "images": [
                            {"url": "https://example.com/image1.png"},
                            {"url": "https://example.com/image2.png"}
                        ]
                    }
                }
            ],
            "usage": {}
        }

        response = MockModelResponse(response_data)
        message = await converter.convert_response(response)

        # Should have 1 text block + 2 image blocks
        assert len(message.content) == 3
        assert isinstance(message.content[0], TextBlock)
        assert isinstance(message.content[1], ImageBlock)
        assert isinstance(message.content[2], ImageBlock)
        assert message.content[1].media.url == "https://example.com/image1.png"
        assert message.content[2].media.url == "https://example.com/image2.png"

    @patch("agentflow.runtime.adapters.llm.openai_converter.HAS_OPENAI", True)
    @pytest.mark.asyncio
    async def test_tool_calls_conversion(self, converter):
        """Test tool call conversion."""
        response_data = {
            "id": "chatcmpl-tools",
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            type('ToolCall', (), {
                                'id': 'call_123',
                                'type': 'function',
                                'function': type('Function', (), {
                                    'name': 'get_weather',
                                    'arguments': json.dumps({"location": "SF"})
                                })
                            })
                        ]
                    }
                }
            ],
            "usage": {}
        }

        response = MockModelResponse(response_data)
        message = await converter.convert_response(response)

        assert len(message.content) == 1
        assert isinstance(message.content[0], ToolCallBlock)
        assert message.content[0].name == "get_weather"
        assert message.content[0].args == {"location": "SF"}
        assert message.content[0].id == "call_123"

    @patch("agentflow.runtime.adapters.llm.openai_converter.HAS_OPENAI", True)
    @pytest.mark.asyncio
    async def test_empty_content(self, converter):
        """Test handling of empty/null content."""
        response_data = {
            "id": "chatcmpl-empty",
            "choices": [
                {
                    "message": {
                        "content": None,
                    }
                }
            ],
            "usage": {}
        }

        response = MockModelResponse(response_data)
        message = await converter.convert_response(response)

        assert isinstance(message, Message)
        assert len(message.content) == 0

    @patch("agentflow.runtime.adapters.llm.openai_converter.HAS_OPENAI", True)
    @pytest.mark.asyncio
    async def test_token_usage(self, converter):
        """Test token usage extraction."""
        response_data = {
            "id": "chatcmpl-usage",
            "choices": [
                {
                    "message": {
                        "content": "Test",
                    }
                }
            ],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150
            }
        }

        response = MockModelResponse(response_data)
        message = await converter.convert_response(response)

        assert isinstance(message.usages, TokenUsages)
        assert message.usages.prompt_tokens == 100
        assert message.usages.completion_tokens == 50
        assert message.usages.total_tokens == 150

    @patch("agentflow.runtime.adapters.llm.openai_converter.HAS_OPENAI", True)
    @pytest.mark.asyncio
    async def test_metadata_extraction(self, converter):
        """Test metadata extraction."""
        response_data = {
            "id": "chatcmpl-meta",
            "model": "gpt-4o-2024-05-13",
            "choices": [
                {
                    "message": {
                        "content": "Test",
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {}
        }

        response = MockModelResponse(response_data)
        message = await converter.convert_response(response)

        assert message.metadata["provider"] == "openai"
        assert message.metadata["model"] == "gpt-4o-2024-05-13"
        assert message.metadata["finish_reason"] == "stop"


# ---------------------------------------------------------------------------
# Reasoning extraction (4-layer cascade in OpenAIConverter)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestOpenAIReasoningExtraction:
    """Test reasoning block extraction from ChatCompletion responses."""

    @patch("agentflow.runtime.adapters.llm.openai_converter.HAS_OPENAI", True)
    async def test_reasoning_content_field(self):
        """reasoning_content field → ReasoningBlock."""
        response = MockModelResponse({
            "choices": [{
                "message": {
                    "content": "x³/3 + C",
                    "reasoning_content": "Apply power rule: ∫x² dx = x³/3 + C",
                },
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        })
        converter = OpenAIConverter()
        msg = await converter.convert_response(response)

        text_blocks = [b for b in msg.content if isinstance(b, TextBlock)]
        reasoning_blocks = [b for b in msg.content if isinstance(b, ReasoningBlock)]
        assert len(text_blocks) == 1
        assert len(reasoning_blocks) == 1
        assert "power rule" in reasoning_blocks[0].summary
        assert msg.reasoning != ""

    @patch("agentflow.runtime.adapters.llm.openai_converter.HAS_OPENAI", True)
    async def test_think_tag_extraction(self):
        """<think> tags stripped from text, reasoning extracted."""
        response = MockModelResponse({
            "choices": [{
                "message": {
                    "content": "<think>Rayleigh scattering</think>The sky is blue.",
                    "reasoning_content": None,
                },
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15},
        })
        converter = OpenAIConverter()
        msg = await converter.convert_response(response)

        text_blocks = [b for b in msg.content if isinstance(b, TextBlock)]
        reasoning_blocks = [b for b in msg.content if isinstance(b, ReasoningBlock)]
        assert len(text_blocks) == 1
        assert "<think>" not in text_blocks[0].text
        assert len(reasoning_blocks) == 1
        assert "Rayleigh" in reasoning_blocks[0].summary

    @patch("agentflow.runtime.adapters.llm.openai_converter.HAS_OPENAI", True)
    async def test_reasoning_content_takes_precedence(self):
        """Field value wins over <think> tags when both present."""
        response = MockModelResponse({
            "choices": [{
                "message": {
                    "content": "<think>Tag reasoning</think>Answer.",
                    "reasoning_content": "Field reasoning",
                },
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
        })
        converter = OpenAIConverter()
        msg = await converter.convert_response(response)

        reasoning_blocks = [b for b in msg.content if isinstance(b, ReasoningBlock)]
        assert len(reasoning_blocks) == 1
        assert reasoning_blocks[0].summary == "Field reasoning"

    @patch("agentflow.runtime.adapters.llm.openai_converter.HAS_OPENAI", True)
    async def test_no_reasoning_no_block(self):
        """Standard response produces zero ReasoningBlocks."""
        response = MockModelResponse({
            "choices": [{
                "message": {
                    "content": "Hello, world!",
                    "reasoning_content": None,
                },
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
        })
        converter = OpenAIConverter()
        msg = await converter.convert_response(response)

        assert not any(isinstance(b, ReasoningBlock) for b in msg.content)
        assert msg.reasoning == ""


# ---------------------------------------------------------------------------
# Token usage details (reasoning tokens, cache tokens)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestTokenUsageDetails:
    """Test token usage extraction including reasoning and cache tokens."""

    @patch("agentflow.runtime.adapters.llm.openai_converter.HAS_OPENAI", True)
    async def test_reasoning_tokens_extracted(self):
        """completion_tokens_details.reasoning_tokens mapped correctly."""
        response = MockModelResponse({
            "choices": [{"message": {"content": "Reasoned answer"}}],
            "usage": {
                "prompt_tokens": 25,
                "completion_tokens": 60,
                "total_tokens": 85,
                "completion_tokens_details": {"reasoning_tokens": 35},
            },
        })
        converter = OpenAIConverter()
        msg = await converter.convert_response(response)

        assert msg.usages.reasoning_tokens == 35
        assert msg.usages.completion_tokens == 60

    @patch("agentflow.runtime.adapters.llm.openai_converter.HAS_OPENAI", True)
    async def test_cache_read_token_mapping(self):
        """cached_tokens → cache_read_input_tokens (not cache_creation)."""
        response = MockModelResponse({
            "choices": [{"message": {"content": "Cached"}}],
            "usage": {
                "prompt_tokens": 50,
                "completion_tokens": 20,
                "total_tokens": 70,
                "prompt_tokens_details": {"cached_tokens": 40},
            },
        })
        converter = OpenAIConverter()
        msg = await converter.convert_response(response)

        assert msg.usages.cache_read_input_tokens == 40
        assert msg.usages.cache_creation_input_tokens == 0

    @patch("agentflow.runtime.adapters.llm.openai_converter.HAS_OPENAI", True)
    async def test_no_token_details_defaults_to_zero(self):
        """None details → reasoning_tokens == 0, cache_read == 0."""
        response = MockModelResponse({
            "choices": [{"message": {"content": "Test"}}],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
                "completion_tokens_details": None,
                "prompt_tokens_details": None,
            },
        })
        converter = OpenAIConverter()
        msg = await converter.convert_response(response)

        assert msg.usages.reasoning_tokens == 0
        assert msg.usages.cache_read_input_tokens == 0
        assert msg.usages.cache_creation_input_tokens == 0


@pytest.fixture
def converter():
    return OpenAIConverter()


@pytest.mark.asyncio
async def test_openai_converter_import_errors(monkeypatch):
    monkeypatch.setattr("agentflow.runtime.adapters.llm.openai_converter.HAS_OPENAI", False)
    converter = OpenAIConverter()
    
    with pytest.raises(ImportError, match="openai is not installed"):
        await converter.convert_response(Mock())
        
    with pytest.raises(ImportError, match="openai is not installed"):
        async for _ in converter.convert_streaming_response({}, "node", Mock()):
            pass


def test_extract_audio_block_exceptions(converter):
    assert converter._extract_audio_block({"transcript": "hi"}) is None
    assert converter._extract_audio_block(object()) is None


def test_extract_image_blocks_various_types(converter):
    img = "https://example.com/single.png"
    blocks = converter._extract_image_blocks(img)
    assert len(blocks) == 1
    assert blocks[0].media.url == img
    
    with patch("agentflow.runtime.adapters.llm.openai_converter.isinstance", side_effect=TypeError("mock error")):
        blocks2 = converter._extract_image_blocks("url")
        assert blocks2 == []


@pytest.mark.asyncio
async def test_streaming_conversion(converter):
    class MockChunk:
        def __init__(self, id, content=None, reasoning=None, tool_calls=None):
            self.id = id
            self.model = "gpt-4o"
            
            delta_obj = type("Delta", (), {
                "content": content,
                "reasoning_content": reasoning,
                "tool_calls": tool_calls,
                "audio": None,
                "images": None
            })
            choice = type("Choice", (), {
                "delta": delta_obj
            })
            self.choices = [choice]

    chunk1 = MockChunk("chat-1", reasoning="Thinking...")
    chunk2 = MockChunk("chat-1", content="Hello ")
    chunk3 = MockChunk("chat-1", content="world!")
    
    tool_call_mock = type("ToolCall", (), {
        "id": "tc-123",
        "type": "function",
        "function": type("Func", (), {
            "name": "calc",
            "arguments": '{"x": 1}'
        })
    })
    chunk4 = MockChunk("chat-1", tool_calls=[tool_call_mock])

    async def mock_async_stream():
        yield chunk1
        yield chunk2
        yield chunk3
        yield chunk4

    messages = []
    async for msg in converter.convert_streaming_response({}, "my_node", mock_async_stream()):
        messages.append(msg)

    assert len(messages) == 5
    assert messages[0].reasoning == "Thinking..."
    assert messages[1].content[0].text == "Hello "
    assert messages[2].content[0].text == "world!"
    assert messages[3].tools_calls[0]["id"] == "tc-123"
    
    final_msg = messages[-1]
    assert final_msg.delta is False
    assert final_msg.reasoning == "Thinking..."
    assert final_msg.content[0].text == "Hello world!"
    assert final_msg.content[1].summary == "Thinking..."
    assert final_msg.content[2].name == "calc"


@pytest.mark.asyncio
async def test_streaming_inline_think_thought(converter):
    class MockChunk:
        def __init__(self, id, content):
            self.id = id
            delta_obj = type("Delta", (), {
                "content": content,
                "reasoning_content": None,
                "tool_calls": None,
                "audio": None,
                "images": None
            })
            self.choices = [type("Choice", (), {"delta": delta_obj})]

    async def mock_stream():
        yield MockChunk("chat-2", "<think>Inline thoughts</think>Actual content")

    messages = []
    async for msg in converter.convert_streaming_response({}, "node", mock_stream()):
        messages.append(msg)

    final_msg = messages[-1]
    assert final_msg.reasoning == "Inline thoughts"
    assert final_msg.content[0].text == "Actual content"


@pytest.mark.asyncio
async def test_streaming_sync_iterator(converter):
    class MockChunk:
        def __init__(self, id, content):
            self.id = id
            delta_obj = type("Delta", (), {
                "content": content,
                "reasoning_content": None,
                "tool_calls": None,
                "audio": None,
                "images": None
            })
            self.choices = [type("Choice", (), {"delta": delta_obj})]

    class SyncStream:
        def __init__(self, chunks):
            self.chunks = chunks
        def __iter__(self):
            return iter(self.chunks)

    stream = SyncStream([MockChunk("chat-3", "Sync chunk")])
    messages = []
    async for msg in converter.convert_streaming_response({}, "node", stream):
        messages.append(msg)

    assert len(messages) == 2
    assert messages[0].content[0].text == "Sync chunk"


@pytest.mark.asyncio
async def test_convert_streaming_response_chat_completion(converter):
    response_data = {
        "id": "chatcmpl-123",
        "model": "gpt-4o",
        "choices": [{"message": {"role": "assistant", "content": "Hello"}}],
        "usage": {}
    }
    response = MockModelResponse(response_data)
    messages = []
    with patch("agentflow.runtime.adapters.llm.openai_converter.ChatCompletion", MockModelResponse):
        async for msg in converter.convert_streaming_response({}, "node", response):
            messages.append(msg)
    
    assert len(messages) == 1
    assert messages[0].content[0].text == "Hello"


@pytest.mark.asyncio
async def test_convert_streaming_response_unsupported(converter):
    with pytest.raises(Exception, match="Unsupported response type"):
        async for _ in converter.convert_streaming_response({}, "node", object()):
            pass

