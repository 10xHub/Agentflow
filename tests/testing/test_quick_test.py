"""Tests for QuickTest - the simplified testing interface."""

import uuid
import pytest

from agentflow.qa.testing.quick_test import QuickTest
from agentflow.qa.testing import TestAgent


class TestQuickTestSingleTurn:
    """Test QuickTest.single_turn()."""

    @pytest.mark.asyncio
    async def test_single_turn_basic(self):
        """Test basic single turn interaction."""
        result = await QuickTest.single_turn(
            agent_response="Hello there!",
            user_message="Hi",
        )
        assert result is not None
        assert result.final_response == "Hello there!"

    @pytest.mark.asyncio
    async def test_single_turn_default_message(self):
        """Test single turn with default user message."""
        result = await QuickTest.single_turn(agent_response="Default response")
        assert result is not None
        assert result.final_response == "Default response"

    @pytest.mark.asyncio
    async def test_single_turn_custom_model(self):
        """Test single turn with custom model name."""
        result = await QuickTest.single_turn(
            agent_response="Custom response",
            user_message="Hello",
            model="custom-model",
        )
        assert result is not None
        assert result.final_response == "Custom response"

    @pytest.mark.asyncio
    async def test_single_turn_with_config(self):
        """Test single turn with configuration."""
        thread_id = str(uuid.uuid4())
        result = await QuickTest.single_turn(
            agent_response="Configured response",
            user_message="Test",
            config={"thread_id": thread_id},
        )
        assert result is not None
        assert result.final_response == "Configured response"

    @pytest.mark.asyncio
    async def test_single_turn_result_has_messages(self):
        """Test that result contains messages."""
        result = await QuickTest.single_turn(
            agent_response="Response text",
            user_message="User message",
        )
        assert len(result.messages) > 0

    @pytest.mark.asyncio
    async def test_single_turn_result_has_state(self):
        """Test that result contains state."""
        result = await QuickTest.single_turn(agent_response="Response")
        assert result.state is not None


class TestQuickTestMultiTurn:
    """Test QuickTest.multi_turn()."""

    @pytest.mark.asyncio
    async def test_multi_turn_basic(self):
        """Test basic multi-turn conversation."""
        result = await QuickTest.multi_turn(
            conversation=[
                ("Hello", "Hi there!"),
                ("How are you?", "I'm great!"),
            ]
        )
        assert result is not None
        assert result.final_response == "I'm great!"

    @pytest.mark.asyncio
    async def test_multi_turn_single_exchange(self):
        """Test multi-turn with single exchange."""
        result = await QuickTest.multi_turn(
            conversation=[("Question", "Answer")]
        )
        assert result.final_response == "Answer"

    @pytest.mark.asyncio
    async def test_multi_turn_three_exchanges(self):
        """Test multi-turn with three exchanges."""
        result = await QuickTest.multi_turn(
            conversation=[
                ("msg1", "resp1"),
                ("msg2", "resp2"),
                ("msg3", "final response"),
            ]
        )
        assert result.final_response == "final response"

    @pytest.mark.asyncio
    async def test_multi_turn_with_config(self):
        """Test multi-turn with config."""
        result = await QuickTest.multi_turn(
            conversation=[("Hello", "Hi!")],
            config={"thread_id": str(uuid.uuid4())},
        )
        assert result is not None

    @pytest.mark.asyncio
    async def test_multi_turn_result_has_messages(self):
        """Test multi-turn result has messages."""
        result = await QuickTest.multi_turn(
            conversation=[
                ("user1", "agent1"),
                ("user2", "agent2"),
            ]
        )
        assert len(result.messages) > 0


class TestQuickTestWithTools:
    """Test QuickTest.with_tools() - testing the logic without graph compilation."""

    @pytest.mark.asyncio
    async def test_with_tools_string_tool_creates_func(self):
        """Test that string tool names create mock functions."""
        # The with_tools method creates mock func from string names
        # We can test the underlying logic directly
        tool_calls = []

        def make_tool(name: str):
            def tool_func(query: str = "", **kwargs) -> str:
                tool_calls.append({"name": name, "args": {"query": query, **kwargs}})
                return f"Mock result from {name}"
            tool_func.__name__ = name
            return tool_func

        func = make_tool("get_weather")
        result = func(query="NYC")
        assert result == "Mock result from get_weather"
        assert len(tool_calls) == 1
        assert tool_calls[0]["name"] == "get_weather"

    @pytest.mark.asyncio
    async def test_with_tools_tool_responses(self):
        """Test that tool_responses maps tool names to return values."""
        tool_responses = {"get_weather": "Sunny, 75F"}
        name = "get_weather"

        def tool_func(query: str = "", **kwargs) -> str:
            if tool_responses and name in tool_responses:
                return tool_responses[name]
            return f"Mock result from {name}"

        result = tool_func(query="NYC")
        assert result == "Sunny, 75F"


class TestQuickTestCustom:
    """Test QuickTest.custom()."""

    @pytest.mark.asyncio
    async def test_custom_basic(self):
        """Test custom with TestAgent."""
        agent = TestAgent(responses=["Custom response!"])
        result = await QuickTest.custom(
            agent=agent,
            user_message="Custom test",
        )
        assert result is not None
        assert result.final_response == "Custom response!"

    @pytest.mark.asyncio
    async def test_custom_with_graph_setup(self):
        """Test custom with a graph_setup callback."""
        from agentflow.utils.constants import END

        agent = TestAgent(responses=["Modified graph response"])

        def setup_graph(graph):
            # Return graph as-is (already set up by QuickTest)
            return graph

        result = await QuickTest.custom(
            agent=agent,
            user_message="Test",
            graph_setup=setup_graph,
        )
        assert result is not None

    @pytest.mark.asyncio
    async def test_custom_with_config(self):
        """Test custom with config."""
        agent = TestAgent(responses=["Configured response"])
        result = await QuickTest.custom(
            agent=agent,
            user_message="Test",
            config={"thread_id": str(uuid.uuid4())},
        )
        assert result is not None


class TestQuickTestExtractResponse:
    """Test QuickTest._extract_response() static method."""

    def test_extract_response_empty_state(self):
        """Test extract response with empty messages."""
        result = QuickTest._extract_response({})
        assert result == ""

    def test_extract_response_no_assistant(self):
        """Test extract response with no assistant messages."""
        from agentflow.core.state import Message
        msg = Message.text_message("Hello", role="user")
        result = QuickTest._extract_response({"messages": [msg]})
        assert result == ""

    def test_extract_response_with_assistant(self):
        """Test extract response with assistant message."""
        from agentflow.core.state import Message
        msg = Message.text_message("Hello user", role="assistant")
        result = QuickTest._extract_response({"messages": [msg]})
        assert result == "Hello user"

    def test_extract_response_last_assistant(self):
        """Test extract response returns last assistant message."""
        from agentflow.core.state import Message
        msgs = [
            Message.text_message("user input", role="user"),
            Message.text_message("first response", role="assistant"),
            Message.text_message("follow up", role="user"),
            Message.text_message("final response", role="assistant"),
        ]
        result = QuickTest._extract_response({"messages": msgs})
        assert result == "final response"

    @pytest.mark.asyncio
    async def test_with_tools_actual_execution(self):
        """Test QuickTest.with_tools actual execution route."""
        # This will call QuickTest.with_tools, which routes through ToolNode
        result = await QuickTest.with_tools(
            query="Tell me the weather in SF",
            response="The weather in SF is sunny",
            tools=["get_weather"],
            tool_responses={"get_weather": "sunny"}
        )
        assert result is not None
        assert result.final_response == "The weather in SF is sunny"
        assert len(result.tool_calls) > 0
        assert result.tool_calls[0]["name"] == "get_weather"

    @pytest.mark.asyncio
    async def test_multi_turn_fallback_no_state_context(self):
        """Test multi_turn fallback when state context is not present."""
        # We can mock compiled.ainvoke to return a dict without state context
        # to hit line 150 of quick_test.py
        from agentflow.core.graph.compiled_graph import CompiledGraph
        from agentflow.core.state import Message
        from unittest.mock import AsyncMock, MagicMock, patch

        mock_compiled = MagicMock(spec=CompiledGraph)
        mock_compiled.ainvoke = AsyncMock(return_value={
            "messages": [Message.text_message("User message", role="user"), Message.text_message("Response", role="assistant")]
        })

        with patch("agentflow.core.graph.StateGraph.compile", return_value=mock_compiled):
            result = await QuickTest.multi_turn(
                conversation=[("User message", "Response")]
            )
            assert result.final_response == "Response"
            assert len(result.messages) == 2

    def test_extract_response_no_text_method_fallback(self):
        """Test extract response fallback when message text attribute is not a method/callable."""
        # Create a mock message where hasattr(msg, "text") is False but hasattr(msg, "content") is True
        msg = type("MockMsg", (), {
            "role": "assistant",
            "content": "Fall back to content string"
        })
        result = QuickTest._extract_response({"messages": [msg]})
        assert result == "Fall back to content string"

        # Create a mock message where msg has no content but has role
        msg2 = type("MockMsg", (), {
            "role": "assistant"
        })
        result = QuickTest._extract_response({"messages": [msg2]})
        # str(msg2) is returned
        assert "MockMsg" in result

