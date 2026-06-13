"""Tests for token tracking functionality in graph execution handlers.

This module tests:
- Token usage calculation from agent state
- Token tracking in StreamHandler
- Token tracking in InvokeHandler
"""

import pytest

from agentflow.core.graph import StateGraph
from agentflow.core.graph.utils.utils import calculate_token_usage
from agentflow.core.state import AgentState, Message, TokenUsages
from agentflow.core.state.message_block import TextBlock
from agentflow.utils import END, ResponseGranularity


class TestCalculateTokenUsage:
    """Test the calculate_token_usage utility function."""

    def test_empty_context(self):
        """Test token calculation with no messages."""
        state = AgentState()
        usage = calculate_token_usage(state.context if state.context else [])

        assert usage["total_input_tokens"] == 0  # noqa: S101
        assert usage["total_output_tokens"] == 0  # noqa: S101
        assert usage["total_reasoning_tokens"] == 0  # noqa: S101
        assert usage["total_tokens"] == 0  # noqa: S101

    def test_single_message_with_usage(self):
        """Test token calculation with a single message."""
        message = Message(
            role="assistant",
            content=[TextBlock(text="Hello")],
            usages=TokenUsages(
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=30,
                reasoning_tokens=5,
            ),
        )

        state = AgentState(context=[message])
        usage = calculate_token_usage(state.context)

        assert usage["total_input_tokens"] == 10  # noqa: S101
        assert usage["total_output_tokens"] == 20  # noqa: S101
        assert usage["total_reasoning_tokens"] == 5  # noqa: S101
        assert usage["total_tokens"] == 30  # noqa: S101

    def test_multiple_messages_with_usage(self):
        """Test token calculation with multiple messages."""
        messages = [
            Message(
                role="user",
                content=[TextBlock(text="Hello")],
                usages=TokenUsages(
                    prompt_tokens=5,
                    completion_tokens=0,
                    total_tokens=5,
                    reasoning_tokens=0,
                ),
            ),
            Message(
                role="assistant",
                content=[TextBlock(text="Hi there!")],
                usages=TokenUsages(
                    prompt_tokens=10,
                    completion_tokens=15,
                    total_tokens=25,
                    reasoning_tokens=3,
                ),
            ),
            Message(
                role="user",
                content=[TextBlock(text="How are you?")],
                usages=TokenUsages(
                    prompt_tokens=8,
                    completion_tokens=0,
                    total_tokens=8,
                    reasoning_tokens=0,
                ),
            ),
            Message(
                role="assistant",
                content=[TextBlock(text="I'm doing well, thanks!")],
                usages=TokenUsages(
                    prompt_tokens=12,
                    completion_tokens=20,
                    total_tokens=32,
                    reasoning_tokens=5,
                ),
            ),
        ]

        state = AgentState(context=messages)
        usage = calculate_token_usage(state.context)

        # Total: 5 + 10 + 8 + 12 = 35 input tokens
        assert usage["total_input_tokens"] == 35  # noqa: S101
        # Total: 0 + 15 + 0 + 20 = 35 output tokens
        assert usage["total_output_tokens"] == 35  # noqa: S101
        # Total: 0 + 3 + 0 + 5 = 8 reasoning tokens
        assert usage["total_reasoning_tokens"] == 8  # noqa: S101
        # Total: 35 + 35 = 70 tokens
        assert usage["total_tokens"] == 70  # noqa: S101

    def test_messages_without_usage(self):
        """Test token calculation with messages that don't have usage data."""
        messages = [
            Message(
                role="user",
                content=[TextBlock(text="Hello")],
                usages=None,
            ),
            Message(
                role="assistant",
                content=[TextBlock(text="Hi there!")],
                usages=TokenUsages(
                    prompt_tokens=10,
                    completion_tokens=15,
                    total_tokens=25,
                    reasoning_tokens=3,
                ),
            ),
        ]

        state = AgentState(context=messages)
        usage = calculate_token_usage(state.context)

        # Only the second message has usage
        assert usage["total_input_tokens"] == 10  # noqa: S101
        assert usage["total_output_tokens"] == 15  # noqa: S101
        assert usage["total_reasoning_tokens"] == 3  # noqa: S101
        assert usage["total_tokens"] == 25  # noqa: S101

    def test_mixed_messages_with_and_without_usage(self):
        """Test token calculation with mix of messages with and without usage."""
        messages = [
            Message(role="user", content=[TextBlock(text="Hello")], usages=None),
            Message(
                role="assistant",
                content=[TextBlock(text="Hi")],
                usages=TokenUsages(
                    prompt_tokens=10, completion_tokens=5, total_tokens=15, reasoning_tokens=2
                ),
            ),
            Message(role="user", content=[TextBlock(text="Thanks")], usages=None),
            Message(
                role="assistant",
                content=[TextBlock(text="You're welcome")],
                usages=TokenUsages(
                    prompt_tokens=8, completion_tokens=12, total_tokens=20, reasoning_tokens=1
                ),
            ),
        ]

        state = AgentState(context=messages)
        usage = calculate_token_usage(state.context)

        assert usage["total_input_tokens"] == 18  # noqa: S101
        assert usage["total_output_tokens"] == 17  # noqa: S101
        assert usage["total_reasoning_tokens"] == 3  # noqa: S101
        assert usage["total_tokens"] == 35  # noqa: S101


@pytest.mark.asyncio
class TestStreamHandlerTokenTracking:
    """Test token tracking in StreamHandler during graph execution."""

    async def test_stream_handler_includes_token_usage_in_final_chunk(self):
        """Test that StreamHandler includes token usage in final stream chunk."""
        from agentflow.core.state.stream_chunks import StreamEvent

        # Define a simple node that returns a message with token usage
        def agent_node(state: AgentState, config: dict) -> list[Message]:
            return [
                Message(
                    role="assistant",
                    content=[TextBlock(text="Hello from agent")],
                    usages=TokenUsages(
                        prompt_tokens=100,
                        completion_tokens=50,
                        total_tokens=150,
                        reasoning_tokens=10,
                    ),
                )
            ]

        # Build and compile graph
        graph = StateGraph(AgentState)
        graph.add_node("agent", agent_node)
        graph.set_entry_point("agent")
        graph.add_edge("agent", END)
        compiled_graph = graph.compile()

        # Execute with streaming
        chunks = []
        async for chunk in compiled_graph.astream(
            {"messages": [Message(role="user", content=[TextBlock(text="Hi")])]},
            response_granularity=ResponseGranularity.FULL,
        ):
            chunks.append(chunk)

        # Find the final UPDATES chunk
        final_chunks = [c for c in chunks if c.event == StreamEvent.UPDATES and c.data.get("status") == "graph_invoked"]
        assert len(final_chunks) > 0  # noqa: S101

        final_chunk = final_chunks[-1]

        # Verify token usage is included
        assert "total_input_tokens" in final_chunk.data  # noqa: S101
        assert "total_output_tokens" in final_chunk.data  # noqa: S101
        assert "total_reasoning_tokens" in final_chunk.data  # noqa: S101
        assert "total_tokens" in final_chunk.data  # noqa: S101

        # Verify token counts (should match what we set in the message)
        assert final_chunk.data["total_input_tokens"] == 100  # noqa: S101
        assert final_chunk.data["total_output_tokens"] == 50  # noqa: S101
        assert final_chunk.data["total_reasoning_tokens"] == 10  # noqa: S101
        assert final_chunk.data["total_tokens"] == 150  # noqa: S101


@pytest.mark.asyncio
class TestInvokeHandlerTokenTracking:
    """Test token tracking in InvokeHandler during graph execution."""

    async def test_invoke_handler_includes_token_usage_in_response(self):
        """Test that InvokeHandler includes token usage in response metadata."""

        # Define a simple node that returns a message with token usage
        def agent_node(state: AgentState, config: dict) -> list[Message]:
            return [
                Message(
                    role="assistant",
                    content=[TextBlock(text="Response from agent")],
                    usages=TokenUsages(
                        prompt_tokens=200,
                        completion_tokens=100,
                        total_tokens=300,
                        reasoning_tokens=20,
                    ),
                )
            ]

        # Build and compile graph
        graph = StateGraph(AgentState)
        graph.add_node("agent", agent_node)
        graph.set_entry_point("agent")
        graph.add_edge("agent", END)
        compiled_graph = graph.compile()

        # Execute with invoke
        result = await compiled_graph.ainvoke(
            {"messages": [Message(role="user", content=[TextBlock(text="Hi")])]},
            response_granularity=ResponseGranularity.FULL,
        )

        # The result should contain the final state
        assert "state" in result  # noqa: S101
        assert "token_usage" in result  # noqa: S101

        # Verify token counts match what we set
        token_usage = result["token_usage"]
        assert token_usage["total_input_tokens"] == 200  # noqa: S101
        assert token_usage["total_output_tokens"] == 100  # noqa: S101
        assert token_usage["total_reasoning_tokens"] == 20  # noqa: S101
        assert token_usage["total_tokens"] == 300  # noqa: S101  # input + output only


@pytest.mark.asyncio
class TestMultipleNodesTokenTracking:
    """Test token tracking across multiple nodes in a graph."""

    async def test_token_accumulation_across_nodes(self):
        """Test that tokens accumulate correctly across multiple nodes."""

        def node1(state: AgentState, config: dict) -> list[Message]:
            return [
                Message(
                    role="assistant",
                    content=[TextBlock(text="Node 1 response")],
                    usages=TokenUsages(
                        prompt_tokens=50,
                        completion_tokens=30,
                        total_tokens=80,
                        reasoning_tokens=5,
                    ),
                )
            ]

        def node2(state: AgentState, config: dict) -> list[Message]:
            return [
                Message(
                    role="assistant",
                    content=[TextBlock(text="Node 2 response")],
                    usages=TokenUsages(
                        prompt_tokens=60,
                        completion_tokens=40,
                        total_tokens=100,
                        reasoning_tokens=8,
                    ),
                )
            ]

        def node3(state: AgentState, config: dict) -> list[Message]:
            return [
                Message(
                    role="assistant",
                    content=[TextBlock(text="Node 3 response")],
                    usages=TokenUsages(
                        prompt_tokens=70,
                        completion_tokens=50,
                        total_tokens=120,
                        reasoning_tokens=10,
                    ),
                )
            ]

        # Build graph with multiple nodes
        graph = StateGraph(AgentState)
        graph.add_node("node1", node1)
        graph.add_node("node2", node2)
        graph.add_node("node3", node3)
        graph.set_entry_point("node1")
        graph.add_edge("node1", "node2")
        graph.add_edge("node2", "node3")
        graph.add_edge("node3", END)
        compiled_graph = graph.compile()

        # Execute
        result = await compiled_graph.ainvoke(
            {"messages": [Message(role="user", content=[TextBlock(text="Start")])]},
            response_granularity=ResponseGranularity.FULL,
        )

        # Verify token usage is included in the result
        assert "token_usage" in result  # noqa: S101
        token_usage = result["token_usage"]

        # Verify accumulated tokens (50 + 60 + 70 = 180 input, 30 + 40 + 50 = 120 output, 5 + 8 + 10 = 23 reasoning)
        assert token_usage["total_input_tokens"] == 180  # noqa: S101
        assert token_usage["total_output_tokens"] == 120  # noqa: S101
        assert token_usage["total_reasoning_tokens"] == 23  # noqa: S101
        assert token_usage["total_tokens"] == 300  # noqa: S101  # input + output only
