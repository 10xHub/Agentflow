"""
Comprehensive test for custom state functionality.
Tests state transitions, checkpointing, and type safety.
"""

from dataclasses import dataclass, field
from typing import Any

from agentflow.storage.checkpointer import InMemoryCheckpointer
from agentflow.core.graph import StateGraph
from agentflow.core.state.agent_state import AgentState
from agentflow.core.state import Message
from agentflow.utils.converter import convert_messages


@dataclass
class TestCustomState(AgentState):
    """Test state with various field types."""

    name: str = "Test"
    count: int = 0
    items: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    score: float = 0.0


@dataclass
class SimpleState(AgentState):
    """Simple state for basic testing."""

    value: str = "default"


async def increment_agent(state: TestCustomState, config: dict[str, Any]) -> TestCustomState:
    """Agent that increments count and updates state."""
    print(f"increment_agent received state type: {type(state)}")
    print(f"increment_agent state.items type: {type(state.items)}")
    print(f"increment_agent state.items value: {state.items}")
    print(f"increment_agent state.count: {state.count}")

    state.count += 1
    print(f"Count incremented to: {state.count}")

    try:
        print(f"About to append to items (type: {type(state.items)})")
        state.items.append(f"item_{state.count}")
        print(f"Successfully appended. Items now: {state.items}")
    except Exception as e:
        print(f"Failed to append: {e}")
        print(f"Items dir: {dir(state.items)[:10]}")
        raise

    state.metadata[f"step_{state.count}"] = {"processed": True}
    state.score += 0.1

    # Add a message to context
    message = Message.text_message(f"Processed step {state.count}")
    state.context.append(message)

    return state


async def chat_agent(state: TestCustomState, config: dict[str, Any]) -> Message:
    """Agent that generates a deterministic chat response."""
    messages = convert_messages(
        system_prompts=[{"role": "system", "content": f"You are processing item #{state.count}"}],
        state=state,
    )
    return Message.text_message(
        f"Processed item #{state.count} with {len(messages)} total messages.",
    )


def test_basic_custom_state():
    """Test basic custom state functionality."""
    print("=== Testing Basic Custom State ===")

    # Create graph with custom state
    graph = StateGraph[TestCustomState](TestCustomState())
    graph.add_node("INCREMENT", increment_agent)
    graph.set_entry_point("INCREMENT")

    checkpointer = InMemoryCheckpointer[TestCustomState]()
    app = graph.compile(checkpointer=checkpointer)

    # Test state updates
    config = {"thread_id": "test1"}
    result = app.invoke({"messages": [Message.from_text("Start test")]}, config=config)

    # Retrieve state from checkpointer
    saved_state = checkpointer.get_state(config)
    assert saved_state is not None
    assert saved_state.count == 1
    assert len(saved_state.items) == 1
    assert saved_state.items[0] == "item_1"
    assert "step_1" in saved_state.metadata
    print(f"✓ State updated correctly: count={saved_state.count}, items={saved_state.items}")

    # Test continuation
    result2 = app.invoke({"messages": [Message.from_text("Continue test")]}, config=config)
    saved_state2 = checkpointer.get_state(config)
    assert saved_state2.count == 2
    assert len(saved_state2.items) == 2
    print(f"✓ State continuation works: count={saved_state2.count}, items={saved_state2.items}")


def test_state_type_safety():
    """Test type safety with different state types."""
    print("\n=== Testing State Type Safety ===")

    # Test with SimpleState
    simple_graph = StateGraph[SimpleState](SimpleState())

    async def simple_agent(state: SimpleState, config: dict[str, Any]) -> SimpleState:
        state.value = "updated"
        return state

    simple_graph.add_node("SIMPLE", simple_agent)
    simple_graph.set_entry_point("SIMPLE")

    simple_checkpointer = InMemoryCheckpointer[SimpleState]()
    simple_app = simple_graph.compile(checkpointer=simple_checkpointer)

    config = {"thread_id": "simple_test"}
    result = simple_app.invoke({"messages": [Message.from_text("Simple test")]}, config=config)

    saved_state = simple_checkpointer.get_state(config)
    assert saved_state is not None
    assert saved_state.value == "updated"
    print(f"✓ SimpleState works correctly: value={saved_state.value}")


def test_state_serialization():
    """Test state serialization/deserialization."""
    print("\n=== Testing State Serialization ===")

    state = TestCustomState()
    state.name = "Test State"
    state.count = 42
    state.items = ["a", "b", "c"]
    state.metadata = {"key": "value", "nested": {"inner": True}}
    state.score = 3.14

    # Test model_dump
    state_dict = state.model_dump()
    assert "context" in state_dict
    assert "context_summary" in state_dict
    assert "active_node" in state_dict
    assert "step" in state_dict
    print("✓ State serialization works")

    # Test with internal metadata
    state_dict_full = state.to_dict(include_internal=True)
    assert "execution_meta" in state_dict_full
    print("✓ Full state serialization (with internals) works")


def test_mixed_return_types():
    """Test agents returning different types."""
    print("\n=== Testing Mixed Return Types ===")

    graph = StateGraph[TestCustomState](TestCustomState())

    # Agent that returns updated state
    graph.add_node("STATE_RETURN", increment_agent)

    # Agent that returns a message
    graph.add_node("MESSAGE_RETURN", chat_agent)

    # Connect them
    graph.add_edge("STATE_RETURN", "MESSAGE_RETURN")
    graph.set_entry_point("STATE_RETURN")

    checkpointer = InMemoryCheckpointer[TestCustomState]()
    app = graph.compile(checkpointer=checkpointer)

    config = {"thread_id": "mixed_test"}
    result = app.invoke({"messages": [Message.from_text("Test mixed returns")]}, config=config)

    saved_state = checkpointer.get_state(config)
    assert saved_state is not None
    assert saved_state.count == 1  # From increment_agent
    assert len(saved_state.context) >= 2  # Original message + increment message + chat response
    print(
        f"✓ Mixed return types work: count={saved_state.count}, context_len={len(saved_state.context)}"
    )


def test_default_agent_state():
    """Test that default AgentState still works."""
    print("\n=== Testing Default AgentState ===")

    # Create graph without specifying state type
    graph = StateGraph()  # Should default to AgentState

    async def default_agent(state: AgentState, config: dict[str, Any]) -> str:
        return f"Hello from default agent! Context has {len(state.context)} messages."

    graph.add_node("DEFAULT", default_agent)
    graph.set_entry_point("DEFAULT")

    checkpointer = InMemoryCheckpointer()  # Should work with default AgentState
    app = graph.compile(checkpointer=checkpointer)

    config = {"thread_id": "default_test"}
    result = app.invoke({"messages": [Message.from_text("Test default state")]}, config=config)

    saved_state = checkpointer.get_state(config)
    assert saved_state is not None
    assert len(saved_state.context) >= 2  # Original + response
    print(f"✓ Default AgentState works: context_len={len(saved_state.context)}")


if __name__ == "__main__":
    print("Running comprehensive custom state tests...")

    try:
        test_basic_custom_state()
        test_state_type_safety()
        test_state_serialization()
        test_mixed_return_types()
        test_default_agent_state()

        print("\n🎉 All comprehensive tests passed!")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise
