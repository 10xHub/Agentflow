"""Testing utilities for Agentflow.

This module provides utilities for testing Agentflow graphs and agents,
including:

- TestAgent: Mock agent that returns predefined responses
- TestContext: Helper for test environment setup
- MockToolRegistry: Tool call tracking for testing
- MockMCPClient: Mock MCP client for testing MCP tool integrations

Example:
    ```python
    from agentflow.qa.testing import TestAgent, TestContext, MockToolRegistry

    # Use TestAgent as a drop-in replacement for Agent
    test_agent = TestAgent(responses=["Hello from test!"])

    # Use TestContext for isolated test setup
    with TestContext() as ctx:
        graph = ctx.create_graph()
        graph.add_node("MAIN", ctx.create_test_agent(responses=["Hi!"]))
        # ... run tests

    # Use MockToolRegistry for tracking tool calls
    tools = MockToolRegistry()
    tools.register("get_weather", lambda city: f"Sunny in {city}")
    # ... after test
    assert tools.was_called("get_weather")

    # Use MockMCPClient for testing MCP tools
    from agentflow.qa.testing import MockMCPClient

    mock_mcp = MockMCPClient()
    mock_mcp.add_tool(
        name="mcp_weather",
        description="Get weather",
        parameters={"city": {"type": "string"}},
        handler=lambda city: f"Weather in {city}: Sunny",
    )
    tool_node = ToolNode([], client=mock_mcp)
    ```
"""

import logging
from typing import Any

from injectq import InjectQ

from .in_memory_store import InMemoryStore
from .mock_mcp import MockMCPClient
from .mock_tools import MockToolRegistry
from .quick_test import QuickTest
from .test_agent import TestAgent
from .test_result import TestResult


logger = logging.getLogger("agentflow.testing")


class TestContext:
    """Helper for test environment setup.

    Provides convenience methods for common test patterns including:
    - Isolated dependency container
    - In-memory store
    - Graph and agent factory methods

    Note: This is optional - users can set up tests manually without it.

    Example:
        ```python
        with TestContext() as ctx:
            graph = ctx.create_graph()
            graph.add_node("MAIN", ctx.create_test_agent(responses=["Hi!"]))
            graph.set_entry_point("MAIN")
            graph.add_edge("MAIN", END)
            compiled = graph.compile()
            result = await compiled.ainvoke({"messages": [...]})
        ```
    """

    def __init__(self):
        """Initialize a test context with isolated container and store."""
        self.container = InjectQ()
        self.store = InMemoryStore()
        self._mock_tools = MockToolRegistry()
        logger.debug("TestContext initialized")

    def __enter__(self) -> "TestContext":
        """Enter context manager - activates isolated container."""
        self.container.activate()
        logger.debug("TestContext entered, container activated")
        return self

    def __exit__(self, *args: Any) -> None:
        """Exit context manager - cleans up resources."""
        # InjectQ doesn't have deactivate, so we just clear the store
        self.store.clear()
        self._mock_tools.clear()
        logger.debug("TestContext exited, resources cleaned up")

    def create_graph(self, state: Any | None = None) -> "StateGraph":
        """Create a StateGraph with the test container.

        Args:
            state: Optional initial state

        Returns:
            New StateGraph instance with test container
        """
        from agentflow.core.graph import StateGraph

        return StateGraph(state=state, container=self.container)

    def create_test_agent(
        self,
        responses: list[str] | None = None,
        model: str = "test-model",
        system_prompt: list[dict[str, Any]] | None = None,
    ) -> TestAgent:
        """Create a TestAgent with predefined responses.

        Args:
            responses: List of responses to return (cycles on multiple calls)
            model: Model identifier (for compatibility)
            system_prompt: Optional system prompt

        Returns:
            New TestAgent instance
        """
        return TestAgent(
            model=model,
            system_prompt=system_prompt,
            responses=responses,
        )

    def get_store(self) -> InMemoryStore:
        """Get the in-memory store for this test context.

        Returns:
            InMemoryStore instance
        """
        return self.store

    def get_mock_tools(self) -> MockToolRegistry:
        """Get the mock tool registry for this test context.

        Returns:
            MockToolRegistry instance
        """
        return self._mock_tools

    def register_mock_tool(
        self,
        name: str,
        func: Any,
        description: str | None = None,
    ) -> "TestContext":
        """Register a mock tool in the registry.

        Args:
            name: Tool name
            func: Tool function
            description: Optional tool description

        Returns:
            Self for method chaining
        """
        self._mock_tools.register(name, func, description)
        return self

    def reset(self) -> None:
        """Reset all test state.

        Clears store, mock tools, and resets all tracking.
        """
        self.store.clear()
        self._mock_tools.clear()
        logger.debug("TestContext reset")


# Type alias for documentation
StateGraph = Any  # Avoid circular import


__all__ = [
    "InMemoryStore",
    "MockMCPClient",
    "MockToolRegistry",
    "QuickTest",
    "TestAgent",
    "TestContext",
    "TestResult",
]
