"""Tests for the graph module."""

import pytest

from agentflow.core.graph import (
    CompiledGraph,
    Edge,
    Node,
    StateGraph,
    ToolNode,
)
from agentflow.core.state import AgentState, Message
from agentflow.utils import END


class TestEdge:
    """Test the Edge class."""

    def test_edge_creation(self):
        """Test creating an edge."""
        edge = Edge("start", "end")
        assert edge.from_node == "start"  # noqa: S101
        assert edge.to_node == "end"  # noqa: S101

    def test_edge_with_condition(self):
        """Test creating an edge with condition."""

        def condition(state: AgentState) -> bool:
            return len(state.context) > 0

        edge = Edge("start", "end", condition=condition)
        assert edge.condition == condition  # noqa: S101

    def test_edge_attributes(self):
        """Test edge attributes."""
        edge = Edge("a", "b")
        assert hasattr(edge, "from_node")  # noqa: S101
        assert hasattr(edge, "to_node")  # noqa: S101
        assert hasattr(edge, "condition")  # noqa: S101
        assert hasattr(edge, "condition_result")  # noqa: S101


class TestNode:
    """Test the Node class."""

    def test_node_creation_with_function(self):
        """Test creating a node with a regular function."""

        def simple_func(state: AgentState) -> AgentState:
            return state

        node = Node("test_node", simple_func)
        assert node.name == "test_node"  # noqa: S101
        assert node.func == simple_func  # noqa: S101

    def test_node_creation_with_tool_node(self):
        """Test creating a node with a ToolNode."""

        def tool_func(x: str) -> str:
            return x

        tool_node = ToolNode([tool_func])
        node = Node("test_tool", tool_node)
        assert node.name == "test_tool"  # noqa: S101
        assert node.func == tool_node  # noqa: S101

    def test_node_attributes(self):
        """Test node attributes."""

        def simple_func(state: AgentState) -> AgentState:
            return state

        node = Node("test", simple_func)
        assert hasattr(node, "name")  # noqa: S101
        assert hasattr(node, "func")  # noqa: S101
        assert hasattr(node, "publisher")  # noqa: S101


class TestToolNode:
    """Test the ToolNode class."""

    def test_tool_node_creation(self):
        """Test creating a ToolNode."""

        def simple_tool(input_data: str) -> str:
            return f"processed: {input_data}"

        tool_node = ToolNode([simple_tool])
        assert hasattr(tool_node, "mcp_tools")  # noqa: S101

    def test_tool_node_with_multiple_functions(self):
        """Test creating a ToolNode with multiple functions."""

        def tool1(x: str) -> str:
            return x

        def tool2(x: int) -> int:
            return x

        tool_node = ToolNode([tool1, tool2])
        assert hasattr(tool_node, "all_tools")  # noqa: S101
        assert hasattr(tool_node, "invoke")  # noqa: S101
        assert hasattr(tool_node, "stream")  # noqa: S101

    @pytest.mark.asyncio
    async def test_tool_node_all_tools(self):
        """Test getting all tools from ToolNode."""

        def tool_func(param: str) -> str:
            """A test tool function."""
            return param

        tool_node = ToolNode([tool_func])
        tools = await tool_node.all_tools()
        assert isinstance(tools, list)  # noqa: S101
        assert len(tools) > 0  # noqa: S101

    def test_tool_node_invalid_function(self):
        """Test ToolNode with invalid function."""
        with pytest.raises(TypeError):
            ToolNode(["not_a_function"])  # type: ignore


class TestStateGraph:
    """Test the StateGraph class."""

    def test_state_graph_creation(self):
        """Test creating a StateGraph."""
        graph = StateGraph[AgentState](AgentState())
        assert graph._state is not None  # noqa: S101
        assert isinstance(graph._state, AgentState)  # noqa: S101

    def test_state_graph_creation_with_publisher(self):
        """Test creating a StateGraph with publisher."""
        from agentflow.runtime.publisher import ConsolePublisher

        publisher = ConsolePublisher()
        graph = StateGraph[AgentState](AgentState(), publisher=publisher)
        assert graph._publisher is not None  # noqa: S101

    def test_add_node(self):
        """Test adding nodes to the graph."""
        graph = StateGraph[AgentState](AgentState())

        def test_func(state: AgentState) -> AgentState:
            return state

        graph.add_node("test", test_func)
        assert "test" in graph.nodes  # noqa: S101
        assert isinstance(graph.nodes["test"], Node)  # noqa: S101

    def test_add_edge(self):
        """Test adding edges to the graph."""
        graph = StateGraph[AgentState](AgentState())

        def func1(state: AgentState) -> AgentState:
            return state

        def func2(state: AgentState) -> AgentState:
            return state

        graph.add_node("node1", func1)
        graph.add_node("node2", func2)
        graph.add_edge("node1", "node2")

        assert len(graph.edges) > 0  # noqa: S101

    def test_set_entry_point(self):
        """Test setting the entry point."""
        graph = StateGraph[AgentState](AgentState())

        def test_func(state: AgentState) -> AgentState:
            return state

        graph.add_node("start", test_func)
        graph.set_entry_point("start")
        assert graph.entry_point == "start"  # noqa: S101

    def test_add_conditional_edges(self):
        """Test adding conditional edges."""
        graph = StateGraph[AgentState](AgentState())

        def node_func(state: AgentState) -> AgentState:
            return state

        def router(state: AgentState) -> str:
            return "next" if len(state.context) > 0 else END

        graph.add_node("start", node_func)
        graph.add_node("next", node_func)
        graph.add_conditional_edges("start", router, {"next": "next", END: END})

    def test_compile_graph(self):
        """Test compiling the graph."""
        graph = StateGraph[AgentState](AgentState())

        def test_func(state: AgentState) -> AgentState:
            return state

        graph.add_node("start", test_func)
        graph.set_entry_point("start")
        graph.add_edge("start", END)

        compiled = graph.compile()
        assert isinstance(compiled, CompiledGraph)  # noqa: S101


class TestCompiledGraph:
    """Test the CompiledGraph class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.graph = StateGraph[AgentState](AgentState())

        def simple_node(state: AgentState) -> AgentState:
            state.context.append(Message.text_message("processed"))
            return state

        self.graph.add_node("start", simple_node)
        self.graph.set_entry_point("start")
        self.graph.add_edge("start", END)
        self.compiled = self.graph.compile()

    def test_invoke(self):
        """Test synchronous invocation."""
        messages = [Message.text_message("Hello", "user")]
        result = self.compiled.invoke({"messages": messages})
        assert "messages" in result  # noqa: S101

    def test_invoke_exception_handling(self, monkeypatch):
        """Test exception handling in invoke method."""

        def mock_ainvoke(*args, **kwargs):
            raise Exception("test error")

        monkeypatch.setattr(self.compiled, "ainvoke", mock_ainvoke)
        messages = [Message.text_message("Hello", "user")]
        with pytest.raises(Exception, match="test error"):
            self.compiled.invoke({"messages": messages})

    @pytest.mark.asyncio
    async def test_ainvoke(self):
        """Test asynchronous invocation."""
        messages = [Message.text_message("Hello", "user")]
        result = await self.compiled.ainvoke({"messages": messages})
        assert "messages" in result  # noqa: S101

    def test_stream(self):
        """Test streaming execution."""
        messages = [Message.text_message("Hello", "user")]
        events = list(self.compiled.stream({"messages": messages}))
        assert len(events) >= 0  # noqa: S101

    @pytest.mark.asyncio
    async def test_astream(self):
        """Test asynchronous streaming execution."""
        messages = [Message.text_message("Hello", "user")]
        events = []
        async for event in self.compiled.astream({"messages": messages}):
            events.append(event)
        assert len(events) >= 0  # noqa: S101

    @pytest.mark.asyncio
    async def test_aclose(self):
        """Test closing the compiled graph."""
        stats = await self.compiled.aclose()
        assert isinstance(stats, dict)  # noqa: S101
        assert "background_tasks" in stats  # noqa: S101

    @pytest.mark.asyncio
    async def test_aclose_is_idempotent(self):
        """Closing twice is a no-op and reports already_closed."""
        first = await self.compiled.aclose()
        assert "background_tasks" in first  # noqa: S101
        second = await self.compiled.aclose()
        assert second == {"status": "already_closed"}  # noqa: S101

    @pytest.mark.asyncio
    async def test_async_context_manager(self):
        """`async with` returns the graph and closes it on exit."""
        async with self.compiled as ctx:
            assert ctx is self.compiled  # noqa: S101
            assert self.compiled._closed is False  # noqa: S101, SLF001
        assert self.compiled._closed is True  # noqa: S101, SLF001

    @pytest.mark.asyncio
    async def test_async_context_manager_closes_on_exception(self):
        """aclose() still runs when the context body raises."""
        with pytest.raises(ValueError, match="boom"):
            async with self.compiled:
                raise ValueError("boom")
        assert self.compiled._closed is True  # noqa: S101, SLF001

    def test_generate_graph(self):
        """Test generating graph representation."""
        graph_dict = self.compiled.generate_graph()
        assert "info" in graph_dict  # noqa: S101
        assert "nodes" in graph_dict  # noqa: S101
        assert "edges" in graph_dict  # noqa: S101
        assert graph_dict["info"]["node_count"] == len(graph_dict["nodes"])  # noqa: S101
        assert graph_dict["info"]["edge_count"] == len(graph_dict["edges"])  # noqa: S101


def test_graph_module_imports():
    """Test that all graph module imports work correctly."""
    # Basic smoke test - just ensure imports work
    assert CompiledGraph is not None  # noqa: S101
    assert Edge is not None  # noqa: S101
    assert Node is not None  # noqa: S101
    assert StateGraph is not None  # noqa: S101
    assert ToolNode is not None  # noqa: S101
