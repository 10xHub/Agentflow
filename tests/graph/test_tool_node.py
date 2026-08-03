"""Tests for the tool_node module."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from injectq import Inject

from agentflow.core.graph.tool_node import ToolNode
from agentflow.core.state import AgentState
from agentflow.core.state import ToolResult
from agentflow.utils import CallbackManager
from agentflow.core.state.message import Message, ToolResultBlock


class TestToolNode:
    """Test the ToolNode class."""

    def test_tool_node_init_with_functions(self):
        """Test ToolNode initialization with functions."""

        def sample_func(x: int, y: str = "default") -> str:
            """Sample function for testing."""
            return f"{x}_{y}"

        tool_node = ToolNode([sample_func])
        assert len(tool_node._funcs) == 1
        assert "sample_func" in tool_node._funcs

    def test_tool_node_init_with_non_callable(self):
        """Test ToolNode initialization with non-callable raises error."""
        with pytest.raises(TypeError, match="ToolNode only accepts callables"):
            ToolNode(["not_callable"])  # type: ignore

    def test_tool_node_init_with_mcp_client_missing_deps(self):
        """Test ToolNode initialization with MCP client but missing dependencies."""
        mock_client = MagicMock()

        # Mock the imports to simulate missing dependencies
        with patch("agentflow.core.graph.tool_node.base.deps.HAS_FASTMCP", False):
            with pytest.raises(ImportError, match="MCP client functionality requires"):
                ToolNode([], client=mock_client)

    @pytest.mark.asyncio
    async def test_get_local_tools(self):
        """Test getting tool descriptions for local functions."""

        def func_with_params(a: int, b: str = "test", c: bool = True) -> str:
            """Function with various parameter types."""
            return f"{a}_{b}_{c}"

        def func_no_params() -> str:
            """Function with no parameters."""
            return "no_params"

        tool_node = ToolNode([func_with_params, func_no_params])
        tools = tool_node.get_local_tool()

        assert len(tools) == 2

        # Check first tool
        tool1 = tools[0]
        assert tool1["type"] == "function"
        assert tool1["function"]["name"] in ["func_with_params", "func_no_params"]
        assert "parameters" in tool1["function"]

        # Check parameters schema
        params = tool1["function"]["parameters"]
        assert "type" in params
        assert "properties" in params

    @pytest.mark.asyncio
    async def test_get_local_tools_with_injectable_params(self):
        """Test that injectable parameters are excluded from tool schema."""

        def func_with_injectables(
            user_input: str,
            state: AgentState | None = None,
            config: dict | None = None,
            tool_call_id: str | None = None,
        ) -> str:
            """Function with injectable parameters."""
            return user_input

        tool_node = ToolNode([func_with_injectables])
        tools = tool_node.get_local_tool()

        assert len(tools) == 1
        params = tools[0]["function"]["parameters"]

        # Only user_input should be in properties, injectables should be excluded
        assert "user_input" in params["properties"]
        assert "state" not in params["properties"]
        assert "config" not in params["properties"]
        assert "tool_call_id" not in params["properties"]

    @pytest.mark.asyncio
    async def test_get_local_tools_excludes_inject_defaults(self):
        """Test that Inject defaults are excluded before schema serialization."""

        class CatalogService:
            pass

        def func_with_service(
            query: str,
            service: CatalogService = Inject[CatalogService],
        ) -> str:
            """Function with Inject default parameter."""
            return query

        tool_node = ToolNode([func_with_service])
        tools = tool_node.get_local_tool()

        assert len(tools) == 1
        params = tools[0]["function"]["parameters"]

        assert "query" in params["properties"]
        assert "service" not in params["properties"]

    def test_annotation_to_schema_primitives(self):
        """Test annotation to schema conversion for primitive types."""
        schema = ToolNode._annotation_to_schema(str, "default")
        assert schema == {"type": "string", "default": "default"}

        schema = ToolNode._annotation_to_schema(int, 42)
        assert schema == {"type": "integer", "default": 42}

        schema = ToolNode._annotation_to_schema(bool, True)
        assert schema == {"type": "boolean", "default": True}

    def test_annotation_to_schema_complex(self):
        """Test annotation to schema conversion for complex types."""
        from typing import Literal

        # A nested item schema must not inherit the parameter's default, and a null
        # default is dropped entirely since `required` already conveys optionality.
        schema = ToolNode._annotation_to_schema(list[str], None)
        assert schema == {"type": "array", "items": {"type": "string"}}

        schema = ToolNode._annotation_to_schema(Literal["a", "b", "c"], None)
        assert schema == {"type": "string", "enum": ["a", "b", "c"]}

    @pytest.mark.asyncio
    async def test_invoke_local_tool_success(self):
        """Test successful invocation of local tool."""

        def sample_tool(x: int, y: str = "test") -> str:
            """Sample tool that returns a string."""
            return f"result_{x}_{y}"

        tool_node = ToolNode([sample_tool])

        state = AgentState()
        config = {"test": "config"}
        callback_mgr = MagicMock(spec=CallbackManager)
        callback_mgr.execute_before_invoke = AsyncMock(return_value={"x": 42, "y": "hello"})
        callback_mgr.execute_after_invoke = AsyncMock(return_value="result_42_hello")

        result = await tool_node.invoke(
            name="sample_tool",
            args={"x": 42, "y": "hello"},
            tool_call_id="test_call_123",
            config=config,
            state=state,
            callback_manager=callback_mgr,
        )

        assert isinstance(result, Message)

    @pytest.mark.asyncio
    async def test_invoke_local_tool_return_types(self):
        """Test different return types from local tools."""

        def tool_return_string() -> str:
            return "string_result"

        def tool_return_dict() -> dict:
            return {"key": "value"}

        def tool_return_message() -> Message:
            return Message.text_message(content="message_result")

        tool_node = ToolNode([tool_return_string, tool_return_dict, tool_return_message])

        state = AgentState()
        config = {}
        callback_mgr = MagicMock(spec=CallbackManager)
        callback_mgr.execute_before_invoke = AsyncMock(side_effect=lambda ctx, data: data)
        callback_mgr.execute_after_invoke = AsyncMock(
            side_effect=lambda ctx, input_data, result: result
        )

        # Test string return
        result = await tool_node.invoke(
            name="tool_return_string",
            args={},
            tool_call_id="call_1",
            config=config,
            state=state,
            callback_manager=callback_mgr,
        )
        assert isinstance(result, Message)
        assert "string_result" in result.content[0].output

        # Test dict return
        result = await tool_node.invoke(
            name="tool_return_dict",
            args={},
            tool_call_id="call_2",
            config=config,
            state=state,
            callback_manager=callback_mgr,
        )
        assert isinstance(result, Message)

        # Test Message return
        result = await tool_node.invoke(
            name="tool_return_message",
            args={},
            tool_call_id="call_3",
            config=config,
            state=state,
            callback_manager=callback_mgr,
        )
        assert isinstance(result, Message)

    @pytest.mark.asyncio
    async def test_invoke_local_tool_tool_result_updates_state(self):
        """Test ToolResult updates state fields and returns a tool message."""

        class CustomState(AgentState):
            status: str = "pending"

        def tool_return_tool_result() -> ToolResult:
            return ToolResult(message="updated", state={"status": "complete"})

        tool_node = ToolNode([tool_return_tool_result])

        state = CustomState()
        config = {}
        callback_mgr = MagicMock(spec=CallbackManager)
        callback_mgr.execute_before_invoke = AsyncMock(side_effect=lambda ctx, data: data)
        callback_mgr.execute_after_invoke = AsyncMock(
            side_effect=lambda ctx, input_data, result: result
        )

        result = await tool_node.invoke(
            name="tool_return_tool_result",
            args={},
            tool_call_id="call_4",
            config=config,
            state=state,
            callback_manager=callback_mgr,
        )

        # ToolResult returns a dict with state and messages
        assert isinstance(result, dict)
        assert "state" in result
        assert "messages" in result
        assert state.status == "complete"
        msg = result["messages"]
        assert isinstance(msg, Message)
        assert msg.content[0].output == "updated"

    @pytest.mark.asyncio
    async def test_invoke_tool_not_found(self):
        """Test invoking a tool that doesn't exist."""
        tool_node = ToolNode([])

        state = AgentState()
        config = {}
        callback_mgr = MagicMock(spec=CallbackManager)

        result = await tool_node.invoke(
            name="nonexistent_tool",
            args={},
            tool_call_id="test_call",
            config=config,
            state=state,
            callback_manager=callback_mgr,
        )

        assert isinstance(result, Message)

    @pytest.mark.asyncio
    async def test_invoke_tool_error_handling(self):
        """Test error handling during tool invocation."""

        def failing_tool() -> str:
            raise ValueError("Tool failed")

        tool_node = ToolNode([failing_tool])

        state = AgentState()
        config = {}
        callback_mgr = MagicMock(spec=CallbackManager)
        callback_mgr.execute_before_invoke = AsyncMock(side_effect=lambda ctx, data: data)
        callback_mgr.execute_on_error = AsyncMock(return_value=None)  # No recovery

        # Test that error message is returned (not raised as exception)
        result = await tool_node.invoke(
            name="failing_tool",
            args={},
            tool_call_id="test_call",
            config=config,
            state=state,
            callback_manager=callback_mgr,
        )

        # Should return an error message instead of raising exception
        assert isinstance(result, Message)
        assert result.content is not None

    @pytest.mark.asyncio
    async def test_invoke_tool_error_recovery(self):
        """Test error recovery during tool invocation."""

        def failing_tool() -> str:
            raise ValueError("Tool failed")

        tool_node = ToolNode([failing_tool])

        state = AgentState()
        config = {}
        recovery_message = Message.tool_message(
            content=[ToolResultBlock(call_id="test_call", output="recovered_result")]
        )
        callback_mgr = MagicMock(spec=CallbackManager)
        callback_mgr.execute_before_invoke = AsyncMock(side_effect=lambda ctx, data: data)
        callback_mgr.execute_on_error = AsyncMock(return_value=recovery_message)

        result = await tool_node.invoke(
            name="failing_tool",
            args={},
            tool_call_id="test_call",
            config=config,
            state=state,
            callback_manager=callback_mgr,
        )

        assert result == recovery_message

    @pytest.mark.asyncio
    async def test_stream_tool_not_found(self):
        """Test streaming a tool that doesn't exist."""
        tool_node = ToolNode([])

        state = AgentState()
        config = {"run_id": "test_run"}
        callback_mgr = MagicMock(spec=CallbackManager)

        chunks = []
        async for chunk in tool_node.stream(
            name="nonexistent_tool",
            args={},
            tool_call_id="test_call",
            config=config,
            state=state,
            callback_manager=callback_mgr,
        ):
            chunks.append(chunk)

        # Should have 2 chunks: error chunk and error message
        assert len(chunks) > 0

        # Second chunk should be error message
        assert isinstance(chunks[0], Message)

    def test_prepare_input_data_tool(self):
        """Test preparing input data for tool execution."""

        def sample_func(
            a: int,
            b: str = "default",
            state: AgentState | None = None,
            config: dict | None = None,
            tool_call_id: str | None = None,
        ):
            pass

        tool_node = ToolNode([sample_func])

        default_data = {
            "state": AgentState(),
            "config": {"test": "config"},
            "tool_call_id": "test_id",
        }

        args = {"a": 42, "b": "custom"}

        input_data = tool_node._prepare_input_data_tool(
            sample_func, "sample_func", args, default_data
        )

        assert input_data["a"] == 42
        assert input_data["b"] == "custom"
        assert input_data["state"] == default_data["state"]
        assert input_data["config"] == default_data["config"]
        assert input_data["tool_call_id"] == default_data["tool_call_id"]

    def test_prepare_input_data_tool_missing_required(self):
        """Test preparing input data with missing required parameter."""

        def sample_func(required_param: int):
            pass

        tool_node = ToolNode([sample_func])

        default_data = {"state": AgentState(), "config": {}, "tool_call_id": "test"}

        with pytest.raises(TypeError, match="Missing required parameter"):
            tool_node._prepare_input_data_tool(sample_func, "sample_func", {}, default_data)

    @pytest.mark.asyncio
    async def test_all_tools_combines_local_and_mcp(self):
        """Test that all_tools combines local and MCP tools."""

        def local_func() -> str:
            """Local function."""
            return "local"

        tool_node = ToolNode([local_func])

        # Mock MCP tools
        with patch.object(tool_node, "_get_mcp_tool", new_callable=AsyncMock) as mock_mcp:
            mock_mcp.return_value = [
                {
                    "type": "function",
                    "function": {
                        "name": "mcp_tool",
                        "description": "MCP tool",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ]

            tools = await tool_node.all_tools()

            # Should have both local and MCP tools
            assert len(tools) == 2
            tool_names = [t["function"]["name"] for t in tools]
            assert "local_func" in tool_names
            assert "mcp_tool" in tool_names


class TestToolNodeMCPUserInfo:
    """Test the pass_user_info_to_mcp feature."""

    def test_tool_node_init_with_pass_user_info_to_mcp_default(self):
        """Test that pass_user_info_to_mcp defaults to False."""

        def sample_func() -> str:
            return "result"

        tool_node = ToolNode([sample_func])
        assert tool_node._pass_user_info_to_mcp is False

    def test_tool_node_init_with_pass_user_info_to_mcp_true(self):
        """Test ToolNode initialization with pass_user_info_to_mcp=True."""

        def sample_func() -> str:
            return "result"

        tool_node = ToolNode([sample_func], pass_user_info_to_mcp=True)
        assert tool_node._pass_user_info_to_mcp is True

    def test_tool_node_init_with_pass_user_info_to_mcp_false(self):
        """Test ToolNode initialization with pass_user_info_to_mcp=False."""

        def sample_func() -> str:
            return "result"

        tool_node = ToolNode([sample_func], pass_user_info_to_mcp=False)
        assert tool_node._pass_user_info_to_mcp is False

    @pytest.mark.asyncio
    async def test_mcp_execute_passes_user_info_when_enabled(self):
        """Test that _mcp_execute passes user info as meta when pass_user_info_to_mcp=True."""

        def sample_func() -> str:
            return "result"

        # Create mock MCP client
        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.ping = AsyncMock(return_value=True)
        mock_call_tool_result = MagicMock()
        mock_call_tool_result.content = [MagicMock(text="mcp_result")]
        mock_client.call_tool = AsyncMock(return_value=mock_call_tool_result)

        # Patch dependencies to allow MCP client
        with patch("agentflow.core.graph.tool_node.base.deps.HAS_FASTMCP", True):
            with patch("agentflow.core.graph.tool_node.base.deps.HAS_MCP", True):
                tool_node = ToolNode(
                    [sample_func], client=mock_client, pass_user_info_to_mcp=True
                )
                # Register MCP tool
                tool_node.mcp_tools = ["mcp_tool"]

                callback_mgr = MagicMock(spec=CallbackManager)
                callback_mgr.execute_before_invoke = AsyncMock(
                    side_effect=lambda ctx, data: data
                )
                callback_mgr.execute_after_invoke = AsyncMock(
                    side_effect=lambda ctx, input_data, result: result
                )

                user_info = {"id": "user123", "name": "John", "role": "admin"}
                config = {"user": user_info, "other_config": "value"}

                result = await tool_node._mcp_execute(
                    name="mcp_tool",
                    args={"param": "value"},
                    tool_call_id="call_123",
                    config=config,
                    callback_mgr=callback_mgr,
                )

                # Verify call_tool was called with meta containing user info
                mock_client.call_tool.assert_called_once()
                call_args = mock_client.call_tool.call_args
                assert call_args[0][0] == "mcp_tool"  # name
                assert call_args[0][1]["user"] == user_info  # input_data contains user info

    @pytest.mark.asyncio
    async def test_mcp_execute_does_not_pass_user_info_when_disabled(self):
        """Test that _mcp_execute does not pass user info when pass_user_info_to_mcp=False."""

        def sample_func() -> str:
            return "result"

        # Create mock MCP client
        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.ping = AsyncMock(return_value=True)
        mock_call_tool_result = MagicMock()
        mock_call_tool_result.content = [MagicMock(text="mcp_result")]
        mock_client.call_tool = AsyncMock(return_value=mock_call_tool_result)

        # Patch dependencies to allow MCP client
        with patch("agentflow.core.graph.tool_node.base.deps.HAS_FASTMCP", True):
            with patch("agentflow.core.graph.tool_node.base.deps.HAS_MCP", True):
                tool_node = ToolNode(
                    [sample_func], client=mock_client, pass_user_info_to_mcp=False
                )
                # Register MCP tool
                tool_node.mcp_tools = ["mcp_tool"]

                callback_mgr = MagicMock(spec=CallbackManager)
                callback_mgr.execute_before_invoke = AsyncMock(
                    side_effect=lambda ctx, data: data
                )
                callback_mgr.execute_after_invoke = AsyncMock(
                    side_effect=lambda ctx, input_data, result: result
                )

                user_info = {"id": "user123", "name": "John", "role": "admin"}
                config = {"user": user_info, "other_config": "value"}

                result = await tool_node._mcp_execute(
                    name="mcp_tool",
                    args={"param": "value"},
                    tool_call_id="call_123",
                    config=config,
                    callback_mgr=callback_mgr,
                )

                # Verify call_tool was called without meta
                mock_client.call_tool.assert_called_once()
                call_args = mock_client.call_tool.call_args
                assert call_args[0][0] == "mcp_tool"  # name
                assert call_args[0][1] == {"param": "value"}  # input_data
                assert "meta" not in call_args[1]  # no meta kwarg

    @pytest.mark.asyncio
    async def test_mcp_execute_handles_missing_user_in_config(self):
        """Test that _mcp_execute handles missing 'user' key in config gracefully."""

        def sample_func() -> str:
            return "result"

        # Create mock MCP client
        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.ping = AsyncMock(return_value=True)
        mock_call_tool_result = MagicMock()
        mock_call_tool_result.content = [MagicMock(text="mcp_result")]
        mock_client.call_tool = AsyncMock(return_value=mock_call_tool_result)

        # Patch dependencies to allow MCP client
        with patch("agentflow.core.graph.tool_node.base.deps.HAS_FASTMCP", True):
            with patch("agentflow.core.graph.tool_node.base.deps.HAS_MCP", True):
                tool_node = ToolNode(
                    [sample_func], client=mock_client, pass_user_info_to_mcp=True
                )
                # Register MCP tool
                tool_node.mcp_tools = ["mcp_tool"]

                callback_mgr = MagicMock(spec=CallbackManager)
                callback_mgr.execute_before_invoke = AsyncMock(
                    side_effect=lambda ctx, data: data
                )
                callback_mgr.execute_after_invoke = AsyncMock(
                    side_effect=lambda ctx, input_data, result: result
                )

                # Config without 'user' key
                config = {"other_config": "value"}

                result = await tool_node._mcp_execute(
                    name="mcp_tool",
                    args={"param": "value"},
                    tool_call_id="call_123",
                    config=config,
                    callback_mgr=callback_mgr,
                )

                # Verify call_tool was called without meta since user is missing
                mock_client.call_tool.assert_called_once()
                call_args = mock_client.call_tool.call_args
                assert call_args[0][0] == "mcp_tool"  # name
                assert call_args[0][1] == {"param": "value"}  # input_data
                assert "meta" not in call_args[1]  # no meta kwarg

    @pytest.mark.asyncio
    async def test_mcp_execute_handles_non_dict_user(self):
        """Test that _mcp_execute handles non-dict 'user' value gracefully."""

        def sample_func() -> str:
            return "result"

        # Create mock MCP client
        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.ping = AsyncMock(return_value=True)
        mock_call_tool_result = MagicMock()
        mock_call_tool_result.content = [MagicMock(text="mcp_result")]
        mock_client.call_tool = AsyncMock(return_value=mock_call_tool_result)

        # Patch dependencies to allow MCP client
        with patch("agentflow.core.graph.tool_node.base.deps.HAS_FASTMCP", True):
            with patch("agentflow.core.graph.tool_node.base.deps.HAS_MCP", True):
                tool_node = ToolNode(
                    [sample_func], client=mock_client, pass_user_info_to_mcp=True
                )
                # Register MCP tool
                tool_node.mcp_tools = ["mcp_tool"]

                callback_mgr = MagicMock(spec=CallbackManager)
                callback_mgr.execute_before_invoke = AsyncMock(
                    side_effect=lambda ctx, data: data
                )
                callback_mgr.execute_after_invoke = AsyncMock(
                    side_effect=lambda ctx, input_data, result: result
                )

                # Config with non-dict 'user' value
                config = {"user": "just_a_string", "other_config": "value"}

                result = await tool_node._mcp_execute(
                    name="mcp_tool",
                    args={"param": "value"},
                    tool_call_id="call_123",
                    config=config,
                    callback_mgr=callback_mgr,
                )

                # Verify call_tool was called without meta since user is not a dict
                mock_client.call_tool.assert_called_once()
                call_args = mock_client.call_tool.call_args
                assert call_args[0][0] == "mcp_tool"  # name
                assert call_args[0][1] == {"param": "value"}  # input_data
                assert "meta" not in call_args[1]  # no meta kwarg


class TestToolNodeRemoteToolParity:
    """Regression tests for issue #148.

    ``all_tools_sync`` used to rebuild the tool list from local + MCP tools only and
    never appended ``remote_tools``, so remote (client-side) tools silently vanished
    for any caller that went through the sync accessor.
    """

    @staticmethod
    def _remote_tool(name: str) -> dict:
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": f"Remote tool {name}.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        }

    def test_all_tools_sync_includes_remote_tools(self):
        """Remote tools must be advertised by the sync accessor."""

        def local_func() -> str:
            """Local function."""
            return "local"

        tool_node = ToolNode([local_func])
        tool_node.set_remote_tool([self._remote_tool("write_file")])

        names = [t["function"]["name"] for t in tool_node.all_tools_sync()]
        assert "local_func" in names
        assert "write_file" in names

    def test_sync_and_async_tool_sets_match(self):
        """The sync and async builders must not drift apart.

        Deliberately a sync test: ``all_tools_sync`` uses ``asyncio.run`` internally
        and cannot be called from inside a running event loop.
        """

        def local_func() -> str:
            """Local function."""
            return "local"

        mcp_tool = {
            "type": "function",
            "function": {
                "name": "mcp_tool",
                "description": "MCP tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }

        tool_node = ToolNode([local_func], client=MagicMock())
        tool_node.set_remote_tool([self._remote_tool("write_file")])

        with patch.object(tool_node, "_get_mcp_tool", new_callable=AsyncMock) as mock_mcp:
            mock_mcp.return_value = [mcp_tool]

            async_names = {t["function"]["name"] for t in asyncio.run(tool_node.all_tools())}
            sync_names = {t["function"]["name"] for t in tool_node.all_tools_sync()}

        assert sync_names == async_names
        assert sync_names == {"local_func", "mcp_tool", "write_file"}

    def test_all_tools_sync_without_remote_tools_is_unchanged(self):
        """No remote tools registered means nothing extra is appended."""

        def local_func() -> str:
            """Local function."""
            return "local"

        tool_node = ToolNode([local_func])

        assert [t["function"]["name"] for t in tool_node.all_tools_sync()] == ["local_func"]
