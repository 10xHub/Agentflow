from __future__ import annotations

import asyncio
import contextlib
import datetime
import logging
from collections.abc import AsyncIterator, Callable, Generator
from typing import TYPE_CHECKING, Any, TypeVar
from uuid import uuid4

from injectq import InjectQ

from agentflow.core.exceptions.graph_error import GraphError
from agentflow.core.graph.base_agent import BaseAgent
from agentflow.core.graph.tool_node.base import ToolNode
from agentflow.core.state import AgentState
from agentflow.core.state.execution_state import StopRequestStatus
from agentflow.core.state.stream_chunks import StreamChunk
from agentflow.runtime.publisher.base_publisher import BasePublisher
from agentflow.storage.checkpointer.base_checkpointer import BaseCheckpointer
from agentflow.storage.store.base_store import BaseStore
from agentflow.utils import (
    CallbackManager,
    ResponseGranularity,
)
from agentflow.utils.background_task_manager import BackgroundTaskManager
from agentflow.utils.constants import DEFAULT_ANONYMOUS_USER_ID

from .node import Node
from .utils.invoke_handler import InvokeHandler
from .utils.stream_handler import StreamHandler


if TYPE_CHECKING:
    from types import TracebackType

    from .state_graph import StateGraph


StateT = TypeVar("StateT", bound=AgentState)

logger = logging.getLogger("agentflow.graph")


class CompiledGraph[StateT: AgentState]:
    """A fully compiled and executable graph ready for workflow execution.

    CompiledGraph represents the final executable form of a StateGraph after compilation.
    It encapsulates all the execution logic, handlers, and services needed to run
    agent workflows. The graph supports both synchronous and asynchronous execution
    with comprehensive state management, checkpointing, event publishing, and
    streaming capabilities.

    This class is generic over state types to support custom AgentState subclasses,
    ensuring type safety throughout the execution process.

    Key Features:
    - Synchronous and asynchronous execution methods
    - Real-time streaming with incremental results
    - State persistence and checkpointing
    - Interrupt and resume capabilities
    - Event publishing for monitoring and debugging
    - Background task management
    - Graceful error handling and recovery

    Attributes:
        _state: The initial/template state for graph executions.
        _invoke_handler: Handler for non-streaming graph execution.
        _stream_handler: Handler for streaming graph execution.
        _checkpointer: Optional state persistence backend.
        _publisher: Optional event publishing backend.
        _store: Optional data storage backend.
        _state_graph: Reference to the source StateGraph.
        _interrupt_before: Nodes where execution should pause before execution.
        _interrupt_after: Nodes where execution should pause after execution.
        _task_manager: Manager for background async tasks.

    Example:
        ```python
        # After building and compiling a StateGraph
        compiled = graph.compile()

        # Synchronous execution
        result = compiled.invoke({"messages": [Message.text_message("Hello")]})

        # Asynchronous execution with streaming
        async for chunk in compiled.astream({"messages": [message]}):
            print(f"Streamed: {chunk.content}")

        # Graceful cleanup
        await compiled.aclose()
        ```

    Note:
        CompiledGraph instances should be properly closed using aclose() to
        release resources like database connections, background tasks, and
        event publishers.
    """

    def __init__(
        self,
        state: StateT,
        checkpointer: BaseCheckpointer[StateT] | None,
        publisher: BasePublisher | None,
        store: BaseStore | None,
        state_graph: StateGraph[StateT],
        interrupt_before: list[str],
        interrupt_after: list[str],
        task_manager: BackgroundTaskManager,
        shutdown_timeout: float = 30.0,
    ):
        logger.info(
            f"Initializing CompiledGraph with nodes: {list(state_graph.nodes.keys())}",
        )

        # Save initial state
        self._state = state
        self._shutdown_timeout = shutdown_timeout

        # create handlers
        self._invoke_handler: InvokeHandler[StateT] = InvokeHandler[StateT](
            nodes=state_graph.nodes,
            edges=state_graph.edges,
            interrupt_after=interrupt_after,
            interrupt_before=interrupt_before,
        )
        self._stream_handler: StreamHandler[StateT] = StreamHandler[StateT](
            nodes=state_graph.nodes,
            edges=state_graph.edges,
            interrupt_after=interrupt_after,
            interrupt_before=interrupt_before,
        )

        self._checkpointer: BaseCheckpointer[StateT] | None = checkpointer
        self._publisher: BasePublisher | None = publisher
        self._store: BaseStore | None = store
        self._state_graph: StateGraph[StateT] = state_graph
        self._interrupt_before: list[str] = interrupt_before
        self._interrupt_after: list[str] = interrupt_after
        # generate task manager
        self._task_manager = task_manager
        # Guards aclose() against being run more than once (e.g. an explicit
        # aclose() inside an ``async with`` block followed by __aexit__).
        self._closed = False

    async def __aenter__(self) -> CompiledGraph[StateT]:
        """Enter an async context; returns this graph unchanged.

        Enables ``async with compiled_graph as graph: ...``, which guarantees
        :meth:`aclose` runs on exit even if the body raises.
        """
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Exit the async context, releasing all resources via :meth:`aclose`."""
        await self.aclose()

    def _prepare_config(
        self,
        config: dict[str, Any] | None,
        is_stream: bool = False,
    ) -> dict[str, Any]:
        cfg = dict(config or {})

        if "thread_id" not in cfg:
            cfg["thread_id"] = InjectQ.get_instance().try_get("generated_id") or str(uuid4())
            # A caller who omits thread_id gets a fresh random one, which means the
            # run cannot be resumed and `stop()` can never target it -- the caller
            # has no way to name it. That is fine for a one-shot script but is
            # almost never what a server wants, so say so rather than fail silently.
            logger.warning(
                "No thread_id in config; generated '%s'. This run cannot be resumed "
                "or stopped, because the caller does not know its thread_id. Pass an "
                "explicit thread_id for any run you may want to resume or stop.",
                cfg["thread_id"],
            )

        if "is_stream" not in cfg:
            cfg["is_stream"] = is_stream
        if "user_id" not in cfg:
            # Was "test-user-id" -- a placeholder that looks like a real user. With
            # per-user isolation enabled that silently filed every un-authenticated
            # run under one fake account. "anonymous" matches what the API layer
            # uses when no auth is configured, so the two agree.
            cfg["user_id"] = DEFAULT_ANONYMOUS_USER_ID
            logger.debug(
                "No user_id in config; using '%s'. Enable auth if you need per-user "
                "isolation -- every anonymous run shares one identity.",
                DEFAULT_ANONYMOUS_USER_ID,
            )
        if "run_id" not in cfg:
            cfg["run_id"] = InjectQ.get_instance().try_get("generated_id") or str(uuid4())

        if "timestamp" not in cfg:
            cfg["timestamp"] = datetime.datetime.now().isoformat()

        return cfg

    def invoke(
        self,
        input_data: dict[str, Any],
        config: dict[str, Any] | None = None,
        response_granularity: ResponseGranularity = ResponseGranularity.LOW,
    ) -> dict[str, Any]:
        """Execute the graph synchronously and return the final results.

        Runs the complete graph workflow from start to finish, handling state
        management, node execution, and result formatting. This method automatically
        detects whether to start a fresh execution or resume from an interrupted state.

        The execution is synchronous but internally uses async operations, making it
        suitable for use in non-async contexts while still benefiting from async
        capabilities for I/O operations.

        Args:
            input_data: Input dictionary for graph execution. For new executions,
                should contain 'messages' key with list of initial messages.
                For resumed executions, can contain additional data to merge.
            config: Optional configuration dictionary containing execution settings:
                - user_id: Identifier for the user/session
                - thread_id: Unique identifier for this execution thread
                - run_id: Unique identifier for this specific run
                - recursion_limit: Maximum steps before stopping (default: 25)
            response_granularity: Level of detail in the response:
                - LOW: Returns only messages (default)
                - PARTIAL: Returns context, summary, and messages
                - FULL: Returns complete state and messages

        Returns:
            Dictionary containing execution results formatted according to the
            specified granularity level. Always includes execution messages
            and may include additional state information.

        Raises:
            ValueError: If input_data is invalid for new execution.
            GraphRecursionError: If execution exceeds recursion limit.
            Various exceptions: Depending on node execution failures.

        Example:
            ```python
            # Basic execution
            result = compiled.invoke({"messages": [Message.text_message("Process this data")]})
            print(result["messages"])  # Final execution messages

            # With configuration and full details
            result = compiled.invoke(
                input_data={"messages": [message]},
                config={"user_id": "user123", "thread_id": "session456", "recursion_limit": 50},
                response_granularity=ResponseGranularity.FULL,
            )
            print(result["state"])  # Complete final state
            ```

        Note:
            This method uses asyncio.run() internally, so it should not be called
            from within an async context. Use ainvoke() instead for async execution.
        """
        logger.info(
            "Starting synchronous graph execution with %d input keys, granularity=%s",
            len(input_data) if input_data else 0,
            response_granularity,
        )
        logger.debug("Input data keys: %s", list(input_data.keys()) if input_data else [])
        # Async Will Handle Event Publish

        try:
            result = asyncio.run(self.ainvoke(input_data, config, response_granularity))
            logger.info("Synchronous graph execution completed successfully")
            return result
        except Exception as e:
            logger.exception("Synchronous graph execution failed: %s", e)
            raise

    async def ainvoke(
        self,
        input_data: dict[str, Any],
        config: dict[str, Any] | None = None,
        response_granularity: ResponseGranularity = ResponseGranularity.LOW,
    ) -> dict[str, Any]:
        """Execute the graph asynchronously.

        Auto-detects whether to start fresh execution or resume from interrupted state
        based on the AgentState's execution metadata.

        Args:
            input_data: Input dict with 'messages' key (for new execution) or
                       additional data for resuming
            config: Configuration dictionary
            response_granularity: Response parsing granularity

        Returns:
            Response dict based on granularity
        """
        self._guard_not_realtime()
        cfg = self._prepare_config(config, is_stream=False)

        return await self._invoke_handler.invoke(
            input_data,
            cfg,
            self._state,
            response_granularity,
        )

    def stop(self, config: dict[str, Any]) -> dict[str, Any]:
        """Request the current graph execution to stop (sync helper).

        This sets a stop flag in the checkpointer's thread store keyed by thread_id.
        Handlers periodically check this flag and interrupt execution.
        Returns a small status dict.
        """
        return asyncio.run(self.astop(config))

    async def astop(self, config: dict[str, Any]) -> dict[str, Any]:
        """Request the current graph execution to stop (async).

        Contract:
        - Requires a valid thread_id in config
        - If no active thread or no checkpointer, returns not-running
        - If state exists and is running, set stop_requested flag in thread info
        """
        cfg = self._prepare_config(config, is_stream=bool((config or {}).get("is_stream", False)))
        if not self._checkpointer:
            return {"ok": False, "reason": "no-checkpointer"}

        # Load state to see if this thread is running
        # Lets load from the cache, incase if not available lets load from the db
        # In prebuilt implement like pgsql, there its called internally, no need this call
        # Still lets do that incase user use their own
        state = await self._checkpointer.aget_state_cache(
            cfg
        ) or await self._checkpointer.aget_state(cfg)
        if not state:
            return {"ok": False, "running": False, "reason": "no-state"}

        running = state.is_running() and not state.is_interrupted()
        # Set stop flag regardless; handlers will act if running
        if running:
            # Record the stop in its own cache entry FIRST. The running graph
            # rewrites the state cache after every node with its own flag-less
            # copy of the state (at the same durable version, so a version-guarded
            # cache accepts it), which would silently swallow a flag written only
            # into the state blob. Nothing in the run loop writes this key, so a
            # stop recorded here always survives to the next stop check.
            await self._checkpointer.arequest_stop(cfg)

            # Also mirror it into the cached state. Redundant for checkpointers
            # that back the stop key, but it keeps stop working for a custom
            # checkpointer that has not implemented the generic cache (whose
            # arequest_stop inherits the no-op default).
            state.execution_meta.stop_current_execution = StopRequestStatus.STOP_REQUESTED
            await self._checkpointer.aput_state_cache(cfg, state)
            logger.info("Set stop_current_execution flag for thread_id: %s", cfg.get("thread_id"))
            return {"ok": True, "running": running}

        logger.info(
            "No running execution to stop for thread_id: %s (running=%s, interrupted=%s)",
            cfg.get("thread_id"),
            running,
            state.is_interrupted(),
        )
        return {"ok": True, "running": running, "reason": "not-running"}

    def override_node(
        self,
        name: str,
        func: Callable | ToolNode | BaseAgent,
    ) -> CompiledGraph[StateT]:
        """Override a node in an already-compiled graph.

        Useful for testing pre-built production graphs by swapping
        nodes with test doubles after compilation.

        Args:
            name: Name of the existing node to override
            func: New function, ToolNode, or Agent to use

        Returns:
            CompiledGraph: The graph instance for method chaining.

        Raises:
            KeyError: If the node doesn't exist

        Example:
            ```python
            # Production factory
            def create_production_workflow():
                graph = StateGraph()
                graph.add_node("MAIN", production_agent)
                # ... complex setup
                return graph.compile()


            # Test
            compiled = create_production_workflow()
            compiled.override_node("MAIN", test_agent)  # Override after compile
            result = await compiled.ainvoke(...)
            ```
        """
        if name not in self._state_graph.nodes:
            raise KeyError(f"Node '{name}' does not exist")

        # Create new Node and update the graph's dict
        new_node = Node(name, func)
        self._state_graph.nodes[name] = new_node

        # Re-register in container so handlers pick up the change
        # The handlers read from container's "get_node" factory
        self._state_graph._container.bind_factory(
            "get_node",
            lambda x: self._state_graph.nodes[x],
        )

        logger.debug("Overrode node '%s' in compiled graph", name)
        return self

    def stream(
        self,
        input_data: dict[str, Any],
        config: dict[str, Any] | None = None,
        response_granularity: ResponseGranularity = ResponseGranularity.LOW,
    ) -> Generator[StreamChunk]:
        """Execute the graph synchronously with streaming support.

        Yields Message objects containing incremental responses.
        If nodes return streaming responses, yields them directly.
        If nodes return complete responses, simulates streaming by chunking.

        Args:
            input_data: Input dict
            config: Configuration dictionary
            response_granularity: Response parsing granularity

        Yields:
            Message objects with incremental content
        """

        # For sync streaming, we'll use asyncio.run to handle the async implementation
        async def _async_stream():
            async for chunk in self.astream(input_data, config, response_granularity):
                yield chunk

        # Convert async generator to sync iteration with a dedicated event loop
        gen = _async_stream()
        loop = asyncio.new_event_loop()
        policy = asyncio.get_event_loop_policy()
        try:
            previous_loop = policy.get_event_loop()
        except Exception:
            previous_loop = None
        asyncio.set_event_loop(loop)
        logger.info("Synchronous streaming started")

        try:
            while True:
                try:
                    chunk = loop.run_until_complete(gen.__anext__())
                    yield chunk
                except StopAsyncIteration:
                    break
        finally:
            # Attempt to close the async generator cleanly
            with contextlib.suppress(Exception):
                loop.run_until_complete(gen.aclose())  # type: ignore[attr-defined]
            # Restore previous loop if any, then close created loop
            try:
                if previous_loop is not None:
                    asyncio.set_event_loop(previous_loop)
            finally:
                loop.close()
        logger.info("Synchronous streaming completed")

    async def astream(
        self,
        input_data: dict[str, Any],
        config: dict[str, Any] | None = None,
        response_granularity: ResponseGranularity = ResponseGranularity.LOW,
    ) -> AsyncIterator[StreamChunk]:
        """Execute the graph asynchronously with streaming support.

        Yields Message objects containing incremental responses.
        If nodes return streaming responses, yields them directly.
        If nodes return complete responses, simulates streaming by chunking.

        Args:
            input_data: Input dict
            config: Configuration dictionary
            response_granularity: Response parsing granularity

        Yields:
            Message objects with incremental content
        """
        self._guard_not_realtime()
        cfg = self._prepare_config(config, is_stream=True)

        async for chunk in self._stream_handler.stream(
            input_data,
            cfg,
            self._state,
            response_granularity,
        ):
            yield chunk

    def attach_remote_tools(
        self,
        tools: list[dict],
        node_name: str,
    ):
        """Attach remote tools to a ToolNode in the graph.

        Args:
            tools: List of tool configurations to attach.
            node_name: Name of the ToolNode to attach tools to.

        Raises:
            GraphError: If the specified node is not a ToolNode.

        Example:
            >>> tool_configs = [
            ...     {"name": "search", "type": "SearchTool", "config": {...}},
            ...     {"name": "calculator", "type": "CalculatorTool", "config": {...}},
            ... ]
            >>> graph.attach_remote_tools(tool_configs, "tool_node")
        """
        logger.debug(
            "Attaching remote tools to node '%s': %s",
            node_name,
            tools,
        )
        node: Node | None = self._state_graph.nodes.get(node_name)
        if not node:
            raise GraphError(
                message=f"Node '{node_name}' not found in graph",
                error_code="GRAPH_004",
                context={"node_name": node_name},
            )

        if not isinstance(node.func, ToolNode):
            error_msg = f"Node '{node_name}' is not a ToolNode"
            logger.error(error_msg)
            raise GraphError(
                message=error_msg,
                error_code="GRAPH_005",
                context={"node_name": node_name},
            )

        tool_node: ToolNode = node.func
        tool_node.set_remote_tool(tools)
        logger.info(
            "Attached %d remote tools to ToolNode '%s'",
            len(tools),
            node_name,
        )

    # ------------------------------------------------------------------ #
    # Realtime runtime (audio-to-audio). A separate runtime from the
    # super-step invoke/stream loop: the live agent owns the turn loop.
    # ------------------------------------------------------------------ #
    def _find_live_nodes(self) -> list[tuple[str, Node]]:
        from agentflow.core.realtime.live_agent import LiveAgent

        return [
            (name, node)
            for name, node in self._state_graph.nodes.items()
            if isinstance(node.func, LiveAgent)
        ]

    def is_realtime(self) -> bool:
        """Whether this graph is a realtime (live) graph.

        True when the graph contains at least one ``LiveAgent`` node, meaning it must be
        driven via ``arealtime()`` / ``realtime()`` rather than invoke/stream.
        """
        return bool(self._find_live_nodes())

    def _guard_not_realtime(self) -> None:
        """Forcing rule: a graph containing a LiveAgent must use arealtime()."""
        if self._find_live_nodes():
            raise RuntimeError(
                "This graph contains a LiveAgent; use .arealtime() / .realtime() instead of "
                "invoke/ainvoke/stream/astream."
            )

    async def arealtime(
        self,
        input_queue: Any,
        config: dict[str, Any] | None = None,
        state: AgentState | None = None,
    ) -> AsyncIterator[Any]:
        """Run the graph's realtime (audio) session, yielding normalized RealtimeEvents.

        Forcing rule: the graph must contain exactly one LiveAgent (the root controller);
        ordinary turn-based graphs must use invoke/stream.
        """
        live = self._find_live_nodes()
        if not live:
            raise RuntimeError(
                "arealtime() requires a graph rooted at a LiveAgent (e.g. AudioAgent); "
                "this graph has none. Use invoke/stream for turn-based graphs."
            )
        if len(live) > 1:
            raise RuntimeError(
                "Only one LiveAgent is allowed per realtime run in v1 "
                f"(found {len(live)}: {[name for name, _ in live]})."
            )

        name, node = live[0]
        agent = node.func
        agent._node_name = name
        cfg = self._prepare_config(config, is_stream=True)
        callback_manager = InjectQ.get_instance().try_get(CallbackManager)
        context_manager = self._state_graph._context_manager
        run_state = state if state is not None else (self._state or AgentState())

        async for event in agent.arun(
            input_queue,
            cfg,
            run_state,
            checkpointer=self._checkpointer,
            callback_manager=callback_manager,
            context_manager=context_manager,
        ):
            yield event

    def realtime(
        self,
        input_queue: Any,
        config: dict[str, Any] | None = None,
        state: AgentState | None = None,
    ) -> Generator[Any]:
        """Synchronous wrapper over :meth:`arealtime` for non-async consumers.

        Must be called from a thread with no running event loop; from inside an async
        context (FastAPI handler, Jupyter), use :meth:`arealtime` directly.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass  # no running loop: safe to drive a private one below
        else:
            raise RuntimeError(
                "realtime() (sync) cannot be called from a running event loop; "
                "await arealtime() instead."
            )

        agen = self.arealtime(input_queue, config, state)
        loop = asyncio.new_event_loop()
        try:
            while True:
                try:
                    yield loop.run_until_complete(agen.__anext__())
                except StopAsyncIteration:
                    break
        finally:
            with contextlib.suppress(Exception):
                loop.run_until_complete(agen.aclose())
            loop.close()

    async def aclose(self) -> dict[str, Any]:  # noqa: PLR0915
        """
        Close the graph and release all resources gracefully.

        This method ensures proper cleanup of all components with timeout handling.
        It follows a structured shutdown sequence:
        1. Background task manager (running tasks)
        2. Checkpointer (state persistence)
        3. Publisher (event publishing)
        4. Store (data storage)

        Returns:
            Dictionary with detailed shutdown statistics for each component.

        Calling this more than once is a no-op; the second call returns
        ``{"status": "already_closed"}``. Prefer the async-context-manager form,
        which calls this automatically on exit:

        Example:
            ```python
            async def main():
                async with await build_and_compile_graph() as graph:
                    await graph.ainvoke(input_data)
                # graph.aclose() has run here, even if ainvoke raised


            # Or manage the lifecycle manually:
            async def main_manual():
                graph = await build_and_compile_graph()
                try:
                    await graph.ainvoke(input_data)
                finally:
                    stats = await graph.aclose()
                    print(f"Shutdown completed: {stats}")
            ```
        """
        from agentflow.utils.shutdown import shutdown_with_timeout

        if self._closed:
            logger.debug("CompiledGraph.aclose() called again; already closed")
            return {"status": "already_closed"}
        self._closed = True

        logger.info("Initiating graceful shutdown of CompiledGraph")
        stats: dict[str, Any] = {}
        start_time = asyncio.get_event_loop().time()

        # 1. Shutdown background task manager (handles all pending async writes)
        try:
            logger.debug("Shutting down background task manager...")
            shutdown_stats = await self._task_manager.shutdown(timeout=self._shutdown_timeout)
            logger.info("Background task manager shutdown completed")
            stats["background_tasks"] = shutdown_stats
        except Exception as e:
            stats["background_tasks"] = {"status": "error", "error": str(e)}
            logger.exception("Error shutting down background tasks: %s", e)

        # 2. Close checkpointer (may have pending writes)
        if self._checkpointer:
            try:
                logger.debug("Releasing checkpointer...")
                result = await shutdown_with_timeout(
                    self._checkpointer.arelease(),
                    timeout=self._shutdown_timeout / 3,  # Give 1/3 of total timeout
                    task_name="checkpointer",
                )
                logger.info("Checkpointer closed successfully")
                stats["checkpointer"] = result
            except Exception as e:
                stats["checkpointer"] = {"status": "error", "error": str(e)}
                logger.exception("Error closing checkpointer: %s", e)
        else:
            stats["checkpointer"] = {"status": "skipped", "reason": "no checkpointer"}

        # 3. Close publisher (event delivery)
        if self._publisher:
            try:
                logger.debug("Closing publisher...")
                result = await shutdown_with_timeout(
                    self._publisher.close(),
                    timeout=self._shutdown_timeout / 3,
                    task_name="publisher",
                )
                logger.info("Publisher closed successfully")
                stats["publisher"] = result
            except Exception as e:
                stats["publisher"] = {"status": "error", "error": str(e)}
                logger.exception("Error closing publisher: %s", e)
        else:
            stats["publisher"] = {"status": "skipped", "reason": "no publisher"}

        # 4. Close store (data persistence)
        if self._store:
            try:
                logger.debug("Releasing store...")
                result = await shutdown_with_timeout(
                    self._store.arelease(),
                    timeout=self._shutdown_timeout / 3,
                    task_name="store",
                )
                logger.info("Store closed successfully")
                stats["store"] = result
            except Exception as e:
                stats["store"] = {"status": "error", "error": str(e)}
                logger.exception("Error closing store: %s", e)
        else:
            stats["store"] = {"status": "skipped", "reason": "no store"}

        total_duration = asyncio.get_event_loop().time() - start_time
        stats["total_duration"] = total_duration
        logger.info(f"CompiledGraph shutdown completed in {total_duration:.2f}s. Stats: {stats}")
        return stats

    def generate_graph(self) -> dict[str, Any]:
        """Generate the graph representation.

        Returns:
            A dictionary representing the graph structure.
        """
        graph = {
            "info": {},
            "nodes": [],
            "edges": [],
        }
        # Populate the graph with nodes and edges
        for node_name in self._state_graph.nodes:
            graph["nodes"].append(
                {
                    "id": str(uuid4()),
                    "name": node_name,
                }
            )

        for edge in self._state_graph.edges:
            graph["edges"].append(
                {
                    "id": str(uuid4()),
                    "source": edge.from_node,
                    "target": edge.to_node,
                }
            )

        # Add few more extra info
        graph["info"] = {
            "node_count": len(graph["nodes"]),
            "edge_count": len(graph["edges"]),
            "checkpointer": self._checkpointer is not None,
            "checkpointer_type": type(self._checkpointer).__name__ if self._checkpointer else None,
            "publisher": self._publisher is not None,
            "store": self._store is not None,
            "interrupt_before": self._interrupt_before,
            "interrupt_after": self._interrupt_after,
            "context_type": self._state_graph._context_manager.__class__.__name__,
            "id_generator": self._state_graph._id_generator.__class__.__name__,
            "id_type": self._state_graph._id_generator.id_type.value,
            "state_type": self._state.__class__.__name__,
            "state_fields": list(self._state.model_dump().keys()),
        }
        return graph
