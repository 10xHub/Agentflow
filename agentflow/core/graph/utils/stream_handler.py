"""Streaming graph execution handler for TAF workflows.

This module provides the StreamHandler class, which manages the execution of graph workflows
with support for streaming output, interrupts, state persistence, and event publishing.
It enables incremental result processing, pause/resume capabilities, and robust error handling
for agent workflows that require real-time or chunked responses.
"""

from __future__ import annotations  # isort: skip_file

import logging
import time
from collections.abc import AsyncGenerator, AsyncIterable
from typing import Any, TypeVar

from collections.abc import Callable

from injectq import inject, Inject

from agentflow.core.exceptions import GraphRecursionError
from agentflow.core.graph.edge import Edge
from agentflow.core.graph.node import Node
from agentflow.runtime.publisher.events import ContentType, Event, EventModel, EventType
from agentflow.runtime.publisher.publish import publish_event
from agentflow.core.state import AgentState, Message, ErrorBlock
from agentflow.core.state.message_block import RemoteToolCallBlock
from agentflow.core.state.stream_chunks import StreamChunk, StreamEvent
from agentflow.utils import END, ResponseGranularity, add_messages
from agentflow.utils.callbacks import CallbackManager, GraphLifecycleContext
from .handler_utils import (
    check_and_handle_interrupt,
    check_interrupted,
    check_stop_requested,
    interrupt_graph,
)

from .handler_mixins import (
    BaseLoggingMixin,
    InterruptConfigMixin,
)
from .utils import (
    calculate_token_usage,
    call_realtime_sync,
    get_next_node,
    load_or_create_state,
    process_node_result,
    sync_data,
)


StateT = TypeVar("StateT", bound=AgentState)

logger = logging.getLogger("agentflow.graph")


class StreamHandler[StateT: AgentState](
    BaseLoggingMixin,
    InterruptConfigMixin,
):
    """Handles streaming execution for graph workflows in TAF.

    StreamHandler manages the execution of agent workflows as directed graphs,
    supporting streaming output, pause/resume via interrupts, state persistence,
    and event publishing for monitoring and debugging. It enables incremental
    result processing and robust error handling for complex agent workflows.

    Attributes:
        nodes: Dictionary mapping node names to Node instances.
        edges: List of Edge instances defining graph connections and routing.
        interrupt_before: List of node names where execution should pause before execution.
        interrupt_after: List of node names where execution should pause after execution.

    Example:
        ```python
        handler = StreamHandler(nodes, edges)
        async for chunk in handler.stream(input_data, config, state):
            print(chunk)
        ```
    """

    @inject
    def __init__(
        self,
        nodes: dict[str, Node],
        edges: list[Edge],
        interrupt_before: list[str] | None = None,
        interrupt_after: list[str] | None = None,
        get_node_factory: Callable[[str], Node] | None = None,
        callback_mgr: CallbackManager = Inject[CallbackManager],
    ):
        self.nodes: dict[str, Node] = nodes
        self.edges: list[Edge] = edges
        # Store factory for node lookup - enables override_node after compile
        # Uses the nodes dict reference directly - modifications to the dict are reflected
        self._get_node = get_node_factory if get_node_factory else lambda x: nodes[x]
        self.interrupt_before = interrupt_before or []
        self.interrupt_after = interrupt_after or []
        self._set_interrupts(interrupt_before, interrupt_after)
        self.callback_mgr = callback_mgr

    async def _execute_graph(  # noqa: PLR0912, PLR0915
        self,
        state: StateT,
        input_data: dict[str, Any],
        config: dict[str, Any],
    ) -> AsyncIterable[StreamChunk]:
        """
        Execute the entire graph with support for interrupts and resuming.

        Why so many chunks are yielded?
        We allow user to set response type, if they want low granularity
        Only few chunks like Message will be sent to user
        """
        logger.info(
            "Starting graph execution from node '%s' at step %d",
            state.execution_meta.current_node,
            state.execution_meta.step,
        )
        messages: list[Message] = []
        messages_ids = set()
        max_steps = config.get("recursion_limit", 25)
        logger.debug("Max steps limit set to %d", max_steps)

        last_human_messages = input_data.get("messages", []) or []
        # Stream initial input messages (e.g., human messages) so callers see full conversation
        # Only emit when present and avoid duplicates by tracking message_ids and existing context
        for m in last_human_messages:
            if m.message_id not in messages_ids:
                messages.append(m)
                messages_ids.add(m.message_id)
                yield StreamChunk(
                    event=StreamEvent.MESSAGE,
                    message=m,
                    metadata={
                        "status": "invoking_graph",
                        "reason": "initial human message",
                    },
                    thread_id=config.get("thread_id"),
                    run_id=config.get("run_id"),
                )

        # Get current execution info from state
        current_node = state.execution_meta.current_node
        step = state.execution_meta.step

        # Build lifecycle context (shared across all hooks in this execution)
        lifecycle_context = GraphLifecycleContext(config=config)

        # Create event for graph execution
        event = EventModel.default(
            config,
            data={"state": state.model_dump(exclude={"execution_meta"})},
            content_type=[ContentType.STATE],
            extra={"step": step, "current_node": current_node},
            event=Event.GRAPH_EXECUTION,
            node_name=current_node,
        )

        yield StreamChunk(
            event=StreamEvent.UPDATES,
            data={
                "status": "invoking_graph",
                "node": current_node,
                "step": step,
                "max_steps": max_steps,
            },
            thread_id=config.get("thread_id"),
            run_id=config.get("run_id"),
        )

        try:
            while current_node != END and step < max_steps:
                logger.debug("Executing step %d at node '%s'", step, current_node)

                # TODO: check if ai called for a tool in that case we should remove last message
                res = await check_stop_requested(
                    state,
                    current_node,
                    event,
                    messages,
                    config,
                )
                if res:
                    event.event_type = EventType.INTERRUPTED
                    event.metadata["status"] = "Graph execution stopped by request"
                    event.data["state"] = state.model_dump()
                    publish_event(event)
                    # stream updated state and updates
                    yield StreamChunk(
                        event=StreamEvent.UPDATES,
                        data={
                            "status": "invoking_node",
                            "node": current_node,
                            "step": step,
                            "max_steps": max_steps,
                            "reason": "Graph execution stopped by request",
                        },
                        thread_id=config.get("thread_id"),
                        run_id=config.get("run_id"),
                    )
                    return

                # Update execution metadata
                state.set_current_node(current_node)
                state.execution_meta.step = step
                await call_realtime_sync(state, config)

                # Update event with current step info
                event.data["step"] = step
                event.data["current_node"] = current_node
                event.event_type = EventType.PROGRESS
                event.metadata["status"] = f"Executing step {step} at node '{current_node}'"
                publish_event(event)

                # Check for interrupt_before
                if await check_and_handle_interrupt(
                    current_node,
                    "before",
                    state,
                    config,
                    interrupt_before=self.interrupt_before,
                    interrupt_after=self.interrupt_after,
                ):
                    logger.info("Graph execution interrupted before node '%s'", current_node)
                    event.event_type = EventType.INTERRUPTED
                    event.metadata["status"] = "Graph execution interrupted before node execution"
                    event.metadata["interrupted"] = "Before"
                    event.data["interrupted"] = "Before"
                    publish_event(event)
                    yield StreamChunk(
                        event=StreamEvent.UPDATES,
                        data={
                            "status": "invoking_node",
                            "node": current_node,
                            "step": step,
                            "max_steps": max_steps,
                            "reason": "Graph execution interrupted before node execution",
                        },
                        thread_id=config.get("thread_id"),
                        run_id=config.get("run_id"),
                    )
                    return

                # Execute current node - use factory for override support
                logger.debug("Executing node '%s'", current_node)
                node = self._get_node(current_node)

                # Snapshot state before node execution for on_state_update hook
                old_state_snapshot = state.model_copy(deep=True)

                node_event = EventModel.default(
                    config,
                    data={"step": step},
                    event=Event.NODE_EXECUTION,
                    event_type=EventType.START,
                    content_type=[ContentType.STATE],
                    node_name=current_node,
                )
                publish_event(node_event)

                ####################################################
                ############ Execute Node ##########################
                ####################################################
                config["_node_name"] = current_node
                result = node.stream(config, state)  # type: ignore
                logger.debug("Node '%s' execution completed", current_node)
                ####################################################
                ############ Execute Node ##########################
                ####################################################

                res = await check_stop_requested(
                    state,
                    current_node,
                    event,
                    messages,
                    config,
                )
                if res:
                    event.event_type = EventType.INTERRUPTED
                    event.metadata["status"] = "Graph execution stopped by request"
                    event.data["state"] = state.model_dump()
                    publish_event(event)
                    yield StreamChunk(
                        event=StreamEvent.UPDATES,
                        data={
                            "status": "invoking_node",
                            "node": current_node,
                            "step": step,
                            "max_steps": max_steps,
                            "reason": "Graph execution stopped by request",
                        },
                        thread_id=config.get("thread_id"),
                        run_id=config.get("run_id"),
                    )
                    return

                # Process result and get next node
                yield StreamChunk(
                    event=StreamEvent.UPDATES,
                    data={
                        "status": "node_invoked",
                        "node": current_node,
                        "step": step,
                        "max_steps": max_steps,
                    },
                    thread_id=config.get("thread_id"),
                    run_id=config.get("run_id"),
                )

                # From Here message no need to stream its already streamed
                # from execute node function, only stream updates and state
                next_node = None
                async for rs in result:
                    # Allow stop to break inner result loop as well
                    if isinstance(rs, StreamChunk):
                        yield rs

                    # if message and remote tool call then yield immediately
                    elif isinstance(rs, Message) and RemoteToolCallBlock in rs.content:
                        # now interrupt the graph
                        await interrupt_graph(
                            current_node,
                            state,
                            config,
                        )
                        yield StreamChunk(
                            event=StreamEvent.UPDATES,
                            data={
                                "status": "node_invoked",
                                "node": current_node,
                                "step": step,
                                "max_steps": max_steps,
                                "reason": "Remote tool call - graph interrupted",
                            },
                            thread_id=config.get("thread_id"),
                            run_id=config.get("run_id"),
                        )
                        return

                    elif isinstance(rs, Message) and not rs.delta:
                        if rs.message_id not in messages_ids:
                            messages.append(rs)
                            messages_ids.add(rs.message_id)

                    elif isinstance(rs, dict) and "is_non_streaming" in rs:
                        if rs["is_non_streaming"]:
                            new_state = rs.get("state", None)
                            if new_state:
                                state = new_state
                                yield StreamChunk(
                                    event=StreamEvent.STATE,
                                    state=state,
                                    metadata={
                                        "node": current_node,
                                        "step": step,
                                    },
                                    thread_id=config.get("thread_id"),
                                    run_id=config.get("run_id"),
                                )

                            new_messages = rs.get("messages", [])
                            for m in new_messages:
                                if m.message_id not in messages_ids and not m.delta:
                                    messages.append(m)
                                    messages_ids.add(m.message_id)
                            next_node = rs.get("next_node", next_node)
                        else:
                            # Streaming path completed: ensure any collected messages are persisted
                            new_messages = rs.get("messages", [])
                            for m in new_messages:
                                if m.message_id not in messages_ids and not m.delta:
                                    messages.append(m)
                                    messages_ids.add(m.message_id)
                            next_node = rs.get("next_node", next_node)
                    else:
                        # Process as node result (non-streaming path)
                        try:
                            state, new_messages, next_node = await process_node_result(
                                rs,
                                state,
                                [],
                            )
                            for m in new_messages:
                                if m.message_id not in messages_ids and not m.delta:
                                    messages.append(m)
                                    messages_ids.add(m.message_id)
                        except Exception as e:
                            logger.error("Failed to process node result: %s", e)

                logger.debug(
                    "Node result processed, next_node=%s, total_messages=%d",
                    next_node,
                    len(messages),
                )

                node_event.event_type = EventType.END
                node_event.data["messages"] = [m.model_dump() for m in messages] if messages else []
                node_event.content_type = [ContentType.MESSAGE]
                publish_event(node_event)

                # Add collected messages to state context
                if messages:
                    state.context = add_messages(state.context, messages)
                    logger.debug("Added %d messages to state context", len(messages))
                    yield StreamChunk(
                        event=StreamEvent.STATE,
                        state=state,
                        metadata={
                            "node": current_node,
                            "step": step,
                        },
                        thread_id=config.get("thread_id"),
                        run_id=config.get("run_id"),
                    )

                # Fire on_state_update hook after state has been merged
                if self.callback_mgr and self.callback_mgr._lifecycle_hooks:
                    state = await self.callback_mgr.fire_on_state_update(
                        lifecycle_context,
                        node_name=current_node,
                        old_state=old_state_snapshot,
                        new_state=state,
                        step=step,
                    )

                # Call realtime sync after node execution
                await call_realtime_sync(state, config)
                event.event_type = EventType.UPDATE
                event.data["state"] = state.model_dump()
                event.data["messages"] = [m.model_dump() for m in messages] if messages else []
                if messages:
                    lm = messages[-1]
                    event.content = str(lm.text() if isinstance(lm.content, list) else lm.content)
                    if isinstance(lm.content, list):
                        event.content_blocks = lm.content
                event.content_type = [ContentType.STATE, ContentType.MESSAGE]
                publish_event(event)

                is_interrupted_requested = False

                # Check for interrupt_after
                if await check_and_handle_interrupt(
                    current_node,
                    "after",
                    state,
                    config,
                    interrupt_before=self.interrupt_before,
                    interrupt_after=self.interrupt_after,
                ):
                    logger.info("Graph execution interrupted after node '%s'", current_node)
                    # For interrupt_after, advance to next node before pausing
                    if next_node is None:
                        next_node = get_next_node(current_node, state, self.edges)
                    state.set_current_node(next_node)

                    is_interrupted_requested = True

                    event.event_type = EventType.INTERRUPTED
                    event.data["interrupted"] = "After"
                    event.metadata["interrupted"] = "After"
                    event.data["state"] = state.model_dump()
                    publish_event(event)

                    yield StreamChunk(
                        event=StreamEvent.UPDATES,
                        data={
                            "status": "node_invoked",
                            "node": current_node,
                            "step": step,
                            "max_steps": max_steps,
                            "reason": "Graph execution interrupted before node execution",
                        },
                        thread_id=config.get("thread_id"),
                        run_id=config.get("run_id"),
                    )

                # Get next node
                if next_node is None:
                    current_node = get_next_node(current_node, state, self.edges)
                    logger.debug("Next node determined by graph logic: '%s'", current_node)
                else:
                    current_node = next_node
                    logger.debug("Next node determined by command: '%s'", current_node)

                # Advance step after successful node execution
                step += 1
                state.advance_step()
                await call_realtime_sync(state, config)

                event.event_type = EventType.UPDATE
                event.metadata["State_Updated"] = "State Updated"
                event.data["state"] = state.model_dump()
                publish_event(event)

                yield StreamChunk(
                    event=StreamEvent.UPDATES,
                    state=state,
                    data={
                        "status": "node_invoked",
                        "node": current_node,
                        "step": step,
                        "max_steps": max_steps,
                    },
                    thread_id=config.get("thread_id"),
                    run_id=config.get("run_id"),
                )

                if is_interrupted_requested:
                    return

                if step >= max_steps:
                    error_msg = "Graph execution exceeded maximum steps"
                    logger.error(error_msg)
                    state.error(error_msg)
                    await call_realtime_sync(state, config)

                    event.event_type = EventType.ERROR
                    event.data["state"] = state.model_dump()
                    event.metadata["error"] = error_msg
                    event.metadata["step"] = step
                    event.metadata["current_node"] = current_node
                    publish_event(event)

                    yield Message(
                        role="assistant",
                        content=[ErrorBlock(text=error_msg)],  # type: ignore
                    )

                    yield StreamChunk(
                        event=StreamEvent.ERROR,
                        state=state,
                        data={
                            "status": "graph_invoked",
                            "node": current_node,
                            "step": step,
                            "max_steps": max_steps,
                            "reason": error_msg,
                        },
                        thread_id=config.get("thread_id"),
                        run_id=config.get("run_id"),
                    )

                    raise GraphRecursionError(
                        message=f"Graph execution exceeded recursion limit: {max_steps}",
                        error_code="RECURSION_001",
                        context={
                            "max_steps": max_steps,
                            "current_step": step,
                            "current_node": current_node,
                        },
                    )

            # Execution completed successfully
            logger.info(
                "Graph execution completed successfully at node '%s' after %d steps",
                current_node,
                step,
            )
            state.complete()

            # Fire on_graph_end hook before final state sync
            if self.callback_mgr and self.callback_mgr._lifecycle_hooks:
                state = await self.callback_mgr.fire_on_graph_end(
                    lifecycle_context,
                    final_state=state,
                    messages=messages,
                    total_steps=step,
                )

            is_context_trimmed = await sync_data(
                state=state,
                config=config,
                messages=messages,
                trim=True,
            )

            # Create completion event
            event.event_type = EventType.END
            event.data["state"] = state.model_dump()
            event.data["messages"] = [m.model_dump() for m in messages] if messages else []
            if messages:
                fm = messages[-1]
                event.content = str(fm.text() if isinstance(fm.content, list) else fm.content)
                if isinstance(fm.content, list):
                    event.content_blocks = fm.content
            event.content_type = [ContentType.STATE, ContentType.MESSAGE]
            event.metadata["status"] = "Graph execution completed"
            event.metadata["step"] = step
            event.metadata["current_node"] = current_node
            event.metadata["is_context_trimmed"] = is_context_trimmed
            publish_event(event)

            # Include messages list for token calculation in stream method
            yield StreamChunk(
                event=StreamEvent.UPDATES,
                state=state,
                data={
                    "status": "graph_invoked",
                    "node": current_node,
                    "step": step,
                    "max_steps": max_steps,
                    "is_context_trimmed": is_context_trimmed,
                    "reason": "Graph execution completed successfully",
                    # Internal: messages from current run for token calculation
                    "_messages": messages,
                },
                thread_id=config.get("thread_id"),
                run_id=config.get("run_id"),
            )

        except Exception as e:
            # Handle execution errors
            logger.exception("Graph execution failed: %s", e)
            state.error(str(e))

            # Publish error event
            event.event_type = EventType.ERROR
            event.metadata["error"] = str(e)
            event.data["state"] = state.model_dump()
            publish_event(event)

            # Fire on_graph_error hook before persisting the error state
            if self.callback_mgr and self.callback_mgr._lifecycle_hooks:
                state, _ = await self.callback_mgr.fire_on_graph_error(
                    lifecycle_context,
                    error=e,
                    partial_state=state,
                    messages=messages,
                    step=step,
                    node_name=current_node,
                )

            await sync_data(
                state=state,
                config=config,
                messages=messages,
                trim=True,
            )

            yield StreamChunk(
                event=StreamEvent.ERROR,
                state=state,
                data={
                    "status": "invoked_graph",
                    "node": current_node,
                    "step": step,
                    "max_steps": max_steps,
                    "reason": str(e),
                },
                thread_id=config.get("thread_id"),
                run_id=config.get("run_id"),
            )

            raise e

    async def stream(
        self,
        input_data: dict[str, Any],
        config: dict[str, Any],
        default_state: StateT,
        response_granularity: ResponseGranularity = ResponseGranularity.LOW,
    ) -> AsyncGenerator[StreamChunk]:
        """Execute the graph asynchronously with streaming output.

        Runs the graph workflow from start to finish, yielding incremental results
        as they become available. Automatically detects whether to start a fresh
        execution or resume from an interrupted state, supporting pause/resume
        and checkpointing.

        Args:
            input_data: Input dictionary for graph execution. For new executions,
                should contain 'messages' key with initial messages. For resumed
                executions, can contain additional data to merge.
            config: Configuration dictionary containing execution settings and context.
            default_state: Initial or template AgentState for workflow execution.
            response_granularity: Level of detail in the response (LOW, PARTIAL, FULL).

        Yields:
            Message objects representing incremental results from graph execution.
            The exact type and frequency of yields depends on node implementations
            and workflow configuration.

        Raises:
            GraphRecursionError: If execution exceeds recursion limit.
            ValueError: If input_data is invalid for new execution.
            Various exceptions: Depending on node execution failures.

        Example:
            ```python
            async for chunk in handler.stream(input_data, config, state):
                print(chunk)
            ```
        """
        logger.info(
            "Starting asynchronous graph execution with %d input keys, granularity=%s",
            len(input_data) if input_data else 0,
            response_granularity,
        )
        config = config or {}
        input_data = input_data or {}

        start_time = time.time()

        # Load or initialize state
        logger.debug("Loading or creating state from input data")
        new_state = await load_or_create_state(
            input_data,
            config,
            default_state,
        )
        state: StateT = new_state  # type: ignore[assignment]
        logger.debug(
            "State loaded: interrupted=%s, current_node=%s, step=%d",
            state.is_interrupted(),
            state.execution_meta.current_node,
            state.execution_meta.step,
        )

        cfg = config.copy()
        if "user" in cfg:
            # This will be available when you are calling
            # vi agentflow api
            del cfg["user"]

        event = EventModel.default(
            config,
            data={"state": state},
            content_type=[ContentType.STATE],
            extra={
                "is_interrupted": state.is_interrupted(),
                "current_node": state.execution_meta.current_node,
                "step": state.execution_meta.step,
                "config": cfg,
                "response_granularity": response_granularity.value,
            },
        )

        # Publish graph initialization event
        publish_event(event)

        # Check if this is a resume case
        state, config = await check_interrupted(state, input_data, config)

        # Fire on_graph_start hook before execution begins
        if self.callback_mgr and self.callback_mgr._lifecycle_hooks:
            lc_context = GraphLifecycleContext(config=config)
            state = await self.callback_mgr.fire_on_graph_start(lc_context, state)  # type: ignore

        # Now start Execution
        # Execute graph
        logger.debug("Beginning graph execution")
        result = self._execute_graph(state, input_data, config)

        # Track messages from current run for token calculation
        current_run_messages = []

        # Stream results based on response granularity
        async for chunk in result:
            # Extract messages from final completion chunk (internal use only)
            if (
                chunk.event == StreamEvent.UPDATES
                and chunk.data
                and chunk.data.get("status") == "graph_invoked"
            ):
                current_run_messages = chunk.data.pop("_messages", [])

            match response_granularity:
                case ResponseGranularity.FULL:
                    yield chunk
                case ResponseGranularity.PARTIAL:
                    if chunk.event != StreamEvent.UPDATES:
                        yield chunk
                case ResponseGranularity.LOW:
                    if chunk.event in [StreamEvent.MESSAGE, StreamEvent.ERROR]:
                        yield chunk

        # Publish graph completion event
        time_taken = time.time() - start_time
        logger.info("Graph execution finished in %.2f seconds", time_taken)

        # Calculate token usage from current run messages only
        token_usage = calculate_token_usage(current_run_messages)

        event.event_type = EventType.END
        event.metadata.update(
            {
                "time_taken": time_taken,
                "state": state.model_dump(),
                "step": state.execution_meta.step,
                "current_node": state.execution_meta.current_node,
                "is_interrupted": state.is_interrupted(),
                "total_messages": len(state.context) if state.context else 0,
                **token_usage,
            }
        )
        publish_event(event)
        if response_granularity == ResponseGranularity.FULL:
            yield StreamChunk(
                event=StreamEvent.UPDATES,
                state=state,
                data={
                    "status": "graph_invoked",
                    "reason": "Graph execution finished",
                    "time_taken": time_taken,
                    "is_interrupted": state.is_interrupted(),
                    "total_messages": len(state.context) if state.context else 0,
                    **token_usage,
                },
                thread_id=config.get("thread_id"),
                run_id=config.get("run_id"),
            )
