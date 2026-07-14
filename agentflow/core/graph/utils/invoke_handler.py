from __future__ import annotations  # isort: skip_file

import logging
from typing import Any, TypeVar

from collections.abc import Callable

from injectq import inject, Inject

from agentflow.core.exceptions import (
    GraphRecursionError,
    GraphStopRequested,
    NodeTimeoutError,
)
from agentflow.core.graph.edge import Edge
from agentflow.core.graph.node import Node
from agentflow.core.graph.utils.guards import execute_with_guards, resolve_timeout
from agentflow.core.graph.utils.utils import (
    calculate_token_usage,
    call_realtime_sync,
    get_next_node,
    load_or_create_state,
    parse_response,
    sync_data,
)
from agentflow.storage.checkpointer import BaseCheckpointer
from agentflow.utils.constants import DEFAULT_NODE_TIMEOUT_SECONDS
from agentflow.runtime.publisher.events import ContentType, Event, EventModel, EventType
from agentflow.runtime.publisher.publish import publish_event
from agentflow.core.state import AgentState, Message
from agentflow.core.state.message_block import RemoteToolCallBlock
from agentflow.utils import END, ResponseGranularity, metrics
from agentflow.utils.logging import bind_log_context_from_config, set_log_context
from agentflow.core.state.reducers import add_messages
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


StateT = TypeVar("StateT", bound=AgentState)

logger = logging.getLogger("agentflow.graph")


class InvokeHandler[StateT: AgentState](
    BaseLoggingMixin,
    InterruptConfigMixin,
):
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
        # Keep existing attributes for backward-compatibility
        self.interrupt_before = interrupt_before or []
        self.interrupt_after = interrupt_after or []
        # And set via mixin for a single source of truth
        self._set_interrupts(interrupt_before, interrupt_after)
        self.callback_mgr = callback_mgr

    async def _execute_node_guarded(
        self,
        node: Node,
        node_name: str,
        config: dict[str, Any],
        state: StateT,
        checkpointer: BaseCheckpointer = Inject[BaseCheckpointer],
    ):
        """Execute a node under a deadline, cancellable by an out-of-band stop.

        Bare `await node.execute(...)` had no bound and no way in: a node hung on
        a socket blocked the run forever, and because control never returned to
        the loop, the between-nodes stop check never ran. Here the node runs as a
        task we can actually cancel -- on the deadline, or as soon as a stop
        request shows up.
        """
        timeout = resolve_timeout(config, "node_timeout", DEFAULT_NODE_TIMEOUT_SECONDS)

        stop_check = None
        if checkpointer:

            async def stop_check() -> bool:  # noqa: F811
                return await checkpointer.ais_stop_requested(config)

        # Node-level metrics. The engine emitted no counters or histograms at all,
        # so operators had spans but nothing to alert on -- no error rate, no
        # latency distribution. The timer tags each observation with its outcome,
        # so success and failure latencies can be separated.
        attrs = {"node": node_name}
        metrics.counter("agentflow.node.executions").inc(attributes=attrs)

        try:
            with metrics.timer("agentflow.node.duration", attributes=attrs):
                return await execute_with_guards(
                    node.execute(config, state),  # type: ignore[arg-type]
                    timeout=timeout,
                    stop_check=stop_check,
                    on_timeout=lambda: NodeTimeoutError(
                        message=(
                            f"Node '{node_name}' exceeded its timeout of {timeout}s "
                            f"and was cancelled"
                        ),
                        error_code="NODE_TIMEOUT_001",
                        context={"node_name": node_name, "timeout": timeout},
                    ),
                    on_stop=lambda: GraphStopRequested(node_name),
                )
        except NodeTimeoutError:
            metrics.counter("agentflow.node.timeouts").inc(attributes=attrs)
            raise
        except GraphStopRequested:
            metrics.counter("agentflow.node.stopped").inc(attributes=attrs)
            raise
        except Exception:
            metrics.counter("agentflow.node.errors").inc(attributes=attrs)
            raise

    async def _execute_graph(  # noqa: PLR0912, PLR0915
        self,
        state: StateT,
        config: dict[str, Any],
    ) -> tuple[StateT, list[Message]]:
        """Execute the entire graph with support for interrupts and resuming."""
        logger.info(
            "Starting graph execution from node '%s' at step %d",
            state.execution_meta.current_node,
            state.execution_meta.step,
        )
        logger.debug("DEBUG: Current node value: %r", state.execution_meta.current_node)
        logger.debug("DEBUG: END constant value: %r", END)
        logger.debug("DEBUG: Are they equal? %s", state.execution_meta.current_node == END)
        messages: list[Message] = []
        # How many of `messages` have already been durably persisted by a per-step
        # checkpoint, so terminal and subsequent writes only send the new ones.
        persisted_upto = 0
        max_steps = config.get("recursion_limit", 25)
        logger.debug("Max steps limit set to %d", max_steps)

        # get the last message from state as that is human message
        last_human_message = state.context[-1] if state.context else None
        if last_human_message and last_human_message.role != "user":
            msg = [msg for msg in reversed(state.context) if msg.role == "user"]
            last_human_message = msg[0] if msg else None

        if last_human_message:
            logger.debug("Last human message: %s", last_human_message.content)
            messages.append(last_human_message)

        # Get current execution info from state
        current_node = state.execution_meta.current_node
        step = state.execution_meta.step

        # Build lifecycle context (shared across all hooks in this execution)
        lifecycle_context = GraphLifecycleContext(config=config)

        # Create event for graph execution
        event = EventModel.default(
            config,
            data={"state": state.model_dump()},
            event=Event.GRAPH_EXECUTION,
            content_type=[ContentType.STATE],
            node_name=current_node,
            extra={
                "current_node": current_node,
                "step": step,
                "max_steps": max_steps,
            },
        )

        try:
            while current_node != END and step < max_steps:
                logger.debug("Executing step %d at node '%s'", step, current_node)
                # Reload state in each iteration to get latest (in case of external updates)
                res = await check_stop_requested(
                    state,
                    current_node,
                    event,
                    messages,
                    config,
                )
                if res:
                    return state, messages

                # Update execution metadata
                state.set_current_node(current_node)
                state.execution_meta.step = step
                await call_realtime_sync(state, config)
                event.data["state"] = state.model_dump()
                event.metadata["step"] = step
                event.metadata["current_node"] = current_node
                event.event_type = EventType.PROGRESS
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
                    event.metadata["interrupted"] = "Before"
                    event.metadata["status"] = "Graph execution interrupted before node execution"
                    event.data["interrupted"] = "Before"
                    publish_event(event)
                    return state, messages

                # Execute current node - use factory for override support
                logger.debug("Executing node '%s'", current_node)
                node = self._get_node(current_node)

                # Snapshot state before node execution for on_state_update hook
                old_state_snapshot = state.model_copy(deep=True)

                node_event = EventModel.default(
                    config,
                    data={"step": step, "state": state.model_dump()},
                    event=Event.NODE_EXECUTION,
                    event_type=EventType.START,
                    content_type=[ContentType.STATE],
                    node_name=current_node,
                )
                publish_event(node_event)

                ###############################################
                ##### Node Execution Started ##################
                ###############################################

                config["_node_name"] = current_node
                # Narrow the correlation context to the node now executing.
                set_log_context(node=current_node)
                try:
                    result = await self._execute_node_guarded(node, current_node, config, state)
                except GraphStopRequested:
                    # The node was cancelled mid-flight because a stop arrived.
                    # Hand off to the normal stop path so the run is marked
                    # stopped, persisted, and reported exactly as a between-nodes
                    # stop would be -- this is control flow, not a failure.
                    logger.info(
                        "Node '%s' cancelled by stop request; finalizing run",
                        current_node,
                    )
                    await check_stop_requested(state, current_node, event, messages, config)
                    return state, messages

                ###############################################
                ##### Node Execution Finished #################
                ###############################################

                logger.debug("Node '%s' execution completed", current_node)

                next_node = None

                # check frontend nodes
                if isinstance(result, Message) and RemoteToolCallBlock in result.content:
                    # now interrupt the graph
                    await interrupt_graph(
                        current_node,
                        state,
                        config,
                    )
                    messages.append(result)
                    return state, messages

                # Process result and get next node
                if isinstance(result, list):
                    # If result is a list of Message, append to messages
                    messages.extend(result)
                    logger.debug(
                        "Node '%s' returned %d messages, total messages now %d",
                        current_node,
                        len(result),
                        len(messages),
                    )
                    # Add messages to state context so they're visible to subsequent nodes
                    state.context = add_messages(state.context, result)

                # No state change beyond adding messages, just advance to next node
                if isinstance(result, dict):
                    state = result.get("state", state)
                    next_node = result.get("next_node")
                    new_messages = result.get("messages", [])
                    if new_messages:
                        messages.extend(new_messages)
                        logger.debug(
                            "Node '%s' returned %d messages, total messages now %d",
                            current_node,
                            len(new_messages),
                            len(messages),
                        )

                logger.debug(
                    "Node result processed, next_node=%s, total_messages=%d",
                    next_node,
                    len(messages),
                )

                node_event.event_type = EventType.END
                node_event.data["messages"] = [m.model_dump() for m in messages] if messages else []
                node_event.content_type = [ContentType.MESSAGE]
                publish_event(node_event)

                # Check stop again after node execution
                res = await check_stop_requested(
                    state,
                    current_node,
                    event,
                    messages,
                    config,
                )
                if res:
                    return state, messages

                # Fire on_state_update hook after state has been merged
                if self.callback_mgr and self.callback_mgr._lifecycle_hooks:
                    state = await self.callback_mgr.fire_on_state_update(
                        lifecycle_context,
                        node_name=current_node,
                        old_state=old_state_snapshot,
                        new_state=state,
                        step=step,
                    )  # type: ignore

                # Call realtime sync after node execution (if state/messages changed)
                await call_realtime_sync(state, config)
                event.event_type = EventType.UPDATE
                event.data["state"] = state.model_dump()
                event.data["messages"] = [m.model_dump() for m in messages] if messages else []
                if messages:
                    lm = messages[-1]
                    event.content = lm.text() if isinstance(lm.content, list) else lm.content  # type: ignore
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

                # Get next node (only if no explicit navigation from Command)
                if next_node is None:
                    current_node = get_next_node(current_node, state, self.edges)
                    logger.debug("Next node determined by graph logic: '%s'", current_node)
                else:
                    current_node = next_node
                    logger.debug("Next node determined by command: '%s'", current_node)

                # Advance step after successful node execution
                step += 1
                state.advance_step()

                # Durable per-step checkpoint.
                #
                # Previously this only wrote the Redis cache, and the durable
                # write happened solely at terminal points (completion, error,
                # interrupt, stop). A pod killed mid-run therefore resumed from
                # the last terminal checkpoint -- or, once the 24h cache TTL had
                # lapsed or Redis restarted, from the very beginning. Persisting
                # each completed step bounds a crash to replaying one node.
                #
                # Only messages not yet persisted are written, so a long run does
                # not re-upsert its whole history on every step.
                if config.get("durable_checkpoint_every_step", True):
                    pending = messages[persisted_upto:]
                    await sync_data(
                        state=state,
                        config=config,
                        messages=pending,
                        trim=False,
                    )
                    persisted_upto = len(messages)
                else:
                    await call_realtime_sync(state, config)

                event.event_type = EventType.UPDATE

                event.metadata["State_Updated"] = "State Updated"
                event.data["state"] = state.model_dump()
                publish_event(event)

                # If we interrupted after, exit now
                if is_interrupted_requested:
                    return state, messages

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
                )  # type: ignore

            res = await sync_data(
                state=state,
                config=config,
                messages=messages,
                trim=True,
            )
            event.event_type = EventType.END
            event.data["state"] = state.model_dump()
            event.data["messages"] = [m.model_dump() for m in messages] if messages else []
            if messages:
                fm = messages[-1]
                event.content = fm.text() if isinstance(fm.content, list) else fm.content  # type: ignore
                if isinstance(fm.content, list):
                    event.content_blocks = fm.content
            event.content_type = [ContentType.STATE, ContentType.MESSAGE]
            event.metadata["status"] = "Graph execution completed"
            event.metadata["step"] = step
            event.metadata["current_node"] = current_node
            event.metadata["is_context_trimmed"] = res

            publish_event(event)

            return state, messages

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
                )  # type: ignore

            await sync_data(
                state=state,
                config=config,
                messages=messages,
                trim=True,
            )
            raise

    async def invoke(
        self,
        input_data: dict[str, Any],
        config: dict[str, Any],
        default_state: StateT,
        response_granularity: ResponseGranularity = ResponseGranularity.LOW,
    ):
        """Execute the graph asynchronously with event publishing."""
        logger.info(
            "Starting asynchronous graph execution with %d input keys, granularity=%s",
            len(input_data) if input_data else 0,
            response_granularity,
        )
        input_data = input_data or {}

        # Bind run correlation for every log line emitted underneath this run, so a
        # single run can actually be grepped out of a busy multi-tenant server.
        bind_log_context_from_config(config)

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

        # Event publishing logic
        event = EventModel.default(
            config,
            data={"state": state.model_dump()},
            event=Event.GRAPH_EXECUTION,
            content_type=[ContentType.STATE],
            node_name=state.execution_meta.current_node,
            extra={
                "current_node": state.execution_meta.current_node,
                "step": state.execution_meta.step,
            },
        )
        event.event_type = EventType.START
        publish_event(event)

        # Check if this is a resume case
        state, config = await check_interrupted(state, input_data, config)

        # Fire on_graph_start hook before execution begins
        if self.callback_mgr and self.callback_mgr._lifecycle_hooks:
            lc_context = GraphLifecycleContext(config=config)
            state = await self.callback_mgr.fire_on_graph_start(lc_context, state)  # type: ignore

        event.event_type = EventType.UPDATE
        event.metadata["status"] = "Graph invoked"
        publish_event(event)

        try:
            logger.debug("Beginning graph execution")
            event.event_type = EventType.PROGRESS
            event.metadata["status"] = "Graph execution started"
            publish_event(event)

            final_state, messages = await self._execute_graph(state, config)
            logger.info("Graph execution completed with %d final messages", len(messages))

            # Calculate token usage
            token_usage = calculate_token_usage(messages)

            event.event_type = EventType.END
            event.metadata["status"] = "Graph execution completed"
            event.metadata.update(token_usage)
            event.data["state"] = final_state.model_dump()
            event.data["messages"] = [m.model_dump() for m in messages] if messages else []
            publish_event(event)

            return await parse_response(
                final_state,
                messages,
                response_granularity,
                token_usage=token_usage,
            )
        except Exception as e:
            logger.exception("Graph execution failed: %s", e)
            event.event_type = EventType.ERROR
            event.metadata["status"] = f"Graph execution failed: {e}"
            event.data["error"] = str(e)
            publish_event(event)
            raise
