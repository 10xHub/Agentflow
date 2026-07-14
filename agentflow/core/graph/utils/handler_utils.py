"""Core utility functions for graph execution and state management.

This module provides essential utilities for TAF graph execution, including
state management, message processing, response formatting, and execution flow control.
These functions handle the low-level operations that support graph workflow execution.

The utilities in this module are designed to work with TAF's dependency injection
system and provide consistent interfaces for common operations across different
execution contexts.

Key functionality areas:
- State loading, creation, and synchronization
- Message processing and deduplication
- Response formatting based on granularity levels
- Node execution result processing
- Interrupt handling and execution flow control
"""

from __future__ import annotations

import logging
from typing import Any, TypeVar

from injectq import Inject

from agentflow.core.state import AgentState, ExecutionStatus
from agentflow.core.state.execution_state import StopRequestStatus
from agentflow.core.state.message import Message
from agentflow.runtime.publisher.events import EventModel, EventType
from agentflow.runtime.publisher.publish import publish_event
from agentflow.storage.checkpointer import BaseCheckpointer
from agentflow.utils import (
    START,
)
from agentflow.utils.callbacks import CallbackManager, GraphLifecycleContext

from .utils import reload_state, sync_data


StateT = TypeVar("StateT", bound=AgentState)

logger = logging.getLogger("agentflow.graph")


async def check_interrupted[StateT: AgentState](
    state: StateT,
    input_data: dict[str, Any],
    config: dict[str, Any],
    callback_mgr: CallbackManager = Inject[CallbackManager],
) -> tuple[StateT, dict[str, Any]]:
    if state.is_interrupted():
        logger.info(
            "Resuming from interrupted state at node '%s'", state.execution_meta.current_node
        )

        # Fire on_resume hook before clearing the interrupt
        if callback_mgr and callback_mgr._lifecycle_hooks:
            context = GraphLifecycleContext(config=config)
            resumed_node = state.execution_meta.interrupted_node or ""
            modified = await callback_mgr.fire_on_resume(
                context,
                resumed_node=resumed_node,
                state=state,
                resume_data=input_data,
            )
            if modified is not None and modified is not state:
                state = modified  # type: ignore[assignment]

        # Save the interrupted node info before clearing so we don't re-interrupt
        config["_skip_interrupt_at"] = {
            "node": state.execution_meta.interrupted_node,
            "status": state.execution_meta.status,
        }
        # This is a resume case - clear interrupt and merge input data
        if input_data:
            config["resume_data"] = input_data
            logger.debug("Added resume data with %d keys", len(input_data))
        state.clear_interrupt()
    elif not input_data.get("messages") and not state.context:
        # This is a fresh execution - validate input data
        error_msg = "Input data must contain 'messages' for new execution."
        logger.error(error_msg)
        raise ValueError(error_msg)
    elif state.execution_meta.status == ExecutionStatus.COMPLETED:
        # Previous execution completed - reset to entry point for new execution
        logger.info(
            "Previous execution completed. Resetting to entry point for new execution "
            "with %d messages",
            len(input_data.get("messages", [])),
        )
        # Reset execution metadata for fresh start
        state.execution_meta.current_node = START
        state.execution_meta.step = 0
        state.execution_meta.status = ExecutionStatus.RUNNING
        state.execution_meta.interrupted_node = None
        state.execution_meta.interrupt_reason = None
        state.execution_meta.interrupt_data = None
    else:
        # Fresh execution, state is already at START
        logger.info(
            "Starting fresh execution with %d messages", len(input_data.get("messages", []))
        )

    return state, config


async def check_and_handle_interrupt[StateT: AgentState](
    current_node: str,
    interrupt_type: str,
    state: StateT,
    config: dict[str, Any],
    interrupt_before: list[str] | None = None,
    interrupt_after: list[str] | None = None,
    callback_mgr: CallbackManager = Inject[CallbackManager],
) -> bool:
    """Check for interrupts and save state if needed. Returns True if interrupted."""
    interrupt_nodes: list[str] = (
        interrupt_before if interrupt_type == "before" else interrupt_after
    ) or []

    # Check if we just resumed from an interrupt at this node with this type
    skip_info = config.get("_skip_interrupt_at", {})
    if skip_info.get("node") == current_node:
        expected_status = (
            ExecutionStatus.INTERRUPTED_BEFORE
            if interrupt_type == "before"
            else ExecutionStatus.INTERRUPTED_AFTER
        )
        if skip_info.get("status") == expected_status:
            logger.debug(
                "Skipping %s interrupt check for node '%s' - just resumed from it",
                interrupt_type,
                current_node,
            )
            # Clear the flag after using it once
            config.pop("_skip_interrupt_at", None)
            return False

    if current_node in interrupt_nodes:
        status = (
            ExecutionStatus.INTERRUPTED_BEFORE
            if interrupt_type == "before"
            else ExecutionStatus.INTERRUPTED_AFTER
        )
        state.set_interrupt(
            current_node,
            f"interrupt_{interrupt_type}: {current_node}",
            status,
        )

        # Fire on_interrupt hook
        if callback_mgr and callback_mgr._lifecycle_hooks:
            context = GraphLifecycleContext(config=config)
            await callback_mgr.fire_on_interrupt(
                context,
                interrupted_node=current_node,
                interrupt_type=interrupt_type,
                state=state,
            )

        # Save state and interrupt
        await sync_data(
            state=state,
            config=config,
            messages=[],
            trim=True,
        )
        logger.debug("Node '%s' interrupted", current_node)
        return True

    logger.debug(
        "No interrupts found for node '%s', continuing execution",
        current_node,
    )
    return False


async def interrupt_graph[StateT: AgentState](
    current_node: str,
    state: StateT,
    config: dict[str, Any],
    callback_mgr: CallbackManager = Inject[CallbackManager],
) -> bool:
    """Check for interrupts and save state if needed. Returns True if interrupted."""
    status = ExecutionStatus.INTERRUPTED_AFTER
    state.set_interrupt(
        current_node,
        f"interrupt_after: {current_node}",
        status,
    )

    # Fire on_interrupt hook
    if callback_mgr and callback_mgr._lifecycle_hooks:
        context = GraphLifecycleContext(config=config)
        await callback_mgr.fire_on_interrupt(
            context,
            interrupted_node=current_node,
            interrupt_type="remote_tool",
            state=state,
        )

    # Save state and interrupt
    await sync_data(
        state=state,
        config=config,
        messages=[],
        trim=False,
    )
    logger.debug("Node '%s' interrupted", current_node)
    return True


async def check_stop_requested[StateT: AgentState](
    state: StateT,
    current_node: str,
    event: EventModel,
    messages: list[Message],
    config: dict[str, Any],
    callback_mgr: CallbackManager = Inject[CallbackManager],
    checkpointer: BaseCheckpointer = Inject[BaseCheckpointer],
) -> bool:
    """Check if a stop has been requested externally."""
    state = await reload_state(config, state)  # type: ignore

    # A stop request lives in its own checkpointer key, NOT in the cached state.
    # This loop rewrites the state cache after every node, so a flag carried in
    # the state blob can be overwritten by our own flag-less copy before we get
    # here. The dedicated key is never written by the loop, so it survives.
    # The state blob is still consulted, so a custom checkpointer that does not
    # back the stop key keeps working the way it did before.
    stop_requested = state.is_stopped_requested()
    if not stop_requested and checkpointer:
        stop_requested = await checkpointer.ais_stop_requested(config)
        if stop_requested:
            state.execution_meta.stop_current_execution = StopRequestStatus.STOP_REQUESTED

    # Check if a stop was requested externally (e.g., frontend)
    if stop_requested:
        logger.info(
            "Stop requested for thread '%s' at node '%s'",
            config.get("thread_id"),
            current_node,
        )
        # Consume the request so it cannot leak into the next run on this thread.
        if checkpointer:
            await checkpointer.aclear_stop_request(config)

        state.set_interrupt(
            current_node,
            "stop_requested",
            ExecutionStatus.INTERRUPTED_AFTER,
            data={"source": "stop", "info": "requested via is_stopped_requested"},
        )

        # Fire on_interrupt hook
        if callback_mgr and callback_mgr._lifecycle_hooks:
            context = GraphLifecycleContext(config=config)
            await callback_mgr.fire_on_interrupt(
                context,
                interrupted_node=current_node,
                interrupt_type="stop",
                state=state,
            )

        await sync_data(state=state, config=config, messages=messages, trim=True)
        event.event_type = EventType.INTERRUPTED
        event.metadata["interrupted"] = "Stop"
        event.metadata["status"] = "Graph execution stopped by request"
        event.data["state"] = state.model_dump()
        publish_event(event)
        return True
    return False
