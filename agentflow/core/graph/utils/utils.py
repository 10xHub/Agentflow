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

import copy
import logging
from collections.abc import Callable
from typing import Any, TypeVar

from injectq import Inject

from agentflow.core.exceptions import GraphError
from agentflow.core.state import AgentState, ExecutionStatus, Message
from agentflow.core.state.base_context import BaseContextManager
from agentflow.core.state.execution_state import ExecutionState as ExecMeta
from agentflow.core.state.execution_state import StopRequestStatus
from agentflow.runtime.adapters.llm.model_response_converter import ModelResponseConverter
from agentflow.storage.checkpointer import BaseCheckpointer
from agentflow.utils import (
    END,
    START,
    Command,
    ResponseGranularity,
    add_messages,
)
from agentflow.utils.callbacks import CallbackManager, GraphLifecycleContext


StateT = TypeVar("StateT", bound=AgentState)

logger = logging.getLogger("agentflow.graph")


async def parse_response(
    state: AgentState,
    messages: list[Message],
    response_granularity: ResponseGranularity = ResponseGranularity.LOW,
    token_usage: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Parse and format execution response based on specified granularity level.

    Formats the final response from graph execution according to the requested
    granularity level, allowing clients to receive different levels of detail
    depending on their needs.

    Args:
        state: The final agent state after graph execution.
        messages: List of messages generated during execution.
        response_granularity: Level of detail to include in the response:
            - FULL: Returns complete state object and all messages
            - PARTIAL: Returns context, summary, and messages
            - LOW: Returns only the messages (default)

    Returns:
        Dictionary containing the formatted response with keys depending on
        granularity level. Always includes 'messages' key with execution results.

    Example:
        ```python
        # LOW granularity (default)
        response = await parse_response(state, messages)
        # Returns: {"messages": [Message(...), ...]}

        # FULL granularity
        response = await parse_response(state, messages, ResponseGranularity.FULL)
        # Returns: {"state": AgentState(...), "messages": [Message(...), ...]}
        ```
    """
    match response_granularity:
        case ResponseGranularity.FULL:
            # Return full state and messages
            return {"state": state, "messages": messages, "token_usage": token_usage}
        case ResponseGranularity.PARTIAL:
            # Return state and summary of messages
            return {
                "context": state.context,
                "summary": state.context_summary,
                "messages": messages,
                "token_usage": token_usage,
            }
        case ResponseGranularity.LOW:
            # Return all messages from state context
            return {"messages": messages, "token_usage": token_usage}

    return {"messages": messages, "token_usage": token_usage}


# Utility to update only provided fields in state
def _update_state_fields(state, partial: dict):
    """Update only the provided fields in the state object."""
    for k, v in partial.items():
        # Avoid updating special fields
        if k in ("context", "context_summary", "execution_meta"):
            continue
        if hasattr(state, k):
            setattr(state, k, v)


async def validate_message_content(
    message: list[Message],
    config: dict[str, Any] | None = None,
    callback_mgr: CallbackManager = Inject[CallbackManager],  # will be auto-injected
) -> bool:
    """Validate message content using registered validators in callback manager.

    Executes all validators registered in the callback manager.
    This allows for flexible, configurable validation strategies.

    Args:
        message: List of Message objects to validate.
        config: Execution config passed through to validators for event publishing.
        callback_manager: CallbackManager instance (required).

    Returns:
        True if all validators pass.

    Raises:
        ValidationError: If any validator fails.
    """
    # Require callback manager to be provided
    if not callback_mgr:
        logger.debug("No callback manager provided, skipping validation")
        return True

    # Execute validation
    await callback_mgr.execute_validators(message, config=config)

    logger.debug("All validators passed")
    return True


async def load_or_create_state[StateT: AgentState](  # noqa: PLR0912, PLR0915
    input_data: dict[str, Any],
    config: dict[str, Any],
    old_state: StateT,
    checkpointer: BaseCheckpointer = Inject[BaseCheckpointer],  # will be auto-injected
) -> StateT:
    """Load existing state from checkpointer or create new state.

    Attempts to fetch a realtime-synced state first, then falls back to
    the persistent checkpointer. If no existing state is found, creates
    a new state from the `StateGraph`'s prototype state and merges any
    incoming messages. Supports partial state update via 'state' in input_data.
    """
    logger.debug("Loading or creating state with thread_id=%s", config.get("thread_id", "default"))

    # Try to load existing state if checkpointer is available
    if checkpointer:
        # A stop request that was never consumed (the previous run finished or
        # died before reaching its next stop check) would otherwise be waiting to
        # kill this run at its first node. A new run supersedes it.
        await checkpointer.aclear_stop_request(config)

        logger.debug("Attempting to load existing state from checkpointer")
        # first check realtime-synced state
        existing_state: StateT | None = await checkpointer.aget_state_cache(config)
        if not existing_state:
            logger.debug("No synced state found, trying persistent checkpointer")
            # If no synced state, try to get from persistent checkpointer
            existing_state = await checkpointer.aget_state(config)

        if existing_state:
            logger.info(
                "Loaded existing state with %d context messages, current_node=%s, step=%d",
                len(existing_state.context) if existing_state.context else 0,
                existing_state.execution_meta.current_node,
                existing_state.execution_meta.step,
            )
            # A stop that interrupted an earlier run is persisted in the state and
            # is never cleared by set_interrupt, so a thread stopped mid-run would
            # carry STOP_REQUESTED into its next run and stop itself again at the
            # first node. Starting a run supersedes any prior stop request.
            existing_state.execution_meta.stop_current_execution = StopRequestStatus.NONE

            # Normalize legacy node names (backward compatibility)
            # Some older runs may have persisted 'start'/'end' instead of '__start__'/'__end__'
            if existing_state.execution_meta.current_node == "start":
                existing_state.execution_meta.current_node = START
                logger.debug("Normalized legacy current_node 'start' to '%s'", START)
            elif existing_state.execution_meta.current_node == "end":
                existing_state.execution_meta.current_node = END
                logger.debug("Normalized legacy current_node 'end' to '%s'", END)
            elif existing_state.execution_meta.current_node == "__start__":
                existing_state.execution_meta.current_node = START
                logger.debug("Normalized legacy current_node '__start__' to '%s'", START)
            elif existing_state.execution_meta.current_node == "__end__":
                existing_state.execution_meta.current_node = END
                logger.debug("Normalized legacy current_node '__end__' to '%s'", END)

            # Reset execution if graph previously completed (current_node == END)
            # This allows multi-turn conversations to continue from a completed state
            if existing_state.execution_meta.current_node == END:
                logger.info(
                    "Resetting completed state for multi-turn continuation (was at END, step=%d)",
                    existing_state.execution_meta.step,
                )
                existing_state.execution_meta.current_node = START
                existing_state.execution_meta.step = 0
                existing_state.execution_meta.status = ExecutionStatus.RUNNING
                existing_state.execution_meta.clear_interrupt()
                existing_state.execution_meta.stop_current_execution = StopRequestStatus.NONE
            # Merge new messages with existing context
            new_messages = input_data.get("messages", [])
            if new_messages:
                logger.debug("Merging %d new messages with existing context", len(new_messages))
                # Validate message content before adding
                await validate_message_content(new_messages, config=config)
                pre_merge_len = len(existing_state.context)
                existing_state.context = add_messages(existing_state.context, new_messages)

                # Fallback: when all input messages were filtered out by ID deduplication
                # (e.g. frontend sends full history with placeholder id="0") and the last
                # input message is a user message while the context ends with an assistant
                # message, forcibly add the last user message so the conversation continues.
                if len(existing_state.context) == pre_merge_len:
                    last_input = new_messages[-1]
                    last_ctx = existing_state.context[-1] if existing_state.context else None
                    if last_input.role == "user" and (last_ctx is None or last_ctx.role != "user"):
                        from agentflow.core.state.message import generate_id

                        new_msg = last_input.model_copy(update={"message_id": generate_id(None)})
                        existing_state.context = [*existing_state.context, new_msg]
                        logger.info(
                            "Added last user message with new ID to continue conversation "
                            "(fallback for duplicate message_id in input)"
                        )

            # Merge partial state fields if provided
            partial_state = input_data.get("state", {})
            if partial_state and isinstance(partial_state, dict):
                logger.debug("Merging partial state with %d fields", len(partial_state))
                _update_state_fields(existing_state, partial_state)
            # Update current node if available
            if "current_node" in partial_state and partial_state["current_node"] is not None:
                existing_state.set_current_node(partial_state["current_node"])
            return existing_state
    else:
        logger.debug("No checkpointer available, will create new state")

    # Create new state by deep copying the graph's prototype state
    logger.info("Creating new state from graph prototype")
    state = copy.deepcopy(old_state)

    # Ensure core AgentState fields are properly initialized
    if hasattr(state, "context") and not isinstance(state.context, list):
        state.context = []
        logger.debug("Initialized empty context list")
    if hasattr(state, "context_summary") and state.context_summary is None:
        state.context_summary = None
        logger.debug("Initialized context_summary as None")
    if hasattr(state, "execution_meta"):
        # Create a fresh execution metadata
        state.execution_meta = ExecMeta(current_node=START)
        logger.debug("Created fresh execution metadata starting at %s", START)

    # Set thread_id in execution metadata
    thread_id = config.get("thread_id", "default")
    state.execution_meta.thread_id = thread_id
    logger.debug("Set thread_id to %s", thread_id)

    # Merge new messages with context
    new_messages = input_data.get("messages", [])
    if new_messages:
        logger.debug("Adding %d new messages to fresh state", len(new_messages))
        # Validate message content before adding
        await validate_message_content(new_messages, config=config)
        state.context = add_messages(state.context, new_messages)
    # Merge partial state fields if provided
    partial_state = input_data.get("state", {})
    if partial_state and isinstance(partial_state, dict):
        logger.debug("Merging partial state with %d fields", len(partial_state))
        _update_state_fields(state, partial_state)

    logger.info(
        "Created new state with %d context messages", len(state.context) if state.context else 0
    )
    if "current_node" in partial_state and partial_state["current_node"] is not None:
        # Normalize legacy values if provided in partial state
        next_node = partial_state["current_node"]
        if next_node == "__start__":
            next_node = START
        elif next_node == "__end__":
            next_node = END
        state.set_current_node(next_node)
    return state  # type: ignore[return-value]


async def reload_state[StateT: AgentState](
    config: dict[str, Any],
    old_state: StateT,
    checkpointer: BaseCheckpointer = Inject[BaseCheckpointer],  # will be auto-injected
) -> StateT:
    """Load existing state from checkpointer or create new state.

    Attempts to fetch a realtime-synced state first, then falls back to
    the persistent checkpointer. If no existing state is found, creates
    a new state from the `StateGraph`'s prototype state and merges any
    incoming messages. Supports partial state update via 'state' in input_data.
    """
    logger.debug("Loading or creating state with thread_id=%s", config.get("thread_id", "default"))

    if not checkpointer:
        return old_state

    # first check realtime-synced state
    existing_state: AgentState | None = await checkpointer.aget_state_cache(config)
    if not existing_state:
        logger.debug("No synced state found, trying persistent checkpointer")
        # If no synced state, try to get from persistent checkpointer
        existing_state = await checkpointer.aget_state(config)

    if not existing_state:
        logger.warning("No existing state found to reload, returning old state")
        return old_state

    logger.info(
        "Loaded existing state with %d context messages, current_node=%s, step=%d",
        len(existing_state.context) if existing_state.context else 0,
        existing_state.execution_meta.current_node,
        existing_state.execution_meta.step,
    )
    # Normalize legacy node names (backward compatibility)
    # Some older runs may have persisted 'start'/'end' instead of '__start__'/'__end__'
    if existing_state.execution_meta.current_node == "start":
        existing_state.execution_meta.current_node = START
        logger.debug("Normalized legacy current_node 'start' to '%s'", START)
    elif existing_state.execution_meta.current_node == "end":
        existing_state.execution_meta.current_node = END
        logger.debug("Normalized legacy current_node 'end' to '%s'", END)
    elif existing_state.execution_meta.current_node == "__start__":
        existing_state.execution_meta.current_node = START
        logger.debug("Normalized legacy current_node '__start__' to '%s'", START)
    elif existing_state.execution_meta.current_node == "__end__":
        existing_state.execution_meta.current_node = END
        logger.debug("Normalized legacy current_node '__end__' to '%s'", END)
    return existing_state


async def process_node_result[StateT: AgentState](  # noqa: PLR0915
    result: Any,
    state: StateT,
    messages: list[Message],
) -> tuple[StateT, list[Message], str | None]:
    """
    Processes the result from a node execution, updating the agent state, message list,
    and determining the next node.

    Supports:
    - Handling results of type Command, AgentState, Message, list, str, dict,
            or other types.
        - Deduplicating messages by message_id.
        - Updating the agent state and its context with new messages.
        - Extracting navigation information (next node) from Command results.

    Args:
        result (Any): The output from a node execution. Can be a Command, AgentState, Message,
            list, str, dict, ModelResponse, or other types.
        state (StateT): The current agent state.
        messages (list[Message]): The list of messages accumulated so far.

    Returns:
        tuple[StateT, list[Message], str | None]:
            - The updated agent state.
            - The updated list of messages (with new, unique messages added).
            - The identifier of the next node to execute, if specified; otherwise, None.
    """
    next_node = None
    existing_ids = {msg.message_id for msg in messages}
    new_messages = []

    def add_unique_message(msg: Message) -> None:
        """Add message only if it doesn't already exist."""
        if msg.message_id not in existing_ids:
            new_messages.append(msg)
            existing_ids.add(msg.message_id)

    async def create_and_add_message(content: Any) -> Message:
        """Create message from content and add if unique."""
        if isinstance(content, Message):
            msg = content
        elif isinstance(content, ModelResponseConverter):
            msg = await content.invoke()
        elif isinstance(content, str):
            msg = Message.text_message(
                content,
                role="assistant",
            )

        else:
            err = f"""
            Unsupported content type for message: {type(content)}.
            Supported types are: AgentState, Message, ModelResponseConverter, Command, str,
            dict (OpenAI style/Native Message).
            """
            raise ValueError(err)

        add_unique_message(msg)
        return msg

    def handle_state_message(old_state: StateT, new_state: StateT) -> None:
        """Handle state messages by updating the context."""
        old_messages = {}
        if old_state.context:
            old_messages = {msg.message_id: msg for msg in old_state.context}

        if not new_state.context:
            return
        # now save all the new messages
        for msg in new_state.context:
            if msg.message_id in old_messages:
                continue
            # otherwise save it
            add_unique_message(msg)

    # Process different result types
    if isinstance(result, Command):
        # Handle state updates
        if result.update:
            if isinstance(result.update, AgentState):
                handle_state_message(state, result.update)  # type: ignore[assignment]
                state = result.update  # type: ignore[assignment]
            elif isinstance(result.update, list):
                for item in result.update:
                    await create_and_add_message(item)
            else:
                await create_and_add_message(result.update)

        # Handle navigation
        next_node = result.goto

    elif isinstance(result, AgentState):
        handle_state_message(state, result)  # type: ignore[assignment]
        state = result  # type: ignore[assignment]

    elif isinstance(result, Message):
        add_unique_message(result)

    elif isinstance(result, list):
        # Handle list of items (convert each to message)
        for item in result:
            await create_and_add_message(item)
    else:
        # Handle single items (str, dict, model_dump-capable, or other)
        await create_and_add_message(result)

    # Add new messages to the main list and state context
    if new_messages:
        messages.extend(new_messages)
        state.context = add_messages(state.context, new_messages)

    return state, messages, next_node


async def check_and_handle_interrupt(
    interrupt_before: list[str],
    interrupt_after: list[str],
    current_node: str,
    interrupt_type: str,
    state: AgentState,
    config: dict[str, Any],
    _sync_data: Callable,
) -> bool:
    """Check for interrupts and save state if needed. Returns True if interrupted."""
    interrupt_nodes = interrupt_before if interrupt_type == "before" else interrupt_after

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
        # Save state and interrupt
        await _sync_data(state, config, [])
        logger.debug("Node '%s' interrupted", current_node)
        return True

    logger.debug(
        "No interrupts found for node '%s', continuing execution",
        current_node,
    )
    return False


def get_next_node(
    current_node: str,
    state: AgentState,
    edges: list,
) -> str:
    """Get the next node to execute based on edges."""
    # Find outgoing edges from current node
    outgoing_edges = [e for e in edges if e.from_node == current_node]

    if not outgoing_edges:
        logger.debug("No outgoing edges from node '%s', ending execution", current_node)
        return END

    # Handle conditional edges
    for edge in outgoing_edges:
        if edge.condition:
            # A condition that raises used to be swallowed with `continue`, so the
            # graph quietly fell through to the first static edge -- or to END --
            # and the run took a path nobody chose. Silent misrouting is worse than
            # a failed run: the caller gets a confident, wrong answer. Routing is a
            # correctness decision, so a broken condition fails the run instead.
            try:
                condition_result = edge.condition(state)
            except Exception as e:
                logger.exception(
                    "Condition for edge %s -> %s raised; cannot route",
                    edge.from_node,
                    edge.to_node,
                )
                raise GraphError(
                    message=(
                        f"Conditional edge from '{edge.from_node}' raised while routing: {e}"
                    ),
                    error_code="GRAPH_ROUTING_001",
                    context={
                        "from_node": edge.from_node,
                        "to_node": edge.to_node,
                        "error_type": type(e).__name__,
                    },
                ) from e

            if hasattr(edge, "condition_result") and edge.condition_result is not None:
                # Mapped conditional edge
                if condition_result == edge.condition_result:
                    return edge.to_node
            elif isinstance(condition_result, str):
                return condition_result
            elif condition_result:
                return edge.to_node

    # Return first static edge if no conditions matched
    static_edges = [e for e in outgoing_edges if not e.condition]
    if static_edges:
        return static_edges[0].to_node

    logger.debug("No valid edges found from node '%s', ending execution", current_node)
    return END


async def call_realtime_sync(
    state: AgentState,
    config: dict[str, Any],
    checkpointer: BaseCheckpointer = Inject[BaseCheckpointer],  # will be auto-injected
) -> None:
    """Call the realtime state sync hook if provided."""
    if checkpointer:
        logger.debug("Calling realtime state sync hook")
        # await call_sync_or_async(checkpointer.a, config, state)
        await checkpointer.aput_state_cache(config, state)


async def sync_data(
    state: AgentState,
    config: dict[str, Any],
    messages: list[Message],
    trim: bool = False,
    checkpointer: BaseCheckpointer = Inject[BaseCheckpointer],  # will be auto-injected
    context_manager: BaseContextManager = Inject[BaseContextManager],  # will be auto-injected
    callback_mgr: CallbackManager = Inject[CallbackManager],  # will be auto-injected
) -> bool:
    """Sync the current state and messages to the checkpointer."""
    is_context_trimmed = False

    new_state = copy.deepcopy(state)
    # if context manager is available then utilize it
    if context_manager and trim:
        new_state = await context_manager.atrim_context(state)
        is_context_trimmed = True

    # Fire on_checkpoint hook before persisting
    if callback_mgr and callback_mgr._lifecycle_hooks:
        lifecycle_context = GraphLifecycleContext(config=config)
        new_state, messages = await callback_mgr.fire_on_checkpoint(
            lifecycle_context,
            state=new_state,
            messages=messages,
            is_context_trimmed=is_context_trimmed,
        )

    if checkpointer:
        logger.debug("Persisting state and %d messages to checkpointer", len(messages))
        # Durable store is the source of truth: write state + messages atomically
        # FIRST (audit M7), so a failure surfaces before the cache is updated and
        # the cache can never hold state that was never persisted (audit B5).
        await checkpointer.aput_checkpoint(config, new_state, messages)
        # Then refresh the realtime cache with the SAME trimmed state we just
        # persisted, so cache and durable store never diverge in shape (audit M6).
        await checkpointer.aput_state_cache(config, new_state)

    return is_context_trimmed


def calculate_token_usage(messages: list[Message]) -> dict[str, int]:
    """Calculate total token usage from all messages in the state.

    Aggregates token usage across all messages in the state's context,
    including input tokens (prompt_tokens), output tokens (completion_tokens),
    and reasoning tokens.

    Args:
        messages: The list of messages containing token usage information.

    Returns:
        Dictionary containing:
            - total_input_tokens: Total prompt/input tokens used
            - total_output_tokens: Total completion/output tokens used
            - total_reasoning_tokens: Total reasoning tokens used
            - total_tokens: Sum of input and output tokens

    Example:
        ```python
        usage = calculate_token_usage(state)
        # Returns: {
        #     "total_input_tokens": 1500,
        #     "total_output_tokens": 800,
        #     "total_reasoning_tokens": 200,
        #     "total_tokens": 2300
        # }
        ```
    """
    total_input_tokens = 0
    total_output_tokens = 0
    total_reasoning_tokens = 0

    if not messages:
        return {
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_reasoning_tokens": total_reasoning_tokens,
            "total_tokens": total_input_tokens + total_output_tokens,
        }

    for message in messages:
        if message.usages:
            total_input_tokens += message.usages.prompt_tokens
            total_output_tokens += message.usages.completion_tokens
            total_reasoning_tokens += message.usages.reasoning_tokens

    # Note: total_tokens is input + output only (reasoning tracked separately)
    total_tokens = total_input_tokens + total_output_tokens

    return {
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_reasoning_tokens": total_reasoning_tokens,
        "total_tokens": total_tokens,
    }
