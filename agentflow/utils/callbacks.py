"""
Callback system for TAF.

This module provides a comprehensive callback framework that allows users to define
their own validation logic and custom behavior at key points in the execution flow:

- before_invoke: Called before AI/TOOL/MCP invocation for input validation and modification
- after_invoke: Called after AI/TOOL/MCP invocation for output validation and modification
- on_error: Called when errors occur during invocation for error handling and logging

Graph-level lifecycle hooks (GraphLifecycleHook) fire at structural events of the entire
graph run:
- on_graph_start: Before the execution loop begins
- on_graph_end: After successful graph completion, before final state sync
- on_graph_error: When an unhandled error escapes the execution loop
- on_interrupt: When execution pauses at an interrupt point
- on_resume: When a previously interrupted execution resumes
- on_checkpoint: Before every durable checkpoint write
- on_state_update: After each node result is merged into state

The system is generic and type-safe, supporting different callback types for different
invocation contexts.
"""

import inspect
import logging
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, TypeVar, Union, cast

from agentflow.core.state.message import Message


if TYPE_CHECKING:
    from agentflow.core.state import AgentState


logger = logging.getLogger("agentflow.utils")

StateT = TypeVar("StateT", bound="AgentState")


class BaseValidator(ABC):
    """Abstract base class for message validators.

    Validators are used to validate message content before processing.
    They provide a simpler interface than callbacks, focused specifically
    on message validation.

    Example:
        ```python
        class MyValidator(BaseValidator):
            async def validate(self, messages: list[Message]) -> bool:
                for msg in messages:
                    if "bad_word" in msg.text():
                        raise ValidationError("Bad word detected", "content_policy")
                return True


        # Register with callback manager
        from agentflow.utils.callbacks import CallbackManager

        callback_manager = CallbackManager()
        callback_manager.register_input_validator(MyValidator())
        ```
    """

    @abstractmethod
    async def validate(self, messages: list[Message]) -> bool:
        """Validate a list of messages.

        Args:
            messages: List of Message objects to validate

        Returns:
            True if validation passes

        Raises:
            ValidationError: If validation fails
        """
        ...


class InvocationType(StrEnum):
    """Types of invocations that can trigger callbacks."""

    AI = "ai"
    TOOL = "tool"
    MCP = "mcp"
    INPUT_VALIDATION = "input_validation"
    SKILL = "skill"


@dataclass
class CallbackContext:
    """Context information passed to callbacks."""

    invocation_type: InvocationType
    node_name: str
    function_name: str | None = None
    metadata: dict[str, Any] | None = None


class BeforeInvokeCallback[T, R](ABC):
    """Abstract base class for before_invoke callbacks.

    Called before the AI model, tool, or MCP function is invoked.
    Allows for input validation and modification.
    """

    @abstractmethod
    async def __call__(self, context: CallbackContext, input_data: T) -> T | R:
        """Execute the before_invoke callback.

        Args:
            context: Context information about the invocation
            input_data: The input data about to be sent to the invocation

        Returns:
            Modified input data (can be same type or different type)

        Raises:
            Exception: If validation fails or modification cannot be performed
        """
        ...


class AfterInvokeCallback[T, R](ABC):
    """Abstract base class for after_invoke callbacks.

    Called after the AI model, tool, or MCP function is invoked.
    Allows for output validation and modification.
    """

    @abstractmethod
    async def __call__(self, context: CallbackContext, input_data: T, output_data: Any) -> Any | R:
        """Execute the after_invoke callback.

        Args:
            context: Context information about the invocation
            input_data: The original input data that was sent
            output_data: The output data returned from the invocation

        Returns:
            Modified output data (can be same type or different type)

        Raises:
            Exception: If validation fails or modification cannot be performed
        """
        ...


class OnErrorCallback(ABC):
    """Abstract base class for on_error callbacks.

    Called when an error occurs during invocation.
    Allows for error handling and logging.
    """

    @abstractmethod
    async def __call__(
        self, context: CallbackContext, input_data: Any, error: Exception
    ) -> Any | None:
        """Execute the on_error callback.

        Args:
            context: Context information about the invocation
            input_data: The input data that caused the error
            error: The exception that occurred

        Returns:
            Optional recovery value or None to re-raise the error

        Raises:
            Exception: If error handling fails or if the error should be re-raised
        """
        ...


# Type aliases for cleaner type hints
BeforeInvokeCallbackType = Union[
    BeforeInvokeCallback[Any, Any], Callable[[CallbackContext, Any], Union[Any, Awaitable[Any]]]
]

AfterInvokeCallbackType = Union[
    AfterInvokeCallback[Any, Any], Callable[[CallbackContext, Any, Any], Union[Any, Awaitable[Any]]]
]

OnErrorCallbackType = Union[
    OnErrorCallback,
    Callable[[CallbackContext, Any, Exception], Union[Any | None, Awaitable[Any | None]]],
]


# ─── Graph Lifecycle Hooks ──────────────────────────────────────────────────


@dataclass
class GraphLifecycleContext:
    """Context passed to all graph lifecycle hook methods.

    Carries the full config dict so hooks can access thread_id, run_id, and
    any application-specific keys set on the config.
    """

    config: dict[str, Any]

    @property
    def thread_id(self) -> str:
        """The thread ID for this execution."""
        return self.config.get("thread_id", "")

    @property
    def run_id(self) -> str:
        """The run ID for this execution."""
        return self.config.get("run_id", "")


class GraphLifecycleHook[StateT]:
    """Abstract base class for graph-level lifecycle hooks.

    All methods have default no-op implementations so users only need to
    override the specific hooks they care about.

    Example:
        ```python
        class MyHook(GraphLifecycleHook):
            async def on_graph_start(self, ctx, state):
                print(f"Graph started for thread {ctx.thread_id}")

            async def on_graph_end(self, ctx, final_state, messages, total_steps):
                print(f"Completed in {total_steps} steps")
        ```
    """

    async def on_graph_start(
        self,
        context: "GraphLifecycleContext",
        state: StateT,
    ) -> "StateT | None":
        """Called after state is loaded, before the execution loop starts.

        Return a modified StateT to replace the initial state, or None to keep it.
        """
        return None

    async def on_graph_end(
        self,
        context: "GraphLifecycleContext",
        final_state: StateT,
        messages: list[Message],
        total_steps: int,
    ) -> "StateT | None":
        """Called after successful graph completion, before final state sync.

        Return a modified StateT to persist, or None to keep the current state.
        """
        return None

    async def on_graph_error(
        self,
        context: "GraphLifecycleContext",
        error: Exception,
        partial_state: StateT,
        messages: list[Message],
        step: int,
        node_name: str,
    ) -> "tuple[StateT, str] | None":
        """Called when an unhandled error escapes the execution loop.

        Return (modified_state, error_message) to change the persisted error snapshot,
        or None to keep the current error state. The exception is always re-raised.
        """
        return None

    async def on_interrupt(
        self,
        context: "GraphLifecycleContext",
        interrupted_node: str,
        interrupt_type: str,
        state: StateT,
    ) -> "StateT | None":
        """Called when execution pauses at an interrupt point.

        Return a modified StateT to persist at interrupt, or None to keep the current state.
        In-place mutation of the passed state is also supported.
        """
        return None

    async def on_resume(
        self,
        context: "GraphLifecycleContext",
        resumed_node: str,
        state: StateT,
        resume_data: dict[str, Any],
    ) -> "StateT | None":
        """Called when a previously interrupted execution is about to resume.

        Called before clear_interrupt(). Return a modified StateT to continue with,
        or None to use the loaded state unchanged.
        """
        return None

    async def on_checkpoint(
        self,
        context: "GraphLifecycleContext",
        state: StateT,
        messages: list[Message],
        is_context_trimmed: bool,
    ) -> "tuple[StateT, list[Message]] | StateT | None":
        """Called before every durable checkpoint write.

        Return (state, messages) to modify both, StateT to modify state only,
        or None to persist without modification.
        """
        return None

    async def on_state_update(
        self,
        context: "GraphLifecycleContext",
        node_name: str,
        old_state: StateT,
        new_state: StateT,
        step: int,
    ) -> "StateT | None":
        """Called after each node's result is merged into state.

        Return a modified StateT to replace new_state, or None to use new_state unchanged.
        """
        return None

    async def on_turn_start(
        self,
        context: "GraphLifecycleContext",
        state: StateT,
        turn_index: int,
    ) -> "StateT | None":
        """Called when a new realtime conversation turn begins (realtime sessions only).

        A "turn" spans one model generation, bounded by the provider's turn-complete (or a
        barge-in). ``turn_index`` is 1-based. Return a modified StateT to replace the current
        state, or None to keep it. Never fired by turn-based invoke/stream runs.
        """
        return None

    async def on_turn_end(
        self,
        context: "GraphLifecycleContext",
        state: StateT,
        turn_index: int,
    ) -> "StateT | None":
        """Called when a realtime conversation turn completes or is interrupted.

        ``turn_index`` is the 1-based index of the turn that just ended. Return a modified
        StateT to replace the current state, or None to keep it. Never fired by turn-based
        invoke/stream runs.
        """
        return None


class CallbackManager:
    """
    Manages registration and execution of callbacks for different invocation types.

    Handles before_invoke, after_invoke, and on_error callbacks for AI, TOOL, and MCP invocations.
    """

    def __init__(self):
        """
        Initialize the CallbackManager with empty callback registries.
        """
        self._before_callbacks: dict[InvocationType, list[BeforeInvokeCallbackType]] = {
            InvocationType.AI: [],
            InvocationType.TOOL: [],
            InvocationType.MCP: [],
            InvocationType.INPUT_VALIDATION: [],
            InvocationType.SKILL: [],
        }
        self._after_callbacks: dict[InvocationType, list[AfterInvokeCallbackType]] = {
            InvocationType.AI: [],
            InvocationType.TOOL: [],
            InvocationType.MCP: [],
            InvocationType.INPUT_VALIDATION: [],
            InvocationType.SKILL: [],
        }
        self._error_callbacks: dict[InvocationType, list[OnErrorCallbackType]] = {
            InvocationType.AI: [],
            InvocationType.TOOL: [],
            InvocationType.MCP: [],
            InvocationType.INPUT_VALIDATION: [],
            InvocationType.SKILL: [],
        }
        # Validator registry
        self._validators: list[BaseValidator] = []
        # Graph lifecycle hooks
        self._lifecycle_hooks: list[GraphLifecycleHook] = []

    def register_lifecycle_hook(self, hook: GraphLifecycleHook) -> None:
        """Register a graph lifecycle hook.

        The hook's methods are called at structural events of the graph run
        (start, end, error, interrupt, resume, checkpoint, state update).

        Args:
            hook: A GraphLifecycleHook instance. Only the methods you override are called.

        Example:
            ```python
            class MyHook(GraphLifecycleHook):
                async def on_graph_start(self, ctx, state):
                    print("Graph started")


            callback_mgr = CallbackManager()
            callback_mgr.register_lifecycle_hook(MyHook())
            ```
        """
        self._lifecycle_hooks.append(hook)
        logger.debug("Registered lifecycle hook: %s", hook.__class__.__name__)

    def register_before_invoke(
        self, invocation_type: InvocationType, callback: BeforeInvokeCallbackType
    ) -> None:
        """
        Register a before_invoke callback for a specific invocation type.

        Args:
            invocation_type (InvocationType): The type of invocation (AI, TOOL, MCP).
            callback (BeforeInvokeCallbackType): The callback to register.
        """
        self._before_callbacks[invocation_type].append(callback)

    def register_after_invoke(
        self, invocation_type: InvocationType, callback: AfterInvokeCallbackType
    ) -> None:
        """
        Register an after_invoke callback for a specific invocation type.

        Args:
            invocation_type (InvocationType): The type of invocation (AI, TOOL, MCP).
            callback (AfterInvokeCallbackType): The callback to register.
        """
        self._after_callbacks[invocation_type].append(callback)

    def register_on_error(
        self, invocation_type: InvocationType, callback: OnErrorCallbackType
    ) -> None:
        """
        Register an on_error callback for a specific invocation type.

        Args:
            invocation_type (InvocationType): The type of invocation (AI, TOOL, MCP).
            callback (OnErrorCallbackType): The callback to register.
        """
        self._error_callbacks[invocation_type].append(callback)

    async def execute_before_invoke(self, context: CallbackContext, input_data: Any) -> Any:
        """
        Execute all before_invoke callbacks for the given context.

        Args:
            context (CallbackContext): Context information about the invocation.
            input_data (Any): The input data to be validated or modified.

        Returns:
            Any: The modified input data after all callbacks.

        Raises:
            Exception: If any callback fails.
        """
        current_data = input_data

        for callback in self._before_callbacks[context.invocation_type]:
            try:
                if isinstance(callback, BeforeInvokeCallback):
                    current_data = await callback(context, current_data)
                elif callable(callback):
                    result = callback(context, current_data)
                    if inspect.isawaitable(result):
                        current_data = await result
                    else:
                        current_data = result
            except Exception as e:
                await self.execute_on_error(context, input_data, e)
                raise

        return current_data

    async def execute_after_invoke(
        self,
        context: CallbackContext,
        input_data: Any,
        output_data: Any,
    ) -> Any:
        """
        Execute all after_invoke callbacks for the given context.

        Args:
            context (CallbackContext): Context information about the invocation.
            input_data (Any): The original input data sent to the invocation.
            output_data (Any): The output data returned from the invocation.

        Returns:
            Any: The modified output data after all callbacks.

        Raises:
            Exception: If any callback fails.
        """
        current_output = output_data

        for callback in self._after_callbacks[context.invocation_type]:
            try:
                if isinstance(callback, AfterInvokeCallback):
                    current_output = await callback(context, input_data, current_output)
                elif callable(callback):
                    result = callback(context, input_data, current_output)
                    if inspect.isawaitable(result):
                        current_output = await result
                    else:
                        current_output = result
            except Exception as e:
                await self.execute_on_error(context, input_data, e)
                raise

        return current_output

    async def execute_on_error(
        self, context: CallbackContext, input_data: Any, error: Exception
    ) -> Message | None:
        """
        Execute all on_error callbacks for the given context.

        Args:
            context (CallbackContext): Context information about the invocation.
            input_data (Any): The input data that caused the error.
            error (Exception): The exception that occurred.

        Returns:
            Message | None: Recovery value from callbacks, or None if not handled.
        """
        recovery_value = None

        for callback in self._error_callbacks[context.invocation_type]:
            try:
                result = None
                if isinstance(callback, OnErrorCallback):
                    result = await callback(context, input_data, error)
                elif callable(callback):
                    result = callback(context, input_data, error)
                    if inspect.isawaitable(result):
                        result = await result  # type: ignore

                if result is None or isinstance(result, Message):
                    recovery_value = result
                else:
                    logger.warning(
                        "Error callback %s returned non-Message value %r; ignoring",
                        callback.__class__.__name__,
                        result,
                    )
            except Exception as exc:
                logger.exception("Error callback failed: %s", exc)
                continue

        return recovery_value

    def register_input_validator(self, validator: BaseValidator) -> None:
        """
        Register a message validator for input validation.

        Validators provide a simpler interface for message validation
        compared to callbacks. They only need to implement validate(messages).

        Args:
            validator: BaseValidator instance to register

        Example:
            ```python
            from agentflow.utils.validators import PromptInjectionValidator

            callback_manager = CallbackManager()
            validator = PromptInjectionValidator()
            callback_manager.register_input_validator(validator)
            ```
        """
        self._validators.append(validator)
        logger.debug("Registered input validator: %s", validator.__class__.__name__)

    async def execute_validators(
        self,
        messages: list[Message],
        config: dict[str, Any] | None = None,
    ) -> bool:
        """
        Execute all registered validators on the given messages.

        Args:
            messages: List of Message objects to validate
            config: Optional execution config; when provided, publishes an error event
                    on validation failure so the publisher stream captures the rejection.

        Returns:
            True if all validators pass

        Raises:
            ValidationError: If any validator fails
        """
        if not self._validators:
            logger.debug("No validators registered, skipping validation")
            return True

        logger.debug("Running %d validators on %d messages", len(self._validators), len(messages))

        try:
            for validator in self._validators:
                await validator.validate(messages)
        except Exception as e:
            if config:
                from agentflow.runtime.publisher.events import (
                    ContentType,
                    Event,
                    EventModel,
                    EventType,
                )
                from agentflow.runtime.publisher.publish import publish_event

                publish_event(
                    EventModel.default(
                        config,
                        data={"error": str(e), "rejected_count": len(messages)},
                        event=Event.GRAPH_EXECUTION,
                        event_type=EventType.ERROR,
                        content_type=[ContentType.ERROR],
                        extra={"lifecycle": "validation_rejected"},
                    )
                )
            raise

        logger.debug("All validators passed")
        return True

    def clear_callbacks(self, invocation_type: InvocationType | None = None) -> None:
        """
        Clear callbacks for a specific invocation type or all types.

        Args:
            invocation_type (InvocationType | None): The invocation type to clear, or None for all.
        """
        if invocation_type:
            self._before_callbacks[invocation_type].clear()
            self._after_callbacks[invocation_type].clear()
            self._error_callbacks[invocation_type].clear()
        else:
            for inv_type in InvocationType:
                self._before_callbacks[inv_type].clear()
                self._after_callbacks[inv_type].clear()
                self._error_callbacks[inv_type].clear()

    def get_callback_counts(self) -> dict[str, dict[str, int]]:
        """
        Get count of registered callbacks by type for debugging.

        Returns:
            dict[str, dict[str, int]]: Counts of callbacks for each invocation type.
        """
        return {
            inv_type.value: {
                "before_invoke": len(self._before_callbacks[inv_type]),
                "after_invoke": len(self._after_callbacks[inv_type]),
                "on_error": len(self._error_callbacks[inv_type]),
            }
            for inv_type in InvocationType
        }

    # ── Graph Lifecycle Fire Methods ─────────────────────────────────────────

    async def fire_on_graph_start(
        self,
        context: GraphLifecycleContext,
        state: StateT,
    ) -> StateT:
        """Fire all on_graph_start hooks and return the (potentially modified) state."""
        result = state
        for hook in self._lifecycle_hooks:
            try:
                modified = await hook.on_graph_start(context, result)
                if modified is not None:
                    result = modified
            except Exception as e:
                logger.exception(
                    "Lifecycle hook %s.on_graph_start failed: %s",
                    hook.__class__.__name__,
                    e,
                )

        from agentflow.runtime.publisher.events import (
            ContentType,
            Event,
            EventModel,
            EventType,
        )
        from agentflow.runtime.publisher.publish import publish_event

        publish_event(
            EventModel.default(
                context.config,
                data={},
                event=Event.GRAPH_EXECUTION,
                event_type=EventType.UPDATE,
                content_type=[ContentType.STATE],
                extra={"lifecycle": "graph_start"},
            )
        )
        return result

    async def fire_on_graph_end(
        self,
        context: GraphLifecycleContext,
        final_state: StateT,
        messages: list[Message],
        total_steps: int,
    ) -> StateT:
        """Fire all on_graph_end hooks and return the (potentially modified) state."""
        result = final_state
        for hook in self._lifecycle_hooks:
            try:
                modified = await hook.on_graph_end(context, result, messages, total_steps)
                if modified is not None:
                    result = modified
            except Exception as e:
                logger.exception(
                    "Lifecycle hook %s.on_graph_end failed: %s",
                    hook.__class__.__name__,
                    e,
                )

        from agentflow.runtime.publisher.events import (
            ContentType,
            Event,
            EventModel,
            EventType,
        )
        from agentflow.runtime.publisher.publish import publish_event

        publish_event(
            EventModel.default(
                context.config,
                data={"total_steps": total_steps},
                event=Event.GRAPH_EXECUTION,
                event_type=EventType.UPDATE,
                content_type=[ContentType.STATE],
                extra={"lifecycle": "graph_end", "total_steps": total_steps},
            )
        )
        return result

    async def fire_on_graph_error(
        self,
        context: GraphLifecycleContext,
        error: Exception,
        partial_state: StateT,
        messages: list[Message],
        step: int,
        node_name: str,
    ) -> tuple[StateT, str]:
        """
        Fire all on_graph_error hooks. Returns (state, error_message).
        Error is always re-raised.
        """
        result_state = partial_state
        error_message = str(error)
        for hook in self._lifecycle_hooks:
            try:
                result = await hook.on_graph_error(
                    context, error, result_state, messages, step, node_name
                )
                if result is not None:
                    result_state, error_message = result
            except Exception as e:
                logger.exception(
                    "Lifecycle hook %s.on_graph_error failed: %s",
                    hook.__class__.__name__,
                    e,
                )
        from agentflow.runtime.publisher.events import (
            ContentType,
            Event,
            EventModel,
            EventType,
        )
        from agentflow.runtime.publisher.publish import publish_event

        publish_event(
            EventModel.default(
                context.config,
                data={"error": error_message, "step": step, "node_name": node_name},
                event=Event.GRAPH_EXECUTION,
                event_type=EventType.UPDATE,
                node_name=node_name,
                content_type=[ContentType.ERROR],
                extra={"lifecycle": "graph_error", "step": step},
            )
        )
        return result_state, error_message

    async def fire_on_interrupt(
        self,
        context: GraphLifecycleContext,
        interrupted_node: str,
        interrupt_type: str,
        state: StateT,
    ) -> StateT:
        """Fire all on_interrupt hooks and return the (potentially modified) state."""
        result = state
        for hook in self._lifecycle_hooks:
            try:
                modified = await hook.on_interrupt(
                    context, interrupted_node, interrupt_type, result
                )
                if modified is not None:
                    result = modified
            except Exception as e:
                logger.exception(
                    "Lifecycle hook %s.on_interrupt failed: %s",
                    hook.__class__.__name__,
                    e,
                )
        from agentflow.runtime.publisher.events import (
            ContentType,
            Event,
            EventModel,
            EventType,
        )
        from agentflow.runtime.publisher.publish import publish_event

        publish_event(
            EventModel.default(
                context.config,
                data={"interrupted_node": interrupted_node, "interrupt_type": interrupt_type},
                event=Event.GRAPH_EXECUTION,
                event_type=EventType.INTERRUPTED,
                node_name=interrupted_node,
                content_type=[ContentType.STATE],
                extra={"lifecycle": "graph_interrupt", "interrupt_type": interrupt_type},
            )
        )
        return result

    async def fire_on_resume(
        self,
        context: GraphLifecycleContext,
        resumed_node: str,
        state: StateT,
        resume_data: dict[str, Any],
    ) -> StateT:
        """Fire all on_resume hooks and return the (potentially modified) state."""
        result = state
        for hook in self._lifecycle_hooks:
            try:
                modified = await hook.on_resume(context, resumed_node, result, resume_data)
                if modified is not None:
                    result = modified
            except Exception as e:
                logger.exception(
                    "Lifecycle hook %s.on_resume failed: %s",
                    hook.__class__.__name__,
                    e,
                )
        from agentflow.runtime.publisher.events import (
            ContentType,
            Event,
            EventModel,
            EventType,
        )
        from agentflow.runtime.publisher.publish import publish_event

        publish_event(
            EventModel.default(
                context.config,
                data={"resumed_node": resumed_node},
                event=Event.GRAPH_EXECUTION,
                event_type=EventType.UPDATE,
                content_type=[ContentType.STATE],
                extra={"lifecycle": "resume", "resumed_node": resumed_node},
            )
        )
        return result

    async def fire_on_checkpoint(
        self,
        context: GraphLifecycleContext,
        state: StateT,
        messages: list[Message],
        is_context_trimmed: bool,
    ) -> tuple[StateT, list[Message]]:
        """Fire all on_checkpoint hooks and return (state, messages) for persistence."""
        result_state = state
        result_messages = messages
        for hook in self._lifecycle_hooks:
            try:
                result = await hook.on_checkpoint(
                    context, result_state, result_messages, is_context_trimmed
                )
                if isinstance(result, tuple):
                    result_state, result_messages = cast(tuple[StateT, list[Message]], result)
                elif result is not None:
                    result_state = cast(StateT, result)
            except Exception as e:
                logger.exception(
                    "Lifecycle hook %s.on_checkpoint failed: %s",
                    hook.__class__.__name__,
                    e,
                )
        from agentflow.runtime.publisher.events import (
            ContentType,
            Event,
            EventModel,
            EventType,
        )
        from agentflow.runtime.publisher.publish import publish_event

        publish_event(
            EventModel.default(
                context.config,
                data={},
                event=Event.GRAPH_EXECUTION,
                event_type=EventType.UPDATE,
                content_type=[ContentType.STATE],
                extra={"lifecycle": "checkpoint", "trimmed": is_context_trimmed},
            )
        )
        return result_state, result_messages

    async def fire_on_state_update(
        self,
        context: GraphLifecycleContext,
        node_name: str,
        old_state: StateT,
        new_state: StateT,
        step: int,
    ) -> StateT:
        """Fire all on_state_update hooks and return the (potentially modified) state."""
        result = new_state
        for hook in self._lifecycle_hooks:
            try:
                modified = await hook.on_state_update(context, node_name, old_state, result, step)
                if modified is not None:
                    result = modified
            except Exception as e:
                logger.exception(
                    "Lifecycle hook %s.on_state_update failed: %s",
                    hook.__class__.__name__,
                    e,
                )
        from agentflow.runtime.publisher.events import (
            ContentType,
            Event,
            EventModel,
            EventType,
        )
        from agentflow.runtime.publisher.publish import publish_event

        publish_event(
            EventModel.default(
                context.config,
                data={"step": step},
                event=Event.NODE_EXECUTION,
                event_type=EventType.UPDATE,
                node_name=node_name,
                content_type=[ContentType.STATE],
                extra={"lifecycle": "state_update", "step": step},
            )
        )
        return result

    async def fire_on_turn_start(
        self,
        context: GraphLifecycleContext,
        state: StateT,
        turn_index: int,
    ) -> StateT:
        """Fire all on_turn_start hooks and return the (potentially modified) state."""
        result = state
        for hook in self._lifecycle_hooks:
            try:
                modified = await hook.on_turn_start(context, result, turn_index)
                if modified is not None:
                    result = modified
            except Exception as e:
                logger.exception(
                    "Lifecycle hook %s.on_turn_start failed: %s",
                    hook.__class__.__name__,
                    e,
                )
        return result

    async def fire_on_turn_end(
        self,
        context: GraphLifecycleContext,
        state: StateT,
        turn_index: int,
    ) -> StateT:
        """Fire all on_turn_end hooks and return the (potentially modified) state."""
        result = state
        for hook in self._lifecycle_hooks:
            try:
                modified = await hook.on_turn_end(context, result, turn_index)
                if modified is not None:
                    result = modified
            except Exception as e:
                logger.exception(
                    "Lifecycle hook %s.on_turn_end failed: %s",
                    hook.__class__.__name__,
                    e,
                )
        return result
