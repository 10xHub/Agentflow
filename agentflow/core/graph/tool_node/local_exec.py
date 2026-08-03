"""LocalExecMixin — executes locally registered tool functions."""

from __future__ import annotations

import inspect
import logging
import typing as t

from agentflow.core.state import (
    AgentState,
    ContentBlock,
    ErrorBlock,
    Message,
    ToolResult,
    ToolResultBlock,
)
from agentflow.runtime.publisher.events import ContentType, Event, EventModel, EventType
from agentflow.runtime.publisher.publish import publish_event
from agentflow.utils import CallbackContext, CallbackManager, InvocationType, call_sync_or_async

from ._helpers import _extract_block_meta, _safe_serialize
from .coercion import coerce_tool_argument
from .constants import INJECTABLE_PARAMS, has_injected_default
from .schema import _safe_type_hints


if t.TYPE_CHECKING:
    from agentflow.core.state.stream_emitter import StreamEmitter

logger = logging.getLogger("agentflow.graph.tool_node")


class LocalExecMixin:
    _funcs: dict[str, t.Callable]

    def _prepare_input_data_tool(
        self,
        fn: t.Callable,
        name: str,
        args: dict,
        default_data: dict,
    ) -> dict:
        sig = inspect.signature(fn)
        hints = _safe_type_hints(fn)
        input_data = {}
        for param_name, param in sig.parameters.items():
            if param.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue

            if param_name in default_data:
                input_data[param_name] = default_data[param_name]
                continue

            if param_name in INJECTABLE_PARAMS:
                continue

            if has_injected_default(param):
                logger.debug(
                    "Skipping injectable parameter '%s' with Inject syntax",
                    param_name,
                )
                continue

            if param_name in args:
                # The provider returns plain JSON, so a model/dataclass/enum parameter
                # arrives as a dict or str and must be validated into its real type.
                input_data[param_name] = coerce_tool_argument(
                    args[param_name],
                    hints.get(param_name, param.annotation),
                    tool_name=name,
                    param_name=param_name,
                )
            elif param.default is inspect.Parameter.empty:
                raise TypeError(f"Missing required parameter '{param_name}' for function '{name}'")

        return input_data

    def _publish_internal_completion(
        self,
        event: EventModel,
        message: Message,
        status: str,
    ) -> None:
        event.event_type = EventType.END
        event.data["message"] = message.model_dump()
        event.metadata["status"] = status
        event.content_type = [ContentType.TOOL_RESULT, ContentType.MESSAGE]
        publish_event(event)

    def _build_internal_result_blocks(
        self,
        result: t.Any,
        tool_call_id: str,
    ) -> list[ContentBlock]:
        is_error = False

        if isinstance(result, str):
            output: str | list[dict[str, t.Any]] = result
        elif isinstance(result, dict):
            is_error, cleaned = _extract_block_meta(result)
            output = [_safe_serialize(cleaned)]
        elif hasattr(result, "model_dump"):
            dumped = result.model_dump()  # type: ignore
            if isinstance(dumped, dict):
                is_error, cleaned = _extract_block_meta(dumped)
                output = [_safe_serialize(cleaned)]
            else:
                output = [_safe_serialize(dumped)]
        elif hasattr(result, "__dict__"):
            is_error, cleaned = _extract_block_meta(result.__dict__)
            output = [_safe_serialize(cleaned)]
        elif isinstance(result, list):
            output = [_safe_serialize(item) for item in result]
        else:
            output = str(result)

        return [
            ToolResultBlock(
                call_id=tool_call_id,
                output=output,
                status="failed" if is_error else "completed",
                is_error=is_error,
            )
        ]

    def _message_from_internal_result(
        self,
        result: t.Any,
        state: AgentState,
        tool_call_id: str,
        meta: dict[str, t.Any],
    ) -> dict[str, t.Any] | Message:
        if isinstance(result, ToolResult):
            if result.state:
                for key, value in result.state.items():
                    if hasattr(state, key):
                        setattr(state, key, value)

            return {
                "state": state,
                "messages": Message.tool_message(
                    content=[
                        ToolResultBlock(
                            call_id=tool_call_id,
                            output=result.message,
                            status="failed" if result.is_error else "completed",
                            is_error=result.is_error,
                        )
                    ],
                    meta=meta,
                ),
            }

        if isinstance(result, Message):
            result.metadata = {**meta, **(result.metadata or {})}
            return result

        return Message.tool_message(
            content=self._build_internal_result_blocks(result, tool_call_id),
            meta=meta,
        )

    async def _internal_execute(
        self,
        name: str,
        args: dict,
        tool_call_id: str,
        config: dict[str, t.Any],
        state: AgentState,
        callback_mgr: CallbackManager,
        emit: StreamEmitter | None = None,
    ) -> dict[str, t.Any] | Message:
        context = CallbackContext(
            invocation_type=InvocationType.TOOL,
            node_name="ToolNode",
            function_name=name,
            metadata={
                "tool_call_id": tool_call_id,
                "args": args,
                "config": config,
            },
        )

        fn = self._funcs[name]
        input_data = self._prepare_input_data_tool(
            fn,
            name,
            args,
            {
                "tool_call_id": tool_call_id,
                "state": state,
                "config": config,
                "emit": emit,
            },
        )

        meta = {
            "function_name": name,
            "function_argument": args,
            "tool_call_id": tool_call_id,
        }

        event = EventModel.default(
            base_config=config,
            data={
                "tool_call_id": tool_call_id,
                "args": args,
                "function_name": name,
                "is_mcp": False,
            },
            content_type=[ContentType.TOOL_CALL],
            event=Event.TOOL_EXECUTION,
        )
        event.event_type = EventType.PROGRESS
        event.node_name = "ToolNode"
        publish_event(event)

        try:
            input_data = await callback_mgr.execute_before_invoke(context, input_data)

            event.event_type = EventType.UPDATE
            event.metadata["status"] = "before_invoke_complete Invoke internal"
            publish_event(event)

            result = await call_sync_or_async(fn, **input_data)

            result = await callback_mgr.execute_after_invoke(
                context,
                input_data,
                result,
            )

            msg = self._message_from_internal_result(result, state, tool_call_id, meta)
            if isinstance(msg, Message):
                self._publish_internal_completion(event, msg, "Internal tool execution complete")
            elif isinstance(msg, dict):
                self._publish_internal_completion(
                    event,
                    Message.tool_message(
                        content=self._build_internal_result_blocks(
                            msg.get("messages"), tool_call_id
                        ),
                        meta=meta,
                    ),
                    "Internal tool execution complete",
                )
            return msg

        except Exception as e:
            recovery_result = await callback_mgr.execute_on_error(context, input_data, e)

            if isinstance(recovery_result, Message):
                event.event_type = EventType.END
                event.data["message"] = recovery_result.model_dump()
                event.metadata["status"] = "Internal tool execution complete, with recovery"
                event.content_type = [ContentType.TOOL_RESULT, ContentType.MESSAGE]
                publish_event(event)
                return recovery_result

            event.event_type = EventType.END
            event.data["error"] = str(e)
            event.metadata["status"] = "Internal tool execution complete, with error"
            event.content_type = [ContentType.TOOL_RESULT, ContentType.ERROR]
            publish_event(event)

            return Message.tool_message(
                content=[
                    ToolResultBlock(
                        call_id=tool_call_id,
                        output=f"Internal execution error: {e}",
                        status="failed",
                        is_error=True,
                    ),
                    ErrorBlock(message=f"Internal execution error: {e}"),
                ],
                meta=meta,
            )
