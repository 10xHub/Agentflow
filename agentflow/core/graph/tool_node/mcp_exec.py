"""MCPMixin — executes tools via a Model Context Protocol (MCP) client."""

from __future__ import annotations

import json
import logging
import typing as t

from agentflow.core.state import (
    ContentBlock,
    ErrorBlock,
    Message,
    ToolResultBlock,
)
from agentflow.runtime.publisher.events import ContentType, Event, EventModel, EventType
from agentflow.runtime.publisher.publish import publish_event
from agentflow.utils import CallbackContext, CallbackManager, InvocationType

from ._helpers import _normalize_resource_uri


logger = logging.getLogger("agentflow.graph.tool_node")


class MCPMixin:
    _client: t.Any | None
    # The concrete ToolNode defines these
    mcp_tools: list[str]  # type: ignore[assignment]
    _pass_user_info_to_mcp: bool  # type: ignore[assignment]

    def _serialize_result(
        self,
        tool_call_id: str,
        res: t.Any,
    ) -> list[ContentBlock]:
        def safe_serialize(obj: t.Any) -> dict[str, t.Any]:
            try:
                json.dumps(obj)
                return obj if isinstance(obj, dict) else {"content": obj}
            except (TypeError, OverflowError):
                if hasattr(obj, "model_dump"):
                    dumped = obj.model_dump()  # type: ignore
                    return _normalize_resource_uri(dumped)
                return {"content": str(obj), "type": "fallback"}

        for source in [
            getattr(res, "content", None),
            getattr(res, "structured_content", None),
            getattr(res, "data", None),
        ]:
            if source is None:
                continue
            try:
                if isinstance(source, list):
                    result = [safe_serialize(item) for item in source]
                else:
                    result = [safe_serialize(source)]

                return [
                    ToolResultBlock(
                        call_id=tool_call_id,
                        output=result,
                        is_error=False,
                        status="completed",
                    )
                ]
            except Exception as e:
                logger.exception("Serialization failure: %s", e)
                continue

        return [
            ToolResultBlock(
                call_id=tool_call_id,
                output=[
                    {
                        "content": str(res),
                        "type": "fallback",
                    }
                ],
                is_error=False,
                status="completed",
            )
        ]

    async def _get_mcp_tool(self, tags: set[str] | None = None) -> list[dict]:
        """Fetch tools from the MCP client, optionally filtering by tags."""
        tools: list[dict] = []
        if not self._client:
            return tools

        try:
            async with self._client:
                res = await self._client.ping()
                if not res:
                    return tools
                mcp_tools: list[t.Any] = await self._client.list_tools()
                for i in mcp_tools:
                    if tags:
                        meta = i.meta or {}
                        tool_tags = set(meta.get("_fastmcp", {}).get("tags", []))
                        if not tool_tags.intersection(tags):
                            continue
                    self.mcp_tools.append(i.name)  # type: ignore[attr-defined]
                    tools.append(
                        {
                            "type": "function",
                            "function": {
                                "name": i.name,
                                "description": i.description,
                                "parameters": i.inputSchema,
                            },
                        }
                    )
        except Exception as e:
            logger.exception("Failed to fetch MCP tools: %s", e)

        return tools

    async def _mcp_execute(  # noqa: PLR0915
        self,
        name: str,
        args: dict,
        tool_call_id: str,
        config: dict[str, t.Any],
        callback_mgr: CallbackManager,
    ) -> Message:
        context = CallbackContext(
            invocation_type=InvocationType.MCP,
            node_name="ToolNode",
            function_name=name,
            metadata={
                "tool_call_id": tool_call_id,
                "args": args,
                "config": config,
                "mcp_client": bool(self._client),
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
                "is_mcp": True,
            },
            content_type=[ContentType.TOOL_CALL],
            event=Event.TOOL_EXECUTION,
        )
        event.event_type = EventType.PROGRESS
        event.node_name = "ToolNode"
        publish_event(event)

        input_data = {**args}

        try:
            input_data = await callback_mgr.execute_before_invoke(context, input_data)
            event.event_type = EventType.UPDATE
            event.metadata["status"] = "before_invoke_complete Invoke MCP"
            publish_event(event)

            if not self._client:
                error_result = Message.tool_message(
                    content=[
                        ErrorBlock(
                            message="No MCP client configured",
                        ),
                        ToolResultBlock(
                            call_id=tool_call_id,
                            output="No MCP client configured",
                            is_error=True,
                            status="failed",
                        ),
                    ],
                    meta=meta,
                )
                res = await callback_mgr.execute_after_invoke(context, input_data, error_result)
                event.event_type = EventType.ERROR
                event.metadata["error"] = "No MCP client configured"
                publish_event(event)
                return res

            async with self._client:
                if not await self._client.ping():
                    error_result = Message.tool_message(
                        content=[
                            ErrorBlock(message="MCP Server not available. Ping failed."),
                            ToolResultBlock(
                                call_id=tool_call_id,
                                output="MCP Server not available. Ping failed.",
                                is_error=True,
                                status="failed",
                            ),
                        ],
                        meta=meta,
                    )
                    event.event_type = EventType.ERROR
                    event.metadata["error"] = "MCP server not available, ping failed"
                    publish_event(event)
                    return await callback_mgr.execute_after_invoke(
                        context, input_data, error_result
                    )

                if self._pass_user_info_to_mcp:
                    mcp_user_info = config.get("user")
                    if mcp_user_info and isinstance(mcp_user_info, dict):
                        input_data["user"] = mcp_user_info
                    else:
                        user_id = config.get("user_id")
                        if user_id:
                            input_data["user"] = {"user_id": user_id}

                res: t.Any = await self._client.call_tool(name, input_data)

                final_res = self._serialize_result(tool_call_id, res)

                result = Message.tool_message(
                    content=final_res,
                    meta=meta,
                )

                res = await callback_mgr.execute_after_invoke(context, input_data, result)
                event.event_type = EventType.END
                event.data["message"] = result.model_dump()
                event.metadata["status"] = "MCP tool execution complete"
                event.content_type = [ContentType.TOOL_RESULT, ContentType.MESSAGE]
                publish_event(event)
                return res

        except Exception as e:
            recovery_result = await callback_mgr.execute_on_error(context, input_data, e)

            if isinstance(recovery_result, Message):
                event.event_type = EventType.END
                event.data["message"] = recovery_result.model_dump()
                event.metadata["status"] = "MCP tool execution complete, with recovery"
                event.content_type = [ContentType.TOOL_RESULT, ContentType.MESSAGE]
                publish_event(event)
                return recovery_result

            event.event_type = EventType.END
            event.data["error"] = str(e)
            event.metadata["status"] = "MCP tool execution complete, with recovery"
            event.content_type = [ContentType.TOOL_RESULT, ContentType.ERROR]
            publish_event(event)

            return Message.tool_message(
                content=[
                    ToolResultBlock(
                        call_id=tool_call_id,
                        output=f"MCP execution error: {e}",
                        is_error=True,
                        status="failed",
                    ),
                    ErrorBlock(message=f"MCP execution error: {e}"),
                ],
                meta=meta,
            )
