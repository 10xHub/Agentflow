# """
# AgentFlowExecutor — the sole bridge between agentflow and the a2a-sdk.

# This module implements the ``AgentExecutor`` interface from the official
# ``a2a-sdk`` so that **any** agentflow ``CompiledGraph`` can be served as
# a standard A2A agent.  The SDK handles all HTTP, JSON-RPC, SSE, and task
# lifecycle concerns.  Agentflow handles all agent logic.

# Blocking path (default):
#     Uses ``CompiledGraph.ainvoke`` to run the graph to completion, then
#     emits a single ``TextPart`` artifact with the last assistant message.

# Streaming path (``streaming=True``):
#     Uses ``CompiledGraph.astream`` to yield incremental ``StreamChunk``
#     objects.  For each chunk that carries a message, the executor sends a
#     ``TaskState.working`` status update so the A2A client can observe
#     progress.  After the stream finishes the final text is emitted as an
#     artifact.
# """

# from __future__ import annotations

# import logging
# from typing import TYPE_CHECKING, Any

# from agentflow.core.state.message import Message as AFMessage
# from agentflow.core.state.stream_chunks import StreamEvent
# from agentflow.utils.constants import ResponseGranularity

# from ._optional import missing_a2a_sdk_error


# if TYPE_CHECKING:
#     from agentflow.core.graph.compiled_graph import CompiledGraph


# try:
#     from a2a.server.agent_execution import AgentExecutor as _AgentExecutor
#     from a2a.server.agent_execution.context import RequestContext
#     from a2a.server.events.event_queue import EventQueue
#     from a2a.server.tasks.task_updater import TaskUpdater
#     from a2a.types import TaskState, TextPart
# except Exception as exc:
#     _A2A_IMPORT_ERROR: BaseException | None = exc

#     class _AgentExecutor:
#         """Fallback base class used when a2a-sdk is not installed."""

# else:
#     _A2A_IMPORT_ERROR = None

# AgentExecutor = _AgentExecutor


# logger = logging.getLogger("agentflow.a2a")


# class AgentFlowExecutor(AgentExecutor):
#     """Bridges a :class:`CompiledGraph` into the A2A execution model.

#     This is the **only** glue code needed between agentflow and a2a-sdk.
#     The SDK owns the transport; agentflow owns the agent logic.

#     Args:
#         compiled_graph: A fully compiled agentflow graph.
#         config: Optional base config dict forwarded to ``ainvoke`` /
#             ``astream`` (e.g. ``thread_id``, ``recursion_limit``).
#         streaming: When *True* the executor uses ``astream`` instead of
#             ``ainvoke`` and sends ``TaskState.working`` progress events.
#     """

#     def __init__(
#         self,
#         compiled_graph: CompiledGraph,
#         config: dict[str, Any] | None = None,
#         streaming: bool = False,
#     ) -> None:
#         if _A2A_IMPORT_ERROR is not None:
#             raise missing_a2a_sdk_error("AgentFlowExecutor", _A2A_IMPORT_ERROR) from (
#                 _A2A_IMPORT_ERROR
#             )

#         self.graph = compiled_graph
#         self._base_config = config or {}
#         self._streaming = streaming

#     # ------------------------------------------------------------------ #
#     #  A2A AgentExecutor interface                                         #
#     # ------------------------------------------------------------------ #

#     async def execute(
#         self,
#         context: RequestContext,
#         event_queue: EventQueue,
#     ) -> None:
#         """Run the agentflow graph for an incoming A2A request.

#         1. Extract user text from the A2A message parts.
#         2. Run the graph (blocking or streaming).
#         3. Push the result back as an A2A artifact.
#         """
#         updater = TaskUpdater(
#             event_queue=event_queue,
#             task_id=context.task_id or "",
#             context_id=context.context_id or "",
#         )
#         await updater.submit()
#         await updater.start_work()

#         try:
#             # --- extract user text from A2A message parts ----------------
#             user_text = context.get_user_input() if context.message else ""

#             # build agentflow messages list
#             messages = [AFMessage.text_message(user_text, role="user")]

#             # per-request config — use context_id as thread_id so that
#             # conversation history persists across A2A tasks within the
#             # same session.  Falls back to task_id for one-shot callers.
#             run_config: dict[str, Any] = {**self._base_config}
#             if "thread_id" not in run_config:
#                 run_config["thread_id"] = context.context_id or context.task_id or ""

#             if self._streaming:
#                 response_text = await self._execute_streaming(messages, run_config, updater)
#             else:
#                 response_text = await self._execute_blocking(messages, run_config)

#             # --- emit the final artifact ---------------------------------
#             await updater.add_artifact([TextPart(text=response_text)])
#             await updater.complete()

#         except Exception as exc:
#             logger.exception("AgentFlowExecutor: graph execution failed")
#             error_msg = updater.new_agent_message(parts=[TextPart(text=f"Error: {exc!s}")])
#             await updater.failed(message=error_msg)

#     async def cancel(
#         self,
#         context: RequestContext,
#         event_queue: EventQueue,
#     ) -> None:
#         """Cancel is not currently supported by agentflow graphs."""
#         raise NotImplementedError("cancel not supported")

#     # ------------------------------------------------------------------ #
#     #  Internal helpers                                                    #
#     # ------------------------------------------------------------------ #

#     async def _execute_blocking(
#         self,
#         messages: list[AFMessage],
#         config: dict[str, Any],
#     ) -> str:
#         """Run the graph via ``ainvoke`` and return the last assistant text."""
#         result = await self.graph.ainvoke(
#             {"messages": messages},
#             config=config,
#             response_granularity=ResponseGranularity.FULL,
#         )
#         return self._extract_response_text(result)

#     async def _execute_streaming(
#         self,
#         messages: list[AFMessage],
#         config: dict[str, Any],
#         updater: TaskUpdater,
#     ) -> str:
#         """Run the graph via ``astream``, sending progress updates per chunk."""
#         last_text = ""
#         async for chunk in self.graph.astream(
#             {"messages": messages},
#             config=config,
#             response_granularity=ResponseGranularity.FULL,
#         ):
#             if chunk.event == StreamEvent.MESSAGE and chunk.message is not None:
#                 text = chunk.message.text()
#                 if text:
#                     last_text = text
#                     # signal progress with the latest text
#                     progress_msg = updater.new_agent_message(parts=[TextPart(text=text)])
#                     await updater.update_status(TaskState.working, message=progress_msg)
#             elif chunk.event == StreamEvent.STATE and chunk.state is not None:
#                 # final state arrived — extract from it
#                 assistant_text = self._extract_state_text(chunk.state)
#                 if assistant_text:
#                     last_text = assistant_text

#         return last_text or "No response generated."

#     # ------------------------------------------------------------------ #
#     #  Text extraction                                                     #
#     # ------------------------------------------------------------------ #

#     @staticmethod
#     def _extract_response_text(result: dict[str, Any]) -> str:
#         """Pull the last assistant message text from an ``ainvoke`` result.

#         With ``ResponseGranularity.FULL`` the result dict contains
#         ``"state"`` (the complete ``AgentState``) as well as
#         ``"messages"`` (last-step messages).  We prefer the full state
#         because it has all messages across every node.
#         """
#         # Primary: full state context
#         full_state = result.get("state")
#         if full_state is not None:
#             for msg in reversed(full_state.context):
#                 if msg.role == "assistant":
#                     return msg.text() or ""

#         # Fallback: messages list at LOW/PARTIAL granularity
#         for msg in reversed(result.get("messages", [])):
#             if msg.role == "assistant":
#                 return msg.text() or ""

#         return "No response generated."

#     @staticmethod
#     def _extract_state_text(state: Any) -> str:
#         """Extract last assistant text from an ``AgentState``."""
#         for msg in reversed(state.context):
#             if msg.role == "assistant":
#                 return msg.text() or ""
#         return ""
