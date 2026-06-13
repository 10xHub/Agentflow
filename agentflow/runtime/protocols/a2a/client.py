# """
# A2A client helpers for agentflow.

# Provides utilities to call any remote A2A-compliant agent from within
# an agentflow graph.

# Functions:
#     delegate_to_a2a_agent  — async one-shot: send text, get text back.
#     create_a2a_client_node — factory returning a graph-compatible node
#                              function that delegates to a remote A2A
#                              agent.
# """

# from __future__ import annotations

# import logging
# import uuid
# from importlib import import_module
# from typing import Any

# from agentflow.core.state.agent_state import AgentState
# from agentflow.core.state.message import Message as AFMessage

# from ._optional import A2A_EXTRA_INSTALL_HINT, get_a2a_attr, import_a2a_module


# logger = logging.getLogger("agentflow.a2a")


# def _import_client_dependencies():
#     """Load client-only optional dependencies when an A2A call is made."""
#     feature = "A2A client helpers"
#     try:
#         httpx = import_module("httpx")
#     except Exception as exc:
#         raise RuntimeError(
#             f"{feature} requires the optional 'httpx' package. {A2A_EXTRA_INSTALL_HINT}"
#         ) from exc

#     a2a_client = get_a2a_attr("a2a.client", "A2AClient", feature)
#     a2a_types = import_a2a_module("a2a.types", feature)
#     return httpx, a2a_client, a2a_types


# # ---------------------------------------------------------------------- #
# #  Low-level helper                                                        #
# # ---------------------------------------------------------------------- #


# async def delegate_to_a2a_agent(
#     url: str,
#     text: str,
#     *,
#     context_id: str | None = None,
#     timeout: float = 30.0,
# ) -> str:
#     """Call a remote A2A agent and return its text response.

#     This uses the (deprecated but stable) ``A2AClient`` from the a2a-sdk
#     which provides the simplest request/response interface.

#     Args:
#         url: Base URL of the remote agent (e.g. ``http://localhost:9999``).
#         text: The user message to send.
#         timeout: HTTP request timeout in seconds.

#     Returns:
#         The text content of the agent's response.

#     Raises:
#         RuntimeError: If the agent returns an error or no text parts.
#     """
#     httpx, a2a_client, a2a_types = _import_client_dependencies()

#     async with httpx.AsyncClient(timeout=timeout) as http:
#         client = a2a_client(httpx_client=http, url=url)

#         request = a2a_types.SendMessageRequest(
#             id=str(uuid.uuid4()),
#             params=a2a_types.MessageSendParams(
#                 message=a2a_types.Message(
#                     role=a2a_types.Role.user,
#                     message_id=str(uuid.uuid4()),
#                     context_id=context_id,
#                     parts=[a2a_types.TextPart(text=text)],
#                 ),
#             ),
#         )

#         response = await client.send_message(request)

#         # response.root is either SendMessageSuccessResponse or JSONRPCErrorResponse
#         result = response.root
#         if hasattr(result, "error"):
#             raise RuntimeError(f"A2A agent returned error: {result.error}")

#         # result.result is Task | Message
#         payload = result.result

#         # Extract text from the response
#         return _extract_text(payload)


# def _extract_text(payload: Any) -> str:
#     """Pull text from a Task or Message returned by the A2A SDK.

#     The SDK wraps parts in ``Part(root=TextPart(...))``  — a discriminated
#     union.  We check both ``part.text`` (direct TextPart) and
#     ``part.root.text`` (wrapped Part) to be resilient.
#     """
#     parts: list[Any] = []

#     if hasattr(payload, "parts"):
#         # It's an A2A Message
#         parts = payload.parts or []
#     elif hasattr(payload, "artifacts") and payload.artifacts:
#         # It's a Task — text lives in artifact parts
#         for artifact in payload.artifacts:
#             parts.extend(artifact.parts or [])
#     elif hasattr(payload, "status") and payload.status and payload.status.message:
#         # Fallback: check status message
#         parts = payload.status.message.parts or []

#     text_parts: list[str] = []
#     for p in parts:
#         # Direct TextPart (has .text)
#         if hasattr(p, "text") and isinstance(p.text, str):
#             text_parts.append(p.text)
#         # Wrapped Part(root=TextPart(...))
#         elif hasattr(p, "root") and hasattr(p.root, "text") and isinstance(p.root.text, str):
#             text_parts.append(p.root.text)

#     if text_parts:
#         return "\n".join(text_parts)

#     raise RuntimeError("A2A agent response contained no text parts")


# # ---------------------------------------------------------------------- #
# #  Graph node factory                                                      #
# # ---------------------------------------------------------------------- #


# def create_a2a_client_node(
#     url: str,
#     *,
#     timeout: float = 30.0,
#     response_role: str = "assistant",
# ):
#     """Return an async callable that can be used as an agentflow graph node.

#     The node reads the last message from the state, forwards its text to
#     the remote A2A agent at *url*, and returns the response as a new
#     ``Message``.

#     Usage::

#         graph.add_node("remote_agent", create_a2a_client_node("http://localhost:9999"))
#         graph.add_edge("some_node", "remote_agent")
#         graph.add_edge("remote_agent", END)

#     Args:
#         url: Base URL of the remote A2A agent.
#         timeout: HTTP request timeout.
#         response_role: Role to assign to the response message
#             (default ``"assistant"``).

#     Returns:
#         An async function with signature
#         ``(state: AgentState, config: dict) -> list[AFMessage]``
#     """

#     async def _a2a_node(state: AgentState, config: dict) -> list[AFMessage]:
#         # Get text from the last message in the conversation
#         if not state.context:
#             return [
#                 AFMessage.text_message(
#                     "No input provided.",
#                     role=response_role,
#                 ),
#             ]

#         user_text = state.context[-1].text()
#         if not user_text:
#             return [
#                 AFMessage.text_message(
#                     "Empty input.",
#                     role=response_role,
#                 ),
#             ]

#         # Reuse the parent graph's thread_id as context_id so the remote
#         # A2A agent stays in the same session as the whole workflow.
#         # The server uses context_id as its own thread_id for its checkpointer,
#         # so it maintains full conversation history server-side across turns.
#         context_id = config.get("thread_id")

#         try:
#             response = await delegate_to_a2a_agent(
#                 url, user_text, context_id=context_id, timeout=timeout
#             )
#         except Exception as exc:
#             logger.exception("A2A client node failed for url=%s", url)
#             return [
#                 AFMessage.text_message(
#                     f"A2A call failed: {exc!s}",
#                     role=response_role,
#                 ),
#             ]

#         return [
#             AFMessage.text_message(
#                 response,
#                 role=response_role,
#             ),
#         ]

#     # Give the function a useful name for debugging / graph visualization
#     _a2a_node.__name__ = f"a2a_client_node({url})"
#     _a2a_node.__qualname__ = _a2a_node.__name__

#     return _a2a_node
