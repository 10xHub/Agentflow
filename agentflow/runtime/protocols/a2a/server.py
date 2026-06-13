# """
# A2A server helpers for agentflow.

# Provides convenience functions to expose a :class:`CompiledGraph` as an
# A2A-compliant HTTP endpoint using the official ``a2a-sdk``.

# Functions:
#     create_a2a_server  — one-call to start a uvicorn server.
#     build_a2a_app      — returns a Starlette ASGI app (composable).
#     make_agent_card    — builds an ``AgentCard`` with sensible defaults.
# """

# from __future__ import annotations

# from typing import TYPE_CHECKING, Any

# from ._optional import A2A_EXTRA_INSTALL_HINT, import_a2a_module


# if TYPE_CHECKING:
#     from a2a.types import AgentCard, AgentSkill
#     from starlette.applications import Starlette

#     from agentflow.core.graph.compiled_graph import CompiledGraph


# # ---------------------------------------------------------------------- #
# #  AgentCard helper                                                        #
# # ---------------------------------------------------------------------- #


# def make_agent_card(
#     name: str,
#     description: str,
#     url: str,
#     *,
#     skills: list[AgentSkill] | None = None,
#     streaming: bool = False,
#     version: str = "1.0.0",
# ) -> AgentCard:
#     """Build an :class:`AgentCard` with sensible defaults.

#     If *skills* is ``None`` a single ``"run_graph"`` skill is created
#     automatically.

#     Args:
#         name: Human-readable agent name.
#         description: Short description of what the agent does.
#         url: Public URL where the agent is reachable.
#         skills: Optional list of ``AgentSkill`` objects.
#         streaming: Whether the agent supports SSE streaming.
#         version: Semantic version string.

#     Returns:
#         A fully populated ``AgentCard``.
#     """
#     a2a_types = import_a2a_module("a2a.types", "make_agent_card")

#     if skills is None:
#         skills = [
#             a2a_types.AgentSkill(
#                 id="run_graph",
#                 name="Run Graph",
#                 description="Execute the agentflow graph",
#                 tags=["agentflow"],
#             )
#         ]

#     return a2a_types.AgentCard(
#         name=name,
#         description=description,
#         url=url,
#         version=version,
#         capabilities=a2a_types.AgentCapabilities(streaming=streaming),
#         default_input_modes=["text"],
#         default_output_modes=["text"],
#         skills=skills,
#     )


# # ---------------------------------------------------------------------- #
# #  ASGI app builder                                                        #
# # ---------------------------------------------------------------------- #


# def build_a2a_app(
#     compiled_graph: CompiledGraph,
#     agent_card: AgentCard,
#     *,
#     streaming: bool = False,
#     executor_config: dict[str, Any] | None = None,
# ) -> Starlette:
#     """Return a Starlette ASGI app that speaks the A2A protocol.

#     Useful when you want to mount the app inside another ASGI framework
#     (e.g. FastAPI), run it with a custom server, or use it in tests.

#     Args:
#         compiled_graph: A compiled agentflow graph.
#         agent_card: The ``AgentCard`` describing this agent.
#         streaming: Whether to use ``astream`` vs ``ainvoke`` in the
#             executor.
#         executor_config: Optional base config forwarded to the graph
#             (e.g. ``{"recursion_limit": 50}``).

#     Returns:
#         A ``Starlette`` application ready to be served.
#     """
#     a2a_apps = import_a2a_module("a2a.server.apps", "build_a2a_app")
#     request_handlers = import_a2a_module("a2a.server.request_handlers", "build_a2a_app")
#     tasks = import_a2a_module("a2a.server.tasks", "build_a2a_app")

#     from .executor import AgentFlowExecutor

#     executor = AgentFlowExecutor(
#         compiled_graph,
#         config=executor_config,
#         streaming=streaming,
#     )
#     handler = request_handlers.DefaultRequestHandler(
#         agent_executor=executor,
#         task_store=tasks.InMemoryTaskStore(),
#     )
#     a2a_app = a2a_apps.A2AStarletteApplication(
#         agent_card=agent_card,
#         http_handler=handler,
#     )
#     return a2a_app.build()


# # ---------------------------------------------------------------------- #
# #  One-call server                                                         #
# # ---------------------------------------------------------------------- #


# def create_a2a_server(
#     compiled_graph: CompiledGraph,
#     agent_card: AgentCard,
#     *,
#     host: str = "127.0.0.1",
#     port: int = 9999,
#     streaming: bool = False,
#     executor_config: dict[str, Any] | None = None,
# ) -> None:
#     """Build and run an A2A server exposing the given graph.

#     This is a blocking call — it starts uvicorn and does not return until
#     the server is shut down.

#     Args:
#         compiled_graph: A compiled agentflow graph.
#         agent_card: The ``AgentCard`` describing this agent.
#         host: Bind address.
#         port: Bind port.
#         streaming: Whether to use ``astream`` in the executor.
#         executor_config: Optional base config forwarded to the graph.
#     """
#     try:
#         uvicorn = __import__("uvicorn")
#     except Exception as exc:
#         raise RuntimeError(
#             "create_a2a_server requires the optional 'uvicorn' package. "
#             f"{A2A_EXTRA_INSTALL_HINT}"
#         ) from exc

#     app = build_a2a_app(
#         compiled_graph,
#         agent_card,
#         streaming=streaming,
#         executor_config=executor_config,
#     )
#     uvicorn.run(app, host=host, port=port)
