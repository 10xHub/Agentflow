"""AudioAgent -- prebuilt realtime (audio-to-audio) agent, React-style builder.

Mirrors :class:`~agentflow.prebuilt.agent.react.ReactAgent`'s construction surface but
wraps a :class:`~agentflow.core.realtime.live_agent.LiveAgent` as the graph root. The
compiled graph is driven by ``CompiledGraph.arealtime`` (a separate runtime), not
``invoke``/``stream``. No sub-agents / handoff are wired in v1 (a handoff tool is just a
tool, so the door stays open).
"""

from collections.abc import Callable, Iterable
from typing import Any

from agentflow.core.graph.compiled_graph import CompiledGraph
from agentflow.core.graph.state_graph import StateGraph
from agentflow.core.graph.tool_node import ToolNode
from agentflow.core.realtime.base import RealtimeClient, RealtimeConfig
from agentflow.core.realtime.live_agent import LiveAgent
from agentflow.core.state.agent_state import AgentState
from agentflow.core.state.base_context import BaseContextManager
from agentflow.runtime.publisher.base_publisher import BasePublisher
from agentflow.storage.checkpointer.base_checkpointer import BaseCheckpointer
from agentflow.storage.media.storage.base import BaseMediaStore
from agentflow.storage.store.base_store import BaseStore
from agentflow.utils.callbacks import CallbackManager
from agentflow.utils.constants import END
from agentflow.utils.id_generator import BaseIDGenerator, DefaultIDGenerator


class AudioAgent[StateT: AgentState]:
    """Build and compile a single realtime audio agent graph."""

    def __init__(  # noqa: PLR0913
        self,
        model: str,
        state: StateT | None = None,
        context_manager: BaseContextManager[StateT] | None = None,
        publisher: BasePublisher | list[BasePublisher] | None = None,
        id_generator: BaseIDGenerator = DefaultIDGenerator(),
        container: Any | None = None,
        *,
        realtime_config: RealtimeConfig | None = None,
        system_prompt: list[dict[str, Any]] | None = None,
        tools: Iterable[Callable] | None = None,
        client: Any = None,
        pass_user_info_to_mcp: bool = False,
        skills: Any | None = None,
        memory: Any | None = None,
        realtime_client_factory: Callable[[], RealtimeClient] | None = None,
        live_node_name: str = "LIVE",
        **agent_kwargs: Any,
    ) -> None:
        self._state = state
        self._context_manager = context_manager
        self._publisher = publisher
        self._id_generator = id_generator
        self._container = container
        self._live_node_name = live_node_name

        self._tool_node = self._build_tool_node(
            tools=list(tools or []),
            client=client,
            pass_user_info_to_mcp=pass_user_info_to_mcp,
        )

        self._agent = LiveAgent(
            model,
            realtime_config=realtime_config,
            system_prompt=system_prompt,
            tool_node=self._tool_node,
            skills=skills,
            memory=memory,
            realtime_client_factory=realtime_client_factory,
            **agent_kwargs,
        )
        self._graph: StateGraph[StateT] | None = None

    @staticmethod
    def _build_tool_node(
        *, tools: list[Callable], client: Any, pass_user_info_to_mcp: bool
    ) -> ToolNode | None:
        if not tools and client is None:
            return None
        return ToolNode(tools, client=client, pass_user_info_to_mcp=pass_user_info_to_mcp)

    def _create_graph(self) -> StateGraph[StateT]:
        return StateGraph[StateT](
            state=self._state,
            context_manager=self._context_manager,
            publisher=self._publisher,
            id_generator=self._id_generator,
            container=self._container,
        )

    def _configure_graph(self) -> None:
        self._graph = self._create_graph()
        self._graph.add_node(self._live_node_name, self._agent)
        self._graph.set_entry_point(self._live_node_name)
        # The edge is never traversed in realtime (the live node owns the loop); it exists
        # only so the graph is well-formed for compile().
        self._graph.add_edge(self._live_node_name, END)

    def compile(
        self,
        checkpointer: BaseCheckpointer[StateT] | None = None,
        store: BaseStore | None = None,
        interrupt_before: list[str] | None = None,
        interrupt_after: list[str] | None = None,
        callback_manager: CallbackManager | None = None,
        media_store: BaseMediaStore | None = None,
        shutdown_timeout: float = 30.0,
    ) -> CompiledGraph:
        self._configure_graph()
        if self._graph is None:  # pragma: no cover - _configure_graph always assigns
            raise RuntimeError("graph configuration failed")
        return self._graph.compile(
            checkpointer=checkpointer,
            store=store,
            interrupt_before=interrupt_before,
            interrupt_after=interrupt_after,
            callback_manager=callback_manager
            if callback_manager is not None
            else CallbackManager(),
            media_store=media_store,
            shutdown_timeout=shutdown_timeout,
        )
