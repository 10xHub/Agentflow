"""Comprehensive unit tests for SupervisorTeamAgent."""

from __future__ import annotations

import inspect
from unittest.mock import Mock, patch

import pytest

from agentflow.core.graph import CompiledGraph, ToolNode
from agentflow.core.graph.base_agent import BaseAgent
from agentflow.core.state import AgentState, Message
from agentflow.utils import END
from agentflow.utils.callbacks import CallbackManager
from agentflow.prebuilt.agent.supervisor_team import (
    SupervisorTeamAgent,
    WorkerConfig,
    _build_supervisor_prompt,
    _make_supervisor_node,
    _resolve_decision,
    _ROUNDS_KEY,
    _ROUTE_HISTORY_KEY,
    _FINISH_TOKEN,
    _SUPERVISOR_NODE,
)


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class FakeAgent(BaseAgent):
    """Minimal Agent stub — never calls an LLM."""

    def __init__(self, model: str = "fake", **kwargs):
        super().__init__(model=model, **kwargs)

    async def execute(self, state: AgentState, config: dict) -> AgentState:
        return state

    async def _call_llm(self, messages: list[dict], tools: list | None = None, **kwargs):
        raise NotImplementedError


class _FakeConverter:
    """Stands in for ModelResponseConverter — resolves to a fixed Message."""

    def __init__(self, text: str):
        self._text = text

    async def invoke(self) -> Message:
        return Message.text_message(self._text, role="assistant")


class FakeSupervisorAgent:
    """Supervisor stub: returns a fixed decision and records the context it saw."""

    def __init__(self, decision_text: str):
        self._decision_text = decision_text
        self.seen_contexts: list[list[Message]] = []

    async def execute(self, state: AgentState, config: dict) -> _FakeConverter:
        # Snapshot the context (incl. any transient routing-memory tokens).
        self.seen_contexts.append(list(state.context))
        return _FakeConverter(self._decision_text)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_agent(**kwargs) -> FakeAgent:
    return FakeAgent(**kwargs)


def _msg(text: str, role: str = "assistant") -> Message:
    return Message.text_message(text, role=role)  # type: ignore[arg-type]


def _state_with_rounds(rounds: int, *messages: Message) -> AgentState:
    state = AgentState()
    state.context = list(messages)
    state.execution_meta.internal_data[_ROUNDS_KEY] = rounds
    return state


def _two_workers() -> dict[str, WorkerConfig]:
    return {
        "RESEARCHER": WorkerConfig(agent=_fake_agent(), description="Searches the web."),
        "CODER": WorkerConfig(agent=_fake_agent(), description="Writes Python code."),
    }


def _single_worker() -> dict[str, WorkerConfig]:
    return {"ANALYST": WorkerConfig(agent=_fake_agent(), description="Analyses data.")}


# ===========================================================================
# Tests for WorkerConfig
# ===========================================================================


class TestWorkerConfig:
    def test_minimal_config(self):
        agent = _fake_agent()
        cfg = WorkerConfig(agent=agent)
        assert cfg.agent is agent
        assert cfg.description == ""

    def test_full_config(self):
        agent = _fake_agent()
        cfg = WorkerConfig(agent=agent, description="Specialist worker.")
        assert cfg.agent is agent
        assert cfg.description == "Specialist worker."

    def test_accepts_any_base_agent_subclass(self):
        cfg = WorkerConfig(agent=_fake_agent())
        assert isinstance(cfg.agent, BaseAgent)


# ===========================================================================
# Tests for _build_supervisor_prompt
# ===========================================================================


class TestBuildSupervisorPrompt:
    def test_includes_worker_names(self):
        workers = _two_workers()
        prompt = _build_supervisor_prompt(workers)
        assert len(prompt) == 1
        assert prompt[0]["role"] == "system"
        content = prompt[0]["content"]
        assert "RESEARCHER" in content
        assert "CODER" in content

    def test_includes_worker_descriptions(self):
        workers = _two_workers()
        prompt = _build_supervisor_prompt(workers)
        content = prompt[0]["content"]
        assert "Searches the web." in content
        assert "Writes Python code." in content

    def test_includes_finish_token(self):
        prompt = _build_supervisor_prompt(_two_workers())
        assert _FINISH_TOKEN in prompt[0]["content"]

    def test_single_worker(self):
        prompt = _build_supervisor_prompt(_single_worker())
        assert "ANALYST" in prompt[0]["content"]
        assert "Analyses data." in prompt[0]["content"]


# ===========================================================================
# Tests for _resolve_decision (pure decision logic; expects normalised raw)
# ===========================================================================


class TestResolveDecision:
    def test_routes_to_worker_by_name(self):
        assert _resolve_decision("RESEARCHER", ["RESEARCHER", "CODER"], 0) == "RESEARCHER"

    def test_routes_to_end_on_finish(self):
        assert _resolve_decision("FINISH", ["RESEARCHER"], 0) == END

    def test_routes_to_end_when_no_match(self):
        assert _resolve_decision("I DON'T KNOW WHAT TO DO.", ["RESEARCHER", "CODER"], 0) == END

    def test_first_worker_match_wins(self):
        """When multiple worker names appear, first in list wins."""
        assert _resolve_decision("RESEARCHER AND CODER", ["RESEARCHER", "CODER"], 0) == "RESEARCHER"

    def test_word_boundary_match(self):
        """Worker CODE must not match inside DECODE."""
        assert _resolve_decision("DECODE", ["CODE"], 0) == END
        assert _resolve_decision("CODE", ["CODE"], 0) == "CODE"


# ===========================================================================
# Tests for the SUPERVISOR node (_make_supervisor_node)
# ===========================================================================


class TestSupervisorNode:
    @pytest.mark.asyncio
    async def test_routes_to_worker(self):
        node = _make_supervisor_node(
            FakeSupervisorAgent("RESEARCHER"), ["RESEARCHER", "CODER"], max_rounds=10
        )
        cmd = await node(AgentState(), {})
        assert cmd.goto == "RESEARCHER"

    @pytest.mark.asyncio
    async def test_routes_to_end_on_finish(self):
        node = _make_supervisor_node(
            FakeSupervisorAgent("FINISH"), ["RESEARCHER"], max_rounds=10
        )
        cmd = await node(AgentState(), {})
        assert cmd.goto == END

    @pytest.mark.asyncio
    async def test_case_insensitive_match(self):
        """Node upper-cases the raw response before resolving."""
        node = _make_supervisor_node(
            FakeSupervisorAgent("researcher"), ["RESEARCHER"], max_rounds=10
        )
        cmd = await node(AgentState(), {})
        assert cmd.goto == "RESEARCHER"

    @pytest.mark.asyncio
    async def test_max_rounds_short_circuits_without_calling_agent(self):
        sup = FakeSupervisorAgent("RESEARCHER")
        node = _make_supervisor_node(sup, ["RESEARCHER"], max_rounds=3)
        state = AgentState()
        state.execution_meta.internal_data[_ROUNDS_KEY] = 3
        cmd = await node(state, {})
        assert cmd.goto == END
        assert sup.seen_contexts == []  # agent never invoked at the cap

    @pytest.mark.asyncio
    async def test_records_worker_decisions_in_history(self):
        sup = FakeSupervisorAgent("RESEARCHER")
        node = _make_supervisor_node(sup, ["RESEARCHER"], max_rounds=10)
        state = AgentState()
        await node(state, {})
        assert state.execution_meta.internal_data[_ROUTE_HISTORY_KEY] == ["RESEARCHER"]

    @pytest.mark.asyncio
    async def test_finish_decision_not_recorded(self):
        sup = FakeSupervisorAgent("FINISH")
        node = _make_supervisor_node(sup, ["RESEARCHER"], max_rounds=10)
        state = AgentState()
        await node(state, {})
        assert state.execution_meta.internal_data.get(_ROUTE_HISTORY_KEY) == []

    @pytest.mark.asyncio
    async def test_history_replayed_as_transient_tokens_not_persisted(self):
        """Past decisions are shown to the supervisor's own call but never persisted."""
        sup = FakeSupervisorAgent("FINISH")
        node = _make_supervisor_node(sup, ["RESEARCHER"], max_rounds=10)
        state = AgentState()
        state.execution_meta.internal_data[_ROUTE_HISTORY_KEY] = ["RESEARCHER"]
        original = [_msg("the shortlist", role="assistant")]
        state.context = list(original)

        await node(state, {})

        # The supervisor saw a transient RESEARCHER token appended to its context...
        seen = sup.seen_contexts[0]
        assert any(m.role == "assistant" and m.text() == "RESEARCHER" for m in seen)
        # ...but the real context is restored (breadcrumb not persisted).
        assert [m.text() for m in state.context] == [m.text() for m in original]

    @pytest.mark.asyncio
    async def test_no_routing_tokens_persisted_to_context(self):
        """The supervisor node returns a Command, never an assistant message."""
        sup = FakeSupervisorAgent("RESEARCHER")
        node = _make_supervisor_node(sup, ["RESEARCHER"], max_rounds=10)
        state = AgentState()
        state.context = [_msg("hello", role="user")]
        cmd = await node(state, {})
        # Command carries navigation only — no state update / message.
        assert cmd.update is None
        assert cmd.goto == "RESEARCHER"
        assert [m.text() for m in state.context] == ["hello"]


# ===========================================================================
# Tests for SupervisorTeamAgent.__init__ validation
# ===========================================================================


class TestSupervisorTeamAgentInit:
    def test_basic_init(self):
        with patch("agentflow.prebuilt.agent.supervisor_team.Agent", FakeAgent):
            agent = SupervisorTeamAgent(
                supervisor_model="gpt-4o", workers=_two_workers()
            )
        assert agent is not None

    def test_empty_workers_raises(self):
        with pytest.raises(ValueError, match="at least one worker"):
            SupervisorTeamAgent(supervisor_model="gpt-4o", workers={})

    def test_reserved_supervisor_name_raises(self):
        bad_workers = {_SUPERVISOR_NODE: WorkerConfig(agent=_fake_agent())}
        with pytest.raises(ValueError, match="reserved"):
            SupervisorTeamAgent(supervisor_model="gpt-4o", workers=bad_workers)

    def test_stores_workers_and_model(self):
        with patch("agentflow.prebuilt.agent.supervisor_team.Agent", FakeAgent):
            agent = SupervisorTeamAgent(
                supervisor_model="gpt-4o", workers=_two_workers()
            )
        assert agent._supervisor_model == "gpt-4o"
        assert set(agent._workers.keys()) == {"RESEARCHER", "CODER"}

    def test_default_max_rounds(self):
        with patch("agentflow.prebuilt.agent.supervisor_team.Agent", FakeAgent):
            agent = SupervisorTeamAgent(
                supervisor_model="gpt-4o", workers=_two_workers()
            )
        assert agent._max_rounds == 10

    def test_custom_max_rounds(self):
        with patch("agentflow.prebuilt.agent.supervisor_team.Agent", FakeAgent):
            agent = SupervisorTeamAgent(
                supervisor_model="gpt-4o", workers=_two_workers(), max_rounds=5
            )
        assert agent._max_rounds == 5

    def test_auto_generated_supervisor_prompt(self):
        with patch("agentflow.prebuilt.agent.supervisor_team.Agent", FakeAgent):
            agent = SupervisorTeamAgent(
                supervisor_model="gpt-4o", workers=_two_workers()
            )
        content = agent._supervisor_system_prompt[0]["content"]
        assert "RESEARCHER" in content
        assert "CODER" in content
        assert _FINISH_TOKEN in content

    def test_custom_supervisor_prompt(self):
        custom = [{"role": "system", "content": "Custom supervisor instructions."}]
        with patch("agentflow.prebuilt.agent.supervisor_team.Agent", FakeAgent):
            agent = SupervisorTeamAgent(
                supervisor_model="gpt-4o",
                workers=_two_workers(),
                supervisor_system_prompt=custom,
            )
        assert agent._supervisor_system_prompt == custom

    def test_single_worker_allowed(self):
        with patch("agentflow.prebuilt.agent.supervisor_team.Agent", FakeAgent):
            agent = SupervisorTeamAgent(
                supervisor_model="gpt-4o", workers=_single_worker()
            )
        assert agent is not None


# ===========================================================================
# Tests for graph topology (compile / _configure_graph)
# ===========================================================================


class TestSupervisorTeamAgentCompile:
    def test_compile_returns_compiled_graph(self):
        with patch("agentflow.prebuilt.agent.supervisor_team.Agent", FakeAgent):
            agent = SupervisorTeamAgent(
                supervisor_model="gpt-4o", workers=_two_workers()
            )
        assert isinstance(agent.compile(), CompiledGraph)

    def test_graph_has_supervisor_node(self):
        with patch("agentflow.prebuilt.agent.supervisor_team.Agent", FakeAgent):
            agent = SupervisorTeamAgent(
                supervisor_model="gpt-4o", workers=_two_workers()
            )
        agent._configure_graph()
        assert _SUPERVISOR_NODE in agent._graph.nodes

    def test_graph_has_all_worker_nodes(self):
        with patch("agentflow.prebuilt.agent.supervisor_team.Agent", FakeAgent):
            agent = SupervisorTeamAgent(
                supervisor_model="gpt-4o", workers=_two_workers()
            )
        agent._configure_graph()
        assert "RESEARCHER" in agent._graph.nodes
        assert "CODER" in agent._graph.nodes

    def test_worker_nodes_are_the_provided_agents(self):
        workers = _two_workers()
        with patch("agentflow.prebuilt.agent.supervisor_team.Agent", FakeAgent):
            sta = SupervisorTeamAgent(supervisor_model="gpt-4o", workers=workers)
        sta._configure_graph()
        for name, cfg in workers.items():
            assert sta._graph.nodes[name].func is cfg.agent

    def test_compile_with_checkpointer(self):
        checkpointer = Mock()
        with patch("agentflow.prebuilt.agent.supervisor_team.Agent", FakeAgent):
            agent = SupervisorTeamAgent(
                supervisor_model="gpt-4o", workers=_two_workers()
            )
        assert isinstance(agent.compile(checkpointer=checkpointer), CompiledGraph)

    def test_compile_with_callback_manager(self):
        with patch("agentflow.prebuilt.agent.supervisor_team.Agent", FakeAgent):
            agent = SupervisorTeamAgent(
                supervisor_model="gpt-4o", workers=_two_workers()
            )
        assert isinstance(agent.compile(callback_manager=CallbackManager()), CompiledGraph)

    def test_compile_with_interrupt_options(self):
        with patch("agentflow.prebuilt.agent.supervisor_team.Agent", FakeAgent):
            agent = SupervisorTeamAgent(
                supervisor_model="gpt-4o", workers=_two_workers()
            )
        compiled = agent.compile(
            interrupt_before=[_SUPERVISOR_NODE],
            interrupt_after=["RESEARCHER"],
        )
        assert isinstance(compiled, CompiledGraph)

    def test_compile_forwards_media_store_and_timeout(self):
        media_store = Mock()
        compiled_graph = Mock(spec=CompiledGraph)

        with patch("agentflow.prebuilt.agent.supervisor_team.Agent", FakeAgent):
            agent = SupervisorTeamAgent(
                supervisor_model="gpt-4o", workers=_two_workers()
            )

        with patch(
            "agentflow.prebuilt.agent.supervisor_team.StateGraph.compile",
            autospec=True,
            return_value=compiled_graph,
        ) as compile_mock:
            result = agent.compile(media_store=media_store, shutdown_timeout=60.0)

        assert result is compiled_graph
        assert compile_mock.call_args.kwargs["media_store"] is media_store
        assert compile_mock.call_args.kwargs["shutdown_timeout"] == 60.0

    def test_compile_multiple_times_resets_graph(self):
        with patch("agentflow.prebuilt.agent.supervisor_team.Agent", FakeAgent):
            agent = SupervisorTeamAgent(
                supervisor_model="gpt-4o", workers=_two_workers()
            )
        c1 = agent.compile()
        c2 = agent.compile()
        assert isinstance(c1, CompiledGraph)
        assert isinstance(c2, CompiledGraph)


# ===========================================================================
# Tests for supervisor agent configuration
# ===========================================================================


class TestSupervisorAgentConfiguration:
    def test_supervisor_has_correct_model(self):
        # The SUPERVISOR node is now a wrapper that drives the supervisor Agent
        # internally, so assert the model on the agent the builder produces.
        with patch("agentflow.prebuilt.agent.supervisor_team.Agent", FakeAgent):
            agent = SupervisorTeamAgent(
                supervisor_model="gpt-4o-turbo", workers=_two_workers()
            )
            supervisor_agent: FakeAgent = agent._build_supervisor_agent()  # type: ignore
        assert supervisor_agent.model == "gpt-4o-turbo"

    def test_supervisor_system_prompt_is_set(self):
        custom = [{"role": "system", "content": "Custom prompt."}]
        with patch("agentflow.prebuilt.agent.supervisor_team.Agent", FakeAgent):
            sta = SupervisorTeamAgent(
                supervisor_model="gpt-4o",
                workers=_two_workers(),
                supervisor_system_prompt=custom,
            )
            supervisor_agent: FakeAgent = sta._build_supervisor_agent()  # type: ignore
        assert supervisor_agent.system_prompt == custom

    def test_supervisor_kwargs_forwarded(self):
        """Extra kwargs like provider/temperature are forwarded to the supervisor Agent."""
        captured_kwargs: dict = {}

        class CapturingFakeAgent(FakeAgent):
            def __init__(self, model: str = "fake", **kwargs):
                super().__init__(model=model, **kwargs)
                captured_kwargs.update(kwargs)

        with patch(
            "agentflow.prebuilt.agent.supervisor_team.Agent", CapturingFakeAgent
        ):
            sta = SupervisorTeamAgent(
                supervisor_model="gpt-4o",
                workers=_two_workers(),
                temperature=0.2,
                provider="openai",
            )
            # Agent is built lazily inside _configure_graph; call it inside the patch
            sta._configure_graph()

        # temperature and provider should have been captured
        assert "temperature" in captured_kwargs or "provider" in captured_kwargs

    def test_each_worker_independently_configured(self):
        """Workers are added as-is; their agents are the exact objects passed in."""
        agent_a = _fake_agent()
        agent_b = _fake_agent()
        workers = {
            "A": WorkerConfig(agent=agent_a, description="Agent A"),
            "B": WorkerConfig(agent=agent_b, description="Agent B"),
        }
        with patch("agentflow.prebuilt.agent.supervisor_team.Agent", FakeAgent):
            sta = SupervisorTeamAgent(supervisor_model="gpt-4o", workers=workers)
        sta._configure_graph()

        assert sta._graph.nodes["A"].func is agent_a
        assert sta._graph.nodes["B"].func is agent_b


# ===========================================================================
# Integration-style routing tests
# ===========================================================================


class TestSupervisorIntegration:
    @pytest.mark.asyncio
    async def test_routing_to_researcher(self):
        node = _make_supervisor_node(
            FakeSupervisorAgent("RESEARCHER"), ["RESEARCHER", "CODER"], max_rounds=5
        )
        cmd = await node(AgentState(), {})
        assert cmd.goto == "RESEARCHER"

    @pytest.mark.asyncio
    async def test_routing_to_coder(self):
        node = _make_supervisor_node(
            FakeSupervisorAgent("CODER"), ["RESEARCHER", "CODER"], max_rounds=5
        )
        cmd = await node(AgentState(), {})
        assert cmd.goto == "CODER"

    @pytest.mark.asyncio
    async def test_finish_ends_graph(self):
        node = _make_supervisor_node(
            FakeSupervisorAgent("FINISH"), ["RESEARCHER", "CODER"], max_rounds=5
        )
        cmd = await node(AgentState(), {})
        assert cmd.goto == END

    @pytest.mark.asyncio
    async def test_max_rounds_ends_graph(self):
        node = _make_supervisor_node(
            FakeSupervisorAgent("RESEARCHER"), ["RESEARCHER"], max_rounds=2
        )
        state = _state_with_rounds(2)
        cmd = await node(state, {})
        assert cmd.goto == END

    def test_compiled_swarm_has_correct_nodes(self):
        with patch("agentflow.prebuilt.agent.supervisor_team.Agent", FakeAgent):
            sta = SupervisorTeamAgent(
                supervisor_model="gpt-4o", workers=_two_workers()
            )
        sta.compile()
        assert _SUPERVISOR_NODE in sta._graph.nodes
        assert "RESEARCHER" in sta._graph.nodes
        assert "CODER" in sta._graph.nodes
