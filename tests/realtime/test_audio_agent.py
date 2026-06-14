"""Unit tests for the AudioAgent prebuilt (wraps LiveAgent as a compiled graph root)."""

import pytest

from agentflow.core.graph.compiled_graph import CompiledGraph
from agentflow.core.realtime.live_agent import LiveAgent
from agentflow.prebuilt.agent.audio import AudioAgent

MODEL = "gemini-2.5-flash-live"


class TestAudioAgentBuild:
    def test_compile_returns_compiled_graph_with_live_root(self):
        agent = AudioAgent(MODEL)
        app = agent.compile()

        assert isinstance(app, CompiledGraph)
        live_node = app._state_graph.nodes[agent._live_node_name]
        assert isinstance(live_node.func, LiveAgent)
        assert app._state_graph.entry_point == agent._live_node_name

    def test_tools_are_wired_into_the_live_agent(self):
        def get_weather(city: str) -> str:
            return f"sunny in {city}"

        agent = AudioAgent(MODEL, tools=[get_weather])
        agent.compile()

        assert agent._agent._resolve_tool_node() is not None

    def test_realtime_config_passthrough(self):
        from agentflow.core.realtime.base import RealtimeConfig

        cfg = RealtimeConfig(model=MODEL, voice="Puck", response_modalities=["AUDIO"])
        agent = AudioAgent(MODEL, realtime_config=cfg)

        assert agent._agent.realtime_config.voice == "Puck"

    @pytest.mark.asyncio
    async def test_compiled_live_agent_runs_via_arun(self):
        # End-to-end-ish: the LiveAgent inside the compiled graph drives a fake socket.
        from agentflow.core.realtime.base import AudioDeltaEvent, TurnCompleteEvent
        from agentflow.core.realtime.queue import LiveInputQueue
        from tests.realtime.test_live_agent import FakeRealtimeClient, _factory

        client = FakeRealtimeClient([AudioDeltaEvent(data=b"\x01"), TurnCompleteEvent()])
        agent = AudioAgent(MODEL, realtime_client_factory=_factory(client))
        agent.compile()

        q = LiveInputQueue()
        q.close()
        events = [e async for e in agent._agent.arun(q, {"thread_id": "t1"})]

        assert [e.type for e in events] == ["audio_delta", "turn_complete"]
