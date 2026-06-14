"""Phase 3: CompiledGraph.arealtime/realtime runtime + forcing-rule guards.

Pure-SDK end-to-end: AudioAgent -> compile -> arealtime drives a fake provider socket,
no server and no live LLM involved.
"""

import pytest

from agentflow.core.realtime.base import (
    AudioDeltaEvent,
    InputTranscriptEvent,
    OutputTranscriptEvent,
    ToolCallEvent,
    TurnCompleteEvent,
)
from agentflow.core.realtime.queue import LiveInputQueue
from agentflow.prebuilt.agent import AudioAgent, ReactAgent
from agentflow.storage.checkpointer import InMemoryCheckpointer
from tests.realtime.test_live_agent import FakeRealtimeClient, _factory

MODEL = "gemini-2.5-flash-live"


class TestArealtimeRuntime:
    @pytest.mark.asyncio
    async def test_arealtime_drives_live_agent_end_to_end(self):
        client = FakeRealtimeClient([AudioDeltaEvent(data=b"\x01"), TurnCompleteEvent()])
        app = AudioAgent(MODEL, realtime_client_factory=_factory(client)).compile()

        q = LiveInputQueue()
        q.close()
        events = [e async for e in app.arealtime(q, {"thread_id": "t1"})]

        assert [e.type for e in events] == ["audio_delta", "turn_complete"]

    @pytest.mark.asyncio
    async def test_arealtime_persists_transcripts_through_compiled_checkpointer(self):
        client = FakeRealtimeClient(
            [
                InputTranscriptEvent(text="hello", finished=True),
                OutputTranscriptEvent(text="hi", finished=True),
                TurnCompleteEvent(),
            ]
        )
        cp = InMemoryCheckpointer()
        app = AudioAgent(MODEL, realtime_client_factory=_factory(client)).compile(checkpointer=cp)

        config = {"thread_id": "t-cp", "user_id": "u1"}
        async for _ in app.arealtime(q := LiveInputQueue(), config):
            q.close()  # close after first event so the loop can finish

        persisted = await cp.alist_messages(config)
        assert {m.content[0].text for m in persisted} == {"hello", "hi"}

    @pytest.mark.asyncio
    async def test_arealtime_tool_loop_uses_compiled_toolnode(self):
        def get_time() -> str:
            return "12:00"

        client = FakeRealtimeClient([ToolCallEvent(id="c1", name="get_time", args={})])
        app = AudioAgent(
            MODEL, tools=[get_time], realtime_client_factory=_factory(client)
        ).compile()

        q = LiveInputQueue()
        q.close()
        events = [e async for e in app.arealtime(q, {"thread_id": "t1"})]

        assert any(e.type == "tool_result" and e.result == {"result": "12:00"} for e in events)
        assert client.tool_responses[0][1] == "get_time"


class TestForcingRule:
    @pytest.mark.asyncio
    async def test_invoke_on_live_graph_raises(self):
        app = AudioAgent(MODEL, realtime_client_factory=_factory(FakeRealtimeClient())).compile()
        with pytest.raises(RuntimeError, match="arealtime"):
            await app.ainvoke({"messages": []}, {"thread_id": "t1"})

    @pytest.mark.asyncio
    async def test_astream_on_live_graph_raises(self):
        app = AudioAgent(MODEL, realtime_client_factory=_factory(FakeRealtimeClient())).compile()
        with pytest.raises(RuntimeError, match="arealtime"):
            async for _ in app.astream({"messages": []}, {"thread_id": "t1"}):
                pass

    @pytest.mark.asyncio
    async def test_arealtime_on_non_live_graph_raises(self):
        app = ReactAgent(model="gemini-2.5-flash").compile()
        q = LiveInputQueue()
        q.close()
        with pytest.raises(RuntimeError, match="LiveAgent"):
            async for _ in app.arealtime(q, {"thread_id": "t1"}):
                pass


class TestRealtimeSyncWrapper:
    def test_realtime_sync_iteration(self):
        client = FakeRealtimeClient([AudioDeltaEvent(data=b"\x02"), TurnCompleteEvent()])
        app = AudioAgent(MODEL, realtime_client_factory=_factory(client)).compile()

        q = LiveInputQueue()
        q.close()
        events = list(app.realtime(q, {"thread_id": "t1"}))

        assert [e.type for e in events] == ["audio_delta", "turn_complete"]

    @pytest.mark.asyncio
    async def test_realtime_sync_rejected_inside_running_loop(self):
        app = AudioAgent(MODEL, realtime_client_factory=_factory(FakeRealtimeClient())).compile()
        q = LiveInputQueue()
        q.close()
        with pytest.raises(RuntimeError, match="running event loop"):
            list(app.realtime(q, {"thread_id": "t1"}))
