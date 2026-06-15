"""Unit tests for LiveAgent's duplex loop, tool loop, persistence, and resumption.

No live LLM and no real socket: a FakeRealtimeClient yields scripted RealtimeEvents and
records everything sent upstream. A client *factory* lets reconnect/resume tests hand out
fresh sockets per connection.
"""

import asyncio

import pytest
from injectq import InjectQ

from agentflow.core.realtime.base import (
    AudioDeltaEvent,
    GoAwayEvent,
    InputTranscriptEvent,
    InterruptedEvent,
    OutputTranscriptEvent,
    RealtimeConfig,
    SessionUpdateEvent,
    ToolCallEvent,
    ToolResultEvent,
    TurnCompleteEvent,
)
from agentflow.core.realtime.live_agent import LiveAgent
from agentflow.core.realtime.queue import LiveInputQueue
from agentflow.core.graph.tool_node import ToolNode
from agentflow.core.state import AgentState, Message
from agentflow.runtime.publisher.base_publisher import BasePublisher
from agentflow.utils import CallbackManager
from agentflow.utils.background_task_manager import BackgroundTaskManager

MODEL = "gemini-2.5-flash-live"


class FakeRealtimeClient:
    def __init__(self, events=None):
        self.events = events or []
        self.connected_with: list[str | None] = []
        self.connected_config = None
        self.sent_audio: list[tuple[bytes, int]] = []
        self.sent_text: list[str] = []
        self.activity: list[str] = []
        self.tool_responses: list[tuple[str, str, object]] = []
        self.reseeded = None
        self.closed = False

    async def connect(self, config, resume_handle=None):
        self.connected_with.append(resume_handle)
        self.connected_config = config

    async def send_audio(self, pcm, sample_rate):
        self.sent_audio.append((pcm, sample_rate))

    async def send_text(self, text):
        self.sent_text.append(text)

    async def send_activity_start(self):
        self.activity.append("start")

    async def send_activity_end(self):
        self.activity.append("end")

    async def send_tool_response(self, call_id, name, result):
        self.tool_responses.append((call_id, name, result))

    async def reseed_history(self, messages):
        self.reseeded = list(messages)

    async def receive(self):
        for event in self.events:
            yield event

    async def close(self):
        self.closed = True


def _factory(*clients):
    """Return a factory that hands out the given clients in order."""
    seq = list(clients)

    def make():
        return seq.pop(0)

    return make


def _closed_queue():
    q = LiveInputQueue()
    q.close()  # pump exits immediately; receive script drives the test
    return q


async def _drain(agent, queue, config, **kw):
    return [event async for event in agent.arun(queue, config, **kw)]


class TestForcingRule:
    @pytest.mark.asyncio
    async def test_execute_raises_directing_to_arealtime(self):
        agent = LiveAgent(MODEL, realtime_client_factory=_factory(FakeRealtimeClient()))
        with pytest.raises(RuntimeError, match="arealtime"):
            await agent.execute(AgentState(), {})

    def test_non_google_model_rejected(self):
        with pytest.raises(ValueError, match="google"):
            LiveAgent("gpt-4o-realtime")


class TestDuplexLoop:
    @pytest.mark.asyncio
    async def test_yields_audio_and_turn_events_to_caller(self):
        client = FakeRealtimeClient([AudioDeltaEvent(data=b"\x01"), TurnCompleteEvent()])
        agent = LiveAgent(MODEL, realtime_client_factory=_factory(client))

        events = await _drain(agent, _closed_queue(), {"thread_id": "t1"})

        assert [e.type for e in events] == ["audio_delta", "turn_complete"]
        assert client.closed is True

    @pytest.mark.asyncio
    async def test_pump_maps_queue_frames_to_provider(self):
        client = FakeRealtimeClient()
        agent = LiveAgent(MODEL, realtime_client_factory=_factory(client))
        agent._active_client = client
        q = LiveInputQueue()
        q.send_audio(b"x", sample_rate=16000)
        q.send_text("hi")
        q.send_activity_start()
        q.send_activity_end()
        q.close()

        await agent._pump(q)

        assert client.sent_audio == [(b"x", 16000)]
        assert client.sent_text == ["hi"]
        assert client.activity == ["start", "end"]


class TestToolLoop:
    @pytest.mark.asyncio
    async def test_tool_call_invokes_toolnode_and_sends_response(self):
        def get_weather(city: str) -> str:
            return f"sunny in {city}"

        client = FakeRealtimeClient(
            [ToolCallEvent(id="c1", name="get_weather", args={"city": "Paris"})]
        )
        agent = LiveAgent(
            MODEL,
            tool_node=ToolNode([get_weather]),
            realtime_client_factory=_factory(client),
        )

        events = await _drain(agent, _closed_queue(), {"thread_id": "t1"})

        assert client.tool_responses[0][0] == "c1"
        assert client.tool_responses[0][1] == "get_weather"
        assert client.tool_responses[0][2] == {"result": "sunny in Paris"}
        # caller sees both the tool_call and a synthesized tool_result
        assert any(isinstance(e, ToolCallEvent) for e in events)
        assert any(
            isinstance(e, ToolResultEvent) and e.result == {"result": "sunny in Paris"}
            for e in events
        )


class TestTransparency:
    @pytest.mark.asyncio
    async def test_tool_loop_and_bargein_emit_publisher_events(self):
        class SpyPublisher(BasePublisher):
            def __init__(self):
                self.events = []

            async def publish(self, event):
                self.events.append(event)

            async def close(self):
                pass

            def sync_close(self):
                pass

        spy = SpyPublisher()
        InjectQ.get_instance().bind_instance(BasePublisher, spy, allow_concrete=True)

        def ping() -> str:
            return "pong"

        client = FakeRealtimeClient(
            [ToolCallEvent(id="c1", name="ping", args={}), InterruptedEvent()]
        )
        agent = LiveAgent(MODEL, tool_node=ToolNode([ping]), realtime_client_factory=_factory(client))

        await _drain(agent, _closed_queue(), {"thread_id": "t1", "run_id": "r1"})

        tm = InjectQ.get_instance().try_get(BackgroundTaskManager)
        if tm is not None:
            await tm.wait_for_all(timeout=2.0)

        kinds = {(str(e.event), str(e.event_type)) for e in spy.events}
        # ToolNode publishes tool execution; LiveAgent publishes barge-in.
        assert any(ev == "tool_execution" for ev, _ in kinds)
        assert any(etype == "interrupted" for _, etype in kinds)


class TestBargeIn:
    @pytest.mark.asyncio
    async def test_interruption_propagates_and_pump_survives(self):
        client = FakeRealtimeClient(
            [AudioDeltaEvent(data=b"\x01"), InterruptedEvent(), AudioDeltaEvent(data=b"\x02")]
        )
        agent = LiveAgent(MODEL, realtime_client_factory=_factory(client))
        q = LiveInputQueue()  # left open: pump must stay alive across the interruption

        events = []
        async for event in agent.arun(q, {"thread_id": "t1"}):
            events.append(event)
            if event.type == "interrupted":
                q.send_audio(b"\x03", sample_rate=16000)  # input still accepted mid-session
                await asyncio.sleep(0.02)  # let the still-alive pump task drain it

        assert [e.type for e in events] == ["audio_delta", "interrupted", "audio_delta"]
        assert (b"\x03", 16000) in client.sent_audio


class TestTranscriptPersistence:
    @pytest.mark.asyncio
    async def test_finished_transcripts_persist_as_messages_no_audio(self):
        from agentflow.storage.checkpointer import InMemoryCheckpointer

        client = FakeRealtimeClient(
            [
                AudioDeltaEvent(data=b"\xaa\xbb"),  # must NOT be persisted
                InputTranscriptEvent(text="hello", finished=True),
                OutputTranscriptEvent(text="hi there", finished=True),
                InputTranscriptEvent(text="partial", finished=False),  # must NOT persist
            ]
        )
        agent = LiveAgent(MODEL, realtime_client_factory=_factory(client))
        cp = InMemoryCheckpointer()
        state = AgentState()
        config = {"thread_id": "t-persist", "user_id": "u1"}

        await _drain(agent, _closed_queue(), config, state=state, checkpointer=cp)

        roles = [(m.role, m.content[0].text) for m in state.context]
        assert ("user", "hello") in roles
        assert ("assistant", "hi there") in roles
        assert all(text != "partial" for _, text in roles)
        # audio bytes never become messages
        for m in state.context:
            assert m.metadata.get("modality") == "audio"
        persisted = await cp.alist_messages(config)
        assert {m.content[0].text for m in persisted} == {"hello", "hi there"}


class TestTranscriptAccumulation:
    @pytest.mark.asyncio
    async def test_partial_chunks_accumulate_and_flush_on_finished(self):
        from agentflow.storage.checkpointer import InMemoryCheckpointer

        # Streamed as partials (finished=False) then a finish marker with empty text.
        client = FakeRealtimeClient(
            [
                OutputTranscriptEvent(text="Hello ", finished=False),
                OutputTranscriptEvent(text="there ", finished=False),
                OutputTranscriptEvent(text="friend.", finished=False),
                OutputTranscriptEvent(text="", finished=True),
            ]
        )
        agent = LiveAgent(MODEL, realtime_client_factory=_factory(client))
        cp = InMemoryCheckpointer()
        state = AgentState()
        config = {"thread_id": "t-acc", "user_id": "u1"}

        await _drain(agent, _closed_queue(), config, state=state, checkpointer=cp)

        persisted = await cp.alist_messages(config)
        assert [m.content[0].text for m in persisted] == ["Hello there friend."]

    @pytest.mark.asyncio
    async def test_finished_event_carries_full_text_to_consumer(self):
        # The consumer should get one consolidated finished transcript with the whole text
        # (not the empty finish marker), without having to accumulate partials itself.
        client = FakeRealtimeClient(
            [
                OutputTranscriptEvent(text="Hello ", finished=False),
                OutputTranscriptEvent(text="world", finished=False),
                OutputTranscriptEvent(text="", finished=True),
            ]
        )
        agent = LiveAgent(MODEL, realtime_client_factory=_factory(client))

        events = await _drain(agent, _closed_queue(), {"thread_id": "t-consolidate"})

        finished = [
            e for e in events if e.type == "output_transcript" and e.finished
        ]
        assert len(finished) == 1
        assert finished[0].text == "Hello world"

    @pytest.mark.asyncio
    async def test_interruption_discards_partial_so_restart_is_not_concatenated(self):
        # A barge-in mid-transcript must drop the abandoned partial; the restarted turn's
        # text must not be glued onto it.
        client = FakeRealtimeClient(
            [
                OutputTranscriptEvent(text="The weather is ", finished=False),
                InterruptedEvent(),
                OutputTranscriptEvent(text="It is sunny.", finished=False),
                OutputTranscriptEvent(text="", finished=True),
            ]
        )
        agent = LiveAgent(MODEL, realtime_client_factory=_factory(client))

        events = await _drain(agent, _closed_queue(), {"thread_id": "t-barge"})

        finished = [e for e in events if e.type == "output_transcript" and e.finished]
        assert len(finished) == 1
        assert finished[0].text == "It is sunny."

    @pytest.mark.asyncio
    async def test_empty_transcript_turn_emits_no_finished_event(self):
        # A finished marker with nothing accumulated must not surface an empty transcript.
        client = FakeRealtimeClient([OutputTranscriptEvent(text="", finished=True)])
        agent = LiveAgent(MODEL, realtime_client_factory=_factory(client))

        events = await _drain(agent, _closed_queue(), {"thread_id": "t-empty"})

        assert not any(e.type == "output_transcript" for e in events)


class TestReseedGating:
    @pytest.mark.asyncio
    async def test_reseed_skipped_when_resumed_from_handle(self):
        from agentflow.storage.checkpointer import InMemoryCheckpointer
        from agentflow.utils.thread_info import ThreadInfo

        cp = InMemoryCheckpointer()
        config = {"thread_id": "t-resumed", "user_id": "u1"}
        await cp.aput_messages(config, [Message.text_message("earlier", role="user")])
        # A stored handle means the provider restores context on connect; reseed must NOT
        # replay history again.
        await cp.aput_thread(
            config, ThreadInfo(thread_id="t-resumed", metadata={"resumption_handle": "H1"})
        )

        client = FakeRealtimeClient([TurnCompleteEvent()])
        agent = LiveAgent(MODEL, realtime_client_factory=_factory(client))

        await _drain(agent, _closed_queue(), config, checkpointer=cp)

        assert client.connected_with == ["H1"]
        assert client.reseeded is None


class TestReconnectBackoff:
    @pytest.mark.asyncio
    async def test_error_driven_reconnect_gives_up_and_emits_fatal_error(self):
        # Every socket drops on receive: error-driven reconnect must back off, cap, then
        # surface a fatal ErrorEvent rather than spin forever.
        class DroppingClient(FakeRealtimeClient):
            async def receive(self):
                raise ConnectionError("socket dropped")
                yield  # pragma: no cover - makes this an async generator

        def make():
            return DroppingClient()

        agent = LiveAgent(MODEL, realtime_client_factory=make)
        agent._reconnect_base_delay = 0.0  # no real sleeping in the test
        agent._reconnect_max_attempts = 3
        q = LiveInputQueue()  # left open so the loop is allowed to attempt resume

        events = await _drain(agent, q, {"thread_id": "t-storm"})

        fatal = [e for e in events if e.type == "error"]
        assert len(fatal) == 1
        assert fatal[0].fatal is True
        assert fatal[0].code == "reconnect_failed"


class TestSessionConfig:
    @pytest.mark.asyncio
    async def test_per_session_overrides_merge_over_realtime_config(self):
        client = FakeRealtimeClient([TurnCompleteEvent()])
        agent = LiveAgent(MODEL, realtime_client_factory=_factory(client))

        await _drain(
            agent,
            _closed_queue(),
            {"thread_id": "t1", "realtime": {"voice": "Puck", "response_modalities": ["TEXT"]}},
        )

        # Per-session overrides win; unspecified fields keep the agent's base config.
        assert client.connected_config.voice == "Puck"
        assert client.connected_config.response_modalities == ["TEXT"]
        assert client.connected_config.model == MODEL

    @pytest.mark.asyncio
    async def test_no_overrides_uses_base_config_identity(self):
        client = FakeRealtimeClient([TurnCompleteEvent()])
        agent = LiveAgent(MODEL, realtime_client_factory=_factory(client))

        await _drain(agent, _closed_queue(), {"thread_id": "t1"})

        assert client.connected_config is agent.realtime_config


class TestToolAdvertising:
    @pytest.mark.asyncio
    async def test_tool_node_tools_advertised_to_provider(self):
        def get_weather(city: str) -> str:
            """Get the weather."""
            return "sunny"

        client = FakeRealtimeClient([TurnCompleteEvent()])
        agent = LiveAgent(
            MODEL, tool_node=ToolNode([get_weather]), realtime_client_factory=_factory(client)
        )

        await _drain(agent, _closed_queue(), {"thread_id": "t1"})

        names = [t["function"]["name"] for t in client.connected_config.tools]
        assert "get_weather" in names

    @pytest.mark.asyncio
    async def test_explicit_config_tools_take_precedence(self):
        def unused(x: int) -> int:
            """Unused."""
            return x

        client = FakeRealtimeClient([TurnCompleteEvent()])
        cfg = RealtimeConfig(model=MODEL, tools=[{"sentinel": True}])
        agent = LiveAgent(
            MODEL,
            realtime_config=cfg,
            tool_node=ToolNode([unused]),
            realtime_client_factory=_factory(client),
        )

        await _drain(agent, _closed_queue(), {"thread_id": "t1"})

        assert client.connected_config.tools == [{"sentinel": True}]

    @pytest.mark.asyncio
    async def test_advertised_tools_filtered_by_tools_tags(self):
        from agentflow.utils import tool

        @tool(tags=["weather"])
        def get_weather(city: str) -> str:
            """Weather."""
            return "x"

        @tool(tags=["math"])
        def add(a: int, b: int) -> int:
            """Add."""
            return a + b

        client = FakeRealtimeClient([TurnCompleteEvent()])
        cfg = RealtimeConfig(model=MODEL, tools_tags=["weather"])
        agent = LiveAgent(
            MODEL,
            realtime_config=cfg,
            tool_node=ToolNode([get_weather, add]),
            realtime_client_factory=_factory(client),
        )

        await _drain(agent, _closed_queue(), {"thread_id": "t1"})

        names = [t["function"]["name"] for t in client.connected_config.tools]
        assert names == ["get_weather"]

    @pytest.mark.asyncio
    async def test_no_tool_node_leaves_tools_unset(self):
        client = FakeRealtimeClient([TurnCompleteEvent()])
        agent = LiveAgent(MODEL, realtime_client_factory=_factory(client))

        await _drain(agent, _closed_queue(), {"thread_id": "t1"})

        assert client.connected_config.tools is None


class TestClientHangup:
    @pytest.mark.asyncio
    async def test_closing_queue_ends_idle_session(self):
        # Provider yields one event then goes idle (blocks forever); closing the input
        # queue must end the session rather than hang on receive().
        class IdleClient(FakeRealtimeClient):
            async def receive(self):
                yield AudioDeltaEvent(data=b"\x01")
                await asyncio.Event().wait()  # never resolves

        agent = LiveAgent(MODEL, realtime_client_factory=_factory(IdleClient()))
        q = LiveInputQueue()  # left open

        events = []

        async def run():
            async for event in agent.arun(q, {"thread_id": "t1"}):
                events.append(event)

        task = asyncio.create_task(run())
        await asyncio.sleep(0.05)  # let the first event flow and the provider go idle
        q.close()
        await asyncio.wait_for(task, timeout=1.0)

        assert [e.type for e in events] == ["audio_delta"]

    @pytest.mark.asyncio
    async def test_closed_queue_still_drains_available_events(self):
        # A pre-closed queue must not preempt the provider's already-available events.
        client = FakeRealtimeClient([AudioDeltaEvent(data=b"\x01"), TurnCompleteEvent()])
        agent = LiveAgent(MODEL, realtime_client_factory=_factory(client))

        events = await _drain(agent, _closed_queue(), {"thread_id": "t1"})

        assert [e.type for e in events] == ["audio_delta", "turn_complete"]


class TestResumption:
    @pytest.mark.asyncio
    async def test_session_update_caches_and_persists_handle(self):
        from agentflow.storage.checkpointer import InMemoryCheckpointer

        client = FakeRealtimeClient([SessionUpdateEvent(resumption_handle="H1")])
        agent = LiveAgent(MODEL, realtime_client_factory=_factory(client))
        cp = InMemoryCheckpointer()
        config = {"thread_id": "t-resume", "user_id": "u1"}

        await _drain(agent, _closed_queue(), config, checkpointer=cp)

        assert agent._resume_handle == "H1"
        thread = await cp.aget_thread(config)
        assert thread.metadata["resumption_handle"] == "H1"

    @pytest.mark.asyncio
    async def test_go_away_reconnects_with_stored_handle(self):
        first = FakeRealtimeClient([SessionUpdateEvent(resumption_handle="H1"), GoAwayEvent(time_left="2s")])
        second = FakeRealtimeClient([AudioDeltaEvent(data=b"\x09"), TurnCompleteEvent()])
        agent = LiveAgent(MODEL, realtime_client_factory=_factory(first, second))

        events = await _drain(agent, _closed_queue(), {"thread_id": "t1"})

        # second socket opened with the cached handle, stream continued seamlessly
        assert second.connected_with == ["H1"]
        assert [e.type for e in events] == ["session_update", "go_away", "audio_delta", "turn_complete"]
        assert first.closed and second.closed

    @pytest.mark.asyncio
    async def test_go_away_without_handle_reconnects_fresh(self):
        # go_away before any session_update: must still reconnect (fresh, no handle)
        # instead of terminating the session.
        first = FakeRealtimeClient([GoAwayEvent(time_left="1s")])
        second = FakeRealtimeClient([AudioDeltaEvent(data=b"\x07"), TurnCompleteEvent()])
        agent = LiveAgent(MODEL, realtime_client_factory=_factory(first, second))

        events = await _drain(agent, _closed_queue(), {"thread_id": "t1"})

        assert second.connected_with == [None]
        assert [e.type for e in events] == ["go_away", "audio_delta", "turn_complete"]
        assert first.closed and second.closed

    @pytest.mark.asyncio
    async def test_cross_session_reseeds_history(self):
        from agentflow.storage.checkpointer import InMemoryCheckpointer

        cp = InMemoryCheckpointer()
        config = {"thread_id": "t-cross", "user_id": "u1"}
        await cp.aput_messages(
            config,
            [Message.text_message("earlier turn", role="user")],
        )

        client = FakeRealtimeClient([TurnCompleteEvent()])
        agent = LiveAgent(MODEL, realtime_client_factory=_factory(client))

        await _drain(agent, _closed_queue(), config, checkpointer=cp)

        assert client.reseeded is not None
        assert client.reseeded[0].content[0].text == "earlier turn"
