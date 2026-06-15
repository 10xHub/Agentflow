"""Unit tests for GeminiLiveClient and its LiveServerMessage normalizer.

No live LLM and no real socket: a FakeLiveSession records sent frames and yields
scripted server messages. Server-message normalization is duck-typed (reads attributes)
so we drive it with lightweight stand-ins shaped like google.genai's LiveServerMessage.
"""

from types import SimpleNamespace

import pytest

from agentflow.core.realtime.base import (
    AudioDeltaEvent,
    ErrorEvent,
    GoAwayEvent,
    InputTranscriptEvent,
    InterruptedEvent,
    OutputTranscriptEvent,
    RealtimeConfig,
    SessionUpdateEvent,
    ToolCallEvent,
    TurnCompleteEvent,
)
from agentflow.core.realtime.providers.gemini_live import GeminiLiveClient, normalize_message


# --------------------------------------------------------------------------- #
# Message-shape helpers (mirror google.genai LiveServerMessage attribute names)
# --------------------------------------------------------------------------- #
def _msg(**kw):
    base = dict(
        server_content=None,
        tool_call=None,
        go_away=None,
        session_resumption_update=None,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def _server_content(**kw):
    base = dict(
        model_turn=None,
        input_transcription=None,
        output_transcription=None,
        interrupted=None,
        generation_complete=None,
        turn_complete=None,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def _audio_part(data: bytes, mime: str = "audio/pcm;rate=24000"):
    inline = SimpleNamespace(data=data, mime_type=mime)
    return SimpleNamespace(inline_data=inline, text=None)


class TestNormalizeMessage:
    def test_audio_part_becomes_audio_delta(self):
        content = _server_content(
            model_turn=SimpleNamespace(parts=[_audio_part(b"\x10\x20")])
        )
        events = normalize_message(_msg(server_content=content))
        assert len(events) == 1
        assert isinstance(events[0], AudioDeltaEvent)
        assert events[0].data == b"\x10\x20"
        assert events[0].sample_rate == 24000

    def test_input_and_output_transcription(self):
        content = _server_content(
            input_transcription=SimpleNamespace(text="hi", finished=False),
            output_transcription=SimpleNamespace(text="hello", finished=True),
        )
        events = normalize_message(_msg(server_content=content))
        assert any(
            isinstance(e, InputTranscriptEvent) and e.text == "hi" and e.finished is False
            for e in events
        )
        assert any(
            isinstance(e, OutputTranscriptEvent) and e.text == "hello" and e.finished is True
            for e in events
        )

    def test_interrupted_emitted(self):
        content = _server_content(interrupted=True)
        events = normalize_message(_msg(server_content=content))
        assert any(isinstance(e, InterruptedEvent) for e in events)

    def test_turn_complete_emits_turn_complete_event(self):
        content = _server_content(turn_complete=True)
        events = normalize_message(_msg(server_content=content))
        assert any(isinstance(e, TurnCompleteEvent) for e in events)

    def test_generation_complete_alone_is_not_a_turn_complete(self):
        # generation_complete and turn_complete arrive in separate messages within one turn;
        # only turn_complete is the authoritative end-of-turn signal. Mapping both would
        # double-count turn boundaries.
        content = _server_content(generation_complete=True)
        events = normalize_message(_msg(server_content=content))
        assert not any(isinstance(e, TurnCompleteEvent) for e in events)

    def test_finished_transcript_with_no_text_still_emits_finish_marker(self):
        # Gemini sends the terminating transcript chunk as finished=True, text=None.
        # It must still surface so consumers can flush their accumulated transcript.
        content = _server_content(
            input_transcription=SimpleNamespace(text=None, finished=True),
            output_transcription=SimpleNamespace(text=None, finished=True),
        )
        events = normalize_message(_msg(server_content=content))
        assert any(
            isinstance(e, InputTranscriptEvent) and e.text == "" and e.finished is True
            for e in events
        )
        assert any(
            isinstance(e, OutputTranscriptEvent) and e.text == "" and e.finished is True
            for e in events
        )

    def test_tool_call_function_calls(self):
        fc = SimpleNamespace(id="call-1", name="get_weather", args={"city": "Paris"})
        tool_call = SimpleNamespace(function_calls=[fc])
        events = normalize_message(_msg(tool_call=tool_call))
        assert len(events) == 1
        assert isinstance(events[0], ToolCallEvent)
        assert events[0].id == "call-1"
        assert events[0].name == "get_weather"
        assert events[0].args == {"city": "Paris"}

    def test_tool_call_synthesizes_id_when_missing(self):
        fc = SimpleNamespace(id=None, name="f", args=None)
        events = normalize_message(_msg(tool_call=SimpleNamespace(function_calls=[fc])))
        assert isinstance(events[0], ToolCallEvent)
        assert events[0].id  # non-empty fallback id
        assert events[0].args == {}

    def test_session_resumption_update(self):
        upd = SimpleNamespace(new_handle="h-123", resumable=True)
        events = normalize_message(_msg(session_resumption_update=upd))
        assert len(events) == 1
        assert isinstance(events[0], SessionUpdateEvent)
        assert events[0].resumption_handle == "h-123"

    def test_go_away(self):
        events = normalize_message(_msg(go_away=SimpleNamespace(time_left="5s")))
        assert isinstance(events[0], GoAwayEvent)
        assert events[0].time_left == "5s"

    def test_empty_message_yields_nothing(self):
        assert normalize_message(_msg()) == []


# --------------------------------------------------------------------------- #
# Fake session / connector for client lifecycle and send-mapping tests
# --------------------------------------------------------------------------- #
class FakeLiveSession:
    def __init__(self, scripted=None):
        self.scripted = scripted or []
        self.sent_realtime = []
        self.tool_responses = []
        self.closed = False

    async def send_realtime_input(self, **kwargs):
        self.sent_realtime.append(kwargs)

    async def send_tool_response(self, **kwargs):
        self.tool_responses.append(kwargs)

    async def send_client_content(self, **kwargs):
        self.client_content = kwargs

    async def receive(self):
        # Mirror the real SDK: receive() drains one turn's messages then completes; a
        # subsequent call returns nothing (the client loops receive() across turns).
        batch, self.scripted = self.scripted, []
        for m in batch:
            yield m


class FakeConnector:
    """Stands in for client.aio.live.connect(...) -> async context manager."""

    def __init__(self, session):
        self.session = session
        self.enter_calls = []
        self.exited = False

    def __call__(self, *, model, config):
        self.enter_calls.append({"model": model, "config": config})
        return self

    async def __aenter__(self):
        return self.session

    async def __aexit__(self, *exc):
        self.exited = True
        return False


@pytest.fixture
def config():
    return RealtimeConfig(model="gemini-2.5-flash-live", voice="Puck")


class TestGeminiLiveClientLifecycle:
    @pytest.mark.asyncio
    async def test_connect_opens_session_via_connector(self, config):
        session = FakeLiveSession()
        connector = FakeConnector(session)
        client = GeminiLiveClient(connector=connector)

        await client.connect(config)

        assert connector.enter_calls[0]["model"] == "gemini-2.5-flash-live"
        assert client.connected is True

    @pytest.mark.asyncio
    async def test_connect_with_resume_handle_sets_session_resumption(self, config):
        connector = FakeConnector(FakeLiveSession())
        client = GeminiLiveClient(connector=connector)

        await client.connect(config, resume_handle="handle-xyz")

        live_config = connector.enter_calls[0]["config"]
        assert live_config.session_resumption.handle == "handle-xyz"

    @pytest.mark.asyncio
    async def test_reseed_history_maps_messages_to_send_client_content(self, config):
        from agentflow.core.state import Message

        session = FakeLiveSession()
        client = GeminiLiveClient(connector=FakeConnector(session))
        await client.connect(config)

        await client.reseed_history(
            [
                Message.text_message("hi", role="user"),
                Message.text_message("hello", role="assistant"),
            ]
        )

        turns = session.client_content["turns"]
        assert [t.role for t in turns] == ["user", "model"]
        assert session.client_content["turn_complete"] is True

    @pytest.mark.asyncio
    async def test_reseed_skips_system_and_tool_roles(self, config):
        from agentflow.core.state import Message

        session = FakeLiveSession()
        client = GeminiLiveClient(connector=FakeConnector(session))
        await client.connect(config)

        await client.reseed_history(
            [
                Message.text_message("be nice", role="system"),
                Message.text_message("hi", role="user"),
                Message.text_message("hello", role="assistant"),
            ]
        )

        turns = session.client_content["turns"]
        # system turn is dropped (set via system_instruction, not reseeded as dialogue)
        assert [t.role for t in turns] == ["user", "model"]

    @pytest.mark.asyncio
    async def test_close_exits_context_manager_and_is_idempotent(self, config):
        session = FakeLiveSession()
        connector = FakeConnector(session)
        client = GeminiLiveClient(connector=connector)
        await client.connect(config)

        await client.close()
        await client.close()  # must not raise

        assert connector.exited is True
        assert client.connected is False


class TestClientCredentialResolution:
    def test_api_key_from_env_is_used(self, monkeypatch):
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.setenv("GEMINI_API_KEY", "k-123")
        client = GeminiLiveClient()._build_client()
        assert client is not None  # genai.Client built without raising

    def test_explicit_api_key_overrides_env(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        client = GeminiLiveClient(api_key="explicit")._build_client()
        assert client is not None

    def test_missing_credentials_raises_clear_error(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        with pytest.raises(ValueError, match="GEMINI_API_KEY or GOOGLE_API_KEY"):
            GeminiLiveClient()._build_client()

    def test_vertex_without_project_raises(self, monkeypatch):
        monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
        with pytest.raises(ValueError, match="project"):
            GeminiLiveClient(use_vertex_ai=True)._build_client()

    def test_vertex_uses_project_and_location(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "proj-x")
        monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "europe-west1")
        client = GeminiLiveClient(use_vertex_ai=True)._build_client()
        assert client is not None


class TestBuildConnectConfig:
    @pytest.mark.asyncio
    async def test_voice_and_transcription_mapped_into_live_config(self, config):
        connector = FakeConnector(FakeLiveSession())
        client = GeminiLiveClient(connector=connector)
        await client.connect(config)

        live_config = connector.enter_calls[0]["config"]
        assert [m.value for m in live_config.response_modalities] == ["AUDIO"]
        assert live_config.speech_config is not None
        assert live_config.input_audio_transcription is not None
        assert live_config.output_audio_transcription is not None
        assert live_config.session_resumption is not None

    @pytest.mark.asyncio
    async def test_disabled_vad_sets_manual_activity_detection(self):
        from agentflow.core.realtime.base import RealtimeConfig, VADConfig

        cfg = RealtimeConfig(model="m", vad=VADConfig(enabled=False))
        connector = FakeConnector(FakeLiveSession())
        client = GeminiLiveClient(connector=connector)
        await client.connect(cfg)

        live_config = connector.enter_calls[0]["config"]
        assert live_config.realtime_input_config.automatic_activity_detection.disabled is True

    @pytest.mark.asyncio
    async def test_context_window_compression_enabled(self):
        from agentflow.core.realtime.base import RealtimeConfig

        cfg = RealtimeConfig(model="m", context_window_compression=True)
        connector = FakeConnector(FakeLiveSession())
        client = GeminiLiveClient(connector=connector)
        await client.connect(cfg)

        live_config = connector.enter_calls[0]["config"]
        assert live_config.context_window_compression is not None


class TestBuildConnectConfigTools:
    @pytest.mark.asyncio
    async def test_openai_tool_dicts_become_function_declarations(self):
        cfg = RealtimeConfig(
            model="m",
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                        },
                    },
                }
            ],
        )
        connector = FakeConnector(FakeLiveSession())
        client = GeminiLiveClient(connector=connector)
        await client.connect(cfg)

        tools = connector.enter_calls[0]["config"].tools
        assert len(tools) == 1  # one Tool wrapping the declarations
        decls = tools[0].function_declarations
        assert [d.name for d in decls] == ["get_weather"]

    @pytest.mark.asyncio
    async def test_non_dict_tools_pass_through_untouched(self):
        def raw_callable():
            """A raw callable tool."""

        cfg = RealtimeConfig(model="m", tools=[raw_callable])
        connector = FakeConnector(FakeLiveSession())
        client = GeminiLiveClient(connector=connector)
        await client.connect(cfg)

        tools = connector.enter_calls[0]["config"].tools
        assert tools == [raw_callable]


class TestGeminiLiveClientSend:
    @pytest.mark.asyncio
    async def test_send_audio_maps_to_blob_with_pcm_mime(self, config):
        session = FakeLiveSession()
        client = GeminiLiveClient(connector=FakeConnector(session))
        await client.connect(config)

        await client.send_audio(b"\xaa\xbb", sample_rate=16000)

        assert len(session.sent_realtime) == 1
        blob = session.sent_realtime[0]["audio"]
        assert blob.data == b"\xaa\xbb"
        assert blob.mime_type == "audio/pcm;rate=16000"

    @pytest.mark.asyncio
    async def test_activity_markers_map_to_realtime_input(self, config):
        session = FakeLiveSession()
        client = GeminiLiveClient(connector=FakeConnector(session))
        await client.connect(config)

        await client.send_activity_start()
        await client.send_activity_end()

        assert "activity_start" in session.sent_realtime[0]
        assert "activity_end" in session.sent_realtime[1]

    @pytest.mark.asyncio
    async def test_send_tool_response_maps_to_function_response(self, config):
        session = FakeLiveSession()
        client = GeminiLiveClient(connector=FakeConnector(session))
        await client.connect(config)

        await client.send_tool_response("call-1", "get_weather", {"temp": 20})

        responses = session.tool_responses[0]["function_responses"]
        assert responses[0].id == "call-1"
        assert responses[0].name == "get_weather"
        assert responses[0].response == {"temp": 20}

    @pytest.mark.asyncio
    async def test_send_text_maps_to_realtime_input(self, config):
        session = FakeLiveSession()
        client = GeminiLiveClient(connector=FakeConnector(session))
        await client.connect(config)

        await client.send_text("hello there")

        assert session.sent_realtime[0]["text"] == "hello there"

    @pytest.mark.asyncio
    async def test_send_before_connect_raises(self, config):
        client = GeminiLiveClient(connector=FakeConnector(FakeLiveSession()))
        with pytest.raises(RuntimeError):
            await client.send_audio(b"\x00", sample_rate=16000)


class TestGeminiLiveClientReceive:
    @pytest.mark.asyncio
    async def test_receive_normalizes_scripted_messages_in_order(self, config):
        scripted = [
            _msg(
                server_content=_server_content(
                    model_turn=SimpleNamespace(parts=[_audio_part(b"\x01")])
                )
            ),
            _msg(server_content=_server_content(turn_complete=True)),
        ]
        session = FakeLiveSession(scripted=scripted)
        client = GeminiLiveClient(connector=FakeConnector(session))
        await client.connect(config)

        events = [e async for e in client.receive()]

        assert isinstance(events[0], AudioDeltaEvent)
        assert isinstance(events[1], TurnCompleteEvent)

    @pytest.mark.asyncio
    async def test_receive_before_connect_raises(self, config):
        client = GeminiLiveClient(connector=FakeConnector(FakeLiveSession()))
        with pytest.raises(RuntimeError):
            async for _ in client.receive():
                pass

    @pytest.mark.asyncio
    async def test_receive_spans_multiple_turns(self, config):
        # Gemini's session.receive() completes per turn; the client must loop it so a
        # single receive() call streams events across several turns until the socket idles.
        class MultiTurnSession:
            def __init__(self, batches):
                self.batches = list(batches)

            async def send_realtime_input(self, **kw):
                pass

            async def receive(self):
                if self.batches:
                    for m in self.batches.pop(0):
                        yield m
                # exhausted -> yields nothing -> client loop stops

            async def __aenter__(self):
                return self

        turn1 = [
            _msg(server_content=_server_content(model_turn=SimpleNamespace(parts=[_audio_part(b"\x01")]))),
            _msg(server_content=_server_content(turn_complete=True)),
        ]
        turn2 = [
            _msg(server_content=_server_content(model_turn=SimpleNamespace(parts=[_audio_part(b"\x02")]))),
            _msg(server_content=_server_content(turn_complete=True)),
        ]
        session = MultiTurnSession([turn1, turn2])
        client = GeminiLiveClient(connector=FakeConnector(session))
        await client.connect(config)

        kinds = [e.type async for e in client.receive()]

        # both turns streamed from one receive() call, then it stops cleanly
        assert kinds == ["audio_delta", "turn_complete", "audio_delta", "turn_complete"]
