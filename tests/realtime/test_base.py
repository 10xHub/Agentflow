"""Unit tests for provider-neutral realtime contracts (agentflow.core.realtime.base)."""

import pytest
from pydantic import TypeAdapter, ValidationError

from agentflow.core.realtime.base import (
    AgentChangedEvent,
    AudioDeltaEvent,
    ErrorEvent,
    GoAwayEvent,
    InputTranscriptEvent,
    InterruptedEvent,
    OutputTranscriptEvent,
    RealtimeConfig,
    RealtimeEvent,
    SessionUpdateEvent,
    ToolCallEvent,
    ToolResultEvent,
    TurnCompleteEvent,
)


class TestRealtimeEventDiscrimination:
    def test_audio_delta_carries_pcm_and_sample_rate(self):
        event = AudioDeltaEvent(data=b"\x00\x01", sample_rate=24000)
        assert event.type == "audio_delta"
        assert event.data == b"\x00\x01"
        assert event.sample_rate == 24000

    def test_input_and_output_transcripts_track_finished_flag(self):
        user = InputTranscriptEvent(text="hello", finished=False)
        model = OutputTranscriptEvent(text="hi there", finished=True)
        assert user.type == "input_transcript"
        assert user.finished is False
        assert model.type == "output_transcript"
        assert model.finished is True

    def test_tool_call_and_result_pair_by_id(self):
        call = ToolCallEvent(id="c1", name="get_weather", args={"city": "Paris"})
        result = ToolResultEvent(id="c1", result={"temp": 20})
        assert call.type == "tool_call"
        assert call.name == "get_weather"
        assert call.args == {"city": "Paris"}
        assert result.type == "tool_result"
        assert result.id == call.id

    def test_lifecycle_events_have_no_required_payload(self):
        assert TurnCompleteEvent().type == "turn_complete"
        assert InterruptedEvent().type == "interrupted"

    def test_session_and_goaway_carry_resume_metadata(self):
        update = SessionUpdateEvent(resumption_handle="abc123")
        goaway = GoAwayEvent(time_left="5s")
        assert update.type == "session_update"
        assert update.resumption_handle == "abc123"
        assert goaway.type == "go_away"
        assert goaway.time_left == "5s"

    def test_agent_changed_and_error(self):
        changed = AgentChangedEvent(author="planner")
        err = ErrorEvent(code="quota", message="rate limited")
        assert changed.type == "agent_changed"
        assert changed.author == "planner"
        assert err.type == "error"
        assert err.code == "quota"
        assert err.message == "rate limited"

    def test_union_deserializes_by_type_discriminator(self):
        adapter = TypeAdapter(RealtimeEvent)
        parsed = adapter.validate_python({"type": "interrupted"})
        assert isinstance(parsed, InterruptedEvent)
        parsed_call = adapter.validate_python(
            {"type": "tool_call", "id": "x", "name": "f", "args": {}}
        )
        assert isinstance(parsed_call, ToolCallEvent)

    def test_union_rejects_unknown_type(self):
        adapter = TypeAdapter(RealtimeEvent)
        with pytest.raises(ValidationError):
            adapter.validate_python({"type": "not_a_real_event"})


class TestRealtimeConfig:
    def test_minimal_config_requires_only_model(self):
        config = RealtimeConfig(model="gemini-2.5-flash-live")
        assert config.model == "gemini-2.5-flash-live"
        # Audio-out by default for an audio agent.
        assert config.response_modalities == ["AUDIO"]

    def test_single_response_modality_enforced(self):
        # Gemini Live allows exactly one response modality per session.
        with pytest.raises(ValidationError):
            RealtimeConfig(model="m", response_modalities=["AUDIO", "TEXT"])

    def test_full_config_round_trip(self):
        config = RealtimeConfig(
            model="gemini-2.5-flash-live",
            response_modalities=["TEXT"],
            voice="Puck",
            system_instruction="be terse",
            input_audio_transcription=True,
            output_audio_transcription=True,
            session_resumption=True,
            tools_tags=["weather"],
        )
        assert config.voice == "Puck"
        assert config.input_audio_transcription is True
        assert config.tools_tags == ["weather"]
