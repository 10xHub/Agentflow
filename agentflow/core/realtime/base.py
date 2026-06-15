"""Provider-neutral contracts for realtime (audio-to-audio) sessions.

These types are the seam between Agentflow and any realtime provider (Gemini Live
first, OpenAI Realtime later). Nothing here imports a provider SDK; provider clients
live under ``agentflow.core.realtime.providers`` and normalize their wire messages
into the :data:`RealtimeEvent` union defined below.
"""

from typing import Annotated, Any, Literal, Protocol, Union, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, field_validator


# Audio format facts for Gemini Live: input PCM16 mono @ 16kHz, output PCM16 @ 24kHz.
INPUT_SAMPLE_RATE = 16000
OUTPUT_SAMPLE_RATE = 24000


# --------------------------------------------------------------------------- #
# RealtimeEvent: the normalized event every downstream consumer reads.
# Discriminated union keyed on ``type`` (mirrors core.state ContentBlock).
# --------------------------------------------------------------------------- #
class AudioDeltaEvent(BaseModel):
    """A chunk of model audio output (PCM16)."""

    type: Literal["audio_delta"] = "audio_delta"
    data: bytes
    sample_rate: int = OUTPUT_SAMPLE_RATE


class InputTranscriptEvent(BaseModel):
    """Transcript of the user's speech (from the provider's input transcription)."""

    type: Literal["input_transcript"] = "input_transcript"
    text: str
    finished: bool = False


class OutputTranscriptEvent(BaseModel):
    """Transcript of the model's speech (from the provider's output transcription)."""

    type: Literal["output_transcript"] = "output_transcript"
    text: str
    finished: bool = False


class ToolCallEvent(BaseModel):
    """The provider is requesting a tool invocation."""

    type: Literal["tool_call"] = "tool_call"
    id: str
    name: str
    args: dict[str, Any] = Field(default_factory=dict)


class ToolResultEvent(BaseModel):
    """A tool finished executing (emitted for observability after the result is sent back)."""

    type: Literal["tool_result"] = "tool_result"
    id: str
    result: Any = None


class TurnCompleteEvent(BaseModel):
    """The model finished generating a turn."""

    type: Literal["turn_complete"] = "turn_complete"


class InterruptedEvent(BaseModel):
    """Barge-in: the user spoke over the model; the client should flush playback."""

    type: Literal["interrupted"] = "interrupted"


class SessionUpdateEvent(BaseModel):
    """The provider issued a session-resumption handle."""

    type: Literal["session_update"] = "session_update"
    resumption_handle: str | None = None


class GoAwayEvent(BaseModel):
    """The provider will close the socket soon; reconnect with the resumption handle."""

    type: Literal["go_away"] = "go_away"
    # Provider duration string (e.g. Gemini "5s"); passed through verbatim.
    time_left: str | None = None


class AgentChangedEvent(BaseModel):
    """The active agent/author changed (future multi-agent persona swap)."""

    type: Literal["agent_changed"] = "agent_changed"
    author: str


class ErrorEvent(BaseModel):
    """A normalized provider error. Fatal errors close the session; transient ones continue."""

    type: Literal["error"] = "error"
    code: str | None = None
    message: str
    fatal: bool = False


RealtimeEvent = Annotated[
    Union[
        AudioDeltaEvent,
        InputTranscriptEvent,
        OutputTranscriptEvent,
        ToolCallEvent,
        ToolResultEvent,
        TurnCompleteEvent,
        InterruptedEvent,
        SessionUpdateEvent,
        GoAwayEvent,
        AgentChangedEvent,
        ErrorEvent,
    ],
    Field(discriminator="type"),
]


# --------------------------------------------------------------------------- #
# RealtimeConfig: per-session value object, provider-neutral.
# --------------------------------------------------------------------------- #
ResponseModality = Literal["AUDIO", "TEXT"]


class VADConfig(BaseModel):
    """Voice-activity-detection settings. Disable for push-to-talk (manual activity)."""

    enabled: bool = True
    # Provider-neutral sensitivity hint; mapped per provider. None = provider default.
    start_sensitivity: str | None = None
    end_sensitivity: str | None = None
    prefix_padding_ms: int | None = None
    silence_duration_ms: int | None = None


class RealtimeConfig(BaseModel):
    """Per-session configuration handed to a :class:`RealtimeClient`.

    Gemini Live permits exactly one response modality per session; ``response_modalities``
    is validated to a single entry.
    """

    # validate_default so the default modality list is held to the same one-modality rule
    # as explicit values (otherwise a bad default silently bypasses the validator below).
    model_config = ConfigDict(validate_default=True)

    model: str
    response_modalities: list[ResponseModality] = Field(default_factory=lambda: ["AUDIO"])
    voice: str | None = None
    system_instruction: str | None = None
    input_audio_transcription: bool = True
    output_audio_transcription: bool = True
    vad: VADConfig = Field(default_factory=VADConfig)
    context_window_compression: bool = False
    session_resumption: bool = True
    tools: list[Any] | None = None
    tools_tags: list[str] | None = None

    @field_validator("response_modalities")
    @classmethod
    def _exactly_one_modality(cls, value: list[ResponseModality]) -> list[ResponseModality]:
        if len(value) != 1:
            raise ValueError(
                "response_modalities must contain exactly one modality per session "
                f"(got {value!r}); a realtime session is single-modality."
            )
        return value


# --------------------------------------------------------------------------- #
# RealtimeClient: provider Protocol. One implementation per provider.
# --------------------------------------------------------------------------- #
@runtime_checkable
class RealtimeClient(Protocol):
    """Protocol every provider client implements.

    Owns a single provider WebSocket for the lifetime of a session. ``receive()`` yields
    normalized :data:`RealtimeEvent`s; the send methods push input upstream.
    """

    async def connect(self, config: RealtimeConfig, resume_handle: str | None = None) -> None:
        """Open the provider socket for ``config``, optionally resuming ``resume_handle``."""
        ...

    async def send_audio(self, pcm: bytes, sample_rate: int) -> None:
        """Send a chunk of input audio (PCM16)."""
        ...

    async def send_text(self, text: str) -> None:
        """Send a text turn into the live session."""
        ...

    async def send_activity_start(self) -> None:
        """Manual-VAD / push-to-talk: mark the start of user activity."""
        ...

    async def send_activity_end(self) -> None:
        """Manual-VAD / push-to-talk: mark the end of user activity."""
        ...

    async def send_tool_response(self, call_id: str, name: str, result: Any) -> None:
        """Return a tool result to the model for ``call_id``."""
        ...

    async def reseed_history(self, messages: list[Any]) -> None:
        """Seed an existing conversation history into a fresh session (cross-session resume)."""
        ...

    def receive(self):  # -> AsyncIterator[RealtimeEvent]
        """Async-iterate normalized events from the provider."""
        ...

    async def close(self) -> None:
        """Close the provider socket. Must be safe to call more than once."""
        ...
