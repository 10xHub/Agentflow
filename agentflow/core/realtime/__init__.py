"""Realtime (audio-to-audio) runtime primitives.

Provider-neutral contracts (:data:`RealtimeEvent`, :class:`RealtimeConfig`,
:class:`RealtimeClient`), the upstream :class:`LiveInputQueue`, and the Gemini Live
provider client. Provider SDK imports are lazy, so importing this package never pulls
the ``realtime`` optional dependency.
"""

from .base import (
    AgentChangedEvent,
    AudioDeltaEvent,
    ErrorEvent,
    GoAwayEvent,
    InputTranscriptEvent,
    InterruptedEvent,
    OutputTranscriptEvent,
    RealtimeClient,
    RealtimeConfig,
    RealtimeEvent,
    SessionUpdateEvent,
    ToolCallEvent,
    ToolResultEvent,
    TurnCompleteEvent,
    VADConfig,
)
from .providers.gemini_live import GeminiLiveClient, normalize_message
from .queue import LiveInput, LiveInputKind, LiveInputQueue


__all__ = [
    "AgentChangedEvent",
    "AudioDeltaEvent",
    "ErrorEvent",
    "GeminiLiveClient",
    "GoAwayEvent",
    "InputTranscriptEvent",
    "InterruptedEvent",
    "LiveInput",
    "LiveInputKind",
    "LiveInputQueue",
    "OutputTranscriptEvent",
    "RealtimeClient",
    "RealtimeConfig",
    "RealtimeEvent",
    "SessionUpdateEvent",
    "ToolCallEvent",
    "ToolResultEvent",
    "TurnCompleteEvent",
    "VADConfig",
    "normalize_message",
]
