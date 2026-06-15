"""The realtime package must re-export the Phase 1 public surface (SDK-first usage)."""

import agentflow.core.realtime as rt


def test_public_symbols_are_exported():
    expected = {
        "RealtimeClient",
        "RealtimeConfig",
        "RealtimeEvent",
        "VADConfig",
        "ReconnectConfig",
        "AudioDeltaEvent",
        "InputTranscriptEvent",
        "OutputTranscriptEvent",
        "ToolCallEvent",
        "ToolResultEvent",
        "TurnCompleteEvent",
        "InterruptedEvent",
        "SessionUpdateEvent",
        "GoAwayEvent",
        "AgentChangedEvent",
        "ErrorEvent",
        "LiveInputQueue",
        "LiveInput",
        "GeminiLiveClient",
        "normalize_message",
    }
    assert expected.issubset(set(rt.__all__))
    for name in expected:
        assert hasattr(rt, name), f"{name} missing from agentflow.core.realtime"


def test_queue_constructs_from_top_level_import():
    q = rt.LiveInputQueue()
    q.send_text("hi")
    item = q.get_nowait()
    assert isinstance(item, rt.LiveInput)
    assert item.kind == "text"
