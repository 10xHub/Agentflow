"""Phase 3: the publisher taxonomy gains a REALTIME category + transcript content type
so all publisher backends inherit realtime telemetry without per-backend changes."""

from agentflow.runtime.publisher.events import ContentType, Event, EventType


def test_realtime_event_category_added():
    assert Event.REALTIME.value == "realtime"


def test_transcript_content_type_added():
    assert ContentType.TRANSCRIPT.value == "transcript"


def test_existing_members_unchanged():
    # Regression guard: realtime additions must not perturb the existing taxonomy.
    assert Event.TOOL_EXECUTION.value == "tool_execution"
    assert EventType.INTERRUPTED.value == "interrupted"
    assert ContentType.AUDIO.value == "audio"
