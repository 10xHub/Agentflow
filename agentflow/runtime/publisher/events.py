"""
Event and streaming primitives for agent graph execution.

This module defines event types, content types, and the EventModel for structured streaming
of execution updates, tool calls, state changes, messages, and errors in agent graphs.

Classes:
    Event: Enum for event sources (graph, node, tool, streaming).
    EventType: Enum for event phases (start, progress, result, end, etc.).
    ContentType: Enum for semantic content types (text, message, tool_call, etc.).
    EventModel: Structured event chunk for streaming agent graph execution.
"""

import uuid
from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from agentflow.core.state.message_block import ContentBlock


class Event(StrEnum):
    """Enum for event sources in agent graph execution.

    Values:
        GRAPH_EXECUTION: Event from graph execution.
        NODE_EXECUTION: Event from node execution.
        TOOL_EXECUTION: Event from tool execution.
        STREAMING: Event from streaming updates.
    """

    GRAPH_EXECUTION = "graph_execution"
    NODE_EXECUTION = "node_execution"
    LLM_CALL = "llm_call"
    TOOL_EXECUTION = "tool_execution"
    STREAMING = "streaming"
    REALTIME = "realtime"


class EventType(StrEnum):
    """Enum for event phases in agent graph execution.

    Values:
        START: Event marks start of execution.
        PROGRESS: Event marks progress update.
        RESULT: Event marks result produced.
        END: Event marks end of execution.
        UPDATE: Event marks update.
        ERROR: Event marks error.
        INTERRUPTED: Event marks interruption.
    """

    START = "start"
    PROGRESS = "progress"
    RESULT = "result"
    END = "end"
    UPDATE = "update"
    ERROR = "error"
    INTERRUPTED = "interrupted"


class ContentType(StrEnum):
    """Enum for semantic content types in agent graph streaming.

    Values:
        TEXT: Textual content.
        MESSAGE: Message content.
        REASONING: Reasoning content.
        TOOL_CALL: Tool call content.
        TOOL_RESULT: Tool result content.
        IMAGE: Image content.
        AUDIO: Audio content.
        VIDEO: Video content.
        DOCUMENT: Document content.
        DATA: Data content.
        STATE: State content.
        UPDATE: Update content.
        ERROR: Error content.
    """

    TEXT = "text"
    MESSAGE = "message"
    REASONING = "reasoning"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    IMAGE = "image"
    AUDIO = "audio"
    TRANSCRIPT = "transcript"
    VIDEO = "video"
    DOCUMENT = "document"
    DATA = "data"
    STATE = "state"
    UPDATE = "update"
    ERROR = "error"


class EventModel(BaseModel):
    """
    Structured event chunk for streaming agent graph execution.

    Represents a chunk of streamed data with event and content semantics, supporting both delta
    (incremental) and full content. Used for real-time streaming of execution updates, tool calls,
    state changes, messages, and errors.

    Attributes:
        event: Type of the event source.
        event_type: Phase of the event (start, progress, end, update).
        content: Streamed textual content.
        content_blocks: Structured content blocks for multimodal streaming.
        delta: True if this is a delta update (incremental).
        delta_type: Type of delta when delta=True.
        block_index: Index of the content block this chunk applies to.
        chunk_index: Per-block chunk index for ordering.
        byte_offset: Byte offset for binary/media streaming.
        data: Additional structured data.
        content_type: Semantic type of content.
        sequence_id: Monotonic sequence ID for stream ordering.
        node_name: Name of the node producing this chunk.
        run_id: Unique ID for this stream/run.
        thread_id: Thread ID for this execution.
        timestamp: UNIX timestamp of when chunk was created.
        is_error: Marks this chunk as representing an error state.
        metadata: Optional metadata for consumers.
    """

    # Event metadata
    event: Event = Field(..., description="Type of the event source")
    event_type: EventType = Field(
        ..., description="Phase of the event (start, progress, end, update)"
    )

    # Streamed content
    content: str = Field(default="", description="Streamed textual content")
    # Structured content blocks for multimodal/structured streaming
    content_blocks: list[ContentBlock] | None = Field(
        default=None, description="Structured content blocks carried by this event"
    )

    # Data payload
    data: dict[str, Any] = Field(default_factory=dict, description="Additional structured data")

    # Metadata
    content_type: list[ContentType] | None = Field(
        default=None, description="Semantic type of content"
    )
    node_name: str = Field(default="", description="Name of the node producing this chunk")
    run_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()), description="Unique ID for this stream/run"
    )
    thread_id: str | int = Field(default="", description="Thread ID for this execution")
    user_id: str | int | None = Field(
        default=None, description="User ID associated with this execution"
    )
    timestamp: float = Field(
        default_factory=datetime.now().timestamp,
        description="UNIX timestamp of when chunk was created",
    )
    is_error: bool = Field(
        default=False, description="Marks this chunk as representing an error state"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Optional metadata for consumers"
    )

    model_config = ConfigDict(use_enum_values=True)

    @classmethod
    def default(
        cls,
        base_config: dict,
        data: dict[str, Any],
        content_type: list[ContentType],
        event: Event = Event.GRAPH_EXECUTION,
        event_type=EventType.START,
        node_name: str = "",
        extra: dict[str, Any] | None = None,
    ) -> "EventModel":
        """Create a default EventModel instance with minimal required fields.

        Args:
            base_config: Base configuration for the event (thread/run/timestamp/user).
            data: Structured data payload.
            content_type: Semantic type(s) of content.
            event: Event source type (default: GRAPH_EXECUTION).
            event_type: Event phase (default: START).
            node_name: Name of the node producing the event.
            extra: Additional metadata.

        Returns:
            EventModel: The created event model instance.
        """
        thread_id = base_config.get("thread_id", "")
        run_id = base_config.get("run_id", "")
        user_id = base_config.get("user_id")

        metadata = {
            "run_timestamp": base_config.get("timestamp", ""),
            "is_stream": base_config.get("is_stream", False),
        }
        if extra:
            metadata.update(extra)
        return cls(
            event=event,
            event_type=event_type,
            content_type=content_type,
            data=data,
            thread_id=thread_id,
            node_name=node_name,
            run_id=run_id,
            user_id=user_id,
            metadata=metadata,
        )

    @classmethod
    def stream(
        cls,
        base_config: dict,
        node_name: str = "",
        extra: dict[str, Any] | None = None,
    ) -> "EventModel":
        """Create a default EventModel instance for streaming updates.

        Args:
            base_config: Base configuration for the event (thread/run/timestamp/user).
            node_name: Name of the node producing the event.
            extra: Additional metadata.

        Returns:
            EventModel: The created event model instance for streaming.
        """
        thread_id = base_config.get("thread_id", "")
        run_id = base_config.get("run_id", "")
        user_id = base_config.get("user_id")

        metadata = {
            "run_timestamp": base_config.get("timestamp", ""),
            "is_stream": base_config.get("is_stream", False),
        }
        if extra:
            metadata.update(extra)
        return cls(
            event=Event.STREAMING,
            event_type=EventType.UPDATE,
            content_type=[ContentType.TEXT, ContentType.REASONING],
            data={},
            thread_id=thread_id,
            user_id=user_id,
            node_name=node_name,
            run_id=run_id,
            metadata=metadata,
        )
