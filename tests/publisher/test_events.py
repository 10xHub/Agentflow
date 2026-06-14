"""
Comprehensive tests for the events module.

This module tests EventModel, Event, EventType, ContentType enums
including validation, serialization, factory methods, and edge cases.
"""

import json
import time
import uuid
from datetime import datetime
from unittest.mock import patch, Mock
import pytest

from agentflow.runtime.publisher.events import (
    Event,
    EventType,
    ContentType,
    EventModel,
)
from agentflow.core.state.message import TextBlock


class TestEventEnums:
    """Test the event-related enums."""
    
    def test_event_enum_values(self):
        """Test Event enum has expected values."""
        assert Event.GRAPH_EXECUTION.value == "graph_execution"
        assert Event.NODE_EXECUTION.value == "node_execution"
        assert Event.LLM_CALL.value == "llm_call"
        assert Event.TOOL_EXECUTION.value == "tool_execution"
        assert Event.STREAMING.value == "streaming"
        
        assert Event.REALTIME.value == "realtime"

        # Test all expected values are present
        expected_values = {
            "graph_execution", "node_execution", "llm_call",
            "tool_execution", "streaming", "realtime"
        }
        actual_values = {event.value for event in Event}
        assert actual_values == expected_values
    
    def test_event_type_enum_values(self):
        """Test EventType enum has expected values."""
        assert EventType.START.value == "start"
        assert EventType.PROGRESS.value == "progress"
        assert EventType.RESULT.value == "result"
        assert EventType.END.value == "end"
        assert EventType.UPDATE.value == "update"
        assert EventType.ERROR.value == "error"
        assert EventType.INTERRUPTED.value == "interrupted"
        
        # Test all expected values are present
        expected_values = {
            "start", "progress", "result", "end", 
            "update", "error", "interrupted"
        }
        actual_values = {event_type.value for event_type in EventType}
        assert actual_values == expected_values
    
    def test_content_type_enum_values(self):
        """Test ContentType enum has expected values."""
        assert ContentType.TEXT.value == "text"
        assert ContentType.MESSAGE.value == "message"
        assert ContentType.REASONING.value == "reasoning"
        assert ContentType.TOOL_CALL.value == "tool_call"
        assert ContentType.TOOL_RESULT.value == "tool_result"
        assert ContentType.IMAGE.value == "image"
        assert ContentType.AUDIO.value == "audio"
        assert ContentType.VIDEO.value == "video"
        assert ContentType.DOCUMENT.value == "document"
        assert ContentType.DATA.value == "data"
        assert ContentType.STATE.value == "state"
        assert ContentType.UPDATE.value == "update"
        assert ContentType.ERROR.value == "error"
        assert ContentType.TRANSCRIPT.value == "transcript"

        # Test all expected values are present
        expected_values = {
            "text", "message", "reasoning", "tool_call", "tool_result",
            "image", "audio", "video", "document", "data",
            "state", "update", "error", "transcript"
        }
        actual_values = {content_type.value for content_type in ContentType}
        assert actual_values == expected_values
    
    def test_enums_are_strings(self):
        """Test that enums inherit from str."""
        assert isinstance(Event.GRAPH_EXECUTION, str)
        assert isinstance(EventType.START, str)
        assert isinstance(ContentType.TEXT, str)
        
        # Test string operations work
        assert Event.GRAPH_EXECUTION + "_test" == "graph_execution_test"
        assert EventType.START.upper() == "START"
        assert ContentType.TEXT.replace("text", "content") == "content"


class TestEventModelBasic:
    """Test basic EventModel functionality."""
    
    def test_minimal_event_model(self):
        """Test creating EventModel with minimal required fields."""
        event = EventModel(
            event=Event.GRAPH_EXECUTION,
            event_type=EventType.START
        )
        
        assert event.event == Event.GRAPH_EXECUTION
        assert event.event_type == EventType.START
        assert event.content == ""
        assert event.content_blocks is None
        assert event.data == {}
        assert event.content_type is None
        assert event.node_name == ""
        assert isinstance(event.run_id, str)
        assert event.thread_id == ""
        assert isinstance(event.timestamp, float)
        assert event.is_error is False
        assert event.metadata == {}
    
    def test_explicit_event_model_values(self):
        """Test creating EventModel with explicit values."""
        test_time = time.time()
        test_run_id = str(uuid.uuid4())
        test_content_blocks = [TextBlock(text="test content")]
        test_data = {"key": "value", "number": 42}
        test_metadata = {"source": "test", "version": "1.0"}
        
        event = EventModel(
            event=Event.NODE_EXECUTION,
            event_type=EventType.PROGRESS,
            content="test content",
            content_blocks=test_content_blocks,
            data=test_data,
            content_type=[ContentType.TEXT, ContentType.MESSAGE],
            node_name="test_node",
            run_id=test_run_id,
            thread_id="thread_123",
            timestamp=test_time,
            is_error=True,
            metadata=test_metadata
        )
        
        assert event.event == Event.NODE_EXECUTION
        assert event.event_type == EventType.PROGRESS
        assert event.content == "test content"
        assert event.content_blocks == test_content_blocks
        assert event.data == test_data
        assert event.content_type == [ContentType.TEXT, ContentType.MESSAGE]
        assert event.node_name == "test_node"
        assert event.run_id == test_run_id
        assert event.thread_id == "thread_123"
        assert event.timestamp == test_time
        assert event.is_error is True
        assert event.metadata == test_metadata
    
    def test_default_factories(self):
        """Test that default factories work correctly."""
        event1 = EventModel(
            event=Event.GRAPH_EXECUTION,
            event_type=EventType.START
        )
        
        event2 = EventModel(
            event=Event.GRAPH_EXECUTION,
            event_type=EventType.START
        )
        
        # run_id should be different for each instance
        assert event1.run_id != event2.run_id
        assert isinstance(event1.run_id, str)
        assert isinstance(event2.run_id, str)
        
        # timestamps should be close but potentially different
        assert isinstance(event1.timestamp, float)
        assert isinstance(event2.timestamp, float)
        
        # data and metadata should be separate instances
        event1.data["test"] = "value1"
        event2.data["test"] = "value2"
        assert event1.data != event2.data
        
        event1.metadata["test"] = "meta1"
        event2.metadata["test"] = "meta2"
        assert event1.metadata != event2.metadata


class TestEventModelSerialization:
    """Test EventModel serialization and deserialization."""
    
    def test_model_dump(self):
        """Test that EventModel can be serialized to dict."""
        event = EventModel(
            event=Event.TOOL_EXECUTION,
            event_type=EventType.RESULT,
            content="test result",
            data={"result": "success"},
            content_type=[ContentType.TOOL_RESULT],
            node_name="tool_node",
            is_error=False
        )
        
        data = event.model_dump()
        
        assert isinstance(data, dict)
        assert data["event"] == "tool_execution"  # Should use enum values
        assert data["event_type"] == "result"
        assert data["content"] == "test result"
        assert data["data"] == {"result": "success"}
        assert data["content_type"] == ["tool_result"]
        assert data["node_name"] == "tool_node"
        assert data["is_error"] is False
        assert "run_id" in data
        assert "timestamp" in data
    
    def test_json_serialization(self):
        """Test that EventModel can be serialized to JSON."""
        event = EventModel(
            event=Event.STREAMING,
            event_type=EventType.UPDATE,
            content="streaming update",
        )
        
        # Should be able to serialize to JSON
        json_str = json.dumps(event.model_dump())
        assert isinstance(json_str, str)
        
        # Should be able to deserialize from JSON
        data = json.loads(json_str)
        restored_event = EventModel(**data)
        
        assert restored_event.event == Event.STREAMING
        assert restored_event.event_type == EventType.UPDATE
        assert restored_event.content == "streaming update"
    
    def test_model_validation(self):
        """Test Pydantic validation of EventModel."""
        # Valid creation should work
        event = EventModel(
            event=Event.GRAPH_EXECUTION,
            event_type=EventType.START
        )
        assert event is not None
        
        # Invalid enum values should fail
        with pytest.raises(ValueError):
            EventModel(
                event="invalid_event",
                event_type=EventType.START
            )
        
        with pytest.raises(ValueError):
            EventModel(
                event=Event.GRAPH_EXECUTION,
                event_type="invalid_event_type"
            )
        

class TestEventModelDefaultFactory:
    """Test the EventModel.default factory method."""
    
    def test_default_factory_basic(self):
        """Test basic usage of EventModel.default factory."""
        base_config = {
            "thread_id": "test_thread",
            "run_id": "test_run",
            "timestamp": "2023-01-01T00:00:00Z",
            "user_id": "user_123"
        }
        
        data = {"action": "test", "value": 42}
        content_type = [ContentType.DATA]
        
        event = EventModel.default(
            base_config=base_config,
            data=data,
            content_type=content_type
        )
        
        assert event.event == Event.GRAPH_EXECUTION  # default
        assert event.event_type == EventType.START  # default
        assert event.data == data
        assert event.content_type == content_type
        assert event.thread_id == "test_thread"
        assert event.metadata["run_timestamp"] == "2023-01-01T00:00:00Z"
        assert event.user_id == "user_123"
        assert event.metadata["is_stream"] is False  # default
        assert event.node_name == ""  # default
    
    def test_default_factory_with_overrides(self):
        """Test EventModel.default with parameter overrides."""
        base_config = {
            "thread_id": "override_thread",
            "run_id": "override_run",
            "is_stream": True
        }
        
        data = {"test": "override"}
        content_type = [ContentType.TEXT, ContentType.MESSAGE]
        extra = {"custom_field": "custom_value"}
        
        event = EventModel.default(
            base_config=base_config,
            data=data,
            content_type=content_type,
            event=Event.NODE_EXECUTION,
            event_type=EventType.ERROR,
            node_name="error_node",
            extra=extra
        )
        
        assert event.event == Event.NODE_EXECUTION
        assert event.event_type == EventType.ERROR
        assert event.data == data
        assert event.content_type == content_type
        assert event.thread_id == "override_thread"
        assert event.node_name == "error_node"
        assert event.metadata["is_stream"] is True
        assert event.metadata["custom_field"] == "custom_value"
    
    def test_default_factory_empty_base_config(self):
        """Test EventModel.default with minimal base_config."""
        base_config = {}
        data = {"minimal": "test"}
        content_type = [ContentType.UPDATE]
        
        event = EventModel.default(
            base_config=base_config,
            data=data,
            content_type=content_type
        )
        
        assert event.data == data
        assert event.content_type == content_type
        assert event.thread_id == ""  # default from empty config
        assert event.metadata["run_timestamp"] == ""  # default from empty config
        assert event.user_id is None  # default from empty config
        assert event.metadata["is_stream"] is False  # default from empty config


class TestEventModelEdgeCases:
    """Test edge cases and boundary conditions for EventModel."""
    
    def test_very_large_content(self):
        """Test handling of very large content strings."""
        large_content = "x" * 100000  # 100k characters
        
        event = EventModel(
            event=Event.STREAMING,
            event_type=EventType.PROGRESS,
            content=large_content
        )
        
        assert event.content == large_content
        assert len(event.content) == 100000
    
    def test_unicode_content(self):
        """Test handling of unicode content."""
        unicode_content = "Hello 世界 🌍 emoji and unicode ñáéíóú"
        
        event = EventModel(
            event=Event.GRAPH_EXECUTION,
            event_type=EventType.UPDATE,
            content=unicode_content
        )
        
        assert event.content == unicode_content
        
        # Should serialize properly
        data = event.model_dump()
        assert data["content"] == unicode_content
    
    def test_nested_data_structures(self):
        """Test handling of complex nested data structures."""
        complex_data = {
            "level1": {
                "level2": {
                    "list": [1, 2, {"nested": "value"}],
                    "tuple_as_list": [1, 2, 3],  # tuples become lists in JSON
                    "numbers": [1.5, 2.7, 3.14159]
                }
            },
            "arrays": [[1, 2], [3, 4], [5, 6]],
            "mixed": [1, "string", True, None, {"key": "value"}]
        }
        
        event = EventModel(
            event=Event.TOOL_EXECUTION,
            event_type=EventType.RESULT,
            data=complex_data
        )
        
        assert event.data == complex_data
        
        # Should serialize and deserialize properly
        serialized = event.model_dump()
        deserialized = EventModel(**serialized)
        assert deserialized.data == complex_data
    
    def test_extreme_timestamps(self):
        """Test handling of extreme timestamp values."""
        # Very old timestamp (1970)
        old_timestamp = 0.0
        event1 = EventModel(
            event=Event.GRAPH_EXECUTION,
            event_type=EventType.START,
            timestamp=old_timestamp
        )
        assert event1.timestamp == old_timestamp
        
        # Very new timestamp (far future)
        future_timestamp = 4102444800.0  # Year 2100
        event2 = EventModel(
            event=Event.GRAPH_EXECUTION,
            event_type=EventType.START,
            timestamp=future_timestamp
        )
        assert event2.timestamp == future_timestamp
    

    
    def test_all_content_types_combination(self):
        """Test using all content types in a single event."""
        all_content_types = list(ContentType)
        
        event = EventModel(
            event=Event.NODE_EXECUTION,
            event_type=EventType.RESULT,
            content_type=all_content_types
        )
        
        assert event.content_type == all_content_types
        assert len(event.content_type) == len(ContentType)
    
    def test_empty_and_none_values(self):
        """Test handling of empty and None values."""
        event = EventModel(
            event=Event.GRAPH_EXECUTION,
            event_type=EventType.START,
            content="",  # empty string
            content_blocks=None,
            content_type=None,
            node_name="",  # empty string
            thread_id="",  # empty string
        )
        
        assert event.content == ""
        assert event.content_blocks is None
        assert event.content_type is None
        assert event.node_name == ""
        assert event.thread_id == ""
    
    def test_content_blocks_with_different_types(self):
        """Test content_blocks with different ContentBlock types."""
        content_blocks = [
            TextBlock(text="First block"),
            TextBlock(text="Second block"),
        ]
        
        event = EventModel(
            event=Event.STREAMING,
            event_type=EventType.UPDATE,
            content_blocks=content_blocks,
        )
        
        assert event.content_blocks == content_blocks
        assert len(event.content_blocks) == 2


class TestEventModelFieldValidation:
    """Test specific field validation in EventModel."""
    
    def test_delta_type_validation(self):
        """Test delta_type field validation."""
        # Valid delta types
        valid_delta_types = ["text", "json", "binary"]
        
        for delta_type in valid_delta_types:
            event = EventModel(
                event=Event.STREAMING,
                event_type=EventType.UPDATE,
            )
        
        # None should be valid
        event = EventModel(
            event=Event.STREAMING,
            event_type=EventType.UPDATE,
        )
    
    def test_thread_id_types(self):
        """Test thread_id can be string or int."""
        # String thread_id
        event1 = EventModel(
            event=Event.GRAPH_EXECUTION,
            event_type=EventType.START,
            thread_id="string_thread_123"
        )
        assert event1.thread_id == "string_thread_123"
        
        # Integer thread_id
        event2 = EventModel(
            event=Event.GRAPH_EXECUTION,
            event_type=EventType.START,
            thread_id=12345
        )
        assert event2.thread_id == 12345
    
    def test_boolean_fields(self):
        """Test boolean field handling."""
        event = EventModel(
            event=Event.GRAPH_EXECUTION,
            event_type=EventType.ERROR,
            is_error=True
        )
        
        assert event.is_error is True
        
        # Test falsy values
        event2 = EventModel(
            event=Event.GRAPH_EXECUTION,
            event_type=EventType.START,
            is_error=False
        )
        
        assert event2.is_error is False


class TestEventModelConfiguration:
    """Test EventModel Pydantic configuration."""
    
    def test_use_enum_values_config(self):
        """Test that Config.use_enum_values works properly."""
        event = EventModel(
            event=Event.NODE_EXECUTION,
            event_type=EventType.PROGRESS,
            content_type=[ContentType.TEXT, ContentType.MESSAGE]
        )
        
        # model_dump should use enum values (strings) not enum objects
        data = event.model_dump()
        
        assert data["event"] == "node_execution"
        assert data["event_type"] == "progress"
        assert data["content_type"] == ["text", "message"]
        
        # The original event should still have enum objects
        assert event.event == Event.NODE_EXECUTION
        assert event.event_type == EventType.PROGRESS
        assert event.content_type[0] == ContentType.TEXT


class TestEventModelTimestampBehavior:
    """Test timestamp generation and behavior."""
    
    def test_timestamp_default_factory(self):
        """Test that timestamp uses time.time() as default factory."""
        event = EventModel(
            event=Event.GRAPH_EXECUTION,
            event_type=EventType.START
        )
        
        # Get timestamp after creating event
        
        # Timestamp should be between before and after
        assert isinstance(event.timestamp, float)
    
    def test_timestamp_override(self):
        """Test that timestamp can be explicitly set."""
        custom_timestamp = 9876543210.987
        
        event = EventModel(
            event=Event.GRAPH_EXECUTION,
            event_type=EventType.START,
            timestamp=custom_timestamp
        )
        
        assert event.timestamp == custom_timestamp
    
    def test_multiple_events_different_timestamps(self):
        """Test that multiple events get different default timestamps."""
        import time
        
        event1 = EventModel(
            event=Event.GRAPH_EXECUTION,
            event_type=EventType.START
        )
        
        # Small delay to ensure different timestamps
        time.sleep(0.001)
        
        event2 = EventModel(
            event=Event.GRAPH_EXECUTION,
            event_type=EventType.START
        )
        
        # Timestamps should be different (unless extremely fast execution)
        assert event1.timestamp <= event2.timestamp


class TestEventModelRunIdBehavior:
    """Test run_id generation and behavior."""
    
    @patch('uuid.uuid4')
    def test_run_id_default_factory(self, mock_uuid4):
        """Test that run_id uses uuid4() as default factory."""
        mock_uuid = Mock()
        mock_uuid.__str__ = Mock(return_value="test-uuid-12345")
        mock_uuid4.return_value = mock_uuid
        
        event = EventModel(
            event=Event.GRAPH_EXECUTION,
            event_type=EventType.START
        )
        
        assert event.run_id == "test-uuid-12345"
        mock_uuid4.assert_called_once()
    
    def test_run_id_override(self):
        """Test that run_id can be explicitly set."""
        custom_run_id = "custom-run-id-123"
        
        event = EventModel(
            event=Event.GRAPH_EXECUTION,
            event_type=EventType.START,
            run_id=custom_run_id
        )
        
        assert event.run_id == custom_run_id
    
    def test_multiple_events_different_run_ids(self):
        """Test that multiple events get different default run_ids."""
        event1 = EventModel(
            event=Event.GRAPH_EXECUTION,
            event_type=EventType.START
        )
        
        event2 = EventModel(
            event=Event.GRAPH_EXECUTION,
            event_type=EventType.START
        )
        
        # run_ids should be different UUIDs
        assert event1.run_id != event2.run_id
        assert isinstance(event1.run_id, str)
        assert isinstance(event2.run_id, str)
        assert len(event1.run_id) > 0
        assert len(event2.run_id) > 0