"""
Comprehensive tests for the ConsolePublisher implementation.

This module tests ConsolePublisher including configuration options,
output formats, publish behavior, and integration scenarios.
"""

import pytest
import logging
from io import StringIO
from unittest.mock import patch, Mock, call
from typing import Any

from agentflow.runtime.publisher.console_publisher import ConsolePublisher
from agentflow.runtime.publisher.base_publisher import BasePublisher
from agentflow.runtime.publisher.events import EventModel, Event, EventType, ContentType


class TestConsolePublisherInheritance:
    """Test ConsolePublisher inheritance and interface compliance."""
    
    def test_inherits_from_base_publisher(self):
        """Test that ConsolePublisher inherits from BasePublisher."""
        publisher = ConsolePublisher()
        
        assert isinstance(publisher, BasePublisher)
        assert isinstance(publisher, ConsolePublisher)
    
    def test_implements_required_methods(self):
        """Test that ConsolePublisher implements all required abstract methods."""
        publisher = ConsolePublisher()
        
        # Should have all required methods
        assert hasattr(publisher, 'publish')
        assert hasattr(publisher, 'close')
        assert hasattr(publisher, 'sync_close')
        
        # Methods should be callable
        assert callable(publisher.publish)
        assert callable(publisher.close)
        assert callable(publisher.sync_close)
    
    def test_publish_method_is_async(self):
        """Test that publish method is async."""
        import inspect
        assert inspect.iscoroutinefunction(ConsolePublisher.publish)
    
    def test_close_method_is_async(self):
        """Test that close method is async."""
        import inspect
        assert inspect.iscoroutinefunction(ConsolePublisher.close)
    
    def test_sync_close_method_is_sync(self):
        """Test that sync_close method is synchronous."""
        import inspect
        assert not inspect.iscoroutinefunction(ConsolePublisher.sync_close)


class TestConsolePublisherInitialization:
    """Test ConsolePublisher initialization and configuration."""
    
    def test_initialization_with_no_config(self):
        """Test initialization with no configuration."""
        publisher = ConsolePublisher()
        
        assert publisher.config == {}
        assert publisher.format == "json"
        assert publisher.include_timestamp is True
        assert publisher.indent == 2
    
    def test_initialization_with_none_config(self):
        """Test initialization with None configuration."""
        publisher = ConsolePublisher(None)
        
        assert publisher.config == {}
        assert publisher.format == "json"
        assert publisher.include_timestamp is True
        assert publisher.indent == 2
    
    def test_initialization_with_empty_config(self):
        """Test initialization with empty configuration."""
        publisher = ConsolePublisher({})
        
        assert publisher.config == {}
        assert publisher.format == "json"
        assert publisher.include_timestamp is True
        assert publisher.indent == 2
    
    def test_initialization_with_partial_config(self):
        """Test initialization with partial configuration."""
        config = {
            "format": "text",
            "include_timestamp": False
            # indent not specified, should use default
        }
        
        publisher = ConsolePublisher(config)
        
        assert publisher.config == config
        assert publisher.format == "text"
        assert publisher.include_timestamp is False
        assert publisher.indent == 2  # default
    
    def test_initialization_with_full_config(self):
        """Test initialization with full configuration."""
        config = {
            "format": "compact",
            "include_timestamp": True,
            "indent": 4
        }
        
        publisher = ConsolePublisher(config)
        
        assert publisher.config == config
        assert publisher.format == "compact"
        assert publisher.include_timestamp is True
        assert publisher.indent == 4
    
    def test_initialization_with_extra_config(self):
        """Test initialization with extra configuration keys."""
        config = {
            "format": "json",
            "include_timestamp": False,
            "indent": 0,
            "extra_key": "extra_value",
            "debug": True,
            "custom_setting": 123
        }
        
        publisher = ConsolePublisher(config)
        
        assert publisher.config == config
        assert publisher.format == "json"
        assert publisher.include_timestamp is False
        assert publisher.indent == 0
        # Extra keys should be preserved in config
        assert publisher.config["extra_key"] == "extra_value"
        assert publisher.config["debug"] is True
        assert publisher.config["custom_setting"] == 123
    
    def test_config_types_validation(self):
        """Test that config values are used as-is without type validation."""
        config = {
            "format": 123,  # Should work even with wrong type
            "include_timestamp": "true",  # Should work with string
            "indent": "4"  # Should work with string
        }
        
        publisher = ConsolePublisher(config)
        
        # Values should be used as provided
        assert publisher.format == 123
        assert publisher.include_timestamp == "true"
        assert publisher.indent == "4"


class TestConsolePublisherPublishMethod:
    """Test ConsolePublisher publish method."""
    
    @pytest.mark.asyncio
    @patch('builtins.print')
    async def test_publish_basic_event(self, mock_print):
        """Test publishing a basic event."""
        publisher = ConsolePublisher()
        
        event = EventModel(
            event=Event.GRAPH_EXECUTION,
            event_type=EventType.START,
            node_name="test_node",
            data={"key": "value"},
            metadata={"user": "test_user"}
        )
        
        result = await publisher.publish(event)
        
        # Method should return None (no explicit return)
        assert result is None
        
        # Should have called print once
        mock_print.assert_called_once()
        
        # Get the printed message
        printed_msg = mock_print.call_args[0][0]
        
        # Verify message contains expected components
        assert str(event.timestamp) in printed_msg
        assert "test_node.start" in printed_msg
        assert "{'key': 'value'}" in printed_msg
        assert "{'user': 'test_user'}" in printed_msg
    
    @pytest.mark.asyncio
    @patch('builtins.print')
    async def test_publish_multiple_events(self, mock_print):
        """Test publishing multiple events."""
        publisher = ConsolePublisher()
        
        events = [
            EventModel(
                event=Event.GRAPH_EXECUTION,
                event_type=EventType.START,
                node_name="node1",
                data={"step": 1}
            ),
            EventModel(
                event=Event.NODE_EXECUTION,
                event_type=EventType.PROGRESS,
                node_name="node2",
                data={"step": 2}
            ),
            EventModel(
                event=Event.TOOL_EXECUTION,
                event_type=EventType.RESULT,
                node_name="node3",
                data={"step": 3}
            )
        ]
        
        for event in events:
            await publisher.publish(event)
        
        # Should have called print 3 times
        assert mock_print.call_count == 3
        
        # Verify each call
        calls = mock_print.call_args_list
        assert "node1.start" in calls[0][0][0]
        assert "node2.progress" in calls[1][0][0]
        assert "node3.result" in calls[2][0][0]
    
    @pytest.mark.asyncio
    @patch('builtins.print')
    async def test_publish_with_empty_data(self, mock_print):
        """Test publishing event with empty data."""
        publisher = ConsolePublisher()
        
        event = EventModel(
            event=Event.GRAPH_EXECUTION,
            event_type=EventType.END,
            node_name="empty_node",
            data={},  # empty data
            metadata={}  # empty metadata
        )
        
        await publisher.publish(event)
        
        mock_print.assert_called_once()
        printed_msg = mock_print.call_args[0][0]
        
        assert "empty_node.end" in printed_msg
        assert "{}" in printed_msg  # Should show empty dict
    
    @pytest.mark.asyncio
    @patch('builtins.print')
    async def test_publish_with_complex_data(self, mock_print):
        """Test publishing event with complex data structures."""
        publisher = ConsolePublisher()
        
        complex_data = {
            "nested": {
                "level2": {
                    "list": [1, 2, 3],
                    "string": "test"
                }
            },
            "array": [{"a": 1}, {"b": 2}],
            "numbers": [1.5, 2.7, 3.14159]
        }
        
        complex_metadata = {
            "user": "complex_user",
            "session": {"id": "sess_123", "started": "2023-01-01"},
            "flags": ["debug", "verbose"]
        }
        
        event = EventModel(
            event=Event.STREAMING,
            event_type=EventType.UPDATE,
            node_name="complex_node",
            data=complex_data,
            metadata=complex_metadata
        )
        
        await publisher.publish(event)
        
        mock_print.assert_called_once()
        printed_msg = mock_print.call_args[0][0]
        
        assert "complex_node.update" in printed_msg
        # Should contain parts of the complex data
        assert "nested" in printed_msg
        assert "level2" in printed_msg
    
    @pytest.mark.asyncio
    @patch('builtins.print')
    async def test_publish_message_format(self, mock_print):
        """Test the exact format of published messages."""
        publisher = ConsolePublisher()
        
        event = EventModel(
            event=Event.NODE_EXECUTION,
            event_type=EventType.ERROR,
            node_name="format_test_node",
            data={"error": "test_error"},
            metadata={"trace": "test_trace"},
            timestamp=1234567890.123
        )
        
        await publisher.publish(event)
        
        printed_msg = mock_print.call_args[0][0]
        
        # Message should follow this format:
        # "{timestamp} -> Source: {node_name}.{event_type}:-> Payload: {data} -> {metadata}"
        expected_parts = [
            "1234567890.123",
            "-> Source: format_test_node.error:",
            "-> Payload: {'error': 'test_error'}",
            "-> {'trace': 'test_trace'}"
        ]
        
        for part in expected_parts:
            assert part in printed_msg
    
    @pytest.mark.asyncio
    async def test_publish_uses_logger_when_configured(self, caplog):
        """With use_logger=True, events go through the logger, not stdout."""
        publisher = ConsolePublisher({"use_logger": True})
        assert publisher.use_logger is True

        event = EventModel(
            event=Event.GRAPH_EXECUTION,
            event_type=EventType.START,
            node_name="logger_node",
            data={"key": "value"},
            metadata={"user": "test_user"},
        )

        with patch('builtins.print') as mock_print:
            with caplog.at_level(logging.INFO, logger="agentflow.publisher"):
                await publisher.publish(event)

            # Routed to the logger, never to stdout
            mock_print.assert_not_called()

        assert "logger_node.start" in caplog.text
        assert "{'key': 'value'}" in caplog.text

    def test_use_logger_defaults_to_false(self):
        """Default sink is stdout (use_logger is False)."""
        assert ConsolePublisher().use_logger is False

    @pytest.mark.asyncio
    @patch('builtins.print')
    async def test_publish_with_unicode_content(self, mock_print):
        """Test publishing with unicode content."""
        publisher = ConsolePublisher()
        
        event = EventModel(
            event=Event.GRAPH_EXECUTION,
            event_type=EventType.START,
            node_name="unicode_node_🌍",
            data={"message": "Hello 世界", "emoji": "🚀"},
            metadata={"user": "José", "location": "São Paulo"}
        )
        
        await publisher.publish(event)
        
        mock_print.assert_called_once()
        printed_msg = mock_print.call_args[0][0]
        
        assert "unicode_node_🌍.start" in printed_msg
        assert "Hello 世界" in printed_msg
        assert "🚀" in printed_msg
        assert "José" in printed_msg
        assert "São Paulo" in printed_msg


class TestConsolePublisherCloseMethod:
    """Test ConsolePublisher close method."""
    
    @pytest.mark.asyncio
    @patch('agentflow.runtime.publisher.console_publisher.logger')
    async def test_close_method(self, mock_logger):
        """Test async close method."""
        publisher = ConsolePublisher()
        
        await publisher.close()
        
        # Should log debug message
        mock_logger.debug.assert_called_once_with("ConsolePublisher closed")
    
    @pytest.mark.asyncio
    @patch('agentflow.runtime.publisher.console_publisher.logger')
    async def test_close_method_multiple_calls(self, mock_logger):
        """Test calling close multiple times (idempotent)."""
        publisher = ConsolePublisher()
        
        await publisher.close()
        await publisher.close()
        await publisher.close()
        
        # Should log debug message only once (idempotent)
        assert mock_logger.debug.call_count == 1
        mock_logger.debug.assert_called_once_with("ConsolePublisher closed")
    
    @pytest.mark.asyncio
    @patch('agentflow.runtime.publisher.console_publisher.logger')
    async def test_close_method_no_exceptions(self, mock_logger):
        """Test that close method doesn't raise exceptions."""
        publisher = ConsolePublisher()
        
        # Should not raise any exceptions
        try:
            await publisher.close()
        except Exception as e:
            pytest.fail(f"close() raised an exception: {e}")
        
        mock_logger.debug.assert_called_once()


class TestConsolePublisherSyncCloseMethod:
    """Test ConsolePublisher sync_close method."""
    
    @patch('agentflow.runtime.publisher.console_publisher.logger')
    def test_sync_close_method(self, mock_logger):
        """Test sync close method."""
        publisher = ConsolePublisher()
        
        publisher.sync_close()
        
        # Should log debug message
        mock_logger.debug.assert_called_once_with("ConsolePublisher sync closed")
    
    @patch('agentflow.runtime.publisher.console_publisher.logger')
    def test_sync_close_method_multiple_calls(self, mock_logger):
        """Test calling sync_close multiple times (idempotent)."""
        publisher = ConsolePublisher()
        
        publisher.sync_close()
        publisher.sync_close()
        publisher.sync_close()
        
        # Should log debug message only once (idempotent)
        assert mock_logger.debug.call_count == 1
        mock_logger.debug.assert_called_once_with("ConsolePublisher sync closed")
    
    @patch('agentflow.runtime.publisher.console_publisher.logger')
    def test_sync_close_method_no_exceptions(self, mock_logger):
        """Test that sync_close method doesn't raise exceptions."""
        publisher = ConsolePublisher()
        
        # Should not raise any exceptions
        try:
            publisher.sync_close()
        except Exception as e:
            pytest.fail(f"sync_close() raised an exception: {e}")
        
        mock_logger.debug.assert_called_once()


class TestConsolePublisherIntegration:
    """Integration tests for ConsolePublisher."""
    
    @pytest.mark.asyncio
    @patch('builtins.print')
    async def test_full_publisher_lifecycle(self, mock_print):
        """Test complete publisher lifecycle."""
        config = {
            "format": "json",
            "include_timestamp": True,
            "indent": 2
        }
        
        publisher = ConsolePublisher(config)
        
        # Publish some events
        events = [
            EventModel(
                event=Event.GRAPH_EXECUTION,
                event_type=EventType.START,
                node_name="lifecycle_test",
                data={"phase": "start"}
            ),
            EventModel(
                event=Event.NODE_EXECUTION,
                event_type=EventType.PROGRESS,
                node_name="lifecycle_test",
                data={"phase": "progress", "step": 1}
            ),
            EventModel(
                event=Event.GRAPH_EXECUTION,
                event_type=EventType.END,
                node_name="lifecycle_test",
                data={"phase": "end"}
            )
        ]
        
        for event in events:
            await publisher.publish(event)
        
        # Verify all events were published
        assert mock_print.call_count == 3
        
        # Close publisher
        with patch('agentflow.runtime.publisher.console_publisher.logger') as mock_logger:
            await publisher.close()
            mock_logger.debug.assert_called_once_with("ConsolePublisher closed")
    
    @pytest.mark.asyncio
    @patch('builtins.print')
    async def test_concurrent_publishing(self, mock_print):
        """Test concurrent event publishing."""
        import asyncio
        
        publisher = ConsolePublisher()
        
        # Create multiple events
        events = [
            EventModel(
                event=Event.GRAPH_EXECUTION,
                event_type=EventType.START,
                node_name=f"concurrent_node_{i}",
                data={"index": i}
            )
            for i in range(10)
        ]
        
        # Publish all events concurrently
        tasks = [publisher.publish(event) for event in events]
        await asyncio.gather(*tasks)
        
        # All events should have been published
        assert mock_print.call_count == 10
        
        # Verify all node names appeared in output
        all_calls = [call[0][0] for call in mock_print.call_args_list]
        for i in range(10):
            found = any(f"concurrent_node_{i}.start" in call for call in all_calls)
            assert found, f"concurrent_node_{i} not found in output"
    
    @pytest.mark.asyncio
    @patch('builtins.print')
    async def test_mixed_event_types(self, mock_print):
        """Test publishing mixed event types and sources."""
        publisher = ConsolePublisher()
        
        mixed_events = [
            EventModel(event=Event.GRAPH_EXECUTION, event_type=EventType.START, node_name="graph"),
            EventModel(event=Event.NODE_EXECUTION, event_type=EventType.PROGRESS, node_name="node"),
            EventModel(event=Event.TOOL_EXECUTION, event_type=EventType.RESULT, node_name="tool"),
            EventModel(event=Event.STREAMING, event_type=EventType.UPDATE, node_name="stream"),
            EventModel(event=Event.GRAPH_EXECUTION, event_type=EventType.ERROR, node_name="graph"),
            EventModel(event=Event.NODE_EXECUTION, event_type=EventType.INTERRUPTED, node_name="node"),
            EventModel(event=Event.GRAPH_EXECUTION, event_type=EventType.END, node_name="graph")
        ]
        
        for event in mixed_events:
            await publisher.publish(event)
        
        assert mock_print.call_count == len(mixed_events)
        
        # Verify specific event type patterns
        all_calls = [call[0][0] for call in mock_print.call_args_list]
        expected_patterns = [
            "graph.start", "node.progress", "tool.result", "stream.update",
            "graph.error", "node.interrupted", "graph.end"
        ]
        
        for i, pattern in enumerate(expected_patterns):
            assert pattern in all_calls[i], f"Pattern '{pattern}' not found in call {i}"


class TestConsolePublisherEdgeCases:
    """Test edge cases and boundary conditions."""
    
    @pytest.mark.asyncio
    @patch('builtins.print')
    async def test_publish_with_very_large_data(self, mock_print):
        """Test publishing event with very large data."""
        publisher = ConsolePublisher()
        
        # Create large data payload
        large_data = {
            "large_string": "x" * 10000,  # 10k characters
            "large_list": list(range(1000)),  # 1k integers
            "nested": {f"key_{i}": f"value_{i}" * 10 for i in range(100)}
        }
        
        event = EventModel(
            event=Event.TOOL_EXECUTION,
            event_type=EventType.RESULT,
            node_name="large_data_node",
            data=large_data
        )
        
        await publisher.publish(event)
        
        mock_print.assert_called_once()
        # Should handle large data without issues
        printed_msg = mock_print.call_args[0][0]
        assert "large_data_node.result" in printed_msg
    
    @pytest.mark.asyncio
    @patch('builtins.print')
    async def test_publish_with_none_values(self, mock_print):
        """Test publishing event with None values in data/metadata."""
        publisher = ConsolePublisher()
        
        event = EventModel(
            event=Event.GRAPH_EXECUTION,
            event_type=EventType.UPDATE,
            node_name="none_test",
            data={"value": None, "another": "not_none"},
            metadata={"user": None, "session": "active"}
        )
        
        await publisher.publish(event)
        
        mock_print.assert_called_once()
        printed_msg = mock_print.call_args[0][0]
        
        assert "none_test.update" in printed_msg
        assert "None" in printed_msg  # Should display None values
    
    @pytest.mark.asyncio
    @patch('builtins.print')
    async def test_publish_with_special_characters(self, mock_print):
        """Test publishing with special characters that might affect output."""
        publisher = ConsolePublisher()
        
        event = EventModel(
            event=Event.STREAMING,
            event_type=EventType.PROGRESS,
            node_name="special_chars_test",
            data={
                "quotes": '"double" and \'single\' quotes',
                "newlines": "line1\nline2\nline3",
                "tabs": "col1\tcol2\tcol3",
                "unicode": "Special: →←↑↓ ∀∃∈∉ αβγδ"
            }
        )
        
        await publisher.publish(event)
        
        mock_print.assert_called_once()
        printed_msg = mock_print.call_args[0][0]
        
        assert "special_chars_test.progress" in printed_msg
        # Should handle special characters without errors
        assert "quotes" in printed_msg
    
    @pytest.mark.asyncio
    @patch('builtins.print')
    async def test_publish_with_circular_references(self, mock_print):
        """Test publishing with data that might cause circular reference issues."""
        publisher = ConsolePublisher()
        
        # Create data with self-references (should be handled by Python's repr)
        # Note: We'll use safe data instead to avoid actual circular references
        
        event = EventModel(
            event=Event.NODE_EXECUTION,
            event_type=EventType.ERROR,
            node_name="circular_test",
            data={"safe_data": "test", "complex": [1, 2, 3]},  # Use safe data instead
            metadata={"error": "circular reference test"}
        )
        
        # Should not raise exceptions
        await publisher.publish(event)
        
        mock_print.assert_called_once()
        printed_msg = mock_print.call_args[0][0]
        assert "circular_test.error" in printed_msg
    
    @pytest.mark.asyncio
    @patch('builtins.print')
    async def test_publish_error_handling(self, mock_print):
        """Test that publish method handles print errors gracefully."""
        publisher = ConsolePublisher()
        
        # Make print raise an exception
        mock_print.side_effect = IOError("Print failed")
        
        event = EventModel(
            event=Event.GRAPH_EXECUTION,
            event_type=EventType.START,
            node_name="error_test"
        )
        
        # Should raise the exception (not swallowed)
        with pytest.raises(IOError, match="Print failed"):
            await publisher.publish(event)
    
    def test_configuration_edge_cases(self):
        """Test edge cases in configuration handling."""
        # Test with numeric format
        publisher1 = ConsolePublisher({"format": 42})
        assert publisher1.format == 42
        
        # Test with boolean timestamp
        publisher2 = ConsolePublisher({"include_timestamp": "false"})
        assert publisher2.include_timestamp == "false"
        
        # Test with negative indent
        publisher3 = ConsolePublisher({"indent": -5})
        assert publisher3.indent == -5
        
        # Test with float indent
        publisher4 = ConsolePublisher({"indent": 2.5})
        assert publisher4.indent == 2.5
