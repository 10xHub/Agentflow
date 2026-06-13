"""Console publisher implementation for debugging and testing.

This module provides a publisher that outputs events to the console for development
and debugging purposes.
"""

import logging
from typing import Any

from .base_publisher import BasePublisher
from .events import EventModel


logger = logging.getLogger("agentflow.publisher")


class ConsolePublisher(BasePublisher):
    """Publisher that writes events to the console for debugging and testing.

    This is a development/debugging publisher. It is opt-in: nothing wires it up
    unless you explicitly construct it and pass it to ``compile()``. **For
    production, use a real transport** (Redis, Kafka, RabbitMQ, or OTEL) rather
    than this one.

    By default events are written to stdout via ``print`` so they are visible in
    a quick script without any logging setup. In a server context, where writing
    to stdout is undesirable, set ``use_logger=True`` to route events through the
    ``agentflow.publisher`` logger at ``INFO`` level instead, so they respect your
    logging configuration.

    Attributes:
        format: Output format ('json' by default).
        include_timestamp: Whether to include timestamp (True by default).
        indent: Indentation for output (2 by default).
        use_logger: Emit via the ``agentflow.publisher`` logger instead of stdout
            (False by default).
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize the ConsolePublisher with the given configuration.

        Args:
            config: Configuration dictionary. Supported keys:
                - format: Output format (default: 'json').
                - include_timestamp: Whether to include timestamp (default: True).
                - indent: Indentation for output (default: 2).
                - use_logger: Emit via the logger instead of stdout
                  (default: False).
        """
        super().__init__(config or {})
        self.format = config.get("format", "json") if config else "json"
        self.include_timestamp = config.get("include_timestamp", True) if config else True
        self.indent = config.get("indent", 2) if config else 2
        self.use_logger = config.get("use_logger", False) if config else False

    async def publish(self, event: EventModel) -> Any:
        """Publish an event to the console.

        Writes to stdout by default, or emits via the ``agentflow.publisher``
        logger when ``use_logger=True`` was set in the config.

        Args:
            event: The event to publish.

        Returns:
            None

        Raises:
            RuntimeError: If publisher is closed.
        """
        if self._is_closed:
            raise RuntimeError("Cannot publish to closed ConsolePublisher")

        msg = f"{event.timestamp} -> Source: {event.node_name}.{event.event_type}:"
        msg += f"-> Payload: {event.data}"
        msg += f" -> {event.metadata}"
        if self.use_logger:
            logger.info(msg)
        else:
            print(msg)  # noqa: T201

    async def close(self):
        """Close the publisher and release any resources.

        ConsolePublisher does not require cleanup, but this method is provided for
        interface compatibility. This method is idempotent.
        """
        if not self._is_closed:
            self._is_closed = True
            logger.debug("ConsolePublisher closed")

    def sync_close(self):
        """Synchronously close the publisher and release any resources.

        ConsolePublisher does not require cleanup, but this method is provided for
        interface compatibility. This method is idempotent.
        """
        if not self._is_closed:
            self._is_closed = True
            logger.debug("ConsolePublisher sync closed")
