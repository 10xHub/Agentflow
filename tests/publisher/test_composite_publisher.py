import pytest
from unittest.mock import AsyncMock, MagicMock
from agentflow.runtime.publisher.composite_publisher import CompositePublisher
from agentflow.runtime.publisher.base_publisher import BasePublisher
from agentflow.runtime.publisher.events import EventModel


@pytest.mark.asyncio
async def test_composite_publisher_lifecycle():
    pub1 = AsyncMock(spec=BasePublisher)
    pub2 = AsyncMock(spec=BasePublisher)
    
    # Initialize
    composite = CompositePublisher([pub1])
    assert pub1 in composite._publishers
    assert pub2 not in composite._publishers

    # Add publisher
    composite.add_publisher(pub2)
    assert pub2 in composite._publishers

    # Publish
    event = MagicMock(spec=EventModel)
    await composite.publish(event)
    pub1.publish.assert_called_once_with(event)
    pub2.publish.assert_called_once_with(event)

    # Close
    await composite.close()
    pub1.close.assert_called_once()
    pub2.close.assert_called_once()

    # Sync close
    pub1.sync_close = MagicMock()
    pub2.sync_close = MagicMock()
    composite.sync_close()
    pub1.sync_close.assert_called_once()
    pub2.sync_close.assert_called_once()

    # Remove publisher
    composite.remove_publisher(pub1)
    assert pub1 not in composite._publishers
    assert pub2 in composite._publishers
