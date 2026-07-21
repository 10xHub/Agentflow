import logging

from injectq import Inject

from agentflow.runtime.publisher.base_publisher import BasePublisher
from agentflow.runtime.publisher.events import EventModel
from agentflow.utils.background_task_manager import BackgroundTaskManager


logger = logging.getLogger("agentflow.publisher")


async def _publish_event_task(
    event: EventModel,
    publisher: BasePublisher | None,
) -> None:
    """Publish an event asynchronously if publisher is configured.

    Args:
        event: The event to publish.
        publisher: The publisher instance, or None.
    """
    if publisher:
        try:
            await publisher.publish(event)
            logger.debug("Published event: %s", event)
        except Exception as e:
            logger.error("Failed to publish event: %s", e)


def publish_event(
    event: EventModel,
    publisher: BasePublisher | None = Inject[BasePublisher],
    task_manager: BackgroundTaskManager = Inject[BackgroundTaskManager],
) -> None:
    """Publish an event asynchronously using the background task manager.

    Args:
        event: The event to publish.
        publisher: The publisher instance (injected).
        task_manager: The background task manager (injected).
    """
    # No sink bound -> nothing to publish. Spawning a task per event just to
    # discover there is nowhere to send it cost a task and kept the event alive
    # for no reason, on the hot path of every node in every run.
    if publisher is None:
        return

    # Store the task to prevent it from being garbage collected. May return None
    # when the manager is shedding load (see BackgroundTaskManager.create_task).
    task_manager.create_task(_publish_event_task(event, publisher))
