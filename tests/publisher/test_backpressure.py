"""Publisher backlog must be bounded (audit H7).

Every graph event spawned a fire-and-forget task into an unbounded set. A sink
that was slow or down (unreachable broker) grew that set -- and every event object
it retained -- until the process died of memory exhaustion. There was no drop
policy and no backpressure.
"""

import asyncio

import pytest

from agentflow.utils.background_task_manager import BackgroundTaskManager


class TestBackpressure:
    @pytest.mark.asyncio
    async def test_tasks_are_dropped_once_the_cap_is_reached(self):
        mgr = BackgroundTaskManager(max_pending_tasks=3)
        gate = asyncio.Event()

        async def blocked():
            await gate.wait()

        accepted = [mgr.create_task(blocked(), name=f"t{i}") for i in range(3)]
        assert all(t is not None for t in accepted)
        assert mgr.pending_count == 3

        # The sink is not keeping up; the next task is shed, not queued.
        dropped = mgr.create_task(blocked(), name="overflow")
        assert dropped is None
        assert mgr.dropped_count == 1
        assert mgr.pending_count == 3  # did not grow

        gate.set()
        await mgr.wait_for_all(timeout=2.0)

    @pytest.mark.asyncio
    async def test_capacity_frees_up_as_tasks_complete(self):
        mgr = BackgroundTaskManager(max_pending_tasks=2)

        async def quick():
            return None

        assert mgr.create_task(quick()) is not None
        assert mgr.create_task(quick()) is not None
        await mgr.wait_for_all(timeout=2.0)

        # Once drained, new work is accepted again.
        assert mgr.create_task(quick()) is not None
        await mgr.wait_for_all(timeout=2.0)

    @pytest.mark.asyncio
    async def test_dropping_does_not_leak_an_unawaited_coroutine(self, recwarn):
        mgr = BackgroundTaskManager(max_pending_tasks=1)
        gate = asyncio.Event()

        async def blocked():
            await gate.wait()

        mgr.create_task(blocked())
        assert mgr.create_task(blocked()) is None  # dropped

        gate.set()
        await mgr.wait_for_all(timeout=2.0)

        # The dropped coroutine must be closed, not left to raise
        # "coroutine was never awaited".
        assert not [w for w in recwarn if "never awaited" in str(w.message)]

    @pytest.mark.asyncio
    async def test_cap_can_be_disabled(self):
        mgr = BackgroundTaskManager(max_pending_tasks=0)
        gate = asyncio.Event()

        async def blocked():
            await gate.wait()

        for _ in range(20):
            assert mgr.create_task(blocked()) is not None
        assert mgr.pending_count == 20
        assert mgr.dropped_count == 0

        gate.set()
        await mgr.wait_for_all(timeout=2.0)


class TestNoPublisherShortCircuit:
    def test_publish_event_does_nothing_without_a_publisher(self):
        """No sink bound -> don't build a task per event on the hot path."""
        from agentflow.runtime.publisher.publish import publish_event

        mgr = BackgroundTaskManager()
        publish_event(event=object(), publisher=None, task_manager=mgr)  # type: ignore[arg-type]

        assert mgr.pending_count == 0
