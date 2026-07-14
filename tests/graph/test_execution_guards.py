"""Tests for execution guards: deadlines and mid-node cancellation (audit B4).

These cover the two failures that were previously unreachable from the loop:
a node/tool that hangs forever, and a stop request that arrives while a node is
still running (so the between-nodes stop check never gets a chance to fire).
"""

import asyncio

import pytest

from agentflow.core.exceptions import GraphStopRequested, NodeTimeoutError
from agentflow.core.graph.utils.guards import execute_with_guards, resolve_timeout


class TestResolveTimeout:
    def test_uses_default_when_key_absent(self):
        assert resolve_timeout({}, "node_timeout", 900.0) == 900.0

    def test_explicit_none_disables_timeout(self):
        assert resolve_timeout({"node_timeout": None}, "node_timeout", 900.0) is None

    def test_zero_disables_timeout(self):
        assert resolve_timeout({"node_timeout": 0}, "node_timeout", 900.0) is None

    def test_explicit_value_overrides_default(self):
        assert resolve_timeout({"node_timeout": 5}, "node_timeout", 900.0) == 5.0

    def test_invalid_value_falls_back_to_default(self):
        assert resolve_timeout({"node_timeout": "abc"}, "node_timeout", 900.0) == 900.0


class TestExecuteWithGuards:
    @pytest.mark.asyncio
    async def test_returns_result_when_work_completes(self):
        async def work():
            return "done"

        assert await execute_with_guards(work(), timeout=5) == "done"

    @pytest.mark.asyncio
    async def test_propagates_work_exception(self):
        async def work():
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            await execute_with_guards(work(), timeout=5)

    @pytest.mark.asyncio
    async def test_hung_work_is_cancelled_at_deadline(self):
        """The core B4 case: without this the run blocks forever."""
        started = asyncio.Event()
        cancelled = asyncio.Event()

        async def hangs_forever():
            started.set()
            try:
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                cancelled.set()
                raise

        with pytest.raises(NodeTimeoutError):
            await execute_with_guards(
                hangs_forever(),
                timeout=0.05,
                on_timeout=lambda: NodeTimeoutError(message="timed out"),
            )

        assert started.is_set()
        # The hung work must actually be cancelled, not merely abandoned.
        assert cancelled.is_set()

    @pytest.mark.asyncio
    async def test_no_timeout_means_work_can_run_long(self):
        async def slow():
            await asyncio.sleep(0.05)
            return "finished"

        assert await execute_with_guards(slow(), timeout=None) == "finished"

    @pytest.mark.asyncio
    async def test_stop_request_cancels_running_work(self):
        """A stop arriving mid-node must actually cancel it, not wait it out."""
        started = asyncio.Event()
        cancelled = asyncio.Event()
        stop_flag = {"stop": False}

        async def long_node():
            started.set()
            try:
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                cancelled.set()
                raise

        async def stop_check():
            return stop_flag["stop"]

        async def request_stop_soon():
            await started.wait()
            stop_flag["stop"] = True

        asyncio.create_task(request_stop_soon())

        with pytest.raises(GraphStopRequested):
            await execute_with_guards(
                long_node(),
                timeout=None,  # no deadline: the stop alone must break the hang
                stop_check=stop_check,
                poll_interval=0.01,
                on_stop=lambda: GraphStopRequested("node_a"),
            )

        assert cancelled.is_set()

    @pytest.mark.asyncio
    async def test_stop_check_not_consulted_when_work_finishes_first(self):
        polls = {"count": 0}

        async def quick():
            return 42

        async def stop_check():
            polls["count"] += 1
            return False

        result = await execute_with_guards(
            quick(),
            timeout=5,
            stop_check=stop_check,
            poll_interval=10,
        )
        assert result == 42
        assert polls["count"] == 0
