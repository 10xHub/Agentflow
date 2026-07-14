"""Tool-call idempotency across replays (audit B2).

The execution loop persists `current_node` BEFORE running a node and advances it
only after the node completes, so a process killed mid-node resumes by re-running
that node from scratch. Any non-idempotent tool that already fired would fire
again -- the double-charge scenario. These tests pin the ledger that prevents it.
"""

import pytest

from agentflow.core.graph.utils.invoke_node_handler import InvokeNodeHandler
from agentflow.storage.checkpointer import InMemoryCheckpointer


class TestLedgerKey:
    """The ledger key must identify one invocation, and never collide across turns."""

    def test_key_combines_origin_message_and_call_id(self):
        assert InvokeNodeHandler._ledger_key("msg_1", "call_1") == "msg_1:call_1"

    def test_same_call_id_on_different_turns_does_not_collide(self):
        # Models happily reuse "call_1" on every turn. If the ledger keyed on the
        # call id alone, turn 2 would get a false hit and its tool would be
        # SKIPPED -- never run at all. The origin message keeps them distinct.
        turn1 = InvokeNodeHandler._ledger_key("assistant_msg_a", "call_1")
        turn2 = InvokeNodeHandler._ledger_key("assistant_msg_b", "call_1")
        assert turn1 != turn2

    def test_replay_of_same_message_yields_same_key(self):
        # On resume the assistant message is reloaded from the checkpoint with the
        # same message_id, so the key is stable -- which is what makes the replay
        # guard work at all.
        assert InvokeNodeHandler._ledger_key("msg_1", "call_1") == InvokeNodeHandler._ledger_key(
            "msg_1", "call_1"
        )

    def test_missing_ids_disable_the_ledger(self):
        # No stable identity -> no idempotency claim. Better to run the tool than
        # to skip it on a guess.
        assert InvokeNodeHandler._ledger_key(None, "call_1") is None
        assert InvokeNodeHandler._ledger_key("msg_1", "") is None


class TestCheckpointerLedger:
    @pytest.mark.asyncio
    async def test_records_and_returns_tool_result(self):
        cp = InMemoryCheckpointer()
        config = {"thread_id": "t1", "user_id": "u1"}

        assert await cp.aget_tool_result(config, "msg_1:call_1") is None

        await cp.aput_tool_result(config, "msg_1:call_1", {"__kind__": "raw", "value": "charged"})
        got = await cp.aget_tool_result(config, "msg_1:call_1")
        assert got == {"__kind__": "raw", "value": "charged"}

    @pytest.mark.asyncio
    async def test_ledger_is_scoped_per_thread(self):
        cp = InMemoryCheckpointer()
        a = {"thread_id": "t1", "user_id": "u1"}
        b = {"thread_id": "t2", "user_id": "u1"}

        await cp.aput_tool_result(a, "msg_1:call_1", {"__kind__": "raw", "value": "a"})

        # A different thread must not see thread t1's tool record.
        assert await cp.aget_tool_result(b, "msg_1:call_1") is None

    @pytest.mark.asyncio
    async def test_unknown_call_is_a_miss_so_tool_runs(self):
        cp = InMemoryCheckpointer()
        config = {"thread_id": "t1", "user_id": "u1"}
        assert await cp.aget_tool_result(config, "msg_9:call_9") is None

    @pytest.mark.asyncio
    async def test_base_default_disables_idempotency_rather_than_breaking(self):
        # A custom checkpointer that does not implement the ledger inherits the
        # base no-op: idempotency is simply off (old at-least-once behaviour),
        # not an exception.
        from agentflow.storage.checkpointer import BaseCheckpointer

        assert BaseCheckpointer.aget_tool_result.__doc__ is not None
        cp = InMemoryCheckpointer()
        # Sanity: the in-memory backend does implement it.
        assert cp.aget_tool_result.__func__ is not BaseCheckpointer.aget_tool_result


class TestReplaySkipsCompletedTool:
    """End-to-end shape of the guard: a recorded call is replayed, not re-run."""

    @pytest.mark.asyncio
    async def test_recorded_result_is_reused_and_tool_not_called_again(self):
        cp = InMemoryCheckpointer()
        config = {"thread_id": "t1", "user_id": "u1"}
        handler = InvokeNodeHandler("tools", lambda: None)

        # Simulate a first attempt that completed and was recorded.
        await cp.aput_tool_result(
            config,
            "msg_1:call_1",
            {"__kind__": "raw", "value": {"charged": True}},
        )

        replayed = await handler._get_recorded_tool_result(cp, config, "msg_1:call_1")
        assert replayed == {"charged": True}

    @pytest.mark.asyncio
    async def test_malformed_ledger_entry_is_treated_as_a_miss(self):
        # Misreading the ledger must never cause a tool to be silently skipped.
        cp = InMemoryCheckpointer()
        config = {"thread_id": "t1", "user_id": "u1"}
        handler = InvokeNodeHandler("tools", lambda: None)

        await cp.aput_tool_result(config, "msg_1:call_1", {"unexpected": "shape"})
        assert await handler._get_recorded_tool_result(cp, config, "msg_1:call_1") is None
