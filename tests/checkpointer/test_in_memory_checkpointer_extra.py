"""Tests for InMemoryCheckpointer sync and async methods.

Targets previously uncovered sync methods: put_state, get_state, clear_state,
put_state_cache, get_state_cache, put_messages, get_message, list_messages,
delete_message, get_thread, list_threads, clean_thread, release, etc.
"""

from __future__ import annotations

import pytest

from agentflow.storage.checkpointer.in_memory_checkpointer import InMemoryCheckpointer
from agentflow.core.state.message import Message
from agentflow.utils.thread_info import ThreadInfo


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cp():
    return InMemoryCheckpointer()


@pytest.fixture
def config_t1():
    return {"thread_id": "thread-1"}


@pytest.fixture
def config_t2():
    return {"thread_id": "thread-2"}


def _make_message(mid: str = "msg-1", content: str = "hello") -> Message:
    return Message.text_message(content=content, role="user", message_id=mid)


def _make_state():
    from agentflow.core.state import AgentState

    return AgentState(messages=[_make_message()])


def _make_thread(tid="t1") -> ThreadInfo:
    return ThreadInfo(thread_id=tid, thread_name="test-thread")


# ---------------------------------------------------------------------------
# setup / asetup
# ---------------------------------------------------------------------------

class TestSetup:
    def test_setup_runs_without_error(self, cp):
        cp.setup()  # should not raise

    @pytest.mark.asyncio
    async def test_asetup_runs_without_error(self, cp):
        await cp.asetup()  # should not raise


# ---------------------------------------------------------------------------
# Sync State Methods
# ---------------------------------------------------------------------------

class TestSyncStateMethods:
    def test_put_get_state(self, cp, config_t1):
        state = _make_state()
        result = cp.put_state(config_t1, state)
        assert result is state
        retrieved = cp.get_state(config_t1)
        assert retrieved is state

    def test_get_state_missing_returns_none(self, cp, config_t1):
        assert cp.get_state(config_t1) is None

    def test_clear_state(self, cp, config_t1):
        state = _make_state()
        cp.put_state(config_t1, state)
        cleared = cp.clear_state(config_t1)
        assert cleared is True
        assert cp.get_state(config_t1) is None

    def test_clear_state_nonexistent(self, cp, config_t1):
        # clearing nonexistent key should still return True
        result = cp.clear_state(config_t1)
        assert result is True

    def test_put_get_state_cache(self, cp, config_t1):
        state = _make_state()
        result = cp.put_state_cache(config_t1, state)
        assert result is state
        cached = cp.get_state_cache(config_t1)
        assert cached is state

    def test_get_state_cache_missing_returns_none(self, cp, config_t1):
        assert cp.get_state_cache(config_t1) is None

    def test_multiple_threads_isolated(self, cp, config_t1, config_t2):
        s1 = _make_state()
        s2 = _make_state()
        cp.put_state(config_t1, s1)
        cp.put_state(config_t2, s2)
        assert cp.get_state(config_t1) is s1
        assert cp.get_state(config_t2) is s2


# ---------------------------------------------------------------------------
# Sync Message Methods
# ---------------------------------------------------------------------------

class TestSyncMessageMethods:
    def test_put_get_message(self, cp, config_t1):
        msg = _make_message("msg-a", "hello")
        cp.put_messages(config_t1, [msg])
        retrieved = cp.get_message(config_t1, "msg-a")
        assert retrieved is msg

    def test_get_message_not_found_raises(self, cp, config_t1):
        with pytest.raises(IndexError):
            cp.get_message(config_t1, "nonexistent")

    def test_list_messages_basic(self, cp, config_t1):
        msgs = [_make_message(f"msg-{i}", f"text {i}") for i in range(5)]
        cp.put_messages(config_t1, msgs)
        listed = cp.list_messages(config_t1)
        assert len(listed) == 5

    def test_list_messages_empty(self, cp, config_t1):
        result = cp.list_messages(config_t1)
        assert result == []

    def test_list_messages_with_search(self, cp, config_t1):
        msgs = [
            _make_message("msg-1", "hello world"),
            _make_message("msg-2", "goodbye"),
        ]
        cp.put_messages(config_t1, msgs)
        result = cp.list_messages(config_t1, search="hello")
        assert len(result) == 1

    def test_list_messages_with_offset_limit(self, cp, config_t1):
        msgs = [_make_message(f"msg-{i}", f"text {i}") for i in range(10)]
        cp.put_messages(config_t1, msgs)
        result = cp.list_messages(config_t1, offset=3, limit=4)
        assert len(result) == 4

    def test_list_messages_with_offset_only(self, cp, config_t1):
        msgs = [_make_message(f"msg-{i}") for i in range(5)]
        cp.put_messages(config_t1, msgs)
        result = cp.list_messages(config_t1, offset=2)
        assert len(result) == 3

    def test_delete_message(self, cp, config_t1):
        msg = _make_message("msg-del", "delete me")
        cp.put_messages(config_t1, [msg])
        deleted = cp.delete_message(config_t1, "msg-del")
        assert deleted is True
        with pytest.raises(IndexError):
            cp.get_message(config_t1, "msg-del")

    def test_delete_message_not_found_raises(self, cp, config_t1):
        with pytest.raises(IndexError):
            cp.delete_message(config_t1, "no-such-message")

    def test_put_messages_with_metadata(self, cp, config_t1):
        msg = _make_message("msg-x")
        result = cp.put_messages(config_t1, [msg], metadata={"source": "test"})
        assert result is True
        assert cp._message_metadata[config_t1["thread_id"]]["source"] == "test"

    def test_messages_accumulated_across_puts(self, cp, config_t1):
        cp.put_messages(config_t1, [_make_message("m1", "text1")])
        cp.put_messages(config_t1, [_make_message("m2", "text2")])
        all_msgs = cp.list_messages(config_t1)
        assert len(all_msgs) == 2


# ---------------------------------------------------------------------------
# Sync Thread Methods
# ---------------------------------------------------------------------------

class TestSyncThreadMethods:
    def test_put_get_thread(self, cp, config_t1):
        thread = _make_thread("t1")
        result = cp.put_thread(config_t1, thread)
        assert result is True
        retrieved = cp.get_thread(config_t1)
        assert retrieved is not None
        assert retrieved.thread_id == "t1"

    def test_get_thread_missing_returns_none(self, cp, config_t1):
        assert cp.get_thread(config_t1) is None

    def test_list_threads(self, cp, config_t1, config_t2):
        cp.put_thread(config_t1, _make_thread("t1"))
        cp.put_thread(config_t2, _make_thread("t2"))
        result = cp.list_threads({})
        assert len(result) == 2

    def test_list_threads_with_search(self, cp, config_t1, config_t2):
        cp.put_thread(config_t1, _make_thread("unique-thread"))
        cp.put_thread(config_t2, _make_thread("other-thread"))
        result = cp.list_threads({}, search="unique")
        assert len(result) == 1

    def test_list_threads_with_offset_limit(self, cp):
        for i in range(5):
            cp.put_thread({"thread_id": f"t{i}"}, _make_thread(f"t{i}"))
        result = cp.list_threads({}, offset=1, limit=2)
        assert len(result) == 2

    def test_clean_thread(self, cp, config_t1):
        cp.put_thread(config_t1, _make_thread("t1"))
        cleaned = cp.clean_thread(config_t1)
        assert cleaned is True
        assert cp.get_thread(config_t1) is None

    def test_clean_thread_nonexistent_returns_false(self, cp, config_t1):
        result = cp.clean_thread(config_t1)
        assert result is False


# ---------------------------------------------------------------------------
# Sync Release
# ---------------------------------------------------------------------------

class TestSyncRelease:
    def test_release_clears_all(self, cp, config_t1):
        state = _make_state()
        cp.put_state(config_t1, state)
        cp.put_messages(config_t1, [_make_message()])
        cp.put_thread(config_t1, _make_thread())

        result = cp.release()
        assert result is True
        assert cp.get_state(config_t1) is None
        assert cp.list_messages(config_t1) == []
        assert cp.get_thread(config_t1) is None


# ---------------------------------------------------------------------------
# Async State Methods
# ---------------------------------------------------------------------------

class TestAsyncStateMethods:
    @pytest.mark.asyncio
    async def test_aput_aget_state(self, cp, config_t1):
        state = _make_state()
        await cp.aput_state(config_t1, state)
        retrieved = await cp.aget_state(config_t1)
        assert retrieved is state

    @pytest.mark.asyncio
    async def test_aget_state_missing(self, cp, config_t1):
        assert await cp.aget_state(config_t1) is None

    @pytest.mark.asyncio
    async def test_aclear_state(self, cp, config_t1):
        state = _make_state()
        await cp.aput_state(config_t1, state)
        await cp.aclear_state(config_t1)
        assert await cp.aget_state(config_t1) is None

    @pytest.mark.asyncio
    async def test_aput_get_state_cache(self, cp, config_t1):
        state = _make_state()
        await cp.aput_state_cache(config_t1, state)
        cached = await cp.aget_state_cache(config_t1)
        assert cached is state


# ---------------------------------------------------------------------------
# Async Message Methods
# ---------------------------------------------------------------------------

class TestAsyncMessageMethods:
    @pytest.mark.asyncio
    async def test_aput_aget_message(self, cp, config_t1):
        msg = _make_message("async-msg-1", "async hello")
        await cp.aput_messages(config_t1, [msg])
        retrieved = await cp.aget_message(config_t1, "async-msg-1")
        assert retrieved is msg

    @pytest.mark.asyncio
    async def test_aget_message_not_found(self, cp, config_t1):
        with pytest.raises(IndexError):
            await cp.aget_message(config_t1, "not-found")

    @pytest.mark.asyncio
    async def test_alist_messages_with_search(self, cp, config_t1):
        await cp.aput_messages(config_t1, [
            _make_message("m1", "search me"),
            _make_message("m2", "not matching"),
        ])
        result = await cp.alist_messages(config_t1, search="search")
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_alist_messages_with_offset_limit(self, cp, config_t1):
        await cp.aput_messages(config_t1, [_make_message(f"m{i}") for i in range(6)])
        result = await cp.alist_messages(config_t1, offset=2, limit=3)
        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_adelete_message(self, cp, config_t1):
        msg = _make_message("del-msg", "delete")
        await cp.aput_messages(config_t1, [msg])
        result = await cp.adelete_message(config_t1, "del-msg")
        assert result is True
        with pytest.raises(IndexError):
            await cp.aget_message(config_t1, "del-msg")

    @pytest.mark.asyncio
    async def test_adelete_message_not_found(self, cp, config_t1):
        with pytest.raises(IndexError):
            await cp.adelete_message(config_t1, "no-msg")


# ---------------------------------------------------------------------------
# Async Thread Methods
# ---------------------------------------------------------------------------

class TestAsyncThreadMethods:
    @pytest.mark.asyncio
    async def test_aput_aget_thread(self, cp, config_t1):
        thread = _make_thread("async-t1")
        await cp.aput_thread(config_t1, thread)
        retrieved = await cp.aget_thread(config_t1)
        assert retrieved is not None
        assert retrieved.thread_id == "async-t1"

    @pytest.mark.asyncio
    async def test_aget_thread_missing(self, cp, config_t1):
        assert await cp.aget_thread(config_t1) is None

    @pytest.mark.asyncio
    async def test_alist_threads(self, cp, config_t1, config_t2):
        await cp.aput_thread(config_t1, _make_thread("t1"))
        await cp.aput_thread(config_t2, _make_thread("t2"))
        result = await cp.alist_threads({})
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_alist_threads_with_search(self, cp, config_t1, config_t2):
        await cp.aput_thread(config_t1, _make_thread("find-me"))
        await cp.aput_thread(config_t2, _make_thread("ignore"))
        result = await cp.alist_threads({}, search="find")
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_alist_threads_offset_limit(self, cp):
        for i in range(5):
            await cp.aput_thread({"thread_id": f"at{i}"}, _make_thread(f"at{i}"))
        result = await cp.alist_threads({}, offset=1, limit=2)
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_aclean_thread(self, cp, config_t1):
        await cp.aput_thread(config_t1, _make_thread("t1"))
        cleaned = await cp.aclean_thread(config_t1)
        assert cleaned is True
        assert await cp.aget_thread(config_t1) is None

    @pytest.mark.asyncio
    async def test_aclean_thread_nonexistent(self, cp, config_t1):
        result = await cp.aclean_thread(config_t1)
        assert result is False


# ---------------------------------------------------------------------------
# Async Release
# ---------------------------------------------------------------------------

class TestAsyncRelease:
    @pytest.mark.asyncio
    async def test_arelease_clears_all(self, cp, config_t1):
        state = _make_state()
        await cp.aput_state(config_t1, state)
        await cp.aput_messages(config_t1, [_make_message()])
        await cp.aput_thread(config_t1, _make_thread())

        result = await cp.arelease()
        assert result is True
        assert await cp.aget_state(config_t1) is None
        listed = await cp.alist_messages(config_t1)
        assert listed == []
        assert await cp.aget_thread(config_t1) is None


# ---------------------------------------------------------------------------
# aget_thread_owner — global ownership resolution
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_aget_thread_owner_returns_owner_when_registered(cp):
    cfg = {"thread_id": "t-own", "user_id": "alice"}
    await cp.aput_thread(cfg, ThreadInfo(thread_id="t-own", user_id="alice"))
    assert await cp.aget_thread_owner("t-own") == "alice"


@pytest.mark.asyncio
async def test_aget_thread_owner_none_for_unknown_thread(cp):
    assert await cp.aget_thread_owner("ghost") is None


@pytest.mark.asyncio
async def test_base_default_raises_not_implemented():
    """A backend that does not override the hook must fail loud, never silently
    report 'no owner' (which would defeat owner-based authorization)."""
    from agentflow.storage.checkpointer.base_checkpointer import BaseCheckpointer

    # Bypass ABC instantiation guard to exercise the concrete default body directly.
    inst = object.__new__(InMemoryCheckpointer)
    with pytest.raises(NotImplementedError):
        await BaseCheckpointer.aget_thread_owner(inst, "x")
