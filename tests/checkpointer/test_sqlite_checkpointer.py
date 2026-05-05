"""Unit tests for SqliteCheckpointer.

Covers all public methods (sync + async):
  - setup / asetup
  - state CRUD
  - state cache CRUD
  - generic cache (with TTL)
  - message CRUD + list / search / pagination
  - thread CRUD + list / search / pagination
  - aclean_thread (cascades all data for a thread)
  - arelease / release
  - thread isolation (operations on one thread don't bleed into another)
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest

from agentflow.core.state import AgentState
from agentflow.core.state.message import Message
from agentflow.storage.checkpointer.sqlite_checkpointer import SqliteCheckpointer
from agentflow.utils.thread_info import ThreadInfo


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(content: str = "hello") -> AgentState:
    msg = Message.text_message(content=content, role="user", message_id="seed-msg")
    return AgentState(context=[msg])


def _make_message(mid: str = "msg-1", content: str = "hello") -> Message:
    return Message.text_message(content=content, role="user", message_id=mid)


def _make_thread(tid: str = "t1", name: str = "test-thread") -> ThreadInfo:
    return ThreadInfo(thread_id=tid, thread_name=name)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cp(tmp_path: Path) -> SqliteCheckpointer:
    """Fresh checkpointer backed by a temp SQLite DB for each test."""
    return SqliteCheckpointer(db_path=tmp_path / "test.db")


@pytest.fixture
def cfg1() -> dict:
    return {"thread_id": "thread-1", "state_class": AgentState}


@pytest.fixture
def cfg2() -> dict:
    return {"thread_id": "thread-2", "state_class": AgentState}


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

class TestSetup:
    def test_sync_setup_creates_db(self, cp: SqliteCheckpointer):
        cp.setup()
        assert cp.db_path.exists()

    @pytest.mark.asyncio
    async def test_async_setup_creates_db(self, cp: SqliteCheckpointer):
        await cp.asetup()
        assert cp.db_path.exists()

    @pytest.mark.asyncio
    async def test_idempotent_setup(self, cp: SqliteCheckpointer):
        """Calling asetup twice must not raise (IF NOT EXISTS tables)."""
        await cp.asetup()
        await cp.asetup()

    def test_default_path_under_home(self):
        """Default path should be inside the user home directory."""
        from agentflow.storage.checkpointer.sqlite_checkpointer import DEFAULT_DB_PATH
        from pathlib import Path
        assert Path(DEFAULT_DB_PATH).parts[-2] == ".agentflow"


# ---------------------------------------------------------------------------
# State — async
# ---------------------------------------------------------------------------

class TestAsyncState:
    @pytest.mark.asyncio
    async def test_put_and_get_state(self, cp: SqliteCheckpointer, cfg1: dict):
        await cp.asetup()
        state = _make_state()
        returned = await cp.aput_state(cfg1, state)
        assert returned is state

        loaded = await cp.aget_state(cfg1)
        assert loaded is not None
        assert isinstance(loaded, AgentState)

    @pytest.mark.asyncio
    async def test_get_state_missing_returns_none(self, cp: SqliteCheckpointer, cfg1: dict):
        await cp.asetup()
        result = await cp.aget_state(cfg1)
        assert result is None

    @pytest.mark.asyncio
    async def test_put_state_overwrites(self, cp: SqliteCheckpointer, cfg1: dict):
        await cp.asetup()
        s1 = _make_state("first")
        s2 = _make_state("second")
        await cp.aput_state(cfg1, s1)
        await cp.aput_state(cfg1, s2)

        loaded = await cp.aget_state(cfg1)
        assert loaded is not None
        assert isinstance(loaded, AgentState)
        # The stored messages come from s2
        assert any(
            "second" in str(block)
            for msg in loaded.context
            for block in (msg.content if isinstance(msg.content, list) else [msg.content])
        )

    @pytest.mark.asyncio
    async def test_clear_state(self, cp: SqliteCheckpointer, cfg1: dict):
        await cp.asetup()
        await cp.aput_state(cfg1, _make_state())
        ok = await cp.aclear_state(cfg1)
        assert ok is True
        assert await cp.aget_state(cfg1) is None

    @pytest.mark.asyncio
    async def test_clear_state_nonexistent(self, cp: SqliteCheckpointer, cfg1: dict):
        await cp.asetup()
        result = await cp.aclear_state(cfg1)
        assert result is True  # idempotent


# ---------------------------------------------------------------------------
# State — sync wrappers
# ---------------------------------------------------------------------------

class TestSyncState:
    def test_put_and_get_state(self, cp: SqliteCheckpointer, cfg1: dict):
        cp.setup()
        state = _make_state()
        cp.put_state(cfg1, state)
        loaded = cp.get_state(cfg1)
        assert isinstance(loaded, AgentState)

    def test_get_state_missing_returns_none(self, cp: SqliteCheckpointer, cfg1: dict):
        cp.setup()
        assert cp.get_state(cfg1) is None

    def test_clear_state(self, cp: SqliteCheckpointer, cfg1: dict):
        cp.setup()
        cp.put_state(cfg1, _make_state())
        cp.clear_state(cfg1)
        assert cp.get_state(cfg1) is None


# ---------------------------------------------------------------------------
# State cache — async
# ---------------------------------------------------------------------------

class TestAsyncStateCache:
    @pytest.mark.asyncio
    async def test_put_and_get_cache(self, cp: SqliteCheckpointer, cfg1: dict):
        await cp.asetup()
        state = _make_state()
        await cp.aput_state_cache(cfg1, state)
        cached = await cp.aget_state_cache(cfg1)
        assert cached is not None
        assert isinstance(cached, AgentState)

    @pytest.mark.asyncio
    async def test_cache_missing_returns_none(self, cp: SqliteCheckpointer, cfg1: dict):
        await cp.asetup()
        assert await cp.aget_state_cache(cfg1) is None

    @pytest.mark.asyncio
    async def test_cache_independent_from_state(
        self, cp: SqliteCheckpointer, cfg1: dict
    ):
        await cp.asetup()
        state = _make_state()
        await cp.aput_state(cfg1, state)
        # cache should still be empty
        assert await cp.aget_state_cache(cfg1) is None

    @pytest.mark.asyncio
    async def test_cache_overwrites(self, cp: SqliteCheckpointer, cfg1: dict):
        await cp.asetup()
        await cp.aput_state_cache(cfg1, _make_state("v1"))
        await cp.aput_state_cache(cfg1, _make_state("v2"))
        cached = await cp.aget_state_cache(cfg1)
        assert cached is not None


# ---------------------------------------------------------------------------
# Generic cache
# ---------------------------------------------------------------------------

class TestGenericCache:
    @pytest.mark.asyncio
    async def test_put_and_get(self, cp: SqliteCheckpointer):
        await cp.asetup()
        await cp.aput_cache_value("ns", "k1", {"foo": "bar"})
        val = await cp.aget_cache_value("ns", "k1")
        assert val == {"foo": "bar"}

    @pytest.mark.asyncio
    async def test_get_missing_returns_none(self, cp: SqliteCheckpointer):
        await cp.asetup()
        assert await cp.aget_cache_value("ns", "missing") is None

    @pytest.mark.asyncio
    async def test_ttl_expiry(self, cp: SqliteCheckpointer):
        await cp.asetup()
        await cp.aput_cache_value("ns", "ttl-key", "data", ttl_seconds=1)
        assert await cp.aget_cache_value("ns", "ttl-key") == "data"
        # simulate expiry by manipulating DB directly
        import aiosqlite
        async with aiosqlite.connect(cp.db_path) as db:
            await db.execute(
                "UPDATE af_cache SET expires_at = ? WHERE namespace = ? AND key = ?",
                (time.time() - 1, "ns", "ttl-key"),
            )
            await db.commit()
        assert await cp.aget_cache_value("ns", "ttl-key") is None

    @pytest.mark.asyncio
    async def test_no_ttl_does_not_expire(self, cp: SqliteCheckpointer):
        await cp.asetup()
        await cp.aput_cache_value("ns", "persist", 42)
        assert await cp.aget_cache_value("ns", "persist") == 42

    @pytest.mark.asyncio
    async def test_clear_cache_value(self, cp: SqliteCheckpointer):
        await cp.asetup()
        await cp.aput_cache_value("ns", "del-me", "bye")
        result = await cp.aclear_cache_value("ns", "del-me")
        assert result == "bye"
        assert await cp.aget_cache_value("ns", "del-me") is None

    @pytest.mark.asyncio
    async def test_list_cache_keys(self, cp: SqliteCheckpointer):
        await cp.asetup()
        for i in range(3):
            await cp.aput_cache_value("media", f"img-{i}", f"url-{i}")
        keys = await cp.alist_cache_keys("media")
        assert sorted(keys) == ["img-0", "img-1", "img-2"]

    @pytest.mark.asyncio
    async def test_list_cache_keys_with_prefix(self, cp: SqliteCheckpointer):
        await cp.asetup()
        await cp.aput_cache_value("media", "thumb-a", "x")
        await cp.aput_cache_value("media", "thumb-b", "y")
        await cp.aput_cache_value("media", "full-c", "z")
        keys = await cp.alist_cache_keys("media", prefix="thumb")
        assert sorted(keys) == ["thumb-a", "thumb-b"]

    @pytest.mark.asyncio
    async def test_list_cache_keys_empty_namespace(self, cp: SqliteCheckpointer):
        await cp.asetup()
        assert await cp.alist_cache_keys("empty-ns") == []

    @pytest.mark.asyncio
    async def test_overwrite_cache_value(self, cp: SqliteCheckpointer):
        await cp.asetup()
        await cp.aput_cache_value("ns", "key", "old")
        await cp.aput_cache_value("ns", "key", "new")
        assert await cp.aget_cache_value("ns", "key") == "new"

    @pytest.mark.asyncio
    async def test_cache_namespace_and_key_colons_do_not_collide(
        self, cp: SqliteCheckpointer
    ):
        await cp.asetup()
        await cp.aput_cache_value("media", "temp:x", "first")
        await cp.aput_cache_value("media:temp", "x", "second")

        assert await cp.aget_cache_value("media", "temp:x") == "first"
        assert await cp.aget_cache_value("media:temp", "x") == "second"

    @pytest.mark.asyncio
    async def test_list_cache_keys_prunes_expired_entries(self, cp: SqliteCheckpointer):
        await cp.asetup()
        await cp.aput_cache_value("ns", "expired", "old", ttl_seconds=1)
        await cp.aput_cache_value("ns", "active", "new")

        import aiosqlite
        async with aiosqlite.connect(cp.db_path) as db:
            await db.execute(
                "UPDATE af_cache SET expires_at = ? WHERE namespace = ? AND key = ?",
                (time.time() - 1, "ns", "expired"),
            )
            await db.commit()

        assert await cp.alist_cache_keys("ns") == ["active"]
        async with aiosqlite.connect(cp.db_path) as db:
            async with db.execute(
                "SELECT 1 FROM af_cache WHERE namespace = ? AND key = ?",
                ("ns", "expired"),
            ) as cursor:
                assert await cursor.fetchone() is None

    @pytest.mark.asyncio
    async def test_setup_migrates_legacy_flat_cache_schema(self, tmp_path: Path):
        import aiosqlite

        db_path = tmp_path / "legacy.db"
        async with aiosqlite.connect(db_path) as db:
            await db.execute(
                """
                CREATE TABLE af_cache (
                    cache_key TEXT NOT NULL PRIMARY KEY,
                    value TEXT NOT NULL,
                    expires_at REAL
                )
                """
            )
            await db.execute(
                "INSERT INTO af_cache (cache_key, value, expires_at) VALUES (?, ?, ?)",
                ("ns:key:with:colons", '"legacy-value"', None),
            )
            await db.commit()

        cp = SqliteCheckpointer(db_path=db_path)
        await cp.asetup()

        assert await cp.aget_cache_value("ns", "key:with:colons") == "legacy-value"


# ---------------------------------------------------------------------------
# Messages — async
# ---------------------------------------------------------------------------

class TestAsyncMessages:
    @pytest.mark.asyncio
    async def test_put_and_get_message(self, cp: SqliteCheckpointer, cfg1: dict):
        await cp.asetup()
        msg = _make_message("m1", "hi")
        await cp.aput_messages(cfg1, [msg])
        retrieved = await cp.aget_message(cfg1, "m1")
        assert retrieved.message_id == "m1"

    @pytest.mark.asyncio
    async def test_get_message_not_found_raises(
        self, cp: SqliteCheckpointer, cfg1: dict
    ):
        await cp.asetup()
        with pytest.raises(IndexError):
            await cp.aget_message(cfg1, "no-such-id")

    @pytest.mark.asyncio
    async def test_list_messages_basic(self, cp: SqliteCheckpointer, cfg1: dict):
        await cp.asetup()
        msgs = [_make_message(f"m{i}", f"text {i}") for i in range(5)]
        await cp.aput_messages(cfg1, msgs)
        result = await cp.alist_messages(cfg1)
        assert len(result) == 5

    @pytest.mark.asyncio
    async def test_list_messages_empty(self, cp: SqliteCheckpointer, cfg1: dict):
        await cp.asetup()
        assert await cp.alist_messages(cfg1) == []

    @pytest.mark.asyncio
    async def test_list_messages_search(self, cp: SqliteCheckpointer, cfg1: dict):
        await cp.asetup()
        await cp.aput_messages(cfg1, [
            _make_message("m1", "hello world"),
            _make_message("m2", "goodbye"),
        ])
        result = await cp.alist_messages(cfg1, search="hello")
        assert len(result) == 1
        assert result[0].message_id == "m1"

    @pytest.mark.asyncio
    async def test_list_messages_offset_limit(
        self, cp: SqliteCheckpointer, cfg1: dict
    ):
        await cp.asetup()
        msgs = [_make_message(f"m{i}") for i in range(10)]
        await cp.aput_messages(cfg1, msgs)
        result = await cp.alist_messages(cfg1, offset=3, limit=4)
        assert len(result) == 4

    @pytest.mark.asyncio
    async def test_list_messages_offset_only(
        self, cp: SqliteCheckpointer, cfg1: dict
    ):
        await cp.asetup()
        msgs = [_make_message(f"m{i}") for i in range(5)]
        await cp.aput_messages(cfg1, msgs)
        result = await cp.alist_messages(cfg1, offset=2)
        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_put_messages_idempotent(
        self, cp: SqliteCheckpointer, cfg1: dict
    ):
        """Putting the same message_id twice should upsert, not duplicate."""
        await cp.asetup()
        msg = _make_message("dup-id", "first")
        await cp.aput_messages(cfg1, [msg])
        await cp.aput_messages(cfg1, [_make_message("dup-id", "second")])
        result = await cp.alist_messages(cfg1)
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_delete_message(self, cp: SqliteCheckpointer, cfg1: dict):
        await cp.asetup()
        await cp.aput_messages(cfg1, [_make_message("del-me")])
        ok = await cp.adelete_message(cfg1, "del-me")
        assert ok is True
        with pytest.raises(IndexError):
            await cp.aget_message(cfg1, "del-me")

    @pytest.mark.asyncio
    async def test_delete_message_not_found_raises(
        self, cp: SqliteCheckpointer, cfg1: dict
    ):
        await cp.asetup()
        with pytest.raises(IndexError):
            await cp.adelete_message(cfg1, "ghost")

    @pytest.mark.asyncio
    async def test_messages_accumulate_across_puts(
        self, cp: SqliteCheckpointer, cfg1: dict
    ):
        await cp.asetup()
        await cp.aput_messages(cfg1, [_make_message("a1", "first batch")])
        await cp.aput_messages(cfg1, [_make_message("a2", "second batch")])
        result = await cp.alist_messages(cfg1)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Messages — sync wrappers
# ---------------------------------------------------------------------------

class TestSyncMessages:
    def test_put_and_get_message(self, cp: SqliteCheckpointer, cfg1: dict):
        cp.setup()
        msg = _make_message("s1", "sync test")
        cp.put_messages(cfg1, [msg])
        retrieved = cp.get_message(cfg1, "s1")
        assert retrieved.message_id == "s1"

    def test_list_messages(self, cp: SqliteCheckpointer, cfg1: dict):
        cp.setup()
        cp.put_messages(cfg1, [_make_message("x1"), _make_message("x2")])
        result = cp.list_messages(cfg1)
        assert len(result) == 2

    def test_delete_message(self, cp: SqliteCheckpointer, cfg1: dict):
        cp.setup()
        cp.put_messages(cfg1, [_make_message("rm1")])
        cp.delete_message(cfg1, "rm1")
        with pytest.raises(IndexError):
            cp.get_message(cfg1, "rm1")


# ---------------------------------------------------------------------------
# Threads — async
# ---------------------------------------------------------------------------

class TestAsyncThreads:
    @pytest.mark.asyncio
    async def test_put_and_get_thread(self, cp: SqliteCheckpointer, cfg1: dict):
        await cp.asetup()
        thread = _make_thread("thread-1")
        ok = await cp.aput_thread(cfg1, thread)
        assert ok is True
        loaded = await cp.aget_thread(cfg1)
        assert loaded is not None
        assert loaded.thread_id == "thread-1"

    @pytest.mark.asyncio
    async def test_get_thread_missing_returns_none(
        self, cp: SqliteCheckpointer, cfg1: dict
    ):
        await cp.asetup()
        assert await cp.aget_thread(cfg1) is None

    @pytest.mark.asyncio
    async def test_put_thread_overwrites(
        self, cp: SqliteCheckpointer, cfg1: dict
    ):
        await cp.asetup()
        await cp.aput_thread(cfg1, _make_thread("thread-1", "old-name"))
        await cp.aput_thread(cfg1, _make_thread("thread-1", "new-name"))
        loaded = await cp.aget_thread(cfg1)
        assert loaded is not None
        assert loaded.thread_name == "new-name"

    @pytest.mark.asyncio
    async def test_list_threads(
        self, cp: SqliteCheckpointer, cfg1: dict, cfg2: dict
    ):
        await cp.asetup()
        await cp.aput_thread(cfg1, _make_thread("thread-1"))
        await cp.aput_thread(cfg2, _make_thread("thread-2"))
        threads = await cp.alist_threads({})
        assert len(threads) == 2

    @pytest.mark.asyncio
    async def test_list_threads_search(
        self, cp: SqliteCheckpointer, cfg1: dict, cfg2: dict
    ):
        await cp.asetup()
        await cp.aput_thread(cfg1, _make_thread("thread-1", "alpha-thread"))
        await cp.aput_thread(cfg2, _make_thread("thread-2", "beta-thread"))
        result = await cp.alist_threads({}, search="alpha")
        assert len(result) == 1
        assert result[0].thread_name == "alpha-thread"

    @pytest.mark.asyncio
    async def test_list_threads_offset_limit(self, cp: SqliteCheckpointer):
        await cp.asetup()
        for i in range(5):
            await cp.aput_thread(
                {"thread_id": f"th-{i}"},
                _make_thread(f"th-{i}", f"thread-{i}"),
            )
        result = await cp.alist_threads({}, offset=1, limit=2)
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_clean_thread_removes_all_data(
        self, cp: SqliteCheckpointer, cfg1: dict
    ):
        await cp.asetup()
        await cp.aput_thread(cfg1, _make_thread("thread-1"))
        await cp.aput_state(cfg1, _make_state())
        await cp.aput_state_cache(cfg1, _make_state())
        await cp.aput_messages(cfg1, [_make_message("m1")])

        ok = await cp.aclean_thread(cfg1)
        assert ok is True

        assert await cp.aget_thread(cfg1) is None
        assert await cp.aget_state(cfg1) is None
        assert await cp.aget_state_cache(cfg1) is None
        assert await cp.alist_messages(cfg1) == []

    @pytest.mark.asyncio
    async def test_clean_thread_idempotent(
        self, cp: SqliteCheckpointer, cfg1: dict
    ):
        await cp.asetup()
        ok = await cp.aclean_thread(cfg1)
        assert ok is True

    @pytest.mark.asyncio
    async def test_put_thread_rejects_mismatched_thread_id(
        self, cp: SqliteCheckpointer, cfg1: dict
    ):
        await cp.asetup()
        with pytest.raises(ValueError):
            await cp.aput_thread(cfg1, _make_thread("different-thread"))


# ---------------------------------------------------------------------------
# Threads — sync wrappers
# ---------------------------------------------------------------------------

class TestSyncThreads:
    def test_put_and_get_thread(self, cp: SqliteCheckpointer, cfg1: dict):
        cp.setup()
        cp.put_thread(cfg1, _make_thread("thread-1"))
        loaded = cp.get_thread(cfg1)
        assert loaded is not None
        assert loaded.thread_id == "thread-1"

    def test_get_thread_missing(self, cp: SqliteCheckpointer, cfg1: dict):
        cp.setup()
        assert cp.get_thread(cfg1) is None

    def test_list_threads(
        self, cp: SqliteCheckpointer, cfg1: dict, cfg2: dict
    ):
        cp.setup()
        cp.put_thread(cfg1, _make_thread("thread-1"))
        cp.put_thread(cfg2, _make_thread("thread-2"))
        assert len(cp.list_threads({})) == 2

    def test_clean_thread(self, cp: SqliteCheckpointer, cfg1: dict):
        cp.setup()
        cp.put_thread(cfg1, _make_thread("thread-1"))
        cp.clean_thread(cfg1)
        assert cp.get_thread(cfg1) is None


# ---------------------------------------------------------------------------
# Thread isolation
# ---------------------------------------------------------------------------

class TestThreadIsolation:
    @pytest.mark.asyncio
    async def test_state_isolated_per_thread(
        self, cp: SqliteCheckpointer, cfg1: dict, cfg2: dict
    ):
        await cp.asetup()
        await cp.aput_state(cfg1, _make_state("thread-1-data"))
        assert await cp.aget_state(cfg2) is None

    @pytest.mark.asyncio
    async def test_messages_isolated_per_thread(
        self, cp: SqliteCheckpointer, cfg1: dict, cfg2: dict
    ):
        await cp.asetup()
        await cp.aput_messages(cfg1, [_make_message("m1")])
        assert await cp.alist_messages(cfg2) == []

    @pytest.mark.asyncio
    async def test_clean_one_thread_leaves_other_intact(
        self, cp: SqliteCheckpointer, cfg1: dict, cfg2: dict
    ):
        await cp.asetup()
        await cp.aput_state(cfg1, _make_state())
        await cp.aput_state(cfg2, _make_state())
        await cp.aclean_thread(cfg1)
        assert await cp.aget_state(cfg1) is None
        assert await cp.aget_state(cfg2) is not None


# ---------------------------------------------------------------------------
# Release
# ---------------------------------------------------------------------------

class TestRelease:
    @pytest.mark.asyncio
    async def test_arelease(self, cp: SqliteCheckpointer):
        await cp.asetup()
        ok = await cp.arelease()
        assert ok is True

    def test_release_sync(self, cp: SqliteCheckpointer):
        cp.setup()
        assert cp.release() is True
