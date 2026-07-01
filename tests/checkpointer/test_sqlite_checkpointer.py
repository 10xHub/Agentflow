"""Tests for SqliteCheckpointer (async + sync APIs).

Covers state (durable + cache), generic TTL cache, messages, threads, thread
isolation, subclass recovery via ``__class_path__``, on-disk persistence across
``release``, and the missing-dependency guard.
"""

from __future__ import annotations

import time

import pytest

from agentflow.core.state import AgentState, Message
from agentflow.storage.checkpointer.sqlite_checkpointer import SqliteCheckpointer
from agentflow.utils.thread_info import ThreadInfo


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


class MyState(AgentState):
    counter: int = 0
    label: str = ""


@pytest.fixture
def cp():
    # A single in-memory database per test; schema is created lazily on first
    # async call via ``_ensure_setup``.
    return SqliteCheckpointer(":memory:")


@pytest.fixture
def cfg1():
    return {"thread_id": "thread-1"}


@pytest.fixture
def cfg2():
    return {"thread_id": "thread-2"}


def _msg(mid: str = "m1", content: str = "hello world") -> Message:
    return Message.text_message(content=content, role="user", message_id=mid)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_put_get_state_roundtrip(cp, cfg1):
    await cp.aput_state(cfg1, MyState(counter=7, label="a"))
    got = await cp.aget_state(cfg1)
    assert type(got) is MyState
    assert got.counter == 7
    assert got.label == "a"


@pytest.mark.asyncio
async def test_get_state_missing_returns_none(cp, cfg1):
    assert await cp.aget_state(cfg1) is None


@pytest.mark.asyncio
async def test_put_state_upserts(cp, cfg1):
    await cp.aput_state(cfg1, MyState(counter=1))
    await cp.aput_state(cfg1, MyState(counter=2))
    got = await cp.aget_state(cfg1)
    assert got.counter == 2


@pytest.mark.asyncio
async def test_clear_state(cp, cfg1):
    await cp.aput_state(cfg1, MyState(counter=1))
    assert await cp.aclear_state(cfg1) is True
    assert await cp.aget_state(cfg1) is None


@pytest.mark.asyncio
async def test_state_recovers_exact_subclass(cp, cfg1):
    await cp.aput_state(cfg1, MyState(counter=3, label="sub"))
    got = await cp.aget_state(cfg1)
    assert isinstance(got, MyState)


# ---------------------------------------------------------------------------
# State cache
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_state_cache_roundtrip(cp, cfg1):
    await cp.aput_state_cache(cfg1, MyState(counter=9))
    got = await cp.aget_state_cache(cfg1)
    assert got.counter == 9
    assert type(got) is MyState


@pytest.mark.asyncio
async def test_state_cache_missing_returns_none(cp, cfg1):
    assert await cp.aget_state_cache(cfg1) is None


@pytest.mark.asyncio
async def test_state_and_cache_are_independent(cp, cfg1):
    await cp.aput_state(cfg1, MyState(counter=1))
    await cp.aput_state_cache(cfg1, MyState(counter=2))
    assert (await cp.aget_state(cfg1)).counter == 1
    assert (await cp.aget_state_cache(cfg1)).counter == 2


# ---------------------------------------------------------------------------
# Generic cache with TTL
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cache_value_roundtrip(cp):
    await cp.aput_cache_value("ns", "k", {"a": 1})
    assert await cp.aget_cache_value("ns", "k") == {"a": 1}


@pytest.mark.asyncio
async def test_cache_value_missing_returns_none(cp):
    assert await cp.aget_cache_value("ns", "missing") is None


@pytest.mark.asyncio
async def test_cache_value_expires(cp):
    await cp.aput_cache_value("ns", "k", "v", ttl_seconds=1)
    # Force expiry by rewriting expires_at in the past via a fresh put.
    await cp.aput_cache_value("ns", "k", "v", ttl_seconds=1)
    # Simulate elapsed time.
    conn = await cp._get_conn()
    await conn.execute(
        "UPDATE af_cache SET expires_at = ? WHERE namespace = ? AND key = ?",
        (time.time() - 10, "ns", "k"),
    )
    await conn.commit()
    assert await cp.aget_cache_value("ns", "k") is None
    # Expired entry is pruned.
    assert await cp.alist_cache_keys("ns") == []


@pytest.mark.asyncio
async def test_clear_cache_value_returns_old(cp):
    await cp.aput_cache_value("ns", "k", {"x": 2})
    assert await cp.aclear_cache_value("ns", "k") == {"x": 2}
    assert await cp.aget_cache_value("ns", "k") is None


@pytest.mark.asyncio
async def test_clear_cache_value_missing(cp):
    assert await cp.aclear_cache_value("ns", "nope") is None


@pytest.mark.asyncio
async def test_list_cache_keys_prefix(cp):
    await cp.aput_cache_value("ns", "user:1", 1)
    await cp.aput_cache_value("ns", "user:2", 2)
    await cp.aput_cache_value("ns", "other", 3)
    keys = sorted(await cp.alist_cache_keys("ns", prefix="user:"))
    assert keys == ["user:1", "user:2"]


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_put_list_messages(cp, cfg1):
    await cp.aput_messages(cfg1, [_msg("m1", "hello"), _msg("m2", "world")])
    msgs = await cp.alist_messages(cfg1)
    assert [str(m.message_id) for m in msgs] == ["m1", "m2"]


@pytest.mark.asyncio
async def test_get_message(cp, cfg1):
    await cp.aput_messages(cfg1, [_msg("m1")])
    got = await cp.aget_message(cfg1, "m1")
    assert str(got.message_id) == "m1"


@pytest.mark.asyncio
async def test_get_message_missing_raises(cp, cfg1):
    with pytest.raises(IndexError):
        await cp.aget_message(cfg1, "nope")


@pytest.mark.asyncio
async def test_list_messages_search(cp, cfg1):
    await cp.aput_messages(cfg1, [_msg("m1", "apple"), _msg("m2", "banana")])
    hits = await cp.alist_messages(cfg1, search="apple")
    assert len(hits) == 1
    assert str(hits[0].message_id) == "m1"


@pytest.mark.asyncio
async def test_list_messages_pagination(cp, cfg1):
    await cp.aput_messages(cfg1, [_msg(f"m{i}", f"c{i}") for i in range(5)])
    page = await cp.alist_messages(cfg1, offset=1, limit=2)
    assert [str(m.message_id) for m in page] == ["m1", "m2"]


@pytest.mark.asyncio
async def test_put_messages_upsert(cp, cfg1):
    await cp.aput_messages(cfg1, [_msg("m1", "first")])
    await cp.aput_messages(cfg1, [_msg("m1", "second")])
    msgs = await cp.alist_messages(cfg1)
    assert len(msgs) == 1
    assert "second" in str(msgs[0].content).lower()


@pytest.mark.asyncio
async def test_delete_message(cp, cfg1):
    await cp.aput_messages(cfg1, [_msg("m1"), _msg("m2")])
    assert await cp.adelete_message(cfg1, "m1") is True
    remaining = [str(m.message_id) for m in await cp.alist_messages(cfg1)]
    assert remaining == ["m2"]


@pytest.mark.asyncio
async def test_delete_message_missing_raises(cp, cfg1):
    with pytest.raises(IndexError):
        await cp.adelete_message(cfg1, "nope")


# ---------------------------------------------------------------------------
# Threads
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_put_get_thread(cp, cfg1):
    ti = ThreadInfo(thread_id="thread-1", thread_name="main", user_id="u1")
    await cp.aput_thread(cfg1, ti)
    got = await cp.aget_thread(cfg1)
    assert got.thread_name == "main"
    assert got.user_id == "u1"


@pytest.mark.asyncio
async def test_get_thread_missing_returns_none(cp, cfg1):
    assert await cp.aget_thread(cfg1) is None


@pytest.mark.asyncio
async def test_put_thread_mismatch_raises(cp, cfg1):
    ti = ThreadInfo(thread_id="different", thread_name="x")
    with pytest.raises(ValueError):
        await cp.aput_thread(cfg1, ti)


@pytest.mark.asyncio
async def test_list_threads_and_search(cp, cfg1, cfg2):
    await cp.aput_thread(cfg1, ThreadInfo(thread_id="thread-1", thread_name="alpha"))
    await cp.aput_thread(cfg2, ThreadInfo(thread_id="thread-2", thread_name="beta"))
    assert len(await cp.alist_threads(cfg1)) == 2
    hits = await cp.alist_threads(cfg1, search="alpha")
    assert len(hits) == 1
    assert hits[0].thread_name == "alpha"


@pytest.mark.asyncio
async def test_clean_thread_cascades(cp, cfg1):
    await cp.aput_state(cfg1, MyState(counter=1))
    await cp.aput_state_cache(cfg1, MyState(counter=2))
    await cp.aput_messages(cfg1, [_msg("m1")])
    await cp.aput_thread(cfg1, ThreadInfo(thread_id="thread-1"))
    assert await cp.aclean_thread(cfg1) is True
    assert await cp.aget_state(cfg1) is None
    assert await cp.aget_state_cache(cfg1) is None
    assert await cp.alist_messages(cfg1) == []
    assert await cp.aget_thread(cfg1) is None


# ---------------------------------------------------------------------------
# Isolation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_thread_isolation(cp, cfg1, cfg2):
    await cp.aput_state(cfg1, MyState(counter=1))
    await cp.aput_state(cfg2, MyState(counter=2))
    await cp.aput_messages(cfg1, [_msg("m1")])
    assert (await cp.aget_state(cfg1)).counter == 1
    assert (await cp.aget_state(cfg2)).counter == 2
    assert await cp.alist_messages(cfg2) == []


# ---------------------------------------------------------------------------
# Persistence across release (file-backed)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_persists_to_disk_across_release(tmp_path):
    db = tmp_path / "cp.db"
    cp1 = SqliteCheckpointer(db)
    await cp1.asetup()
    cfg = {"thread_id": "t"}
    await cp1.aput_state(cfg, MyState(counter=42, label="persist"))
    await cp1.aput_messages(cfg, [_msg("m1", "durable")])
    await cp1.arelease()

    cp2 = SqliteCheckpointer(db)
    got = await cp2.aget_state(cfg)
    assert got.counter == 42
    assert got.label == "persist"
    assert len(await cp2.alist_messages(cfg)) == 1
    await cp2.arelease()


@pytest.mark.asyncio
async def test_default_path_used_when_none(monkeypatch, tmp_path):
    fake_default = tmp_path / "home" / ".agentflow" / "checkpointer.db"
    monkeypatch.setattr(
        "agentflow.storage.checkpointer.sqlite_checkpointer.DEFAULT_DB_PATH",
        str(fake_default),
    )
    cp = SqliteCheckpointer()
    await cp.asetup()
    assert fake_default.exists()
    await cp.arelease()


# ---------------------------------------------------------------------------
# Sync wrappers
# ---------------------------------------------------------------------------


def test_sync_state_message_thread_roundtrip():
    cp = SqliteCheckpointer(":memory:")
    cp.setup()
    cfg = {"thread_id": "sync-1"}
    cp.put_state(cfg, MyState(counter=5))
    assert cp.get_state(cfg).counter == 5
    cp.put_state_cache(cfg, MyState(counter=6))
    assert cp.get_state_cache(cfg).counter == 6
    cp.put_messages(cfg, [_msg("m1")])
    assert str(cp.get_message(cfg, "m1").message_id) == "m1"
    assert len(cp.list_messages(cfg)) == 1
    cp.put_thread(cfg, ThreadInfo(thread_id="sync-1", thread_name="s"))
    assert cp.get_thread(cfg).thread_name == "s"
    assert len(cp.list_threads(cfg)) == 1
    cp.put_cache_value("ns", "k", 1)
    assert cp.get_cache_value("ns", "k") == 1
    cp.clear_cache_value("ns", "k")
    assert cp.get_cache_value("ns", "k") is None
    assert cp.clear_state(cfg) is True
    assert cp.clean_thread(cfg) is True
    assert cp.release() is True


# ---------------------------------------------------------------------------
# Missing dependency guard
# ---------------------------------------------------------------------------


def test_missing_aiosqlite_raises(monkeypatch):
    monkeypatch.setattr(
        "agentflow.storage.checkpointer.sqlite_checkpointer.HAS_AIOSQLITE", False
    )
    with pytest.raises(ImportError, match="aiosqlite"):
        SqliteCheckpointer(":memory:")
