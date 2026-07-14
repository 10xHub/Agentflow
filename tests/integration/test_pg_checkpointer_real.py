"""Durable-storage guarantees against a REAL Postgres + Redis (audit B7).

The existing `pg_checkpointer` coverage came from mocks that set
`cp._pg_pool = MagicMock()` and asserted canned rows echoed back -- which proves
the Python branches run, and proves nothing about the SQL, the transactions, or
Redis/Postgres consistency. Every blocker fixed in this file's subject (B1 CAS,
B2 tool ledger, B3 isolation, M7 atomicity) is only *actually* verified here.

Run with real services:

    docker run -d -p 5432:5432 -e POSTGRES_PASSWORD=test -e POSTGRES_USER=test \\
        -e POSTGRES_DB=test_agentflow postgres:16
    docker run -d -p 6379:6379 redis:7

    pytest tests/integration/test_pg_checkpointer_real.py --integration

Connection details come from POSTGRES_DSN / REDIS_URL, defaulting to the CI
service containers.
"""

import os
import uuid

import pytest
import pytest_asyncio

from agentflow.core.exceptions import StaleStateError, StorageError
from agentflow.core.state import AgentState, Message


pytestmark = [pytest.mark.integration, pytest.mark.asyncio]

POSTGRES_DSN = os.environ.get(
    "POSTGRES_DSN", "postgresql://test:test@localhost:5432/test_agentflow"
)
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/1")


@pytest_asyncio.fixture
async def cp():
    """A checkpointer wired to real services, with its schema created."""
    from agentflow.storage.checkpointer import PgCheckpointer

    checkpointer = PgCheckpointer(
        postgres_dsn=POSTGRES_DSN,
        redis_url=REDIS_URL,
        state_history_limit=5,
    )
    await checkpointer.asetup()
    yield checkpointer
    await checkpointer.arelease()


@pytest.fixture
def cfg():
    """A unique thread per test, so tests do not interfere."""
    return {"thread_id": f"t-{uuid.uuid4()}", "user_id": f"u-{uuid.uuid4()}"}


class TestSchemaAndMigrations:
    async def test_setup_creates_schema_at_current_version(self, cp):
        from agentflow.storage.checkpointer.pg_checkpointer import CURRENT_SCHEMA_VERSION

        async with (await cp._get_pg_pool()).acquire() as conn:
            version = await cp._get_recorded_schema_version(conn)
            assert version == CURRENT_SCHEMA_VERSION

            # The v2/v3 migrations must have actually produced their objects.
            cols = await conn.fetch(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'states'"
            )
            assert "version" in {c["column_name"] for c in cols}

            tables = await conn.fetch(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'public'"
            )
            assert "tool_executions" in {t["table_name"] for t in tables}

    async def test_setup_is_idempotent(self, cp):
        # Re-running setup (every worker does on boot) must not blow up.
        await cp.asetup()
        await cp.asetup()


class TestVersioningAndCAS:
    async def test_writes_append_increasing_versions(self, cp, cfg):
        await cp.aput_state(cfg, AgentState())
        assert cfg["_checkpoint_version"] == 1

        await cp.aput_state(cfg, AgentState())
        assert cfg["_checkpoint_version"] == 2

    async def test_read_returns_latest_version_deterministically(self, cp, cfg):
        s1 = AgentState()
        s1.context = [Message.text_message("first", role="user")]
        await cp.aput_state(cfg, s1)

        s2 = AgentState()
        s2.context = [Message.text_message("second", role="user")]
        await cp.aput_state(cfg, s2)

        fresh = {"thread_id": cfg["thread_id"], "user_id": cfg["user_id"]}
        got = await cp.aget_state(fresh)
        assert got.context[0].text() == "second"
        assert fresh["_checkpoint_version"] == 2

    async def test_concurrent_write_on_stale_version_is_rejected(self, cp, cfg):
        """B1: the lost-update scenario, against real SQL."""
        await cp.aput_state(cfg, AgentState())  # v1

        # Two runs both load v1.
        run_a = {**cfg, "_checkpoint_version": 1}
        run_b = {**cfg, "_checkpoint_version": 1}

        await cp.aput_state(run_a, AgentState())  # commits v2
        assert run_a["_checkpoint_version"] == 2

        # B is now stale: committing would silently discard A's work.
        with pytest.raises(StaleStateError):
            await cp.aput_state(run_b, AgentState())

    async def test_history_is_pruned_to_the_limit(self, cp, cfg):
        for _ in range(8):  # limit is 5 in the fixture
            await cp.aput_state(cfg, AgentState())

        async with (await cp._get_pg_pool()).acquire() as conn:
            count = await conn.fetchval(
                'SELECT COUNT(*) FROM "public"."states" WHERE thread_id = $1',
                cfg["thread_id"],
            )
        assert count <= 5


class TestAtomicCheckpoint:
    async def test_state_and_messages_commit_together(self, cp, cfg):
        """M7: one transaction, so state can't advance without its messages."""
        state = AgentState()
        msgs = [Message.text_message("hello", role="user", message_id=str(uuid.uuid4()))]

        await cp.aput_checkpoint(cfg, state, msgs)

        stored = await cp.alist_messages(cfg)
        assert len(stored) == 1
        assert (await cp.aget_state(cfg)) is not None


class TestUserIsolation:
    async def test_other_user_cannot_read_state(self, cp, cfg):
        await cp.aput_state(cfg, AgentState())

        intruder = {"thread_id": cfg["thread_id"], "user_id": "someone-else"}
        assert await cp.aget_state(intruder) is None

    async def test_other_user_cannot_write_state(self, cp, cfg):
        await cp.aput_state(cfg, AgentState())

        intruder = {"thread_id": cfg["thread_id"], "user_id": "someone-else"}
        with pytest.raises(StorageError):
            await cp.aput_state(intruder, AgentState())

    async def test_other_user_cannot_read_messages(self, cp, cfg):
        await cp.aput_checkpoint(
            cfg,
            AgentState(),
            [Message.text_message("secret", role="user", message_id=str(uuid.uuid4()))],
        )

        intruder = {"thread_id": cfg["thread_id"], "user_id": "someone-else"}
        assert await cp.alist_messages(intruder) == []


class TestToolLedger:
    async def test_completed_tool_is_recorded_and_replayed(self, cp, cfg):
        """B2: a replayed node must not re-fire a tool that already ran."""
        await cp.aput_state(cfg, AgentState())  # ensure thread exists

        assert await cp.aget_tool_result(cfg, "msg1:call1") is None

        await cp.aput_tool_result(cfg, "msg1:call1", {"__kind__": "raw", "value": "charged"})

        got = await cp.aget_tool_result(cfg, "msg1:call1")
        assert got == {"__kind__": "raw", "value": "charged"}

    async def test_recording_the_same_call_twice_keeps_the_first_result(self, cp, cfg):
        await cp.aput_state(cfg, AgentState())

        await cp.aput_tool_result(cfg, "msg1:call1", {"__kind__": "raw", "value": "first"})
        await cp.aput_tool_result(cfg, "msg1:call1", {"__kind__": "raw", "value": "second"})

        got = await cp.aget_tool_result(cfg, "msg1:call1")
        assert got["value"] == "first"


class TestCacheConsistency:
    async def test_cache_write_cannot_move_backwards(self, cp, cfg):
        """The wedge fix: a stale run must not stamp its state over a newer one."""
        state = AgentState()
        await cp.aput_state(cfg, state)  # v1
        await cp.aput_state_cache(cfg, state)  # cache at v1

        await cp.aput_state(cfg, state)  # v2
        await cp.aput_state_cache(cfg, state)  # cache at v2

        # A stale run still holding v1 tries to write; it must be refused.
        stale = {**cfg, "_checkpoint_version": 1}
        assert await cp.aput_state_cache(stale, state) is None

        # And the cache must still report v2, not have been dragged back to v1.
        reader = {"thread_id": cfg["thread_id"], "user_id": cfg["user_id"]}
        await cp.aget_state_cache(reader)
        assert reader["_checkpoint_version"] == 2

    async def test_cache_miss_falls_back_to_postgres(self, cp, cfg):
        await cp.aput_state(cfg, AgentState())
        await cp.redis.delete(cp._get_thread_key(cfg["thread_id"], cfg["user_id"]))

        reader = {"thread_id": cfg["thread_id"], "user_id": cfg["user_id"]}
        assert await cp.aget_state_cache(reader) is not None
