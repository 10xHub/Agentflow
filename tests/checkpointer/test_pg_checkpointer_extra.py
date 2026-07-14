import json
from enum import Enum
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentflow.storage.checkpointer.pg_checkpointer import PgCheckpointer
from agentflow.core.state import AgentState


@pytest.fixture
def cp(monkeypatch):
    monkeypatch.setattr("agentflow.storage.checkpointer.pg_checkpointer.HAS_ASYNCPG", True)
    monkeypatch.setattr("agentflow.storage.checkpointer.pg_checkpointer.HAS_REDIS", True)

    class _Redis:
        def __init__(self):
            self.setex = AsyncMock()
            self.get = AsyncMock(return_value=None)
            self.delete = AsyncMock(return_value=1)
            self.eval = AsyncMock(return_value=1)  # version-guarded cache CAS
            self.scan = AsyncMock(side_effect=[(1, [b"generic_cache:ns:a"]), (0, [b"generic_cache:ns:b"])])

    redis = _Redis()
    c = PgCheckpointer(postgres_dsn="postgres://x", redis=redis)
    c._pg_pool = MagicMock()
    return c


def test_table_and_schema_name_validation(cp):
    assert cp._get_table_name("threads") == '"public"."threads"'
    with pytest.raises(ValueError):
        cp._get_table_name("bad-name")


@pytest.mark.asyncio
async def test_generic_cache_methods(cp):
    assert await cp.aput_cache_value("ns", "k", {"x": 1}, ttl_seconds=12) is True
    cp.redis.get.return_value = json.dumps({"x": 1})
    assert await cp.aget_cache_value("ns", "k") == {"x": 1}
    assert await cp.aclear_cache_value("ns", "k") == 1
    keys = await cp.alist_cache_keys("ns")
    assert keys == ["a", "b"]


@pytest.mark.asyncio
async def test_generic_cache_failures_return_safe_values(cp):
    cp.redis.setex.side_effect = RuntimeError("x")
    cp.redis.get.side_effect = RuntimeError("x")
    cp.redis.delete.side_effect = RuntimeError("x")
    cp.redis.scan.side_effect = RuntimeError("x")

    assert await cp.aput_cache_value("ns", "k", {"x": 1}) is None
    assert await cp.aget_cache_value("ns", "k") is None
    assert await cp.aclear_cache_value("ns", "k") is None
    assert await cp.alist_cache_keys("ns") == []


def test_validate_config_errors(cp):
    with pytest.raises(ValueError):
        cp._validate_config({"thread_id": "t"})
    with pytest.raises(ValueError):
        cp._validate_config({"user_id": "u"})


def test_json_serializer_fallback(monkeypatch, cp):
    monkeypatch.setenv("FAST_JSON", "1")
    assert callable(cp._get_json_serializer())


def test_deserialize_state_from_bytes_and_dict(cp):
    state = AgentState()
    payload = cp._serialize_state(state).encode()
    out1 = cp._deserialize_state(payload, AgentState)
    assert isinstance(out1, AgentState)

    out2 = cp._deserialize_state(state.model_dump(), AgentState)
    assert isinstance(out2, AgentState)


def test_row_to_message_handles_string_and_bytes_json(cp):
    row = {
        "message_id": "m1",
        "role": "assistant",
        "content": "plain text",
        "tool_calls": json.dumps([{"name": "x"}]),
        "tool_call_id": None,
        "reasoning": "r",
        "created_at": None,
        "total_tokens": 0,
        "usages": json.dumps({"completion_tokens": 1, "prompt_tokens": 2, "total_tokens": 3}),
        "meta": json.dumps({"k": "v"}),
    }
    msg = cp._row_to_message(row)
    assert msg.role == "assistant"
    assert msg.metadata["k"] == "v"

    row2 = dict(row)
    row2["content"] = b"{\"type\": \"text\", \"text\": \"hi\"}"
    row2["tool_calls"] = b"invalid-json"
    row2["usages"] = b"invalid-json"
    row2["meta"] = b"invalid-json"
    msg2 = cp._row_to_message(row2)
    assert msg2.role == "assistant"


class _AcquireCtx:
    def __init__(self, conn):
        self.conn = conn

    async def __aenter__(self):
        return self.conn

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _Txn:
    """Async context manager standing in for asyncpg's conn.transaction()."""

    async def __aenter__(self):
        return None

    async def __aexit__(self, exc_type, exc, tb):
        return False


def _mock_conn(**kwargs):
    """Build an AsyncMock connection whose transaction() is a real async CM."""
    conn = AsyncMock(**kwargs)
    conn.transaction = MagicMock(return_value=_Txn())
    return conn


@pytest.mark.asyncio
async def test_message_methods_cover_query_paths(cp):
    conn = AsyncMock()
    conn.fetchrow = AsyncMock(
        return_value={
            "message_id": "m1",
            "role": "assistant",
            "content": "hello",
            "tool_calls": None,
            "tool_call_id": None,
            "reasoning": None,
            "created_at": None,
            "total_tokens": 0,
            "usages": None,
            "meta": None,
        }
    )
    conn.fetch = AsyncMock(return_value=[])
    conn.execute = AsyncMock()

    cp._pg_pool = MagicMock()
    cp._pg_pool.acquire.return_value = _AcquireCtx(conn)

    out = await cp.aget_message({"thread_id": "t1"}, "m1")
    assert out.message_id == "m1"

    rows = await cp.alist_messages({"thread_id": "t1"}, search="hello", offset=1, limit=2)
    assert rows == []

    await cp.adelete_message({"thread_id": "t1"}, "m1")
    await cp.adelete_message({}, "m1")
    assert conn.execute.await_count >= 2


@pytest.mark.asyncio
async def test_thread_methods_cover_insert_update_and_list(cp):
    conn = AsyncMock()
    # 1) insert-returning (created), 2) insert conflict -> None,
    # 3) owner-scoped UPDATE ... RETURNING (thread is ours), 4) aget_thread row.
    conn.fetchrow = AsyncMock(side_effect=[{"thread_id": "t1"}, None, {"thread_id": "t1"}, {"meta": json.dumps({"run_id": "r1"}), "thread_name": "T", "updated_at": None}])
    conn.fetch = AsyncMock(
        return_value=[
            {
                "thread_id": "t1",
                "thread_name": "T",
                "user_id": "u1",
                "created_at": None,
                "updated_at": None,
                "meta": json.dumps({"run_id": "r1"}),
            }
        ]
    )
    conn.execute = AsyncMock()

    cp._pg_pool = MagicMock()
    cp._pg_pool.acquire.return_value = _AcquireCtx(conn)

    from agentflow.utils.thread_info import ThreadInfo

    created = await cp.aput_thread({"thread_id": "t1", "user_id": "u1"}, ThreadInfo(thread_id="t1", thread_name="T"))
    assert created is True

    created2 = await cp.aput_thread(
        {"thread_id": "t1", "user_id": "u1"},
        ThreadInfo(thread_id="t1", thread_name=None, metadata={"x": 1}),
    )
    assert created2 is False

    thr = await cp.aget_thread({"thread_id": "t1", "user_id": "u1"})
    assert thr is None or thr.thread_id == "t1"

    listed = await cp.alist_threads({"user_id": "u1"}, search="T", offset=1, limit=1)
    assert len(listed) == 1


@pytest.mark.asyncio
async def test_aget_message_not_found_raises(cp):
    conn = AsyncMock()
    conn.fetchrow = AsyncMock(return_value=None)
    cp._pg_pool = MagicMock()
    cp._pg_pool.acquire.return_value = _AcquireCtx(conn)

    with pytest.raises(ValueError):
        await cp.aget_message({"thread_id": "t1"}, "missing")


def test_import_error_asyncpg(monkeypatch):
    monkeypatch.setattr("agentflow.storage.checkpointer.pg_checkpointer.HAS_ASYNCPG", False)
    with pytest.raises(ImportError) as exc:
        PgCheckpointer(postgres_dsn="postgres://x")
    assert "requires 'asyncpg'" in str(exc.value)


def test_import_error_redis(monkeypatch):
    monkeypatch.setattr("agentflow.storage.checkpointer.pg_checkpointer.HAS_ASYNCPG", True)
    monkeypatch.setattr("agentflow.storage.checkpointer.pg_checkpointer.HAS_REDIS", False)
    with pytest.raises(ImportError) as exc:
        PgCheckpointer(postgres_dsn="postgres://x")
    assert "requires 'redis'" in str(exc.value)


def test_schema_name_validation_on_init(monkeypatch):
    monkeypatch.setattr("agentflow.storage.checkpointer.pg_checkpointer.HAS_ASYNCPG", True)
    monkeypatch.setattr("agentflow.storage.checkpointer.pg_checkpointer.HAS_REDIS", True)
    with pytest.raises(ValueError):
        PgCheckpointer(postgres_dsn="postgres://x", redis=MagicMock(), schema="invalid-schema-name")


def test_init_with_pools(monkeypatch):
    monkeypatch.setattr("agentflow.storage.checkpointer.pg_checkpointer.HAS_ASYNCPG", True)
    monkeypatch.setattr("agentflow.storage.checkpointer.pg_checkpointer.HAS_REDIS", True)
    
    mock_pg_pool = MagicMock()
    mock_redis_pool = MagicMock()
    
    with patch("agentflow.storage.checkpointer.pg_checkpointer.Redis") as mock_redis_class:
        cp = PgCheckpointer(pg_pool=mock_pg_pool, redis_pool=mock_redis_pool)
        assert cp._pg_pool is mock_pg_pool
        mock_redis_class.assert_called_once_with(connection_pool=mock_redis_pool)


def test_create_redis_pool_no_url(monkeypatch):
    monkeypatch.setattr("agentflow.storage.checkpointer.pg_checkpointer.HAS_ASYNCPG", True)
    monkeypatch.setattr("agentflow.storage.checkpointer.pg_checkpointer.HAS_REDIS", True)
    
    cp = PgCheckpointer(postgres_dsn="postgres://x", redis=MagicMock())
    with pytest.raises(ValueError):
        cp._create_redis_pool(redis=None, redis_pool=None, redis_url=None, redis_pool_config={})


def test_create_pg_pool(cp):
    mock_pool = MagicMock()
    assert cp._create_pg_pool(pg_pool=mock_pool, postgres_dsn=None, pool_config={}) is mock_pool
    
    with patch("asyncpg.create_pool") as mock_create_pool:
        cp._create_pg_pool(pg_pool=None, postgres_dsn="postgres://url", pool_config={"min_size": 5})
        mock_create_pool.assert_called_once_with(dsn="postgres://url", min_size=5)


@pytest.mark.asyncio
async def test_get_pg_pool_lazy(monkeypatch):
    monkeypatch.setattr("agentflow.storage.checkpointer.pg_checkpointer.HAS_ASYNCPG", True)
    monkeypatch.setattr("agentflow.storage.checkpointer.pg_checkpointer.HAS_REDIS", True)
    
    cp = PgCheckpointer(postgres_dsn="postgres://dsn", redis=MagicMock())
    assert cp._pg_pool is None
    
    mock_pool = MagicMock()
    async def mock_create_pool(*args, **kwargs):
        return mock_pool
        
    monkeypatch.setattr(cp, "_create_pg_pool", mock_create_pool)
    pool = await cp._get_pg_pool()
    assert pool is mock_pool
    assert cp._pg_pool is mock_pool


def test_json_serializer_fast_json_importers(monkeypatch, cp):
    monkeypatch.setenv("FAST_JSON", "1")
    
    import builtins
    real_import = builtins.__import__
    def mock_import(name, *args, **kwargs):
        if name in ("orjson", "msgspec"):
            raise ImportError
        return real_import(name, *args, **kwargs)
        
    with patch("builtins.__import__", side_effect=mock_import):
        serializer = cp._get_json_serializer()
        assert serializer == json.dumps


@pytest.mark.asyncio
async def test_check_and_apply_schema_version_upgrade(cp):
    # Recorded at v1, target v2 -> runs the v2 migration steps, then records v2.
    conn = AsyncMock()
    conn.fetchrow = AsyncMock(return_value={"version": 1})

    await cp._check_and_apply_schema_version(conn)

    executed = [call.args[0] for call in conn.execute.await_args_list]
    # The v2 migration adds the version column and its unique index...
    assert any("ADD COLUMN IF NOT EXISTS version" in sql for sql in executed)
    assert any("uq_states_thread_version" in sql for sql in executed)
    # ...and finally records the new schema version.
    assert any("INSERT INTO" in sql and "schema_version" in sql for sql in executed)


@pytest.mark.asyncio
async def test_check_and_apply_schema_version_exception(cp):
    # A failing migration must surface as a SchemaVersionError, not be swallowed.
    from agentflow.core.exceptions.storage_exceptions import SchemaVersionError

    conn = AsyncMock()
    conn.fetchrow = AsyncMock(side_effect=RuntimeError("db error"))

    with pytest.raises(SchemaVersionError):
        await cp._check_and_apply_schema_version(conn)


@pytest.mark.asyncio
async def test_initialize_schema_early_return(cp):
    cp._schema_initialized = True
    await cp._initialize_schema()
    assert cp._pg_pool is None or not cp._pg_pool.acquire.called


@pytest.mark.asyncio
async def test_initialize_schema_error(cp):
    cp._schema_initialized = False

    conn = _mock_conn()
    conn.execute = AsyncMock(side_effect=RuntimeError("sql error"))

    cp._pg_pool = MagicMock()
    cp._pg_pool.acquire.return_value = _AcquireCtx(conn)

    with pytest.raises(RuntimeError):
        await cp._initialize_schema()


def test_serialize_state_enum_handler(cp):
    class MockEnum(Enum):
        VAL = "enum_val"
        
    class MockObj:
        def __str__(self):
            return "str_obj"
            
    mock_state = MagicMock()
    mock_state.model_dump.return_value = {"enum": MockEnum.VAL, "obj": MockObj()}
    serialized = cp._serialize_state(mock_state)
    loaded = json.loads(serialized)
    assert loaded["enum"] == "enum_val"
    assert loaded["obj"] == "str_obj"


def test_deserialize_state_errors(cp):
    class BadState:
        @classmethod
        def model_validate(cls, d):
            raise TypeError("validation error")

    with pytest.raises(TypeError):
        cp._deserialize_state({"invalid": "data"}, BadState)

    with pytest.raises(json.JSONDecodeError):
        cp._deserialize_state("invalid-str", BadState)


# ---------------------------------------------------------------------------
# Versioning / optimistic concurrency control (audit B1, M7)
# ---------------------------------------------------------------------------


def _bind_conn(cp, conn):
    cp._pg_pool = MagicMock()
    cp._pg_pool.acquire.return_value = _AcquireCtx(conn)


@pytest.mark.asyncio
async def test_aput_state_appends_next_version_and_records_it(cp):
    conn = _mock_conn()
    # _write_state_row: SELECT user_id FOR UPDATE, then MAX(version).
    conn.fetchrow = AsyncMock(side_effect=[{"user_id": "u1"}, {"v": 4}])
    _bind_conn(cp, conn)

    config = {"thread_id": "t1", "user_id": "u1"}
    await cp.aput_state(config, AgentState())

    # Next version after 4 is 5, recorded on config for a subsequent CAS.
    assert config["_checkpoint_version"] == 5
    inserts = [c.args for c in conn.execute.await_args_list if "INSERT INTO" in c.args[0]]
    state_insert = next(a for a in inserts if "states" in a[0])
    assert state_insert[2] == 5  # version param


@pytest.mark.asyncio
async def test_aput_state_conflict_raises_stale(cp):
    from agentflow.core.exceptions.storage_exceptions import StaleStateError

    conn = _mock_conn()
    # Current version is 7 but the caller based its write on version 5.
    conn.fetchrow = AsyncMock(side_effect=[{"user_id": "u1"}, {"v": 7}])
    _bind_conn(cp, conn)

    config = {"thread_id": "t1", "user_id": "u1", "_checkpoint_version": 5}
    with pytest.raises(StaleStateError):
        await cp.aput_state(config, AgentState())


@pytest.mark.asyncio
async def test_aput_state_rejects_cross_user_thread(cp):
    conn = _mock_conn()
    conn.fetchrow = AsyncMock(side_effect=[{"user_id": "someone_else"}])
    _bind_conn(cp, conn)

    from agentflow.core.exceptions.storage_exceptions import StorageError

    config = {"thread_id": "t1", "user_id": "u1"}
    with pytest.raises(StorageError):
        await cp.aput_state(config, AgentState())


@pytest.mark.asyncio
async def test_aput_checkpoint_writes_state_and_messages_atomically(cp):
    conn = _mock_conn()
    conn.fetchrow = AsyncMock(side_effect=[{"user_id": "u1"}, {"v": 0}])
    _bind_conn(cp, conn)

    config = {"thread_id": "t1", "user_id": "u1"}
    from agentflow.core.state import Message

    msgs = [Message.text_message("hi", role="user", message_id="m1")]
    await cp.aput_checkpoint(config, AgentState(), msgs)

    # Exactly one transaction opened for the combined write.
    assert conn.transaction.call_count == 1
    executed = [c.args[0] for c in conn.execute.await_args_list]
    assert any("states" in sql and "INSERT INTO" in sql for sql in executed)
    assert any("messages" in sql and "INSERT INTO" in sql for sql in executed)
    assert config["_checkpoint_version"] == 1


@pytest.mark.asyncio
async def test_aput_state_prunes_old_versions(cp):
    cp.state_history_limit = 3
    conn = _mock_conn()
    conn.fetchrow = AsyncMock(side_effect=[{"user_id": "u1"}, {"v": 10}])
    _bind_conn(cp, conn)

    config = {"thread_id": "t1", "user_id": "u1"}
    await cp.aput_state(config, AgentState())

    deletes = [c.args for c in conn.execute.await_args_list if c.args[0].startswith("DELETE")]
    assert deletes, "expected a prune DELETE"
    # New version is 11, keep 3 -> delete version <= 8.
    assert deletes[0][2] == 8


@pytest.mark.asyncio
async def test_aget_state_orders_by_version_and_scopes_user(cp):
    conn = _mock_conn()
    payload = cp._serialize_state_payload(AgentState())
    conn.fetchrow = AsyncMock(return_value={"version": 9, "state_data": json.dumps(payload)})
    _bind_conn(cp, conn)

    config = {"thread_id": "t1", "user_id": "u1"}
    state = await cp.aget_state(config)

    assert isinstance(state, AgentState)
    # Read version is recorded so a later write can compare-and-swap against it.
    assert config["_checkpoint_version"] == 9
    sql = conn.fetchrow.await_args_list[0].args[0]
    assert "ORDER BY version DESC" in sql
    assert "user_id = $2" in sql


def test_deserialize_payload_falls_back_on_unknown_class(cp):
    # A renamed/removed state class must not brick history: fall back to AgentState.
    data = AgentState().model_dump(mode="json")
    data["__class_path__"] = "some.removed.module.GoneState"
    state = cp._deserialize_state_payload(data)
    assert isinstance(state, AgentState)


def test_serialize_payload_uses_json_mode(cp):
    # mode="json" must be used so datetime/UUID/enum fields don't crash json.dumps.
    payload = cp._serialize_state_payload(AgentState())
    assert "__class_path__" in payload
    # Round-trips through json without a custom default handler.
    assert json.loads(json.dumps(payload))


# ---------------------------------------------------------------------------
# Cache must never move backwards, and must not wedge a thread
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cache_write_is_version_guarded_cas(cp):
    # The cache write must go through the Lua CAS (never a blind SETEX), passing
    # the current version so a stale run cannot overwrite a newer cached state.
    config = {"thread_id": "t1", "user_id": "u1", "_checkpoint_version": 5}
    await cp.aput_state_cache(config, AgentState())

    cp.redis.setex.assert_not_called()
    cp.redis.eval.assert_awaited_once()
    args = cp.redis.eval.await_args.args
    assert args[1] == 1  # numkeys
    assert args[2] == "state_cache:t1:u1"
    assert args[4] == "5"  # version passed to the guard


@pytest.mark.asyncio
async def test_cache_write_skipped_when_guard_rejects(cp):
    # Lua returned 0 -> the cached entry is newer; we must not claim success.
    cp.redis.eval = AsyncMock(return_value=0)
    config = {"thread_id": "t1", "user_id": "u1", "_checkpoint_version": 3}
    assert await cp.aput_state_cache(config, AgentState()) is None


@pytest.mark.asyncio
async def test_cache_write_with_no_version_uses_sentinel(cp):
    # Brand-new thread: no durable version yet -> sentinel must be below any real
    # version so it can never clobber a newer cached entry.
    config = {"thread_id": "t1", "user_id": "u1"}
    await cp.aput_state_cache(config, AgentState())
    assert cp.redis.eval.await_args.args[4] == "-1"


@pytest.mark.asyncio
async def test_stale_write_invalidates_cache_so_thread_is_not_wedged(cp):
    from agentflow.core.exceptions.storage_exceptions import StaleStateError

    conn = _mock_conn()
    conn.fetchrow = AsyncMock(side_effect=[{"user_id": "u1"}, {"v": 7}])
    _bind_conn(cp, conn)

    config = {"thread_id": "t1", "user_id": "u1", "_checkpoint_version": 5}
    with pytest.raises(StaleStateError):
        await cp.aput_state(config, AgentState())

    # The cache may hold state built on v5; it must be dropped so the next read
    # falls back to Postgres instead of re-seeding the same doomed version.
    cp.redis.delete.assert_awaited_with("state_cache:t1:u1")


# ---------------------------------------------------------------------------
# Cross-tenant isolation on messages / threads
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_messages_is_owner_scoped(cp):
    conn = _mock_conn()
    conn.fetch = AsyncMock(return_value=[])
    _bind_conn(cp, conn)

    await cp.alist_messages({"thread_id": "t1", "user_id": "u1"})

    sql = conn.fetch.await_args.args[0]
    assert "thread_id IN (SELECT thread_id FROM" in sql
    assert "user_id = $2" in sql


@pytest.mark.asyncio
async def test_delete_message_is_owner_scoped(cp):
    conn = _mock_conn()
    _bind_conn(cp, conn)

    await cp.adelete_message({"thread_id": "t1", "user_id": "u1"}, "m1")

    sql = conn.execute.await_args.args[0]
    assert "DELETE FROM" in sql
    assert "thread_id IN (SELECT thread_id FROM" in sql


@pytest.mark.asyncio
async def test_get_message_is_owner_scoped(cp):
    conn = _mock_conn()
    conn.fetchrow = AsyncMock(
        return_value={
            "message_id": "m1",
            "role": "assistant",
            "content": "hi",
            "tool_calls": None,
            "tool_call_id": None,
            "reasoning": None,
            "created_at": None,
            "total_tokens": 0,
            "usages": None,
            "meta": None,
        }
    )
    _bind_conn(cp, conn)

    await cp.aget_message({"thread_id": "t1", "user_id": "u1"}, "m1")

    sql = conn.fetchrow.await_args.args[0]
    assert "thread_id IN (SELECT thread_id FROM" in sql


@pytest.mark.asyncio
async def test_ensure_thread_exists_rejects_thread_owned_by_other_user(cp):
    from agentflow.core.exceptions.storage_exceptions import StorageError

    conn = _mock_conn()
    # INSERT ... ON CONFLICT DO NOTHING no-ops; the thread belongs to someone else.
    conn.fetchval = AsyncMock(return_value="someone_else")
    _bind_conn(cp, conn)

    with pytest.raises(StorageError):
        await cp._ensure_thread_exists("t1", "u1", {})


@pytest.mark.asyncio
async def test_ensure_thread_exists_accepts_own_thread(cp):
    conn = _mock_conn()
    conn.fetchval = AsyncMock(return_value="u1")
    _bind_conn(cp, conn)

    await cp._ensure_thread_exists("t1", "u1", {})  # must not raise


# ---------------------------------------------------------------------------
# enforce_user_isolation: secure by default, but the developer can opt out
# ---------------------------------------------------------------------------


def test_user_isolation_is_on_by_default(cp):
    assert cp.enforce_user_isolation is True
    assert cp._isolation_active("u1") is True


def test_isolation_inactive_without_a_user_id(cp):
    # Scoping on a missing user_id would match nothing -- worse than not scoping.
    assert cp._isolation_active(None) is False
    assert cp._isolation_active("") is False


def test_thread_scope_joins_when_isolation_on(cp):
    sql, params = cp._thread_scope("t1", "u1")
    assert "SELECT thread_id FROM" in sql
    assert params == ["t1", "u1"]


def test_thread_scope_skips_join_when_isolation_off(monkeypatch):
    monkeypatch.setattr("agentflow.storage.checkpointer.pg_checkpointer.HAS_ASYNCPG", True)
    monkeypatch.setattr("agentflow.storage.checkpointer.pg_checkpointer.HAS_REDIS", True)
    cp = PgCheckpointer(
        postgres_dsn="postgres://x",
        redis=MagicMock(),
        enforce_user_isolation=False,
    )
    sql, params = cp._thread_scope("t1", "u1")
    assert sql == "thread_id = $1"
    assert params == ["t1"]  # no join, no user_id param


@pytest.mark.asyncio
async def test_isolation_off_allows_writing_thread_owned_by_other(monkeypatch):
    # Single-tenant / no-real-identity setups must not be forced into ownership
    # errors just because a placeholder user_id differs.
    monkeypatch.setattr("agentflow.storage.checkpointer.pg_checkpointer.HAS_ASYNCPG", True)
    monkeypatch.setattr("agentflow.storage.checkpointer.pg_checkpointer.HAS_REDIS", True)

    redis = MagicMock()
    redis.eval = AsyncMock(return_value=1)
    redis.delete = AsyncMock(return_value=1)
    cp = PgCheckpointer(
        postgres_dsn="postgres://x",
        redis=redis,
        enforce_user_isolation=False,
    )

    conn = _mock_conn()
    conn.fetchrow = AsyncMock(side_effect=[{"user_id": "someone_else"}, {"v": 0}])
    _bind_conn(cp, conn)

    # Would raise STORAGE_FORBIDDEN_001 if isolation were enforced.
    config = {"thread_id": "t1", "user_id": "u1"}
    await cp.aput_state(config, AgentState())
    assert config["_checkpoint_version"] == 1


@pytest.mark.asyncio
async def test_isolation_off_skips_user_filter_on_messages(monkeypatch):
    monkeypatch.setattr("agentflow.storage.checkpointer.pg_checkpointer.HAS_ASYNCPG", True)
    monkeypatch.setattr("agentflow.storage.checkpointer.pg_checkpointer.HAS_REDIS", True)
    cp = PgCheckpointer(
        postgres_dsn="postgres://x",
        redis=MagicMock(),
        enforce_user_isolation=False,
    )

    conn = _mock_conn()
    conn.fetch = AsyncMock(return_value=[])
    _bind_conn(cp, conn)

    await cp.alist_messages({"thread_id": "t1", "user_id": "u1"})
    sql = conn.fetch.await_args.args[0]
    assert "SELECT thread_id FROM" not in sql  # no ownership join


@pytest.mark.asyncio
async def test_schema_init_takes_advisory_lock(cp):
    # Multiple workers/replicas may initialize concurrently; DDL must be serialized
    # across processes, not just within one.
    conn = _mock_conn()
    conn.fetchrow = AsyncMock(return_value=None)
    _bind_conn(cp, conn)
    cp._schema_initialized = False

    await cp._initialize_schema()

    executed = [c.args[0] for c in conn.execute.await_args_list]
    assert any("pg_advisory_xact_lock" in sql for sql in executed)

