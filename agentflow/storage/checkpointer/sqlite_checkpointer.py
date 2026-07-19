"""SQLite-backed checkpointer for client-side and single-user agent workloads.

This module provides :class:`SqliteCheckpointer`, a :class:`BaseCheckpointer`
implementation that keeps *everything* — durable state, the hot "realtime" state
cache, a generic TTL cache, messages, and threads — in a single local SQLite
database file. No external services (Postgres, Redis) are required.

It is the right choice for:

- Client-side / embedded agents: a desktop app (Tauri, Electron, PyInstaller)
  that ships a Python sidecar, or a local CLI agent.
- Dedicated-room deployments where each user has their own process and their own
  database file.

It is **not** the right choice for multi-user servers where many users share one
backend. SQLite serializes writers and does not scale horizontally; use
:class:`~agentflow.storage.checkpointer.pg_checkpointer.PgCheckpointer` there.

Install with::

    pip install 10xscale-agentflow[sqlite_checkpoint]
"""

import asyncio
import importlib
import json
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from enum import Enum
from pathlib import Path
from typing import Any, TypeVar

from agentflow.core.state import AgentState, Message
from agentflow.utils.callable_utils import run_coroutine
from agentflow.utils.thread_info import ThreadInfo

from .base_checkpointer import BaseCheckpointer


try:
    import aiosqlite

    HAS_AIOSQLITE = True
except ImportError:
    HAS_AIOSQLITE = False
    aiosqlite = None  # type: ignore[assignment]

logger = logging.getLogger("agentflow.checkpointer.sqlite")

StateT = TypeVar("StateT", bound="AgentState")

# Default database location for desktop / single-user agents.
DEFAULT_DB_PATH = str(Path.home() / ".agentflow" / "checkpointer.db")

# Sentinel key holding the fully-qualified AgentState subclass path so state can
# be recovered into its real type (mirrors PgCheckpointer behavior).
_CLASS_PATH_KEY = "__class_path__"

DDL_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS af_states (
        thread_id TEXT NOT NULL PRIMARY KEY,
        state_data TEXT NOT NULL,
        updated_at REAL NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS af_state_cache (
        thread_id TEXT NOT NULL PRIMARY KEY,
        state_data TEXT NOT NULL,
        updated_at REAL NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS af_messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        thread_id TEXT NOT NULL,
        message_id TEXT NOT NULL,
        message_data TEXT NOT NULL,
        created_at REAL NOT NULL,
        UNIQUE (thread_id, message_id)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_af_messages_thread ON af_messages (thread_id)",
    """
    CREATE TABLE IF NOT EXISTS af_threads (
        thread_id TEXT NOT NULL PRIMARY KEY,
        thread_data TEXT NOT NULL,
        updated_at REAL NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS af_cache (
        namespace TEXT NOT NULL,
        key TEXT NOT NULL,
        value TEXT NOT NULL,
        expires_at REAL,
        PRIMARY KEY (namespace, key)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_af_cache_namespace ON af_cache (namespace)",
)


def _enum_handler(obj: Any) -> Any:
    if isinstance(obj, Enum):
        return obj.value
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _dumps(data: Any) -> str:
    return json.dumps(data, default=_enum_handler)


class SqliteCheckpointer(BaseCheckpointer[StateT]):
    """SQLite-backed checkpointer that stores all data in one local ``.db`` file.

    Everything a graph needs to persist — durable state, the realtime state
    cache, generic TTL cache, messages, and threads — lives in a single SQLite
    file accessed through ``aiosqlite`` (fully async). A single persistent
    connection is opened lazily; WAL journal mode is enabled for better
    read concurrency and writes are serialized behind an ``asyncio.Lock``.

    State is reconstructed into its exact :class:`AgentState` subclass on read
    via an embedded class path, matching
    :class:`~agentflow.storage.checkpointer.pg_checkpointer.PgCheckpointer`.

    Args:
        db_path: Path to the SQLite database file. Defaults to
            ``~/.agentflow/checkpointer.db``. Parent directories are created on
            setup. Use ``":memory:"`` for an ephemeral in-process database
            (useful for tests).

    Raises:
        ImportError: If ``aiosqlite`` is not installed.

    Example::

        checkpointer = SqliteCheckpointer("agent.db")
        graph = builder.compile(checkpointer=checkpointer)
        await graph.ainvoke({"messages": [...]}, config={"thread_id": "t1"})
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        if not HAS_AIOSQLITE:
            raise ImportError(
                "SqliteCheckpointer requires 'aiosqlite'. "
                "Install with: pip install 10xscale-agentflow[sqlite_checkpoint]"
            )

        if db_path == ":memory:":
            self.db_path: str | Path = ":memory:"
        else:
            self.db_path = Path(db_path) if db_path else Path(DEFAULT_DB_PATH)

        self._conn: aiosqlite.Connection | None = None
        self._setup_done = False
        self._setup_lock = asyncio.Lock()
        self._write_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_config_key(self, config: dict[str, Any]) -> str:
        return str(config.get("thread_id", ""))

    def _serialize_state(self, state: StateT) -> str:
        data = state.model_dump(mode="json")
        cls = state.__class__
        data[_CLASS_PATH_KEY] = f"{cls.__module__}.{cls.__name__}"
        return _dumps(data)

    def _deserialize_state(self, raw: str) -> StateT:
        data = json.loads(raw)
        class_path = data.pop(_CLASS_PATH_KEY, None)
        if not class_path:
            raise ValueError("Missing '__class_path__' in stored state data")
        module_name, class_name = class_path.rsplit(".", 1)
        cls = getattr(importlib.import_module(module_name), class_name)
        return cls.model_validate(data)  # type: ignore[no-any-return]

    async def _ensure_setup(self) -> None:
        if self._setup_done:
            return
        async with self._setup_lock:
            if not self._setup_done:
                await self.asetup()

    async def _get_conn(self) -> "aiosqlite.Connection":
        """Return the shared connection, opening and configuring it if needed."""
        if self._conn is None:
            conn = await aiosqlite.connect(self.db_path)
            conn.row_factory = aiosqlite.Row
            await conn.execute("PRAGMA journal_mode=WAL")
            await conn.execute("PRAGMA synchronous=NORMAL")
            self._conn = conn
        return self._conn

    @asynccontextmanager
    async def _write(self) -> AsyncIterator["aiosqlite.Connection"]:
        """Serialize a write transaction and commit on success."""
        conn = await self._get_conn()
        async with self._write_lock:
            yield conn
            await conn.commit()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self) -> None:
        return run_coroutine(self.asetup())

    async def asetup(self) -> None:
        """Create the database directory (if any) and all required tables."""
        if isinstance(self.db_path, Path):
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = await self._get_conn()
        async with self._write_lock:
            for stmt in DDL_STATEMENTS:
                await conn.execute(stmt)
            await conn.commit()
        self._setup_done = True
        logger.info("SqliteCheckpointer setup complete: %s", self.db_path)

    # ------------------------------------------------------------------
    # State — async
    # ------------------------------------------------------------------

    async def aput_state(self, config: dict[str, Any], state: StateT) -> StateT:
        await self._ensure_setup()
        key = self._get_config_key(config)
        data = self._serialize_state(state)
        async with self._write() as conn:
            await conn.execute(
                """
                INSERT INTO af_states (thread_id, state_data, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(thread_id) DO UPDATE SET
                    state_data = excluded.state_data,
                    updated_at = excluded.updated_at
                """,
                (key, data, time.time()),
            )
        logger.debug("Stored state for thread: %s", key)
        return state

    async def aget_state(self, config: dict[str, Any]) -> StateT | None:
        await self._ensure_setup()
        key = self._get_config_key(config)
        conn = await self._get_conn()
        async with conn.execute(
            "SELECT state_data FROM af_states WHERE thread_id = ?", (key,)
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return None
        return self._deserialize_state(row["state_data"])

    async def aclear_state(self, config: dict[str, Any]) -> bool:
        await self._ensure_setup()
        key = self._get_config_key(config)
        async with self._write() as conn:
            await conn.execute("DELETE FROM af_states WHERE thread_id = ?", (key,))
        return True

    # ------------------------------------------------------------------
    # State cache — async (hot layer, same file, no Redis)
    # ------------------------------------------------------------------

    async def aput_state_cache(self, config: dict[str, Any], state: StateT) -> StateT:
        await self._ensure_setup()
        key = self._get_config_key(config)
        data = self._serialize_state(state)
        async with self._write() as conn:
            await conn.execute(
                """
                INSERT INTO af_state_cache (thread_id, state_data, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(thread_id) DO UPDATE SET
                    state_data = excluded.state_data,
                    updated_at = excluded.updated_at
                """,
                (key, data, time.time()),
            )
        return state

    async def aget_state_cache(self, config: dict[str, Any]) -> StateT | None:
        await self._ensure_setup()
        key = self._get_config_key(config)
        conn = await self._get_conn()
        async with conn.execute(
            "SELECT state_data FROM af_state_cache WHERE thread_id = ?", (key,)
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return None
        return self._deserialize_state(row["state_data"])

    # ------------------------------------------------------------------
    # Generic cache — async (TTL, lazy prune on read)
    # ------------------------------------------------------------------

    async def aput_cache_value(
        self,
        namespace: str,
        key: str,
        value: Any,
        ttl_seconds: int | None = None,
    ) -> Any | None:
        await self._ensure_setup()
        expires_at = time.time() + ttl_seconds if ttl_seconds else None
        async with self._write() as conn:
            await conn.execute(
                """
                INSERT INTO af_cache (namespace, key, value, expires_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(namespace, key) DO UPDATE SET
                    value = excluded.value,
                    expires_at = excluded.expires_at
                """,
                (namespace, key, _dumps(value), expires_at),
            )
        return value

    async def aget_cache_value(self, namespace: str, key: str) -> Any | None:
        await self._ensure_setup()
        conn = await self._get_conn()
        async with conn.execute(
            "SELECT value, expires_at FROM af_cache WHERE namespace = ? AND key = ?",
            (namespace, key),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return None
        expires_at = row["expires_at"]
        if expires_at is not None and expires_at <= time.time():
            async with self._write() as conn:
                await conn.execute(
                    "DELETE FROM af_cache WHERE namespace = ? AND key = ?",
                    (namespace, key),
                )
            return None
        return json.loads(row["value"])

    async def aclear_cache_value(self, namespace: str, key: str) -> Any | None:
        await self._ensure_setup()
        conn = await self._get_conn()
        async with conn.execute(
            "SELECT value FROM af_cache WHERE namespace = ? AND key = ?",
            (namespace, key),
        ) as cursor:
            row = await cursor.fetchone()
        async with self._write() as conn:
            await conn.execute(
                "DELETE FROM af_cache WHERE namespace = ? AND key = ?",
                (namespace, key),
            )
        return json.loads(row["value"]) if row else None

    async def alist_cache_keys(
        self,
        namespace: str,
        prefix: str | None = None,
    ) -> list[str]:
        await self._ensure_setup()
        async with self._write() as conn:
            await conn.execute(
                "DELETE FROM af_cache WHERE namespace = ? AND expires_at IS NOT NULL "
                "AND expires_at <= ?",
                (namespace, time.time()),
            )
        conn = await self._get_conn()
        async with conn.execute(
            "SELECT key FROM af_cache WHERE namespace = ?", (namespace,)
        ) as cursor:
            rows = await cursor.fetchall()
        keys = [row["key"] for row in rows]
        if prefix is not None:
            keys = [k for k in keys if k.startswith(prefix)]
        return keys

    # ------------------------------------------------------------------
    # Messages — async
    # ------------------------------------------------------------------

    async def aput_messages(
        self,
        config: dict[str, Any],
        messages: list[Message],
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        await self._ensure_setup()
        key = self._get_config_key(config)
        now = time.time()
        async with self._write() as conn:
            for msg in messages:
                await conn.execute(
                    """
                    INSERT INTO af_messages (thread_id, message_id, message_data, created_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(thread_id, message_id) DO UPDATE SET
                        message_data = excluded.message_data
                    """,
                    (key, str(msg.message_id), _dumps(msg.model_dump(mode="json")), now),
                )
        logger.debug("Stored %d messages for thread: %s", len(messages), key)
        return True

    async def aget_message(self, config: dict[str, Any], message_id: str | int) -> Message:
        await self._ensure_setup()
        key = self._get_config_key(config)
        conn = await self._get_conn()
        async with conn.execute(
            "SELECT message_data FROM af_messages WHERE thread_id = ? AND message_id = ?",
            (key, str(message_id)),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            raise IndexError(f"Message with ID {message_id} not found for thread: {key}")
        return Message.model_validate(json.loads(row["message_data"]))

    async def alist_messages(
        self,
        config: dict[str, Any],
        search: str | None = None,
        offset: int | None = None,
        limit: int | None = None,
    ) -> list[Message]:
        await self._ensure_setup()
        key = self._get_config_key(config)
        conn = await self._get_conn()
        async with conn.execute(
            "SELECT message_data FROM af_messages WHERE thread_id = ? ORDER BY id", (key,)
        ) as cursor:
            rows = await cursor.fetchall()

        messages: list[Message] = [
            Message.model_validate(json.loads(row["message_data"])) for row in rows
        ]

        if search:
            needle = search.lower()
            messages = [
                m for m in messages if hasattr(m, "content") and needle in str(m.content).lower()
            ]

        start = offset or 0
        end = (start + limit) if limit else None
        return messages[start:end]

    async def adelete_message(self, config: dict[str, Any], message_id: str | int) -> bool:
        await self._ensure_setup()
        key = self._get_config_key(config)
        conn = await self._get_conn()
        async with conn.execute(
            "SELECT id FROM af_messages WHERE thread_id = ? AND message_id = ?",
            (key, str(message_id)),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            raise IndexError(f"Message with ID {message_id} not found for thread: {key}")
        async with self._write() as conn:
            await conn.execute(
                "DELETE FROM af_messages WHERE thread_id = ? AND message_id = ?",
                (key, str(message_id)),
            )
        return True

    # ------------------------------------------------------------------
    # Threads — async
    # ------------------------------------------------------------------

    async def aput_thread(self, config: dict[str, Any], thread_info: ThreadInfo) -> bool:
        await self._ensure_setup()
        key = self._get_config_key(config) or str(thread_info.thread_id)
        if str(thread_info.thread_id) != key:
            raise ValueError(
                "ThreadInfo.thread_id must match config['thread_id']: "
                f"{thread_info.thread_id!r} != {key!r}"
            )
        data = _dumps(thread_info.model_dump(mode="json"))
        async with self._write() as conn:
            await conn.execute(
                """
                INSERT INTO af_threads (thread_id, thread_data, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(thread_id) DO UPDATE SET
                    thread_data = excluded.thread_data,
                    updated_at = excluded.updated_at
                """,
                (key, data, time.time()),
            )
        return True

    async def aget_thread(self, config: dict[str, Any]) -> ThreadInfo | None:
        await self._ensure_setup()
        key = self._get_config_key(config)
        conn = await self._get_conn()
        async with conn.execute(
            "SELECT thread_data FROM af_threads WHERE thread_id = ?", (key,)
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return None
        return ThreadInfo.model_validate(json.loads(row["thread_data"]))

    async def aget_thread_owner(self, thread_id: str | int) -> str | int | None:
        """Return the owning ``user_id`` for ``thread_id`` (global, not owner-scoped)."""
        await self._ensure_setup()
        conn = await self._get_conn()
        async with conn.execute(
            "SELECT thread_data FROM af_threads WHERE thread_id = ?", (str(thread_id),)
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return None
        return json.loads(row["thread_data"]).get("user_id")

    async def alist_threads(
        self,
        config: dict[str, Any],
        search: str | None = None,
        offset: int | None = None,
        limit: int | None = None,
    ) -> list[ThreadInfo]:
        await self._ensure_setup()
        conn = await self._get_conn()
        async with conn.execute(
            "SELECT thread_data FROM af_threads ORDER BY updated_at DESC"
        ) as cursor:
            rows = await cursor.fetchall()

        threads: list[ThreadInfo] = [
            ThreadInfo.model_validate(json.loads(row["thread_data"])) for row in rows
        ]

        if search:
            needle = search.lower()
            threads = [
                t for t in threads if any(needle in str(v).lower() for v in t.model_dump().values())
            ]

        start = offset or 0
        end = (start + limit) if limit else None
        return threads[start:end]

    async def aclean_thread(self, config: dict[str, Any]) -> bool:
        await self._ensure_setup()
        key = self._get_config_key(config)
        async with self._write() as conn:
            await conn.execute("DELETE FROM af_threads WHERE thread_id = ?", (key,))
            await conn.execute("DELETE FROM af_messages WHERE thread_id = ?", (key,))
            await conn.execute("DELETE FROM af_states WHERE thread_id = ?", (key,))
            await conn.execute("DELETE FROM af_state_cache WHERE thread_id = ?", (key,))
        return True

    # ------------------------------------------------------------------
    # Release
    # ------------------------------------------------------------------

    async def arelease(self) -> bool:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None
            self._setup_done = False
        logger.info("SqliteCheckpointer released: %s", self.db_path)
        return True

    def release(self) -> bool:
        return run_coroutine(self.arelease())
