import asyncio
import json
import logging
import os
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
    aiosqlite = None  # type: ignore

logger = logging.getLogger("agentflow.checkpointer.sqlite")

StateT = TypeVar("StateT", bound="AgentState")

# Default SQLite database path for desktop agents
DEFAULT_DB_PATH = os.path.join(os.path.expanduser("~"), ".agentflow", "checkpointer.db")

DDL_STATEMENTS = [
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
]


def _enum_handler(obj: Any) -> Any:
    if isinstance(obj, Enum):
        return obj.value
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _serialize_model(model: Any) -> str:
    """Serialize a Pydantic model to JSON, handling datetime and enums via Pydantic."""
    return json.dumps(model.model_dump(mode="json"))


def _serialize(data: Any) -> str:
    return json.dumps(data, default=_enum_handler)


class SqliteCheckpointer(BaseCheckpointer[StateT]):
    """
    SQLite-backed checkpointer for desktop/client-side agent workloads.

    Stores all agent state, messages, threads, and generic cache in a local
    SQLite database using ``aiosqlite`` for fully async I/O.  No external
    services (Postgres, Redis) are required — a single ``.db`` file is the
    only dependency, making it ideal for desktop agents or single-user
    environments.

    Performance notes:
        - WAL journal mode is enabled on setup for better concurrent reads.
        - All tables are indexed by ``thread_id``.
        - Generic cache uses a dedicated table with TTL-based expiry; expired
          entries are lazily pruned on read.

    Args:
        db_path (str | Path | None): Path to the SQLite database file.
            Defaults to ``~/.agentflow/checkpointer.db``.

    Raises:
        ImportError: If ``aiosqlite`` is not installed.

    Example::

        checkpointer = SqliteCheckpointer()  # default path
        checkpointer = SqliteCheckpointer("agent.db")  # custom path
        await checkpointer.asetup()  # create tables
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        if not HAS_AIOSQLITE:
            raise ImportError(
                "SqliteCheckpointer requires 'aiosqlite'. "
                "Install with: pip install 10xscale-agentflow[sqlite_checkpoint]"
            )

        self.db_path = Path(db_path) if db_path else Path(DEFAULT_DB_PATH)
        self._setup_done = False
        self._setup_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_config_key(self, config: dict[str, Any]) -> str:
        return str(config.get("thread_id", ""))

    @asynccontextmanager
    async def _get_db(self) -> AsyncIterator["aiosqlite.Connection"]:
        """Open (and configure) an aiosqlite connection as an async context manager."""
        async with aiosqlite.connect(self.db_path) as conn:
            conn.row_factory = aiosqlite.Row
            await conn.execute("PRAGMA journal_mode=WAL")
            await conn.execute("PRAGMA synchronous=NORMAL")
            await conn.execute("PRAGMA foreign_keys=ON")
            yield conn

    async def _ensure_setup(self) -> None:
        if self._setup_done:
            return
        async with self._setup_lock:
            if not self._setup_done:
                await self.asetup()

    async def _ensure_cache_schema(self, db: "aiosqlite.Connection") -> None:
        """Migrate the pre-namespace cache schema, if present."""
        async with db.execute("PRAGMA table_info(af_cache)") as cursor:
            columns = {row["name"] for row in await cursor.fetchall()}
        if {"namespace", "key"}.issubset(columns):
            return

        await db.execute("ALTER TABLE af_cache RENAME TO af_cache_legacy")
        await db.execute(
            """
            CREATE TABLE af_cache (
                namespace TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                expires_at REAL,
                PRIMARY KEY (namespace, key)
            )
            """
        )
        await db.execute(
            """
            INSERT OR REPLACE INTO af_cache (namespace, key, value, expires_at)
            SELECT
                substr(cache_key, 1, instr(cache_key, ':') - 1),
                substr(cache_key, instr(cache_key, ':') + 1),
                value,
                expires_at
            FROM af_cache_legacy
            WHERE instr(cache_key, ':') > 0
            """
        )
        await db.execute("DROP TABLE af_cache_legacy")

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self) -> None:
        return run_coroutine(self.asetup())

    async def asetup(self) -> None:
        """Create the database directory and all required tables."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        async with self._get_db() as db:
            for stmt in DDL_STATEMENTS:
                await db.execute(stmt)
            await self._ensure_cache_schema(db)
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_af_cache_namespace ON af_cache (namespace)"
            )
            await db.commit()
        self._setup_done = True
        logger.info("SqliteCheckpointer setup complete: %s", self.db_path)

    # ------------------------------------------------------------------
    # State — async
    # ------------------------------------------------------------------

    async def aput_state(self, config: dict[str, Any], state: StateT) -> StateT:
        await self._ensure_setup()
        key = self._get_config_key(config)
        data = _serialize_model(state)
        async with self._get_db() as db:
            await db.execute(
                """
                INSERT INTO af_states (thread_id, state_data, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(thread_id) DO UPDATE SET
                    state_data = excluded.state_data,
                    updated_at = excluded.updated_at
                """,
                (key, data, time.time()),
            )
            await db.commit()
        logger.debug("Stored state for thread: %s", key)
        return state

    async def aget_state(self, config: dict[str, Any]) -> StateT | None:
        await self._ensure_setup()
        key = self._get_config_key(config)
        async with (
            self._get_db() as db,
            db.execute("SELECT state_data FROM af_states WHERE thread_id = ?", (key,)) as cursor,
        ):
            row = await cursor.fetchone()
        if row is None:
            return None
        state_class = config.get("state_class") or AgentState
        data = json.loads(row["state_data"])
        return state_class.model_validate(data)  # type: ignore[return-value]

    async def aclear_state(self, config: dict[str, Any]) -> bool:
        await self._ensure_setup()
        key = self._get_config_key(config)
        async with self._get_db() as db:
            await db.execute("DELETE FROM af_states WHERE thread_id = ?", (key,))
            await db.commit()
        return True

    # ------------------------------------------------------------------
    # State cache — async (separate table, no Redis needed)
    # ------------------------------------------------------------------

    async def aput_state_cache(self, config: dict[str, Any], state: StateT) -> StateT:
        await self._ensure_setup()
        key = self._get_config_key(config)
        data = _serialize_model(state)
        async with self._get_db() as db:
            await db.execute(
                """
                INSERT INTO af_state_cache (thread_id, state_data, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(thread_id) DO UPDATE SET
                    state_data = excluded.state_data,
                    updated_at = excluded.updated_at
                """,
                (key, data, time.time()),
            )
            await db.commit()
        return state

    async def aget_state_cache(self, config: dict[str, Any]) -> StateT | None:
        await self._ensure_setup()
        key = self._get_config_key(config)
        async with (
            self._get_db() as db,
            db.execute(
                "SELECT state_data FROM af_state_cache WHERE thread_id = ?", (key,)
            ) as cursor,
        ):
            row = await cursor.fetchone()
        if row is None:
            return None
        state_class = config.get("state_class") or AgentState
        data = json.loads(row["state_data"])
        return state_class.model_validate(data)  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Generic cache — async
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
        async with self._get_db() as db:
            await db.execute(
                """
                INSERT INTO af_cache (namespace, key, value, expires_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(namespace, key) DO UPDATE SET
                    value = excluded.value,
                    expires_at = excluded.expires_at
                """,
                (namespace, key, _serialize(value), expires_at),
            )
            await db.commit()
        return value

    async def aget_cache_value(self, namespace: str, key: str) -> Any | None:
        await self._ensure_setup()
        now = time.time()
        async with self._get_db() as db:
            async with db.execute(
                "SELECT value, expires_at FROM af_cache WHERE namespace = ? AND key = ?",
                (namespace, key),
            ) as cursor:
                row = await cursor.fetchone()
            if row is None:
                return None
            expires_at = row["expires_at"]
            if expires_at is not None and expires_at <= now:
                await db.execute(
                    "DELETE FROM af_cache WHERE namespace = ? AND key = ?",
                    (namespace, key),
                )
                await db.commit()
                return None
        return json.loads(row["value"])

    async def aclear_cache_value(self, namespace: str, key: str) -> Any | None:
        await self._ensure_setup()
        async with self._get_db() as db:
            async with db.execute(
                "SELECT value FROM af_cache WHERE namespace = ? AND key = ?",
                (namespace, key),
            ) as cursor:
                row = await cursor.fetchone()
            await db.execute(
                "DELETE FROM af_cache WHERE namespace = ? AND key = ?",
                (namespace, key),
            )
            await db.commit()
        return json.loads(row["value"]) if row else None

    async def alist_cache_keys(
        self,
        namespace: str,
        prefix: str | None = None,
    ) -> list[str]:
        await self._ensure_setup()
        now = time.time()
        async with self._get_db() as db:
            await db.execute(
                "DELETE FROM af_cache WHERE namespace = ? AND expires_at IS NOT NULL "
                "AND expires_at <= ?",
                (namespace, now),
            )
            async with db.execute(
                "SELECT key FROM af_cache WHERE namespace = ?",
                (namespace,),
            ) as cursor:
                rows = await cursor.fetchall()
            await db.commit()
        keys = [row["key"] for row in rows]
        if prefix is not None:
            keys = [key for key in keys if key.startswith(prefix)]
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
        async with self._get_db() as db:
            for msg in messages:
                msg_id = str(msg.message_id)
                msg_data = _serialize_model(msg)
                await db.execute(
                    """
                    INSERT INTO af_messages (thread_id, message_id, message_data, created_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(thread_id, message_id) DO UPDATE SET
                        message_data = excluded.message_data
                    """,
                    (key, msg_id, msg_data, now),
                )
            await db.commit()
        logger.debug("Stored %d messages for thread: %s", len(messages), key)
        return True

    async def aget_message(self, config: dict[str, Any], message_id: str | int) -> Message:
        await self._ensure_setup()
        key = self._get_config_key(config)
        async with (
            self._get_db() as db,
            db.execute(
                "SELECT message_data FROM af_messages WHERE thread_id = ? AND message_id = ?",
                (key, str(message_id)),
            ) as cursor,
        ):
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
        async with (
            self._get_db() as db,
            db.execute(
                "SELECT message_data FROM af_messages WHERE thread_id = ? ORDER BY id",
                (key,),
            ) as cursor,
        ):
            rows = await cursor.fetchall()

        messages: list[Message] = [
            Message.model_validate(json.loads(row["message_data"])) for row in rows
        ]

        if search:
            messages = [
                m
                for m in messages
                if hasattr(m, "content") and search.lower() in str(m.content).lower()
            ]

        start = offset or 0
        end = (start + limit) if limit else None
        return messages[start:end]

    async def adelete_message(self, config: dict[str, Any], message_id: str | int) -> bool:
        await self._ensure_setup()
        key = self._get_config_key(config)
        async with self._get_db() as db:
            async with db.execute(
                "SELECT id FROM af_messages WHERE thread_id = ? AND message_id = ?",
                (key, str(message_id)),
            ) as cursor:
                row = await cursor.fetchone()
            if row is None:
                raise IndexError(f"Message with ID {message_id} not found for thread: {key}")
            await db.execute(
                "DELETE FROM af_messages WHERE thread_id = ? AND message_id = ?",
                (key, str(message_id)),
            )
            await db.commit()
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
        data = _serialize_model(thread_info)
        async with self._get_db() as db:
            await db.execute(
                """
                INSERT INTO af_threads (thread_id, thread_data, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(thread_id) DO UPDATE SET
                    thread_data = excluded.thread_data,
                    updated_at = excluded.updated_at
                """,
                (key, data, time.time()),
            )
            await db.commit()
        return True

    async def aget_thread(self, config: dict[str, Any]) -> ThreadInfo | None:
        await self._ensure_setup()
        key = self._get_config_key(config)
        async with self._get_db() as db:
            async with db.execute(
                "SELECT thread_data FROM af_threads WHERE thread_id = ?", (key,)
            ) as cursor:
                row = await cursor.fetchone()
        if row is None:
            return None
        return ThreadInfo.model_validate(json.loads(row["thread_data"]))

    async def alist_threads(
        self,
        config: dict[str, Any],
        search: str | None = None,
        offset: int | None = None,
        limit: int | None = None,
    ) -> list[ThreadInfo]:
        await self._ensure_setup()
        async with self._get_db() as db:
            async with db.execute(
                "SELECT thread_data FROM af_threads ORDER BY updated_at DESC"
            ) as cursor:
                rows = await cursor.fetchall()

        threads: list[ThreadInfo] = [
            ThreadInfo.model_validate(json.loads(row["thread_data"])) for row in rows
        ]

        if search:
            threads = [
                t
                for t in threads
                if any(search.lower() in str(v).lower() for v in t.model_dump().values())
            ]

        start = offset or 0
        end = (start + limit) if limit else None
        return threads[start:end]

    async def aclean_thread(self, config: dict[str, Any]) -> bool:
        await self._ensure_setup()
        key = self._get_config_key(config)
        async with self._get_db() as db:
            await db.execute("DELETE FROM af_threads WHERE thread_id = ?", (key,))
            await db.execute("DELETE FROM af_messages WHERE thread_id = ?", (key,))
            await db.execute("DELETE FROM af_states WHERE thread_id = ?", (key,))
            await db.execute("DELETE FROM af_state_cache WHERE thread_id = ?", (key,))
            await db.commit()
        return True

    # ------------------------------------------------------------------
    # Release
    # ------------------------------------------------------------------

    async def arelease(self) -> bool:
        logger.info("SqliteCheckpointer released (no persistent connections to close)")
        return True

    def release(self) -> bool:
        return run_coroutine(self.arelease())
