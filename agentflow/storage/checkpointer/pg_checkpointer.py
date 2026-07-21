import asyncio
import importlib
import json
import logging
import os
import re
from enum import Enum
from typing import Any, TypeVar

from injectq import InjectQ

from agentflow.core.exceptions.storage_exceptions import (
    SchemaVersionError,
    StaleStateError,
    StorageError,
    TransientStorageError,
)
from agentflow.utils import ThreadInfo, metrics


try:
    import asyncpg
    from asyncpg import Pool

    HAS_ASYNCPG = True
except ImportError:
    HAS_ASYNCPG = False
    asyncpg = None  # type: ignore
    Pool = None  # type: ignore

try:
    from redis.asyncio import ConnectionPool, Redis

    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False
    ConnectionPool = None  # type: ignore
    Redis = None  # type: ignore

from agentflow.core.state import AgentState, Message

from .base_checkpointer import BaseCheckpointer


logger = logging.getLogger("agentflow.checkpointer.pg")

StateT = TypeVar("StateT", bound="AgentState")

# Default TTL for Redis cache (24 hours)
DEFAULT_CACHE_TTL = 86400

# Current durable schema version. Bump when schema changes and add a matching
# entry to ``PgCheckpointer._migrations``.
CURRENT_SCHEMA_VERSION = 3

# Advisory-lock key used to serialize schema creation/migration across processes.
# Any stable arbitrary 64-bit constant works; it only has to be the same in every
# process that initializes this schema.
SCHEMA_INIT_LOCK_ID = 0x41474E54464C5721  # "AGNTFLW!"

# How many historical state rows to retain per thread. The append-only history
# is useful for debugging/audit, but must be bounded to avoid unbounded growth.
# Older rows beyond this window are pruned on each durable write.
DEFAULT_STATE_HISTORY_LIMIT = 20

# Config key used to carry the state version read at the start of a run through
# to the durable write, so the write can perform an optimistic compare-and-swap.
STATE_VERSION_CONFIG_KEY = "_checkpoint_version"

# Marker embedded in the cached (Redis) state payload so a resume-from-cache can
# recover the durable version for the compare-and-swap. Kept out of the durable
# ``state_data`` column, which uses the authoritative ``version`` column instead.
_CACHE_VERSION_KEY = "__checkpoint_version__"

# Atomic version-guarded cache write.
#
# The realtime cache is written on every step of a run, including by runs whose
# durable write will later lose the optimistic version check. A plain SETEX would
# let such a run stamp its OLDER state over a newer one that another run already
# committed -- and because reads prefer the cache and seed the next write's
# expected version from it, the thread would then fail its compare-and-swap
# forever (wedged until the TTL expired).
#
# This script only writes when the incoming version is >= the cached one, so a
# stale run can never move the cache backwards. Equal versions are allowed: that
# is the normal within-a-run case (progress updates, stop flags) where the state
# advances but the durable version has not been bumped yet. Doing the compare and
# the set in one Lua call keeps it atomic; a GET-then-SETEX from Python would
# still race across coroutines.
_CACHE_CAS_LUA = """
local existing = redis.call('GET', KEYS[1])
if existing then
    local ok, decoded = pcall(cjson.decode, existing)
    if ok and type(decoded) == 'table' and decoded[ARGV[4]] then
        if tonumber(decoded[ARGV[4]]) > tonumber(ARGV[2]) then
            return 0
        end
    end
end
redis.call('SETEX', KEYS[1], ARGV[3], ARGV[1])
return 1
"""

# Sentinel version used when the caller has no durable version yet (brand-new
# thread). Lower than any real version, so it never overwrites a newer entry.
_NO_VERSION = -1

# SQL type mapping for ID types
ID_TYPE_MAP = {
    "string": "VARCHAR(255)",
    "int": "SERIAL",
    "bigint": "BIGSERIAL",
}


class PgCheckpointer(BaseCheckpointer[StateT]):
    """
    Implements a checkpointer using PostgreSQL and Redis for persistent and cached state management.

    This class provides asynchronous and synchronous methods for storing, retrieving, and managing
    agent states, messages, and threads. PostgreSQL is used for durable storage, while Redis
    provides fast caching with TTL.

    Features:
        - Async-first design with sync fallbacks
        - Configurable ID types (string, int, bigint)
        - Connection pooling for both PostgreSQL and Redis
        - Proper error handling and resource management
        - Schema migration support

    Args:
        postgres_dsn (str, optional): PostgreSQL connection string.
        pg_pool (Any, optional): Existing asyncpg Pool instance.
        pool_config (dict, optional): Configuration for new pg pool creation.
        redis_url (str, optional): Redis connection URL.
        redis (Any, optional): Existing Redis instance.
        redis_pool (Any, optional): Existing Redis ConnectionPool.
        redis_pool_config (dict, optional): Configuration for new redis pool creation.
        **kwargs: Additional configuration options:
            - user_id_type: Type for user_id fields ('string', 'int', 'bigint')
            - cache_ttl: Redis cache TTL in seconds
            - release_resources: Whether to release resources on cleanup
            - state_history_limit: State snapshots retained per thread (default 20)
            - enforce_user_isolation: Treat ``user_id`` as an ownership boundary
              (default True). When enabled, threads/state/messages are scoped to
              the requesting user, so knowing a ``thread_id`` is not enough to
              read or delete another user's data. Set to False for single-tenant
              apps or when you have no real user identity (no auth configured, or
              a placeholder user_id) -- then ``user_id`` is ignored for ownership
              and queries key on ``thread_id`` alone.

    Raises:
        ImportError: If required dependencies are missing.
        ValueError: If required connection details are missing.
    """

    def __init__(
        self,
        # postgress connection details
        postgres_dsn: str | None = None,
        pg_pool: Any | None = None,
        pool_config: dict | None = None,
        # redis connection details
        redis_url: str | None = None,
        redis: Any | None = None,
        redis_pool: Any | None = None,
        redis_pool_config: dict | None = None,
        # database schema
        schema: str = "public",
        # other configurations - combine to reduce args
        **kwargs,
    ):
        """
        Initializes PgCheckpointer with PostgreSQL and Redis connections.

        Args:
            postgres_dsn (str, optional): PostgreSQL connection string.
            pg_pool (Any, optional): Existing asyncpg Pool instance.
            pool_config (dict, optional): Configuration for new pg pool creation.
            redis_url (str, optional): Redis connection URL.
            redis (Any, optional): Existing Redis instance.
            redis_pool (Any, optional): Existing Redis ConnectionPool.
            redis_pool_config (dict, optional): Configuration for new redis pool creation.
            schema (str, optional): PostgreSQL schema name. Defaults to "public".
            **kwargs: Additional configuration options.

        Raises:
            ImportError: If required dependencies are missing.
            ValueError: If required connection details are missing.
        """
        # Check for required dependencies
        if not HAS_ASYNCPG:
            raise ImportError(
                "PgCheckpointer requires 'asyncpg' package. "
                "Install with: pip install 10xscale-agentflow[pg_checkpoint]"
            )

        if not HAS_REDIS:
            raise ImportError(
                "PgCheckpointer requires 'redis' package. "
                "Install with: pip install 10xscale-agentflow[pg_checkpoint]"
            )

        self.user_id_type = kwargs.get("user_id_type", "string")
        # allow explicit override via kwargs, fallback to InjectQ, then default
        self.id_type = kwargs.get(
            "id_type", InjectQ.get_instance().try_get("generated_id_type", "string")
        )
        self.cache_ttl = kwargs.get("cache_ttl", DEFAULT_CACHE_TTL)
        self.release_resources = kwargs.get("release_resources", False)

        # Ownership is tracked PER RESOURCE. A single shared flag was wrong: when
        # a caller passed in their own pg_pool but only a redis_url, building the
        # Redis pool flipped the flag and arelease() then closed the caller's
        # Postgres pool -- which the rest of their app was still using. We only
        # ever close what we created ourselves (or what the caller explicitly
        # told us to release via release_resources=True).
        #
        # Decided here, not in _create_pg_pool: that runs lazily on first use, so
        # a checkpointer built from a DSN but released before its first query
        # would otherwise look unowned.
        self._owns_pg_pool = pg_pool is None
        self._owns_redis = False
        # Number of historical state rows to keep per thread (append-only history
        # is pruned to this window on every durable write).
        self.state_history_limit = kwargs.get("state_history_limit", DEFAULT_STATE_HISTORY_LIMIT)
        # Whether ``user_id`` is treated as an ownership boundary.
        #
        # Secure by default: threads, state, and messages are scoped to the
        # ``user_id`` in the config, so an authenticated caller cannot read or
        # write another user's thread even if they know (or guess) its id.
        #
        # This is a framework, not a product, so the decision stays with the
        # developer. Set ``enforce_user_isolation=False`` for a single-tenant app,
        # or when there is no real user identity (no auth configured, or a
        # placeholder/None user_id). With it off, ``user_id`` is ignored for
        # ownership entirely and every query keys on ``thread_id`` alone, which
        # also skips the extra join. Only turn it off when you are not relying on
        # a thread_id being secret.
        self.enforce_user_isolation = kwargs.get("enforce_user_isolation", True)

        # Validate schema name to prevent SQL injection
        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", schema):
            raise ValueError(
                f"Invalid schema name: {schema}. Schema must match pattern ^[a-zA-Z_][a-zA-Z0-9_]*$"
            )
        self.schema = schema

        self._schema_initialized = False
        self._loop: asyncio.AbstractEventLoop | None = None
        # Guards lazy pool creation so a cold-start race cannot create (and leak)
        # two pools for the same checkpointer (see _get_pg_pool).
        self._pg_pool_lock = asyncio.Lock()
        # Guards schema init within this process; a Postgres advisory lock guards
        # it across processes (see _initialize_schema).
        self._schema_init_lock = asyncio.Lock()

        # Store pool configuration for lazy initialization
        self._pg_pool_config = {
            "pg_pool": pg_pool,
            "postgres_dsn": postgres_dsn,
            "pool_config": pool_config or {},
        }

        # Initialize pool immediately if provided, otherwise defer
        if pg_pool is not None:
            self._pg_pool = pg_pool
        else:
            self._pg_pool = None

        # Now check and initialize connections
        if not pg_pool and not postgres_dsn:
            raise ValueError("Either postgres_dsn or pg_pool must be provided.")

        if not redis and not redis_url and not redis_pool:
            raise ValueError("Either redis_url, redis_pool or redis instance must be provided.")

        # Initialize Redis connection (synchronous)
        self.redis = self._create_redis_pool(redis, redis_pool, redis_url, redis_pool_config or {})

    def _create_redis_pool(
        self,
        redis: Any | None,
        redis_pool: Any | None,
        redis_url: str | None,
        redis_pool_config: dict,
    ) -> Any:
        """
        Create or use an existing Redis connection.

        Args:
            redis (Any, optional): Existing Redis instance.
            redis_pool (Any, optional): Existing Redis ConnectionPool.
            redis_url (str, optional): Redis connection URL.
            redis_pool_config (dict): Configuration for new redis pool creation.

        Returns:
            Redis: Redis connection instance.

        Raises:
            ValueError: If redis_url is not provided when creating a new connection.
        """
        # Caller-owned: they passed the client/pool in, so they close it.
        if redis:
            return redis

        if redis_pool:
            return Redis(connection_pool=redis_pool)  # type: ignore

        # We are building the pool from a URL, so we own it and must close it.
        # This marks ONLY Redis as ours -- it must not imply anything about the
        # Postgres pool, which the caller may have supplied.
        if not redis_url:
            raise ValueError("redis_url must be provided when creating new Redis connection")

        self._owns_redis = True
        return Redis(
            connection_pool=ConnectionPool.from_url(  # type: ignore
                redis_url,
                **redis_pool_config,
            )
        )

    def _get_table_name(self, table: str) -> str:
        """
        Get the schema-qualified table name.

        Args:
            table (str): The base table name (e.g., 'threads', 'states', 'messages')

        Returns:
            str: The schema-qualified table name (e.g., '"public"."threads"')
        """
        # Validate table name to prevent SQL injection
        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", table):
            raise ValueError(
                f"Invalid table name: {table}. Table must match pattern ^[a-zA-Z_][a-zA-Z0-9_]*$"
            )
        return f'"{self.schema}"."{table}"'

    def _create_pg_pool(self, pg_pool: Any, postgres_dsn: str | None, pool_config: dict) -> Any:
        """
        Create or use an existing PostgreSQL connection pool.

        Args:
            pg_pool (Any, optional): Existing asyncpg Pool instance.
            postgres_dsn (str, optional): PostgreSQL connection string.
            pool_config (dict): Configuration for new pg pool creation.

        Returns:
            Pool: PostgreSQL connection pool.
        """
        # Caller-owned: they passed the pool in, so they close it.
        if pg_pool:
            return pg_pool
        # Built from a DSN, so we own it. Ownership was already recorded in
        # __init__ (this runs lazily on first use, which may be never).
        return asyncpg.create_pool(dsn=postgres_dsn, **pool_config)  # type: ignore

    async def _get_pg_pool(self) -> Any:
        """
        Get PostgreSQL pool, creating it if necessary.

        Returns:
            Pool: PostgreSQL connection pool.
        """
        if self._pg_pool is None:
            async with self._pg_pool_lock:
                # Re-check under the lock: another coroutine may have created the
                # pool while we were waiting to acquire it.
                if self._pg_pool is None:
                    config = self._pg_pool_config
                    self._pg_pool = await self._create_pg_pool(
                        config["pg_pool"], config["postgres_dsn"], config["pool_config"]
                    )
        return self._pg_pool

    def _get_sql_type(self, type_name: str) -> str:
        """
        Get SQL type for given configuration type.

        Args:
            type_name (str): Type name ('string', 'int', 'bigint').

        Returns:
            str: Corresponding SQL type.
        """
        return ID_TYPE_MAP.get(type_name, "VARCHAR(255)")

    def _get_json_serializer(self):
        """Get optimal JSON serializer based on FAST_JSON env var."""
        if os.environ.get("FAST_JSON", "0") == "1":
            try:
                import orjson

                return orjson.dumps
            except ImportError:
                try:
                    import msgspec  # type: ignore

                    return msgspec.json.encode
                except ImportError:
                    pass
        return json.dumps

    def _get_current_schema_version(self) -> int:
        """Return current expected schema version."""
        return CURRENT_SCHEMA_VERSION

    def _build_create_tables_sql(self) -> list[str]:
        """
        Build SQL statements for table creation with dynamic ID types.

        Returns:
            list[str]: List of SQL statements for table creation.
        """
        thread_id_type = self._get_sql_type(self.id_type)
        user_id_type = self._get_sql_type(self.user_id_type)
        message_id_type = self._get_sql_type(self.id_type)

        # For AUTO INCREMENT types, we need to handle primary key differently
        thread_pk = (
            "thread_id SERIAL PRIMARY KEY"
            if self.id_type == "int"
            else f"thread_id {thread_id_type} PRIMARY KEY"
        )
        message_pk = (
            "message_id SERIAL PRIMARY KEY"
            if self.id_type == "int"
            else f"message_id {message_id_type} PRIMARY KEY"
        )

        return [
            # Schema version tracking table
            f"""
            CREATE TABLE IF NOT EXISTS {self._get_table_name("schema_version")} (
                version INT PRIMARY KEY,
                applied_at TIMESTAMPTZ DEFAULT NOW()
            )
            """,
            # Create message role enum (safe for older Postgres versions)
            (
                "DO $$\n"
                "BEGIN\n"
                "    CREATE TYPE message_role AS ENUM ('user', 'assistant', 'system', 'tool');\n"
                "EXCEPTION\n"
                "    WHEN duplicate_object THEN NULL;\n"
                "END$$;"
            ),
            # Create threads table
            f"""
            CREATE TABLE IF NOT EXISTS {self._get_table_name("threads")} (
                {thread_pk},
                thread_name VARCHAR(255),
                user_id {user_id_type} NOT NULL,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                updated_at TIMESTAMPTZ DEFAULT NOW(),
                meta JSONB DEFAULT '{{}}'::jsonb
            )
            """,
            # Create states table.
            # ``version`` is a per-thread monotonically increasing counter used
            # for optimistic concurrency control (see aput_state/aput_checkpoint).
            # The UNIQUE (thread_id, version) constraint is what makes concurrent
            # appends collide instead of silently duplicating a version.
            f"""
            CREATE TABLE IF NOT EXISTS {self._get_table_name("states")} (
                state_id SERIAL PRIMARY KEY,
                thread_id {thread_id_type} NOT NULL
                    REFERENCES {self._get_table_name("threads")}(thread_id)
                    ON DELETE CASCADE,
                version BIGINT NOT NULL DEFAULT 0,
                state_data JSONB NOT NULL,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                updated_at TIMESTAMPTZ DEFAULT NOW(),
                meta JSONB DEFAULT '{{}}'::jsonb
            )
            """,
            # Create messages table
            f"""
            CREATE TABLE IF NOT EXISTS {self._get_table_name("messages")} (
                {message_pk},
                thread_id {thread_id_type} NOT NULL
                    REFERENCES {self._get_table_name("threads")}(thread_id)
                    ON DELETE CASCADE,
                role message_role NOT NULL,
                content TEXT NOT NULL,
                tool_calls JSONB,
                tool_call_id VARCHAR(255),
                reasoning TEXT,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                updated_at TIMESTAMPTZ DEFAULT NOW(),
                total_tokens INT DEFAULT 0,
                usages JSONB DEFAULT '{{}}'::jsonb,
                meta JSONB DEFAULT '{{}}'::jsonb
            )
            """,
            # Create indexes. Version-dependent indexes on ``states`` are created
            # by the schema migrations (see _build_migrations) so that an upgrade
            # from a pre-version schema adds the column before indexing it.
            f"CREATE INDEX IF NOT EXISTS idx_threads_user_id ON "
            f"{self._get_table_name('threads')}(user_id)",
            f"CREATE INDEX IF NOT EXISTS idx_states_thread_id ON "
            f"{self._get_table_name('states')}(thread_id)",
            f"CREATE INDEX IF NOT EXISTS idx_messages_thread_id ON "
            f"{self._get_table_name('messages')}(thread_id)",
        ]

    def _build_migrations(self) -> dict[int, list[str]]:
        """Return ordered DDL steps keyed by the schema version they produce.

        Every statement must be idempotent (``IF NOT EXISTS`` / guarded), so a
        migration can run safely both on a freshly created schema (where the
        target objects already exist and each step is a no-op) and on an older
        database being upgraded in place. Migrations are applied in ascending
        version order inside a single transaction (see
        ``_check_and_apply_schema_version``).
        """
        states = self._get_table_name("states")
        return {
            # v2: per-thread state versioning for optimistic concurrency control.
            2: [
                f"ALTER TABLE {states} ADD COLUMN IF NOT EXISTS version BIGINT NOT NULL DEFAULT 0",
                # Backfill deterministic per-thread versions for any pre-existing
                # rows so the unique index below can be created. Ordered by
                # created_at then state_id to match the "latest wins" read order.
                f"""
                UPDATE {states} AS s
                SET version = sub.rn
                FROM (
                    SELECT state_id,
                           ROW_NUMBER() OVER (
                               PARTITION BY thread_id
                               ORDER BY created_at, state_id
                           ) AS rn
                    FROM {states}
                ) AS sub
                WHERE s.state_id = sub.state_id
                  AND s.version = 0
                """,  # noqa: S608
                f"CREATE INDEX IF NOT EXISTS idx_states_thread_version ON "
                f"{states}(thread_id, version DESC)",
                f"CREATE UNIQUE INDEX IF NOT EXISTS uq_states_thread_version ON "
                f"{states}(thread_id, version)",
            ],
            # v3: durable tool-execution ledger, so a node replayed after a crash
            # does not re-fire tool calls that already completed (audit B2).
            # The (thread_id, tool_call_id) primary key IS the idempotency key.
            3: [
                f"""
                CREATE TABLE IF NOT EXISTS {self._get_table_name("tool_executions")} (
                    thread_id {self._get_sql_type(self.id_type)} NOT NULL
                        REFERENCES {self._get_table_name("threads")}(thread_id)
                        ON DELETE CASCADE,
                    tool_call_id VARCHAR(255) NOT NULL,
                    result JSONB NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    PRIMARY KEY (thread_id, tool_call_id)
                )
                """,
            ],
        }

    async def _get_recorded_schema_version(self, conn) -> int:
        """Return the highest schema version recorded in the tracking table.

        Returns 0 when no version has been recorded yet (first run).
        """
        row = await conn.fetchrow(
            f"SELECT version FROM {self._get_table_name('schema_version')} "  # noqa: S608
            f"ORDER BY version DESC LIMIT 1"
        )
        return int(row["version"]) if row else 0

    async def _check_and_apply_schema_version(self, conn) -> None:
        """Apply any pending migrations and record the new schema version.

        Runs every migration between the recorded version and the target version
        in ascending order, inside the caller's transaction, then records the
        target version. Idempotent DDL means this is safe on both fresh and
        upgraded databases (see ``_build_migrations``).
        """
        try:
            current_version = await self._get_recorded_schema_version(conn)
            target_version = self._get_current_schema_version()

            if current_version >= target_version:
                return

            logger.info(
                "Upgrading checkpointer schema from version %d to %d",
                current_version,
                target_version,
            )
            migrations = self._build_migrations()
            for version in range(current_version + 1, target_version + 1):
                for statement in migrations.get(version, []):
                    logger.debug("Applying migration step for v%d: %s", version, statement.strip())
                    await conn.execute(statement)

            await conn.execute(
                f"INSERT INTO {self._get_table_name('schema_version')} (version) VALUES ($1) "  # noqa: S608
                f"ON CONFLICT (version) DO NOTHING",
                target_version,
            )
        except Exception as e:
            logger.error("Failed to apply schema migrations: %s", e)
            raise SchemaVersionError(
                message=f"Failed to apply schema migrations: {e}",
                error_code="STORAGE_SCHEMA_001",
                context={"error_type": type(e).__name__},
            ) from e

    async def _initialize_schema(self) -> None:
        """
        Initialize database schema if not already done.

        Returns:
            None
        """
        if self._schema_initialized:
            return

        logger.debug(
            "Initializing database schema with types: id_type=%s, user_id_type=%s",
            self.id_type,
            self.user_id_type,
        )

        async with self._schema_init_lock:
            # Re-check under the in-process lock (another coroutine may have run
            # initialization while we waited).
            if self._schema_initialized:
                return

            async with (await self._get_pg_pool()).acquire() as conn:
                try:
                    async with conn.transaction():
                        # Serialize schema creation ACROSS processes (multiple
                        # gunicorn workers / k8s replicas starting at once). An
                        # in-process flag is not enough: concurrent
                        # CREATE TABLE IF NOT EXISTS / CREATE TYPE can still fail
                        # with "duplicate key value violates unique constraint
                        # pg_type_typname_nsp_index", which the DO-block's
                        # duplicate_object guard does not catch. The advisory lock
                        # is released automatically when the transaction ends.
                        await conn.execute("SELECT pg_advisory_xact_lock($1)", SCHEMA_INIT_LOCK_ID)

                        sql_statements = self._build_create_tables_sql()
                        for sql in sql_statements:
                            logger.debug("Executing SQL: %s", sql.strip())
                            await conn.execute(sql)

                        # Apply pending migrations and record the schema version.
                        await self._check_and_apply_schema_version(conn)

                    self._schema_initialized = True
                    logger.debug("Database schema initialized successfully")
                except Exception as e:
                    logger.error("Failed to initialize database schema: %s", e)
                    raise

    ###########################
    #### SETUP METHODS ########
    ###########################

    async def asetup(self) -> Any:
        """
        Asynchronous setup method. Initializes database schema.

        Returns:
            Any: True if setup completed.
        """
        logger.info(
            "Setting up PgCheckpointer (async)",
            extra={
                "id_type": self.id_type,
                "user_id_type": self.user_id_type,
                "schema": self.schema,
            },
        )
        await self._initialize_schema()
        logger.info("PgCheckpointer setup completed")
        return True

    ###########################
    #### HELPER METHODS #######
    ###########################

    def _validate_config(self, config: dict[str, Any]) -> tuple[str | int, str | int]:
        """
        Extract and validate thread_id and user_id from config.

        Args:
            config (dict): Configuration dictionary.

        Returns:
            tuple: (thread_id, user_id)

        Raises:
            ValueError: If required fields are missing.
        """
        thread_id = config.get("thread_id")
        user_id = config.get("user_id")
        if not user_id:
            raise ValueError("user_id must be provided in config")

        if not thread_id:
            raise ValueError("Both thread_id must be provided in config")

        return thread_id, user_id

    def _get_thread_key(
        self,
        thread_id: str | int,
        user_id: str | int,
    ) -> str:
        """
        Get Redis cache key for thread state.

        Args:
            thread_id (str|int): Thread identifier.
            user_id (str|int): User identifier.

        Returns:
            str: Redis cache key.
        """
        return f"state_cache:{thread_id}:{user_id}"

    def _get_generic_cache_key(self, namespace: str, key: str) -> str:
        """Build a Redis key for shared non-state cache entries."""
        return f"generic_cache:{namespace}:{key}"

    def _serialize_state(self, state: StateT) -> str:
        """
        Serialize state to JSON string for storage.

        Args:
            state (StateT): State object.

        Returns:
            str: JSON string.
        """

        def enum_handler(obj):
            if isinstance(obj, Enum):
                return obj.value
            return str(obj)

        return json.dumps(state.model_dump(), default=enum_handler)

    def _deserialize_state(
        self,
        data: Any,
        state_class: type[StateT],
    ) -> StateT:
        """
        Deserialize JSON/JSONB back to state object.

        Args:
            data (Any): JSON string or dict/list.
            state_class (type): State class type.

        Returns:
            StateT: Deserialized state object.

        Raises:
            Exception: If deserialization fails.
        """
        try:
            if isinstance(data, bytes | bytearray):
                data = data.decode()
            if isinstance(data, str):
                return state_class.model_validate(json.loads(data))
            # Assume it's already a dict/list
            return state_class.model_validate(data)
        except Exception:
            # Last-resort: coerce to string and attempt parse, else raise
            if isinstance(data, str):
                return state_class.model_validate(json.loads(data))
            raise

    async def _retry_on_connection_error(
        self,
        operation,
        *args,
        max_retries=3,
        **kwargs,
    ):
        """
        Retry database operations on connection errors.

        Args:
            operation: Callable operation.
            *args: Arguments.
            max_retries (int): Maximum retries.
            **kwargs: Keyword arguments.

        Returns:
            Any: Result of operation or None.

        Raises:
            Exception: If all retries fail.
        """
        last_exception = None

        # Define exception types to catch (only if asyncpg is available)
        exceptions_to_catch: list[type[Exception]] = [ConnectionError]
        if HAS_ASYNCPG and asyncpg:
            exceptions_to_catch.extend([asyncpg.PostgresConnectionError, asyncpg.InterfaceError])

        exception_tuple = tuple(exceptions_to_catch)

        for attempt in range(max_retries):
            try:
                return await operation(*args, **kwargs)
            except exception_tuple as e:
                last_exception = e
                if attempt < max_retries - 1:
                    wait_time = 2**attempt  # exponential backoff
                    logger.warning(
                        "Database connection error on attempt %d/%d, retrying in %ds: %s",
                        attempt + 1,
                        max_retries,
                        wait_time,
                        e,
                    )
                    await asyncio.sleep(wait_time)
                    continue

                logger.error("Failed after %d attempts: %s", max_retries, e)
                break
            except Exception as e:
                # Don't retry on non-connection errors
                logger.error("Non-retryable error: %s", e)
                raise

        if last_exception:
            raise last_exception
        return None

    async def _run_query(self, fn, *, in_transaction: bool = False):
        """Acquire a pooled connection, run ``fn(conn)``, retrying on conn errors.

        Collapses the ``acquire() as conn`` + ``_retry_on_connection_error``
        boilerplate that every query method previously repeated. When
        ``in_transaction`` is True the callback runs inside ``conn.transaction()``
        so a multi-statement write commits atomically; a connection error retries
        the whole callback on a fresh connection (``max_retries=3``, matching the
        previous per-method wrapping).

        Args:
            fn: Coroutine function taking a single ``conn`` argument.
            in_transaction: Wrap the callback in a database transaction.

        Returns:
            Whatever ``fn`` returns.
        """

        async def _operation():
            async with (await self._get_pg_pool()).acquire() as conn:
                if in_transaction:
                    async with conn.transaction():
                        return await fn(conn)
                return await fn(conn)

        return await self._retry_on_connection_error(_operation, max_retries=3)

    @staticmethod
    def _loads_jsonb(value: Any) -> Any:
        """Normalize a JSONB column to a Python object.

        asyncpg may hand back a JSONB column either already decoded (dict/list)
        or as raw ``str``/``bytes`` depending on codec configuration. This coerces
        both shapes to the decoded value, replacing the ad-hoc
        ``json.loads(x) if isinstance(x, ...) else x`` guards scattered through
        the row-mapping code.
        """
        if isinstance(value, str | bytes | bytearray):
            return json.loads(value)
        return value

    def _thread_row_scope(
        self,
        thread_id: str | int,
        user_id: str | int | None,
        config: dict[str, Any] | None,
    ) -> tuple[str, list[Any]]:
        """Build the WHERE fragment + params for a direct ``threads`` row lookup.

        Unlike :meth:`_thread_scope` (which scopes ``states``/``messages`` through
        a subquery), this keys straight off the ``threads`` table's own ``user_id``
        column. Shared by the thread read/delete paths.
        """
        if self._isolation_active(user_id, config):
            return "thread_id = $1 AND user_id = $2", [thread_id, user_id]
        return "thread_id = $1", [thread_id]

    def _append_owner_scope(
        self,
        query: str,
        params: list[Any],
        user_id: str | int | None,
        config: dict[str, Any] | None,
    ) -> str:
        """Append the thread-ownership filter to a ``messages`` query when active.

        Mutates ``params`` in place (appending ``user_id``) and returns the query
        with the extra ``AND thread_id IN (...)`` clause. A no-op when isolation is
        off, so a single-tenant caller does not pay for the join.
        """
        if self._isolation_active(user_id, config):
            params.append(user_id)
            query += f" AND {self._thread_owned_by_sql(len(params))}"
        return query

    def _row_to_thread_info(self, row) -> ThreadInfo:
        """Build a :class:`ThreadInfo` from a ``threads`` row."""
        meta_dict = self._loads_jsonb(row["meta"]) if row["meta"] else {}
        return ThreadInfo(
            thread_id=row["thread_id"],
            thread_name=row["thread_name"],
            user_id=row["user_id"],
            metadata=meta_dict,
            run_id=meta_dict.get("run_id"),
            updated_at=row["updated_at"],
        )

    async def _ensure_thread_exists(
        self,
        thread_id: str | int,
        user_id: str | int,
        config: dict[str, Any],
    ) -> None:
        """
        Ensure thread exists in database, create if not.

        Args:
            thread_id (str|int): Thread identifier.
            user_id (str|int): User identifier.
            config (dict): Configuration dictionary.

        Returns:
            None

        Raises:
            Exception: If creation fails.
        """
        try:

            async def _check_and_create_thread(conn):
                # Insert-if-absent, then verify ownership from what is actually
                # in the table. Checking existence with (thread_id AND user_id)
                # first was unsafe: for a thread owned by ANOTHER user the
                # lookup missed, the INSERT silently no-opped on conflict, and
                # the caller went on to write into that user's thread.
                # Reading the owner back after the insert is also race-safe:
                # if a concurrent request created the thread, we observe the
                # winner's user_id rather than assuming we created it.
                await conn.execute(
                    f"""
                    INSERT INTO {self._get_table_name("threads")}
                        (thread_id, thread_name, user_id, meta)
                    VALUES ($1, $2, $3, $4)
                    ON CONFLICT DO NOTHING
                    """,  # noqa: S608
                    thread_id,
                    config.get("thread_name", f"Thread {thread_id}"),
                    user_id,
                    json.dumps(config.get("thread_meta", {})),
                )

                owner = await conn.fetchval(
                    f"SELECT user_id FROM {self._get_table_name('threads')} "  # noqa: S608
                    f"WHERE thread_id = $1",
                    thread_id,
                )
                if owner is None:
                    raise StorageError(
                        message="Thread could not be created",
                        error_code="STORAGE_002",
                        context={"thread_id": thread_id},
                    )
                if self._isolation_active(user_id, config) and str(owner) != str(user_id):
                    raise StorageError(
                        message="Thread is owned by a different user",
                        error_code="STORAGE_FORBIDDEN_001",
                        context={"thread_id": thread_id},
                    )

            await self._run_query(_check_and_create_thread)

        except Exception as e:
            logger.error("Failed to ensure thread exists: %s", e)
            raise

    def _get_full_class_path(self, obj: object) -> str:
        cls = obj.__class__
        return f"{cls.__module__}.{cls.__name__}"

    def _import_class_from_path(self, path: str) -> type[AgentState]:
        module_name, class_name = path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        return getattr(module, class_name)

    def _serialize_state_payload(self, state: StateT) -> dict[str, Any]:
        """Build the JSON-safe payload persisted for a state.

        Uses Pydantic ``mode="json"`` so non-primitive fields (datetime, UUID,
        enums) are coerced to JSON-serializable values instead of raising at
        ``json.dumps`` time. The concrete class is recorded under
        ``__class_path__`` so it can be reconstructed on read.
        """
        data = state.model_dump(mode="json")
        data["__class_path__"] = self._get_full_class_path(state)
        return data

    def _deserialize_state_payload(self, data: dict[str, Any]) -> StateT:
        """Reconstruct a state object from a persisted payload.

        If the recorded ``__class_path__`` can no longer be imported (the class
        was renamed or moved), fall back to the base ``AgentState`` with a
        warning instead of failing the whole read, so history stays loadable.
        """
        data = dict(data)
        class_path = data.pop("__class_path__", None)
        data.pop(_CACHE_VERSION_KEY, None)
        cls: type[AgentState] | None = None
        if class_path:
            try:
                cls = self._import_class_from_path(class_path)
            except Exception as e:  # degrade gracefully rather than brick history
                logger.warning(
                    "Could not import persisted state class '%s' (%s); "
                    "falling back to AgentState. History for this thread may be "
                    "missing custom fields.",
                    class_path,
                    e,
                )
        if cls is None:
            cls = AgentState
        return cls.model_validate(data)  # type: ignore[return-value]

    def _thread_scope(
        self,
        thread_id: str | int,
        user_id: str | int | None,
        config: dict[str, Any] | None = None,
    ) -> tuple[str, list[Any]]:
        """Build the WHERE fragment + params selecting a thread's rows.

        ``states``/``messages`` carry a ``thread_id`` but no ``user_id`` of their
        own; ownership lives on ``threads``. When isolation is active (per the request
        policy, or ``enforce_user_isolation`` when there is no policy) we scope through the
        parent table, so a caller can only touch threads they own. Otherwise we key on
        ``thread_id`` alone and skip the join entirely.

        Returns:
            tuple[str, list]: SQL fragment using $1..$n, and the matching params.
        """
        if self._isolation_active(user_id, config):
            return (
                f"thread_id = $1 AND thread_id IN "  # noqa: S608
                f"(SELECT thread_id FROM {self._get_table_name('threads')} "
                f"WHERE user_id = $2)",
                [thread_id, user_id],
            )
        return ("thread_id = $1", [thread_id])

    def _isolation_active(
        self, user_id: str | int | None, config: dict[str, Any] | None = None
    ) -> bool:
        """Whether ownership scoping should be applied for this call.

        Driven by the trusted ``config["authz"]`` policy when present, falling back to this
        checkpointer's ``enforce_user_isolation`` setting otherwise (so ``config`` is
        optional and a caller that omits it keeps the historical behaviour). Scoping still
        requires an actual ``user_id`` -- scoping on a missing one would match nothing.
        """
        return self._isolation_enabled(config, user_id)

    def _thread_owned_by_sql(self, user_param: int) -> str:
        """Return a SQL fragment requiring the row's thread to belong to a user.

        Applied to ``messages`` queries, whose rows carry a ``thread_id`` but no
        ``user_id`` of their own. Without this, any authenticated caller who knows
        (or guesses) a ``thread_id`` could read or delete another user's messages.

        Callers must gate this on :meth:`_isolation_active` so a developer who has
        turned isolation off does not pay for the join.
        """
        # Table name is regex-validated by _get_table_name; user_param is a bind
        # placeholder index, never a value. No interpolation of caller data.
        return (
            f"thread_id IN (SELECT thread_id FROM {self._get_table_name('threads')} "  # noqa: S608
            f"WHERE user_id = ${user_param})"
        )

    async def _lock_thread_for_write(
        self,
        conn,
        thread_id: str | int,
        user_id: str | int,
        config: dict[str, Any],
    ) -> int:
        """Ensure the thread exists, take a row lock, and return its current version.

        Creating the thread (if absent) and locking its row serializes concurrent
        writers on the same thread, so their version numbers cannot collide. The
        ownership check enforces per-user isolation on writes.

        Returns:
            int: The highest existing state ``version`` for the thread (0 if none).

        Raises:
            StorageError: If the thread is owned by a different user.
        """
        await conn.execute(
            f"""
            INSERT INTO {self._get_table_name("threads")}
                (thread_id, thread_name, user_id, meta)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT DO NOTHING
            """,  # noqa: S608
            thread_id,
            config.get("thread_name", f"Thread {thread_id}"),
            user_id,
            json.dumps(config.get("thread_meta", {})),
        )

        owner_row = await conn.fetchrow(
            f"SELECT user_id FROM {self._get_table_name('threads')} "  # noqa: S608
            f"WHERE thread_id = $1 FOR UPDATE",
            thread_id,
        )
        if owner_row is None:
            raise StorageError(
                message="Thread row disappeared during write",
                error_code="STORAGE_002",
                context={"thread_id": thread_id},
            )
        if self._isolation_active(user_id, config) and str(owner_row["user_id"]) != str(user_id):
            raise StorageError(
                message="Thread is owned by a different user",
                error_code="STORAGE_FORBIDDEN_001",
                context={"thread_id": thread_id},
            )

        version_row = await conn.fetchrow(
            f"SELECT COALESCE(MAX(version), 0) AS v "  # noqa: S608
            f"FROM {self._get_table_name('states')} WHERE thread_id = $1",
            thread_id,
        )
        return int(version_row["v"]) if version_row else 0

    async def _write_state_row(
        self,
        conn,
        thread_id: str | int,
        user_id: str | int,
        state: StateT,
        config: dict[str, Any],
        expected_version: int | None,
    ) -> int:
        """Append a new versioned state row inside an existing transaction.

        Performs the optimistic compare-and-swap: if ``expected_version`` is set
        and no longer matches the thread's current version, another execution
        committed in the meantime and this write is rejected as stale rather than
        silently overwriting it. Old rows beyond the history window are pruned.

        Returns:
            int: The new version assigned to the written row.
        """
        current_version = await self._lock_thread_for_write(conn, thread_id, user_id, config)

        if expected_version is not None and int(expected_version) != current_version:
            raise StaleStateError(
                message=(
                    "State was modified by another execution "
                    f"(expected version {expected_version}, found {current_version})"
                ),
                error_code="STORAGE_CONFLICT_001",
                context={
                    "thread_id": thread_id,
                    "expected_version": expected_version,
                    "current_version": current_version,
                },
            )

        new_version = current_version + 1
        state_json = json.dumps(self._serialize_state_payload(state))
        await conn.execute(
            f"""
            INSERT INTO {self._get_table_name("states")}
                (thread_id, version, state_data, meta)
            VALUES ($1, $2, $3, $4)
            """,  # noqa: S608
            thread_id,
            new_version,
            state_json,
            json.dumps(config.get("meta", {})),
        )

        # Prune history beyond the retention window (bounded append-only log).
        if self.state_history_limit and self.state_history_limit > 0:
            await conn.execute(
                f"DELETE FROM {self._get_table_name('states')} "  # noqa: S608
                f"WHERE thread_id = $1 AND version <= $2",
                thread_id,
                new_version - self.state_history_limit,
            )

        return new_version

    ###########################
    #### STATE METHODS ########
    ###########################

    async def aput_state(
        self,
        config: dict[str, Any],
        state: StateT,
    ) -> StateT:
        """
        Store state in PostgreSQL with optimistic concurrency control.

        A new versioned row is appended under a per-thread row lock. When
        ``config['_checkpoint_version']`` is present (set by a prior read within
        the same run), the write performs a compare-and-swap and raises
        :class:`StaleStateError` if another execution advanced the thread first.
        On success the config's version marker is advanced to the new version.

        Args:
            config (dict): Configuration dictionary.
            state (StateT): State object to store.

        Returns:
            StateT: The stored state object.

        Raises:
            StorageError: If storing fails.
            StaleStateError: If the optimistic version check fails.
        """
        thread_id, user_id = self._validate_config(config)

        logger.debug("Storing state for thread_id=%s, user_id=%s", thread_id, user_id)
        metrics.counter("pg_checkpointer.save_state.attempts").inc()

        expected_version = config.get(STATE_VERSION_CONFIG_KEY)

        with metrics.timer("pg_checkpointer.save_state.duration"):
            try:

                async def _store_state(conn):
                    return await self._write_state_row(
                        conn, thread_id, user_id, state, config, expected_version
                    )

                new_version = await self._run_query(_store_state, in_transaction=True)
                config[STATE_VERSION_CONFIG_KEY] = new_version
                logger.debug("State stored for thread_id=%s at version=%s", thread_id, new_version)
                metrics.counter("pg_checkpointer.save_state.success").inc()
                return state

            except StaleStateError:
                metrics.counter("pg_checkpointer.save_state.conflict").inc()
                # The cache may hold state built on the stale version; drop it so
                # the next read falls back to Postgres instead of re-seeding the
                # same doomed expected version.
                await self._invalidate_state_cache(thread_id, user_id)
                raise
            except Exception as e:
                metrics.counter("pg_checkpointer.save_state.error").inc()
                logger.error("Failed to store state for thread_id=%s: %s", thread_id, e)
                if asyncpg and hasattr(asyncpg, "ConnectionDoesNotExistError"):
                    connection_errors = (
                        asyncpg.ConnectionDoesNotExistError,
                        asyncpg.InterfaceError,
                    )
                    if isinstance(e, connection_errors):
                        raise TransientStorageError(
                            message=f"Connection issue storing state: {e}",
                            error_code="STORAGE_TRANSIENT_001",
                            context={
                                "thread_id": thread_id,
                                "error_type": type(e).__name__,
                            },
                        ) from e
                raise StorageError(
                    message=f"Failed to store state: {e}",
                    error_code="STORAGE_001",
                    context={
                        "thread_id": thread_id,
                        "error_type": type(e).__name__,
                    },
                ) from e

    async def aput_checkpoint(
        self,
        config: dict[str, Any],
        state: StateT,
        messages: list[Message] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> StateT:
        """
        Atomically persist a state and its messages in a single transaction.

        This is the durable-checkpoint entry point used by the graph loop. State
        and messages are committed together, so a crash can never leave state
        advanced without the messages that justify it (audit M7). The same
        optimistic version check as :meth:`aput_state` applies.

        Args:
            config (dict): Configuration dictionary.
            state (StateT): State object to store.
            messages (list[Message], optional): Messages to store atomically.
            metadata (dict, optional): Additional message metadata.

        Returns:
            StateT: The stored state object.

        Raises:
            StorageError: If storing fails.
            StaleStateError: If the optimistic version check fails.
        """
        thread_id, user_id = self._validate_config(config)
        messages = messages or []
        expected_version = config.get(STATE_VERSION_CONFIG_KEY)

        metrics.counter("pg_checkpointer.save_checkpoint.attempts").inc()
        with metrics.timer("pg_checkpointer.save_checkpoint.duration"):
            try:

                async def _store_checkpoint(conn):
                    version = await self._write_state_row(
                        conn, thread_id, user_id, state, config, expected_version
                    )
                    if messages:
                        await self._insert_messages(conn, thread_id, messages, metadata)
                    return version

                new_version = await self._run_query(_store_checkpoint, in_transaction=True)
                config[STATE_VERSION_CONFIG_KEY] = new_version
                logger.debug(
                    "Checkpoint stored for thread_id=%s at version=%s (%d messages)",
                    thread_id,
                    new_version,
                    len(messages),
                )
                metrics.counter("pg_checkpointer.save_checkpoint.success").inc()
                return state

            except StaleStateError:
                metrics.counter("pg_checkpointer.save_checkpoint.conflict").inc()
                # See aput_state: drop the possibly-poisoned cache so the thread
                # is not wedged into failing every subsequent compare-and-swap.
                await self._invalidate_state_cache(thread_id, user_id)
                raise
            except Exception as e:
                metrics.counter("pg_checkpointer.save_checkpoint.error").inc()
                logger.error("Failed to store checkpoint for thread_id=%s: %s", thread_id, e)
                raise StorageError(
                    message=f"Failed to store checkpoint: {e}",
                    error_code="STORAGE_001",
                    context={"thread_id": thread_id, "error_type": type(e).__name__},
                ) from e

    async def aget_state(self, config: dict[str, Any]) -> StateT | None:
        """
        Retrieve the latest state for a thread from PostgreSQL.

        Reads the highest-version row deterministically (``version DESC,
        state_id DESC``) scoped to the requesting ``user_id`` so a developer who
        enables auth cannot read another user's thread (audit B1/B3). The read
        version is recorded in ``config['_checkpoint_version']`` so a subsequent
        write in the same run can perform its compare-and-swap.

        Args:
            config (dict): Configuration dictionary.

        Returns:
            StateT | None: Retrieved state or None.

        Raises:
            Exception: If retrieval fails.
        """
        thread_id, user_id = self._validate_config(config)

        logger.debug("Retrieving state for thread_id=%s, user_id=%s", thread_id, user_id)

        try:
            scope_sql, scope_params = self._thread_scope(thread_id, user_id, config)
            query = f"""
                SELECT version, state_data FROM {self._get_table_name("states")}
                WHERE {scope_sql}
                ORDER BY version DESC, state_id DESC
                LIMIT 1
                """  # noqa: S608

            row = await self._run_query(lambda conn: conn.fetchrow(query, *scope_params))

            if row:
                data = json.loads(row["state_data"])
                logger.debug(
                    "State found for thread_id=%s at version=%s", thread_id, row["version"]
                )
                config[STATE_VERSION_CONFIG_KEY] = int(row["version"])
                return self._deserialize_state_payload(data)

            logger.debug("No state found for thread_id=%s", thread_id)
            return None

        except Exception as e:
            logger.error("Failed to retrieve state for thread_id=%s: %s", thread_id, e)
            raise

    async def aclear_state(self, config: dict[str, Any]) -> Any:
        """
        Clear state from PostgreSQL and Redis cache (scoped to the owning user).

        Args:
            config (dict): Configuration dictionary.

        Returns:
            Any: None

        Raises:
            Exception: If clearing fails.
        """
        thread_id, user_id = self._validate_config(config)

        logger.debug("Clearing state for thread_id=%s, user_id=%s", thread_id, user_id)

        try:
            scope_sql, scope_params = self._thread_scope(thread_id, user_id, config)

            # Clear from PostgreSQL with retry logic, scoped to the owner.
            query = f"DELETE FROM {self._get_table_name('states')} WHERE {scope_sql}"  # noqa: S608
            await self._run_query(lambda conn: conn.execute(query, *scope_params))

            # Clear from Redis cache
            cache_key = self._get_thread_key(thread_id, user_id)
            await self.redis.delete(cache_key)
            config.pop(STATE_VERSION_CONFIG_KEY, None)

            logger.debug("State cleared for thread_id=%s", thread_id)

        except Exception as e:
            logger.error("Failed to clear state for thread_id=%s: %s", thread_id, e)
            raise

    async def aput_state_cache(self, config: dict[str, Any], state: StateT) -> Any | None:
        """
        Cache state in Redis with TTL.

        The durable version currently associated with the run (from
        ``config['_checkpoint_version']``) is embedded in the cached payload so a
        resume that reads from cache recovers the right version for its
        compare-and-swap. Caching remains best-effort and never raises.

        Args:
            config (dict): Configuration dictionary.
            state (StateT): State object to cache.

        Returns:
            Any | None: True if cached, None if failed.
        """
        # No DB access, but keep consistent
        thread_id, user_id = self._validate_config(config)

        logger.debug("Caching state for thread_id=%s, user_id=%s", thread_id, user_id)

        try:
            cache_key = self._get_thread_key(thread_id, user_id)
            data = self._serialize_state_payload(state)
            version = config.get(STATE_VERSION_CONFIG_KEY)
            if version is not None:
                data[_CACHE_VERSION_KEY] = version
            state_json = json.dumps(data)

            # Version-guarded write: never move the cache backwards (see
            # _CACHE_CAS_LUA). A run that is about to lose its optimistic version
            # check must not stamp its stale state over a newer committed one.
            written = await self.redis.eval(
                _CACHE_CAS_LUA,
                1,
                cache_key,
                state_json,
                str(version if version is not None else _NO_VERSION),
                str(self.cache_ttl),
                _CACHE_VERSION_KEY,
            )
            if not written:
                logger.debug(
                    "Skipped stale cache write for thread_id=%s (version=%s is behind cache)",
                    thread_id,
                    version,
                )
                return None

            logger.debug("State cached with key=%s, ttl=%d", cache_key, self.cache_ttl)
            return True

        except Exception as e:
            logger.error("Failed to cache state for thread_id=%s: %s", thread_id, e)
            # Don't raise - caching is optional
            return None

    async def _invalidate_state_cache(
        self,
        thread_id: str | int,
        user_id: str | int,
    ) -> None:
        """Drop the cached state for a thread (best-effort).

        Called when a durable write loses its optimistic version check. The cache
        may hold state derived from the now-stale version, and because reads
        prefer the cache and seed the next write's expected version from it, a
        poisoned entry would make every later run fail its compare-and-swap.
        Dropping it forces the next read back to Postgres, so the thread heals
        itself instead of staying wedged until the TTL expires.
        """
        try:
            await self.redis.delete(self._get_thread_key(thread_id, user_id))
            logger.debug("Invalidated stale state cache for thread_id=%s", thread_id)
        except Exception as e:  # invalidation is best-effort
            logger.warning("Failed to invalidate state cache for thread_id=%s: %s", thread_id, e)

    async def aget_state_cache(self, config: dict[str, Any]) -> StateT | None:
        """
        Get state from Redis cache, fallback to PostgreSQL if miss.

        On a cache hit the embedded version is restored into
        ``config['_checkpoint_version']`` so a subsequent write still performs a
        correct compare-and-swap.

        Args:
            config (dict): Configuration dictionary.

        Returns:
            StateT | None: State object or None.
        """
        # Schema might be needed if we fall back to DB
        thread_id, user_id = self._validate_config(config)

        logger.debug("Getting cached state for thread_id=%s, user_id=%s", thread_id, user_id)

        try:
            # Try Redis first
            cache_key = self._get_thread_key(thread_id, user_id)
            cached_data = await self.redis.get(cache_key)

            if cached_data:
                data = json.loads(cached_data)
                logger.debug("Cache hit for thread_id=%s", thread_id)
                version = data.get(_CACHE_VERSION_KEY)
                if version is not None:
                    config[STATE_VERSION_CONFIG_KEY] = int(version)
                return self._deserialize_state_payload(data)

            # Cache miss - fallback to PostgreSQL
            logger.debug("Cache miss for thread_id=%s, falling back to PostgreSQL", thread_id)
            state = await self.aget_state(config)

            # Cache the result for next time
            if state:
                await self.aput_state_cache(config, state)

            return state

        except Exception as e:
            logger.error("Failed to get cached state for thread_id=%s: %s", thread_id, e)
            # Fallback to PostgreSQL on error
            return await self.aget_state(config)

    async def aput_cache_value(
        self,
        namespace: str,
        key: str,
        value: Any,
        ttl_seconds: int | None = None,
    ) -> Any | None:
        """Cache a generic JSON-serializable value in Redis."""
        try:
            cache_key = self._get_generic_cache_key(namespace, key)
            ttl = ttl_seconds or self.cache_ttl
            await self.redis.setex(cache_key, ttl, json.dumps(value))
            return True
        except Exception as e:
            logger.error("Failed to cache value for namespace=%s key=%s: %s", namespace, key, e)
            return None

    async def aget_cache_value(self, namespace: str, key: str) -> Any | None:
        """Retrieve a generic cache value from Redis."""
        try:
            cache_key = self._get_generic_cache_key(namespace, key)
            cached_data = await self.redis.get(cache_key)
            if not cached_data:
                return None
            return json.loads(cached_data)
        except Exception as e:
            logger.error(
                "Failed to get cached value for namespace=%s key=%s: %s", namespace, key, e
            )
            return None

    async def aclear_cache_value(self, namespace: str, key: str) -> Any | None:
        """Delete a generic cache value from Redis."""
        try:
            cache_key = self._get_generic_cache_key(namespace, key)
            return await self.redis.delete(cache_key)
        except Exception as e:
            logger.error(
                "Failed to clear cached value for namespace=%s key=%s: %s",
                namespace,
                key,
                e,
            )
            return None

    async def alist_cache_keys(
        self,
        namespace: str,
        prefix: str | None = None,
    ) -> list[str]:
        """List all cache keys for a namespace using Redis SCAN."""
        try:
            pattern = self._get_generic_cache_key(namespace, prefix or "*")
            keys: list[str] = []
            cursor = 0
            while True:
                cursor, found = await self.redis.scan(
                    cursor=cursor,
                    match=pattern,
                    count=100,
                )
                for raw_key in found:
                    key_str = raw_key.decode() if isinstance(raw_key, bytes) else raw_key
                    # Strip the namespace prefix to return just the key part
                    ns_prefix = f"generic_cache:{namespace}:"
                    if key_str.startswith(ns_prefix):
                        keys.append(key_str[len(ns_prefix) :])
                if cursor == 0:
                    break
            return keys
        except Exception as e:
            logger.error(
                "Failed to list cache keys for namespace=%s: %s",
                namespace,
                e,
            )
            return []

    ##################################
    #### TOOL EXECUTION LEDGER #######
    ##################################

    async def aget_tool_result(
        self,
        config: dict[str, Any],
        tool_call_id: str,
    ) -> dict[str, Any] | None:
        """Return the recorded result of a completed tool call, if there is one.

        Consulted before a tool runs. A hit means this exact tool call already
        completed in an earlier attempt at this node (the process died before the
        node finished, and the node is now being replayed), so the tool must NOT
        be called again -- we return what it returned last time.
        """
        thread_id, _ = self._validate_config(config)

        try:
            query = (
                f"SELECT result FROM {self._get_table_name('tool_executions')} "  # noqa: S608
                f"WHERE thread_id = $1 AND tool_call_id = $2"
            )
            row = await self._run_query(
                lambda conn: conn.fetchrow(query, thread_id, str(tool_call_id))
            )
            if not row:
                return None

            return self._loads_jsonb(row["result"])

        except Exception as e:
            # A ledger read failure must not take the run down. Falling back to
            # "no record" means the tool runs again -- at-least-once, which is the
            # old behaviour, rather than a hard failure.
            logger.error(
                "Failed to read tool ledger for thread_id=%s tool_call_id=%s: %s",
                thread_id,
                tool_call_id,
                e,
            )
            return None

    async def aput_tool_result(
        self,
        config: dict[str, Any],
        tool_call_id: str,
        result: dict[str, Any],
    ) -> Any | None:
        """Durably record that a tool call completed.

        Written as soon as the tool returns, so a crash later in the same node
        cannot cause it to be re-fired on replay. ON CONFLICT DO NOTHING keeps the
        first recorded result authoritative if this ever races with itself.

        Unlike the read side, a write failure here is raised: silently failing to
        record a completed side effect is exactly what leads to a double charge on
        the next replay, so the caller must know.
        """
        thread_id, user_id = self._validate_config(config)

        await self._ensure_thread_exists(thread_id, user_id, config)

        query = f"""
            INSERT INTO {self._get_table_name("tool_executions")}
                (thread_id, tool_call_id, result)
            VALUES ($1, $2, $3)
            ON CONFLICT (thread_id, tool_call_id) DO NOTHING
            """  # noqa: S608
        await self._run_query(
            lambda conn: conn.execute(query, thread_id, str(tool_call_id), json.dumps(result))
        )
        logger.debug(
            "Recorded tool execution thread_id=%s tool_call_id=%s", thread_id, tool_call_id
        )
        return True

    ###########################
    #### MESSAGE METHODS ######
    ###########################

    async def _insert_messages(
        self,
        conn,
        thread_id: str | int,
        messages: list[Message],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Upsert messages on an already-open connection/transaction.

        Shared by :meth:`aput_messages` (standalone) and :meth:`aput_checkpoint`
        (atomic with the state write) so both use identical insert semantics.
        """
        for message in messages:
            await conn.execute(
                f"""
                    INSERT INTO {self._get_table_name("messages")} (
                        message_id, thread_id, role, content, tool_calls,
                        tool_call_id, reasoning, total_tokens, usages, meta
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                    ON CONFLICT (message_id) DO UPDATE SET
                        content = EXCLUDED.content,
                        reasoning = EXCLUDED.reasoning,
                        usages = EXCLUDED.usages,
                        updated_at = NOW()
                    """,  # noqa: S608
                message.message_id,
                thread_id,
                message.role,
                json.dumps([block.model_dump(mode="json") for block in message.content]),
                json.dumps(message.tools_calls) if message.tools_calls else None,
                getattr(message, "tool_call_id", None),
                message.reasoning,
                message.usages.total_tokens if message.usages else 0,
                json.dumps(message.usages.model_dump()) if message.usages else None,
                json.dumps({**(metadata or {}), **(message.metadata or {})}),
            )

    async def aput_messages(
        self,
        config: dict[str, Any],
        messages: list[Message],
        metadata: dict[str, Any] | None = None,
    ) -> Any:
        """
        Store messages in PostgreSQL.

        Args:
            config (dict): Configuration dictionary.
            messages (list[Message]): List of messages to store.
            metadata (dict, optional): Additional metadata.

        Returns:
            Any: None

        Raises:
            Exception: If storing fails.
        """
        # Ensure schema is initialized before accessing tables
        thread_id, user_id = self._validate_config(config)

        if not messages:
            logger.debug("No messages to store for thread_id=%s", thread_id)
            return

        logger.debug("Storing %d messages for thread_id=%s", len(messages), thread_id)

        try:
            # Ensure thread exists
            await self._ensure_thread_exists(thread_id, user_id, config)

            # Store messages in batch with retry logic
            await self._run_query(
                lambda conn: self._insert_messages(conn, thread_id, messages, metadata),
                in_transaction=True,
            )
            logger.debug("Stored %d messages for thread_id=%s", len(messages), thread_id)

        except Exception as e:
            logger.error("Failed to store messages for thread_id=%s: %s", thread_id, e)
            raise

    async def aget_message(self, config: dict[str, Any], message_id: str | int) -> Message:
        """
        Retrieve a single message by ID.

        Args:
            config (dict): Configuration dictionary.
            message_id (str|int): Message identifier.

        Returns:
            Message: Retrieved message object.

        Raises:
            Exception: If retrieval fails.
        """
        thread_id = config.get("thread_id")
        user_id = config.get("user_id")

        logger.debug("Retrieving message_id=%s for thread_id=%s", message_id, thread_id)

        try:
            query = f"""
                SELECT message_id, thread_id, role, content, tool_calls,
                       tool_call_id, reasoning, created_at, total_tokens,
                       usages, meta
                FROM {self._get_table_name("messages")}
                WHERE message_id = $1
            """  # noqa: S608
            params: list[Any] = [message_id]
            if thread_id:
                params.append(thread_id)
                query += f" AND thread_id = ${len(params)}"
            # Owner scoping: a caller may only read messages on threads they own,
            # even if they know the message/thread id.
            query = self._append_owner_scope(query, params, user_id, config)

            row = await self._run_query(lambda conn: conn.fetchrow(query, *params))

            if not row:
                raise ValueError(f"Message not found: {message_id}")

            return self._row_to_message(row)

        except Exception as e:
            logger.error("Failed to retrieve message_id=%s: %s", message_id, e)
            raise

    async def alist_messages(
        self,
        config: dict[str, Any],
        search: str | None = None,
        offset: int | None = None,
        limit: int | None = None,
    ) -> list[Message]:
        """
        List messages for a thread with optional search and pagination.

        Args:
            config (dict): Configuration dictionary.
            search (str, optional): Search string.
            offset (int, optional): Offset for pagination.
            limit (int, optional): Limit for pagination.

        Returns:
            list[Message]: List of message objects.

        Raises:
            Exception: If listing fails.
        """
        thread_id = config.get("thread_id")
        user_id = config.get("user_id")

        if not thread_id:
            raise ValueError("thread_id must be provided in config")

        logger.debug("Listing messages for thread_id=%s", thread_id)

        try:
            # Build query with optional search
            query = f"""
                SELECT message_id, thread_id, role, content, tool_calls,
                       tool_call_id, reasoning, created_at, total_tokens,
                       usages, meta
                FROM {self._get_table_name("messages")}
                WHERE thread_id = $1
            """  # noqa: S608
            params: list[Any] = [thread_id]

            # Owner scoping: listing a thread you do not own returns nothing.
            query = self._append_owner_scope(query, params, user_id, config)

            if search:
                params.append(f"%{search}%")
                query += f" AND content ILIKE ${len(params)}"

            query += " ORDER BY created_at ASC"

            if limit:
                params.append(limit)
                query += f" LIMIT ${len(params)}"

            if offset:
                params.append(offset)
                query += f" OFFSET ${len(params)}"

            rows = await self._run_query(lambda conn: conn.fetch(query, *params))
            if not rows:
                rows = []
            messages = [self._row_to_message(row) for row in rows]

            logger.debug("Found %d messages for thread_id=%s", len(messages), thread_id)
            return messages

        except Exception as e:
            logger.error("Failed to list messages for thread_id=%s: %s", thread_id, e)
            raise

    async def adelete_message(
        self,
        config: dict[str, Any],
        message_id: str | int,
    ) -> Any | None:
        """
        Delete a message by ID.

        Args:
            config (dict): Configuration dictionary.
            message_id (str|int): Message identifier.

        Returns:
            Any | None: None

        Raises:
            Exception: If deletion fails.
        """
        thread_id = config.get("thread_id")
        user_id = config.get("user_id")

        logger.debug("Deleting message_id=%s for thread_id=%s", message_id, thread_id)

        try:
            query = (
                f"DELETE FROM {self._get_table_name('messages')} "  # noqa: S608
                f"WHERE message_id = $1"
            )
            params: list[Any] = [message_id]
            if thread_id:
                params.append(thread_id)
                query += f" AND thread_id = ${len(params)}"
            # Owner scoping: never let a caller delete a message on a thread they
            # do not own, even with a known message_id.
            query = self._append_owner_scope(query, params, user_id, config)

            await self._run_query(lambda conn: conn.execute(query, *params))
            logger.debug("Deleted message_id=%s", message_id)
            return None

        except Exception as e:
            logger.error("Failed to delete message_id=%s: %s", message_id, e)
            raise

    def _row_to_message(self, row) -> Message:  # noqa: PLR0912, PLR0915
        """
        Convert database row to Message object with robust JSON handling.

        Args:
            row: Database row.

        Returns:
            Message: Message object.
        """
        from agentflow.core.state.message import TokenUsages

        # Handle usages JSONB
        usages = None
        if row["usages"]:
            try:
                usages = TokenUsages(**self._loads_jsonb(row["usages"]))
            except Exception:
                usages = None

        # Handle tool_calls JSONB
        tool_calls = None
        if row["tool_calls"]:
            try:
                tool_calls = self._loads_jsonb(row["tool_calls"])
            except Exception:
                tool_calls = None

        # Handle meta JSONB
        metadata = {}
        if row["meta"]:
            try:
                metadata = self._loads_jsonb(row["meta"])
            except Exception:
                metadata = {}

        # Handle content TEXT/JSONB -> list of blocks
        content_raw = row["content"]
        content_value: list[Any] = []
        if content_raw is None:
            content_value = []
        elif isinstance(content_raw, bytes | bytearray):
            try:
                parsed = json.loads(content_raw.decode())
                if isinstance(parsed, list):
                    content_value = parsed
                elif isinstance(parsed, dict):
                    content_value = [parsed]
                else:
                    content_value = [{"type": "text", "text": str(parsed), "annotations": []}]
            except Exception:
                content_value = [
                    {"type": "text", "text": content_raw.decode(errors="ignore"), "annotations": []}
                ]
        elif isinstance(content_raw, str):
            # Try JSON parse first
            try:
                parsed = json.loads(content_raw)
                if isinstance(parsed, list):
                    content_value = parsed
                elif isinstance(parsed, dict):
                    content_value = [parsed]
                else:
                    content_value = [{"type": "text", "text": content_raw, "annotations": []}]
            except Exception:
                content_value = [{"type": "text", "text": content_raw, "annotations": []}]
        elif isinstance(content_raw, list):
            content_value = content_raw
        elif isinstance(content_raw, dict):
            content_value = [content_raw]
        else:
            content_value = [{"type": "text", "text": str(content_raw), "annotations": []}]

        return Message(
            message_id=row["message_id"],
            role=row["role"],
            content=content_value,
            tools_calls=tool_calls,
            reasoning=row["reasoning"],
            timestamp=row["created_at"].timestamp() if row["created_at"] else 0,
            metadata=metadata,
            usages=usages,
        )

    ###########################
    #### THREAD METHODS #######
    ###########################

    async def aput_thread(
        self,
        config: dict[str, Any],
        thread_info: ThreadInfo,
    ) -> Any | None:
        """
        Create or update thread information.

        Args:
            config (dict): Configuration dictionary.
            thread_info (ThreadInfo): Thread information object.

        Returns:
            bool: True if thread was created (inserted).
                  False if thread already existed and was updated.

        Raises:
            Exception: If storing fails.
        """
        # Ensure schema is initialized before accessing tables
        thread_id, user_id = self._validate_config(config)

        logger.debug("Storing thread info for thread_id=%s, user_id=%s", thread_id, user_id)

        try:
            thread_name = thread_info.thread_name
            meta = thread_info.metadata or {}
            user_id = thread_info.user_id or user_id
            meta.update(
                {
                    "run_id": thread_info.run_id,
                }
            )

            async def _put_thread(conn):
                # Try to insert; if inserted, RETURNING will return the row
                row = await conn.fetchrow(
                    f"""
                    INSERT INTO {self._get_table_name("threads")}
                        (thread_id, thread_name, user_id, meta)
                    VALUES ($1, $2, $3, $4)
                    ON CONFLICT (thread_id) DO NOTHING
                    RETURNING thread_id
                    """,  # noqa: S608
                    thread_id,
                    thread_name,
                    user_id,
                    json.dumps(meta),
                )

                if row:
                    # Insert succeeded
                    return True

                # Insert did nothing (conflict) -> update the existing row.
                # When isolation is on, the UPDATE is scoped by user_id too:
                # without that, any caller could rename or re-tag another
                # user's thread. RETURNING lets us detect that case.
                # Only set thread_name when a non-None value is provided.
                scoped = self._isolation_active(user_id, config)

                if thread_name is None:
                    sets, params = "meta = $1, updated_at = NOW()", [json.dumps(meta)]
                else:
                    sets = "thread_name = $1, meta = $2, updated_at = NOW()"
                    params = [thread_name, json.dumps(meta)]

                params.append(thread_id)
                where = f"thread_id = ${len(params)}"
                if scoped:
                    params.append(user_id)
                    where += f" AND user_id = ${len(params)}"

                updated = await conn.fetchrow(
                    f"""
                    UPDATE {self._get_table_name("threads")}
                    SET {sets}
                    WHERE {where}
                    RETURNING thread_id
                    """,  # noqa: S608
                    *params,
                )

                if updated is None and scoped:
                    # The thread exists but is owned by someone else.
                    raise StorageError(
                        message="Thread is owned by a different user",
                        error_code="STORAGE_FORBIDDEN_001",
                        context={"thread_id": thread_id},
                    )

                return False

            created = await self._run_query(_put_thread)
            logger.debug("Thread info stored for thread_id=%s (created=%s)", thread_id, created)
            return bool(created)

        except Exception as e:
            logger.error("Failed to store thread info for thread_id=%s: %s", thread_id, e)
            raise e

    async def aget_thread(
        self,
        config: dict[str, Any],
    ) -> ThreadInfo | None:
        """
        Get thread information.

        Args:
            config (dict): Configuration dictionary.

        Returns:
            ThreadInfo | None: Thread information object or None.

        Raises:
            Exception: If retrieval fails.
        """
        # Ensure schema is initialized before accessing tables
        thread_id, user_id = self._validate_config(config)

        logger.debug("Retrieving thread info for thread_id=%s, user_id=%s", thread_id, user_id)

        try:
            # Owner-scoped unless the developer disabled isolation.
            where, params = self._thread_row_scope(thread_id, user_id, config)
            query = f"""
                SELECT thread_id, thread_name, user_id, created_at, updated_at, meta
                FROM {self._get_table_name("threads")}
                WHERE {where}
                """  # noqa: S608

            row = await self._run_query(lambda conn: conn.fetchrow(query, *params))

            if row:
                meta_dict = self._loads_jsonb(row["meta"]) if row["meta"] else {}
                return ThreadInfo(
                    thread_id=thread_id,
                    thread_name=row["thread_name"],
                    user_id=user_id,
                    metadata=meta_dict,
                    run_id=meta_dict.get("run_id"),
                    updated_at=row["updated_at"],
                )

            logger.debug("Thread not found for thread_id=%s, user_id=%s", thread_id, user_id)
            return None

        except Exception as e:
            logger.error("Failed to retrieve thread info for thread_id=%s: %s", thread_id, e)
            raise e

    async def aget_thread_owner(self, thread_id: str | int) -> str | int | None:
        """Return the ``user_id`` that owns ``thread_id`` (global, not owner-scoped).

        Resolves ownership across all users so an authorization layer can reject a
        different user acting on the thread. Returns None if no such thread exists.
        """

        query = f"""
            SELECT user_id
            FROM {self._get_table_name("threads")}
            WHERE thread_id = $1
            """  # noqa: S608
        return await self._run_query(lambda conn: conn.fetchval(query, thread_id))

    async def alist_threads(
        self,
        config: dict[str, Any],
        search: str | None = None,
        offset: int | None = None,
        limit: int | None = None,
    ) -> list[ThreadInfo]:
        """
        List threads for a user with optional search and pagination.

        Args:
            config (dict): Configuration dictionary.
            search (str, optional): Search string.
            offset (int, optional): Offset for pagination.
            limit (int, optional): Limit for pagination.

        Returns:
            list[ThreadInfo]: List of thread information objects.

        Raises:
            Exception: If listing fails.
        """
        user_id = config.get("user_id")

        # `user_id = user_id or "test-user"` used to sit here, which made the check
        # below dead code and silently listed a fake user's threads whenever the
        # caller forgot user_id. With isolation on that is a confusing empty result;
        # with it off it is a cross-tenant listing. Ask for the id instead.
        if not user_id and self.enforce_user_isolation:
            raise ValueError("user_id must be provided in config to list threads")

        logger.debug("Listing threads for user_id=%s", user_id)

        try:
            # Build query with optional search. With isolation on, a user only
            # ever sees their own threads; with it off (single-tenant) the listing
            # is not restricted by user_id.
            query = f"""
                SELECT thread_id, thread_name, user_id, created_at, updated_at, meta
                FROM {self._get_table_name("threads")}
                WHERE TRUE
            """  # noqa: S608
            params: list[Any] = []

            if self._isolation_active(user_id, config):
                params.append(user_id)
                query += f" AND user_id = ${len(params)}"

            if search:
                params.append(f"%{search}%")
                query += f" AND thread_name ILIKE ${len(params)}"

            query += " ORDER BY updated_at DESC"

            if limit:
                params.append(limit)
                query += f" LIMIT ${len(params)}"

            if offset:
                params.append(offset)
                query += f" OFFSET ${len(params)}"

            rows = await self._run_query(lambda conn: conn.fetch(query, *params))
            if not rows:
                rows = []

            threads = [self._row_to_thread_info(row) for row in rows]
            logger.debug("Found %d threads for user_id=%s", len(threads), user_id)
            return threads

        except Exception as e:
            logger.error("Failed to list threads for user_id=%s: %s", user_id, e)
            raise

    async def aclean_thread(self, config: dict[str, Any]) -> Any | None:
        """
        Clean/delete a thread and all associated data.

        Args:
            config (dict): Configuration dictionary.

        Returns:
            Any | None: None

        Raises:
            Exception: If cleaning fails.
        """
        # Ensure schema is initialized before accessing tables
        thread_id, user_id = self._validate_config(config)

        logger.debug("Cleaning thread thread_id=%s, user_id=%s", thread_id, user_id)

        try:
            # Owner-scoped unless the developer disabled isolation, so one user
            # cannot delete another's thread by id.
            where, params = self._thread_row_scope(thread_id, user_id, config)

            # Delete thread (cascade will handle messages and states) with retry logic
            query = f"DELETE FROM {self._get_table_name('threads')} WHERE {where}"  # noqa: S608
            await self._run_query(lambda conn: conn.execute(query, *params))

            # Clean from Redis cache
            cache_key = self._get_thread_key(thread_id, user_id)
            await self.redis.delete(cache_key)

            logger.debug("Thread cleaned: thread_id=%s, user_id=%s", thread_id, user_id)

        except Exception as e:
            logger.error("Failed to clean thread thread_id=%s: %s", thread_id, e)
            raise

    ###########################
    #### RESOURCE CLEANUP #####
    ###########################

    async def arelease(self) -> Any | None:
        """
        Clean up connections and resources.

        Returns:
            Any | None: None
        """
        logger.info("Releasing PgCheckpointer resources")

        # Close each resource only if we created it. ``release_resources=True``
        # remains an explicit opt-in for callers who want us to close pools they
        # handed in. Deciding per resource is what stops a caller-supplied
        # pg_pool from being closed just because we happened to build the Redis
        # pool from a URL.
        close_redis = self.release_resources or self._owns_redis
        close_pg = self.release_resources or self._owns_pg_pool

        if not close_redis and not close_pg:
            logger.info("No owned resources to release")
            return

        errors = []

        # Close Redis connection
        if close_redis:
            try:
                if hasattr(self.redis, "aclose"):
                    await self.redis.aclose()
                elif hasattr(self.redis, "close"):
                    await self.redis.close()
                logger.debug("Redis connection closed")
            except Exception as e:
                logger.error("Error closing Redis connection: %s", e)
                errors.append(f"Redis: {e}")

        # Close PostgreSQL pool
        if close_pg:
            try:
                if self._pg_pool and not self._pg_pool.is_closing():
                    await self._pg_pool.close()
                logger.debug("PostgreSQL pool closed")
            except Exception as e:
                logger.error("Error closing PostgreSQL pool: %s", e)
                errors.append(f"PostgreSQL: {e}")

        if errors:
            error_msg = f"Errors during resource cleanup: {'; '.join(errors)}"
            logger.warning(error_msg)
            # Don't raise - cleanup should be best effort
        else:
            logger.info("All resources released successfully")
