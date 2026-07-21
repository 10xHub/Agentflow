import asyncio
import logging
import time
from collections import defaultdict
from typing import TYPE_CHECKING, Any, TypeVar

from agentflow.core.exceptions import StorageError
from agentflow.core.state import AgentState, Message
from agentflow.utils.callable_utils import run_coroutine
from agentflow.utils.thread_info import ThreadInfo

from .base_checkpointer import BaseCheckpointer


if TYPE_CHECKING:
    from agentflow.core.state import AgentState, Message

logger = logging.getLogger("agentflow.checkpointer")

StateT = TypeVar("StateT", bound="AgentState")


class InMemoryCheckpointer[StateT: AgentState](BaseCheckpointer[StateT]):
    """
    In-memory implementation of BaseCheckpointer.

    Stores all agent state, messages, and thread info in memory using Python dictionaries.
    Data is lost when the process ends. Designed for testing and ephemeral use cases.
    Async-first design using asyncio locks for concurrent access.

    Args:
        None

    Attributes:
        _states (dict): Stores agent states by thread key.
        _state_cache (dict): Stores cached agent states by thread key.
        _messages (dict): Stores messages by thread key.
        _message_metadata (dict): Stores message metadata by thread key.
        _threads (dict): Stores thread info by thread key.
        _state_lock (asyncio.Lock): Lock for state operations.
        _messages_lock (asyncio.Lock): Lock for message operations.
        _threads_lock (asyncio.Lock): Lock for thread operations.
    """

    def __init__(self):
        """
        Initialize all in-memory storage and locks.
        """
        # State storage
        self._states: dict[str, StateT] = {}
        self._state_cache: dict[str, StateT] = {}
        self._generic_cache: dict[str, tuple[Any, float | None]] = {}

        # Message storage - organized by config key
        self._messages: dict[str, list[Message]] = defaultdict(list)
        self._message_metadata: dict[str, dict[str, Any]] = {}

        # Thread storage
        self._threads: dict[str, dict[str, Any]] = {}

        # Thread ownership: config-key (thread_id) -> owner user_id. The first writer of a
        # thread owns it. Used to enforce owner-only access when the request policy
        # (config["authz"]) asks for it -- see _owner_denied / _record_owner.
        self._thread_owner: dict[str, str] = {}

        # Tool execution ledger (idempotency): "<thread>:<tool_call_id>" -> result
        self._tool_results: dict[str, dict[str, Any]] = {}

        # Async locks for concurrent access
        self._state_lock = asyncio.Lock()
        self._cache_lock = asyncio.Lock()
        self._messages_lock = asyncio.Lock()
        self._threads_lock = asyncio.Lock()
        self._tool_lock = asyncio.Lock()

    def setup(self) -> Any:
        """
        Synchronous setup method. No setup required for in-memory checkpointer.
        """
        logger.debug("InMemoryCheckpointer setup not required")

    async def asetup(self) -> Any:
        """
        Asynchronous setup method. No setup required for in-memory checkpointer.
        """
        logger.debug("InMemoryCheckpointer async setup not required")

    def _get_config_key(self, config: dict[str, Any]) -> str:
        """
        Generate a string key from config dict for storage indexing.

        Args:
            config (dict): Configuration dictionary.

        Returns:
            str: Key for indexing storage.
        """
        # Sort keys for consistent hashing
        thread_id = config.get("thread_id", "")
        return str(thread_id)

    def _record_owner(self, config: dict[str, Any]) -> None:
        """Record the caller as the thread's owner on first write (needs a user_id)."""
        user_id = config.get("user_id") if isinstance(config, dict) else None
        if user_id:
            self._thread_owner.setdefault(self._get_config_key(config), str(user_id))

    def _owner_denied(self, config: dict[str, Any]) -> bool:
        """True when isolation is active and the caller does not own this thread.

        A thread with no recorded owner (brand new) is never denied -- the caller becomes
        its owner on write, and reads simply find nothing.
        """
        user_id = config.get("user_id") if isinstance(config, dict) else None
        if not self._isolation_enabled(config, user_id):
            return False
        owner = self._thread_owner.get(self._get_config_key(config))
        return owner is not None and str(owner) != str(user_id)

    # -------------------------
    # State methods Async
    # -------------------------
    async def aput_state(self, config: dict[str, Any], state: StateT) -> StateT:
        """
        Store state asynchronously.

        Args:
            config (dict): Configuration dictionary.
            state (StateT): State object to store.

        Returns:
            StateT: The stored state object.
        """
        if self._owner_denied(config):
            raise StorageError("Thread is owned by a different user")
        key = self._get_config_key(config)
        async with self._state_lock:
            self._record_owner(config)
            self._states[key] = state
            logger.debug(f"Stored state for key: {key}")
            return state

    async def aget_state(self, config: dict[str, Any]) -> StateT | None:
        """
        Retrieve state asynchronously.

        Args:
            config (dict): Configuration dictionary.

        Returns:
            StateT | None: Retrieved state or None.
        """
        if self._owner_denied(config):
            return None  # not the owner -> as if it does not exist
        key = self._get_config_key(config)
        async with self._state_lock:
            state = self._states.get(key)
            logger.debug(f"Retrieved state for key: {key}, found: {state is not None}")
            return state

    async def aclear_state(self, config: dict[str, Any]) -> bool:
        """
        Clear state asynchronously.

        Args:
            config (dict): Configuration dictionary.

        Returns:
            bool: True if cleared.
        """
        if self._owner_denied(config):
            return True  # not the owner -> no-op, do not reveal existence
        key = self._get_config_key(config)
        async with self._state_lock:
            if key in self._states:
                del self._states[key]
                logger.debug(f"Cleared state for key: {key}")
            return True

    async def aput_state_cache(self, config: dict[str, Any], state: StateT) -> StateT:
        """
        Store state cache asynchronously.

        Args:
            config (dict): Configuration dictionary.
            state (StateT): State object to cache.

        Returns:
            StateT: The cached state object.
        """
        key = self._get_config_key(config)
        async with self._state_lock:
            self._state_cache[key] = state
            logger.debug(f"Stored state cache for key: {key}")
            return state

    async def aget_state_cache(self, config: dict[str, Any]) -> StateT | None:
        """
        Retrieve state cache asynchronously.

        Args:
            config (dict): Configuration dictionary.

        Returns:
            StateT | None: Cached state or None.
        """
        if self._owner_denied(config):
            return None
        key = self._get_config_key(config)
        async with self._state_lock:
            cache = self._state_cache.get(key)
            logger.debug(f"Retrieved state cache for key: {key}, found: {cache is not None}")
            return cache

    async def aput_cache_value(
        self,
        namespace: str,
        key: str,
        value: Any,
        ttl_seconds: int | None = None,
    ) -> Any | None:
        cache_key = f"{namespace}:{key}"
        expires_at = time.time() + ttl_seconds if ttl_seconds else None
        async with self._cache_lock:
            self._generic_cache[cache_key] = (value, expires_at)
        return value

    async def aget_cache_value(self, namespace: str, key: str) -> Any | None:
        cache_key = f"{namespace}:{key}"
        async with self._cache_lock:
            cached = self._generic_cache.get(cache_key)
            if cached is None:
                return None

            value, expires_at = cached
            if expires_at is not None and expires_at <= time.time():
                self._generic_cache.pop(cache_key, None)
                return None
            return value

    # -------------------------
    # Tool execution ledger
    # -------------------------
    def _tool_key(self, config: dict[str, Any], tool_call_id: str) -> str:
        return f"{self._get_config_key(config)}:{tool_call_id}"

    async def aget_tool_result(
        self,
        config: dict[str, Any],
        tool_call_id: str,
    ) -> dict[str, Any] | None:
        """Return a recorded tool result, so a replayed tool call is not re-fired."""
        async with self._tool_lock:
            return self._tool_results.get(self._tool_key(config, tool_call_id))

    async def aput_tool_result(
        self,
        config: dict[str, Any],
        tool_call_id: str,
        result: dict[str, Any],
    ) -> Any | None:
        """Record that a tool call completed, with its result."""
        async with self._tool_lock:
            self._tool_results[self._tool_key(config, tool_call_id)] = result
            return True

    async def aclear_cache_value(self, namespace: str, key: str) -> Any | None:
        cache_key = f"{namespace}:{key}"
        async with self._cache_lock:
            return self._generic_cache.pop(cache_key, None)

    async def alist_cache_keys(
        self,
        namespace: str,
        prefix: str | None = None,
    ) -> list[str]:
        """List all cache keys for a namespace."""
        ns_prefix = f"{namespace}:"
        async with self._cache_lock:
            keys = []
            for full_key in self._generic_cache:
                if full_key.startswith(ns_prefix):
                    key_part = full_key[len(ns_prefix) :]
                    if prefix is None or key_part.startswith(prefix):
                        keys.append(key_part)
            return keys

    # -------------------------
    # State methods Sync
    # -------------------------
    def put_state(self, config: dict[str, Any], state: StateT) -> StateT:
        """Store state synchronously."""
        return run_coroutine(self.aput_state(config, state))

    def get_state(self, config: dict[str, Any]) -> StateT | None:
        """Retrieve state synchronously."""
        return run_coroutine(self.aget_state(config))

    def clear_state(self, config: dict[str, Any]) -> bool:
        """Clear state synchronously."""
        return run_coroutine(self.aclear_state(config))

    def put_state_cache(self, config: dict[str, Any], state: StateT) -> StateT:
        """Store state cache synchronously."""
        return run_coroutine(self.aput_state_cache(config, state))

    def get_state_cache(self, config: dict[str, Any]) -> StateT | None:
        """Retrieve state cache synchronously."""
        return run_coroutine(self.aget_state_cache(config))

    # -------------------------
    # Message methods async
    # -------------------------
    async def aput_messages(
        self,
        config: dict[str, Any],
        messages: list[Message],
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """
        Store messages asynchronously.

        Args:
            config (dict): Configuration dictionary.
            messages (list[Message]): List of messages to store.
            metadata (dict, optional): Additional metadata.

        Returns:
            bool: True if stored.
        """
        if self._owner_denied(config):
            raise StorageError("Thread is owned by a different user")
        key = self._get_config_key(config)
        async with self._messages_lock:
            self._record_owner(config)
            self._messages[key].extend(messages)
            if metadata:
                self._message_metadata[key] = metadata
            logger.debug(f"Stored {len(messages)} messages for key: {key}")
            return True

    async def aget_message(self, config: dict[str, Any], message_id: str | int) -> Message:
        """
        Retrieve a specific message asynchronously.

        Args:
            config (dict): Configuration dictionary.
            message_id (str|int): Message identifier.

        Returns:
            Message: Retrieved message object.

        Raises:
            IndexError: If message not found.
        """
        key = self._get_config_key(config)
        if self._owner_denied(config):
            raise IndexError(f"Message with ID {message_id} not found for config key: {key}")
        async with self._messages_lock:
            messages = self._messages.get(key, [])
            for msg in messages:
                if msg.message_id == message_id:
                    return msg
            raise IndexError(f"Message with ID {message_id} not found for config key: {key}")

    async def alist_messages(
        self,
        config: dict[str, Any],
        search: str | None = None,
        offset: int | None = None,
        limit: int | None = None,
    ) -> list[Message]:
        """
        List messages asynchronously with optional filtering.

        Args:
            config (dict): Configuration dictionary.
            search (str, optional): Search string.
            offset (int, optional): Offset for pagination.
            limit (int, optional): Limit for pagination.

        Returns:
            list[Message]: List of message objects.
        """
        if self._owner_denied(config):
            return []
        key = self._get_config_key(config)
        async with self._messages_lock:
            messages = self._messages.get(key, [])

            # Apply search filter if provided
            if search:
                # Simple string search in message content
                messages = [
                    msg
                    for msg in messages
                    if hasattr(msg, "content") and search.lower() in str(msg.content).lower()
                ]

            # Apply offset and limit
            start = offset or 0
            end = (start + limit) if limit else None
            return messages[start:end]

    async def adelete_message(self, config: dict[str, Any], message_id: str | int) -> bool:
        """
        Delete a specific message asynchronously.

        Args:
            config (dict): Configuration dictionary.
            message_id (str|int): Message identifier.

        Returns:
            bool: True if deleted.

        Raises:
            IndexError: If message not found.
        """
        key = self._get_config_key(config)
        if self._owner_denied(config):
            raise IndexError(f"Message with ID {message_id} not found for config key: {key}")
        async with self._messages_lock:
            messages = self._messages.get(key, [])
            for msg in messages:
                if msg.message_id == message_id:
                    messages.remove(msg)
                    logger.debug(f"Deleted message with ID {message_id} for key: {key}")
                    return True
            raise IndexError(f"Message with ID {message_id} not found for config key: {key}")

    # -------------------------
    # Message methods sync
    # -------------------------
    def put_messages(
        self,
        config: dict[str, Any],
        messages: list[Message],
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Store messages synchronously."""
        return run_coroutine(self.aput_messages(config, messages, metadata))

    def get_message(self, config: dict[str, Any], message_id: str | int) -> Message:
        """Retrieve a specific message synchronously."""
        return run_coroutine(self.aget_message(config, message_id))

    def list_messages(
        self,
        config: dict[str, Any],
        search: str | None = None,
        offset: int | None = None,
        limit: int | None = None,
    ) -> list[Message]:
        """List messages synchronously with optional filtering."""
        return run_coroutine(self.alist_messages(config, search, offset, limit))

    def delete_message(self, config: dict[str, Any], message_id: str | int) -> bool:
        """Delete a specific message synchronously."""
        return run_coroutine(self.adelete_message(config, message_id))

    # -------------------------
    # Thread methods async
    # -------------------------
    async def aput_thread(
        self,
        config: dict[str, Any],
        thread_info: ThreadInfo,
    ) -> bool:
        """
        Store thread info asynchronously.

        Args:
            config (dict): Configuration dictionary.
            thread_info (ThreadInfo): Thread information object.

        Returns:
            bool: True if stored.
        """
        if self._owner_denied(config):
            raise StorageError("Thread is owned by a different user")
        key = self._get_config_key(config)
        async with self._threads_lock:
            self._record_owner(config)
            self._threads[key] = thread_info.model_dump()
            logger.debug(f"Stored thread info for key: {key}")
            return True

    async def aget_thread_owner(self, thread_id: str | int) -> str | int | None:
        """Return the owner ``user_id`` of ``thread_id``, or None if unknown.

        Threads are keyed by ``thread_id`` here, so the stored record's ``user_id``
        is the owner. Note the in-memory registry is only populated by explicit
        ``aput_thread`` calls; a thread that only ever had state written is not
        recorded, so this returns None for it.
        """
        async with self._threads_lock:
            record = self._threads.get(str(thread_id))
        if not record:
            return None
        return record.get("user_id")

    async def aget_thread(
        self,
        config: dict[str, Any],
    ) -> ThreadInfo | None:
        """
        Retrieve thread info asynchronously.

        Args:
            config (dict): Configuration dictionary.

        Returns:
            ThreadInfo | None: Thread information object or None.
        """
        if self._owner_denied(config):
            return None
        key = self._get_config_key(config)
        async with self._threads_lock:
            thread = self._threads.get(key)
            logger.debug(f"Retrieved thread for key: {key}, found: {thread is not None}")
            return ThreadInfo.model_validate(thread) if thread else None

    async def alist_threads(
        self,
        config: dict[str, Any],
        search: str | None = None,
        offset: int | None = None,
        limit: int | None = None,
    ) -> list[ThreadInfo]:
        """
        List all threads asynchronously with optional filtering.

        Args:
            config (dict): Configuration dictionary.
            search (str, optional): Search string.
            offset (int, optional): Offset for pagination.
            limit (int, optional): Limit for pagination.

        Returns:
            list[ThreadInfo]: List of thread information objects.
        """
        user_id = config.get("user_id") if isinstance(config, dict) else None
        isolate = self._isolation_enabled(config, user_id)
        async with self._threads_lock:
            threads = list(self._threads.values())

            # Owner-scope the listing when the policy asks for it.
            if isolate:
                threads = [t for t in threads if str(t.get("user_id")) == str(user_id)]

            # Apply search filter if provided
            if search:
                threads = [
                    thread
                    for thread in threads
                    if any(search.lower() in str(value).lower() for value in thread.values())
                ]

            # Apply offset and limit
            start = offset or 0
            end = (start + limit) if limit else None
            return [ThreadInfo.model_validate(thread) for thread in threads[start:end]]

    async def aclean_thread(self, config: dict[str, Any]) -> bool:
        """
        Clean/delete thread asynchronously.

        Args:
            config (dict): Configuration dictionary.

        Returns:
            bool: True if cleaned.
        """
        if self._owner_denied(config):
            return False  # not the owner -> no-op
        key = self._get_config_key(config)
        async with self._threads_lock:
            if key in self._threads:
                del self._threads[key]
                self._thread_owner.pop(key, None)
                logger.debug(f"Cleaned thread for key: {key}")
                return True
        return False

    # -------------------------
    # Thread methods sync
    # -------------------------
    def put_thread(self, config: dict[str, Any], thread_info: ThreadInfo) -> bool:
        """Store thread info synchronously."""
        return run_coroutine(self.aput_thread(config, thread_info))

    def get_thread(self, config: dict[str, Any]) -> ThreadInfo | None:
        """Retrieve thread info synchronously."""
        return run_coroutine(self.aget_thread(config))

    def list_threads(
        self,
        config: dict[str, Any],
        search: str | None = None,
        offset: int | None = None,
        limit: int | None = None,
    ) -> list[ThreadInfo]:
        """List all threads synchronously with optional filtering."""
        return run_coroutine(self.alist_threads(config, search, offset, limit))

    def clean_thread(self, config: dict[str, Any]) -> bool:
        """Clean/delete thread synchronously."""
        return run_coroutine(self.aclean_thread(config))

    # -------------------------
    # Clean Resources
    # -------------------------
    async def arelease(self) -> bool:
        """
        Release resources asynchronously.

        Returns:
            bool: True if released.
        """
        async with (
            self._state_lock,
            self._cache_lock,
            self._messages_lock,
            self._threads_lock,
            self._tool_lock,
        ):
            self._states.clear()
            self._state_cache.clear()
            self._generic_cache.clear()
            self._messages.clear()
            self._message_metadata.clear()
            self._threads.clear()
            self._tool_results.clear()
            logger.info("Released all in-memory resources")
            return True

    def release(self) -> bool:
        """Release resources synchronously."""
        return run_coroutine(self.arelease())
