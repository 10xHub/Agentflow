import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, TypeVar

from agentflow.core.state import AgentState, Message
from agentflow.utils import run_coroutine
from agentflow.utils.thread_info import ThreadInfo


logger = logging.getLogger("agentflow.checkpointer")

if TYPE_CHECKING:
    from agentflow.core.state import AgentState, Message


StateT = TypeVar("StateT", bound="AgentState")

# Namespace for out-of-band stop requests in the generic cache.
#
# The stop flag deliberately lives in its OWN cache entry rather than inside the
# cached state blob. A running graph rewrites the state cache after every node
# (``call_realtime_sync``), and those in-run writes carry the same durable
# version as the entry a concurrent ``stop()`` just wrote -- so a flag stored in
# the state blob is legally overwritten by the loop's own flag-less copy and the
# stop is silently lost. Nothing in the run loop ever writes this key, so a stop
# recorded here cannot be clobbered by the run it is trying to stop.
STOP_REQUEST_NAMESPACE = "stop_request"

# How long an unconsumed stop request stays live. A stop is meant to be observed
# within a node or two; this bound only keeps an abandoned flag from lingering
# forever if the run dies before consuming it.
STOP_REQUEST_TTL_SECONDS = 3600


class BaseCheckpointer[StateT: AgentState](ABC):
    """
    Abstract base class for checkpointing agent state, messages, and threads.

    This class defines the contract for all checkpointer implementations, supporting both
    async and sync methods.
    Subclasses should implement async methods for optimal performance.
    Sync methods are provided for compatibility.

    Usage:
        - Async-first design: subclasses should implement `async def` methods.
        - If a subclass provides only a sync `def`, it will be executed in a worker thread
            automatically using `asyncio.run`.
        - Callers always use the async APIs (`await cp.put_state(...)`, etc.).

    Type Args:
        StateT: Type of agent state (must inherit from AgentState).
    """

    ###########################
    #### SETUP ################
    ###########################
    def setup(self) -> Any:
        """
        Synchronous setup method for checkpointer.

        Returns:
            Any: Implementation-defined setup result.
        """
        return run_coroutine(self.asetup())

    @abstractmethod
    async def asetup(self) -> Any:
        """
        Asynchronous setup method for checkpointer.

        Returns:
            Any: Implementation-defined setup result.
        """
        raise NotImplementedError

    # -------------------------
    # State methods Async
    # -------------------------
    @abstractmethod
    async def aput_state(self, config: dict[str, Any], state: StateT) -> StateT:
        """
        Store agent state asynchronously.

        Args:
            config (dict): Configuration dictionary.
            state (StateT): State object to store.

        Returns:
            StateT: The stored state object.
        """
        raise NotImplementedError

    @abstractmethod
    async def aget_state(self, config: dict[str, Any]) -> StateT | None:
        """
        Retrieve agent state asynchronously.

        Args:
            config (dict): Configuration dictionary.

        Returns:
            StateT | None: Retrieved state or None.
        """
        raise NotImplementedError

    @abstractmethod
    async def aclear_state(self, config: dict[str, Any]) -> Any:
        """
        Clear agent state asynchronously.

        Args:
            config (dict): Configuration dictionary.

        Returns:
            Any: Implementation-defined result.
        """
        raise NotImplementedError

    async def aput_checkpoint(
        self,
        config: dict[str, Any],
        state: StateT,
        messages: "list[Message] | None" = None,
        metadata: dict[str, Any] | None = None,
    ) -> StateT:
        """
        Persist a state and its messages as one durable checkpoint.

        This default is only a fallback for backends without a shared
        transaction (e.g. in-memory): it writes state then messages sequentially,
        exactly as callers did before. Backends that can commit both together
        (e.g. Postgres) override this so the two writes are atomic and a crash
        cannot leave state advanced without the messages that justify it.

        Args:
            config (dict): Configuration dictionary.
            state (StateT): State object to store.
            messages (list[Message], optional): Messages to persist with the state.
            metadata (dict, optional): Additional message metadata.

        Returns:
            StateT: The stored state object.
        """
        await self.aput_state(config, state)
        if messages:
            await self.aput_messages(config, messages, metadata)
        return state

    @abstractmethod
    async def aput_state_cache(self, config: dict[str, Any], state: StateT) -> Any | None:
        """
        Store agent state in cache asynchronously.

        Args:
            config (dict): Configuration dictionary.
            state (StateT): State object to cache.

        Returns:
            Any | None: Implementation-defined result.
        """
        raise NotImplementedError

    @abstractmethod
    async def aget_state_cache(self, config: dict[str, Any]) -> StateT | None:
        """
        Retrieve agent state from cache asynchronously.

        Args:
            config (dict): Configuration dictionary.

        Returns:
            StateT | None: Cached state or None.
        """
        raise NotImplementedError

    # -------------------------
    # State methods Sync
    # -------------------------
    def put_state(self, config: dict[str, Any], state: StateT) -> StateT:
        """
        Store agent state synchronously.

        Args:
            config (dict): Configuration dictionary.
            state (StateT): State object to store.

        Returns:
            StateT: The stored state object.
        """
        return run_coroutine(self.aput_state(config, state))

    def get_state(self, config: dict[str, Any]) -> StateT | None:
        """
        Retrieve agent state synchronously.

        Args:
            config (dict): Configuration dictionary.

        Returns:
            StateT | None: Retrieved state or None.
        """
        return run_coroutine(self.aget_state(config))

    def clear_state(self, config: dict[str, Any]) -> Any:
        """
        Clear agent state synchronously.

        Args:
            config (dict): Configuration dictionary.

        Returns:
            Any: Implementation-defined result.
        """
        return run_coroutine(self.aclear_state(config))

    def put_state_cache(self, config: dict[str, Any], state: StateT) -> Any | None:
        """
        Store agent state in cache synchronously.

        Args:
            config (dict): Configuration dictionary.
            state (StateT): State object to cache.

        Returns:
            Any | None: Implementation-defined result.
        """
        return run_coroutine(self.aput_state_cache(config, state))

    def get_state_cache(self, config: dict[str, Any]) -> StateT | None:
        """
        Retrieve agent state from cache synchronously.

        Args:
            config (dict): Configuration dictionary.

        Returns:
            StateT | None: Cached state or None.
        """
        return run_coroutine(self.aget_state_cache(config))

    # -------------------------
    # Generic cache methods
    # -------------------------
    async def aput_cache_value(
        self,
        namespace: str,
        key: str,
        value: Any,
        ttl_seconds: int | None = None,
    ) -> Any | None:
        """Store a small JSON-serializable cache value.

        This is intentionally optional so existing checkpointers keep working
        even if they do not offer a shared cache backend.
        """
        return None

    async def aget_cache_value(self, namespace: str, key: str) -> Any | None:
        """Retrieve a cached value previously stored via ``aput_cache_value``."""
        return None

    async def aclear_cache_value(self, namespace: str, key: str) -> Any | None:
        """Delete a previously cached value."""
        return None

    async def alist_cache_keys(
        self,
        namespace: str,
        prefix: str | None = None,
    ) -> list[str]:
        """List all cache keys for a namespace.

        This is intentionally optional — default returns an empty list.
        Subclasses should override if they support key enumeration.

        Args:
            namespace: Cache namespace (e.g. "media:signed-url").
            prefix: Optional prefix to filter keys.

        Returns:
            List of cache key strings.
        """
        return []

    def put_cache_value(
        self,
        namespace: str,
        key: str,
        value: Any,
        ttl_seconds: int | None = None,
    ) -> Any | None:
        """Synchronously store a small cache value."""
        return run_coroutine(self.aput_cache_value(namespace, key, value, ttl_seconds))

    def get_cache_value(self, namespace: str, key: str) -> Any | None:
        """Synchronously retrieve a small cache value."""
        return run_coroutine(self.aget_cache_value(namespace, key))

    def clear_cache_value(self, namespace: str, key: str) -> Any | None:
        """Synchronously delete a small cache value."""
        return run_coroutine(self.aclear_cache_value(namespace, key))

    # -------------------------
    # Stop requests
    # -------------------------
    def _stop_request_key(self, config: dict[str, Any]) -> str:
        """Build the per-thread stop-request key.

        Scoped by user as well as thread so one user cannot stop another user's
        run by guessing a thread_id, matching how state and messages are scoped.
        """
        return f"{config.get('thread_id')}:{config.get('user_id')}"

    async def arequest_stop(self, config: dict[str, Any]) -> Any | None:
        """Record an out-of-band stop request for a thread.

        Stored under :data:`STOP_REQUEST_NAMESPACE` in the generic cache rather
        than in the state blob, so the running graph cannot overwrite it with its
        own state writes. Backends without a shared cache inherit the no-op
        generic-cache defaults, in which case this degrades to doing nothing
        rather than misbehaving.
        """
        return await self.aput_cache_value(
            STOP_REQUEST_NAMESPACE,
            self._stop_request_key(config),
            True,
            STOP_REQUEST_TTL_SECONDS,
        )

    async def ais_stop_requested(self, config: dict[str, Any]) -> bool:
        """Return True when an unconsumed stop request exists for this thread."""
        value = await self.aget_cache_value(
            STOP_REQUEST_NAMESPACE,
            self._stop_request_key(config),
        )
        return bool(value)

    async def aclear_stop_request(self, config: dict[str, Any]) -> Any | None:
        """Drop any pending stop request for a thread.

        Called once a stop has been acted on, and again when a new run starts, so
        a flag left behind by an abandoned run cannot kill the next one.
        """
        return await self.aclear_cache_value(
            STOP_REQUEST_NAMESPACE,
            self._stop_request_key(config),
        )

    def request_stop(self, config: dict[str, Any]) -> Any | None:
        """Synchronously record a stop request."""
        return run_coroutine(self.arequest_stop(config))

    def is_stop_requested(self, config: dict[str, Any]) -> bool:
        """Synchronously check for a pending stop request."""
        return run_coroutine(self.ais_stop_requested(config))

    def clear_stop_request(self, config: dict[str, Any]) -> Any | None:
        """Synchronously drop a pending stop request."""
        return run_coroutine(self.aclear_stop_request(config))

    # -------------------------
    # Tool execution ledger (idempotency)
    # -------------------------
    async def aget_tool_result(
        self,
        config: dict[str, Any],
        tool_call_id: str,
    ) -> dict[str, Any] | None:
        """Return a previously recorded result for this tool call, if any.

        The execution loop persists ``current_node`` *before* running a node and
        only advances it after the node completes. So a process killed mid-node
        (OOM, SIGKILL on deploy) resumes by re-running that node from scratch --
        and re-fires every tool in it. For a non-idempotent tool that means the
        card is charged twice.

        This ledger closes that: a tool call that already completed is recorded
        durably against its ``tool_call_id``, and on replay its recorded result is
        returned instead of the tool being called again.

        Returning None means "no record" -- the tool should run. Backends that do
        not implement the ledger inherit this default, which simply disables
        idempotency rather than breaking (behaviour matches the old at-least-once
        semantics).
        """
        return None

    async def aput_tool_result(
        self,
        config: dict[str, Any],
        tool_call_id: str,
        result: dict[str, Any],
    ) -> Any | None:
        """Durably record that a tool call completed, with its result.

        Written immediately after the tool returns, so a crash later in the same
        node cannot cause the tool to be re-fired on resume.
        """
        return None

    # -------------------------
    # Message methods async
    # -------------------------
    @abstractmethod
    async def aput_messages(
        self,
        config: dict[str, Any],
        messages: list[Message],
        metadata: dict[str, Any] | None = None,
    ) -> Any:
        """
        Store messages asynchronously.

        Args:
            config (dict): Configuration dictionary.
            messages (list[Message]): List of messages to store.
            metadata (dict, optional): Additional metadata.

        Returns:
            Any: Implementation-defined result.
        """
        raise NotImplementedError

    @abstractmethod
    async def aget_message(self, config: dict[str, Any], message_id: str | int) -> Message:
        """
        Retrieve a specific message asynchronously.

        Args:
            config (dict): Configuration dictionary.
            message_id (str|int): Message identifier.

        Returns:
            Message: Retrieved message object.
        """
        raise NotImplementedError

    @abstractmethod
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
        raise NotImplementedError

    @abstractmethod
    async def adelete_message(self, config: dict[str, Any], message_id: str | int) -> Any | None:
        """
        Delete a specific message asynchronously.

        Args:
            config (dict): Configuration dictionary.
            message_id (str|int): Message identifier.

        Returns:
            Any | None: Implementation-defined result.
        """
        raise NotImplementedError

    # -------------------------
    # Message methods sync
    # -------------------------
    def put_messages(
        self,
        config: dict[str, Any],
        messages: list[Message],
        metadata: dict[str, Any] | None = None,
    ) -> Any:
        """
        Store messages synchronously.

        Args:
            config (dict): Configuration dictionary.
            messages (list[Message]): List of messages to store.
            metadata (dict, optional): Additional metadata.

        Returns:
            Any: Implementation-defined result.
        """
        return run_coroutine(self.aput_messages(config, messages, metadata))

    def get_message(self, config: dict[str, Any], message_id: str | int) -> Message:
        """
        Retrieve a specific message synchronously.

        Args:
            config (dict): Configuration dictionary.

        Returns:
            Message: Retrieved message object.
        """
        return run_coroutine(self.aget_message(config, message_id))

    def list_messages(
        self,
        config: dict[str, Any],
        search: str | None = None,
        offset: int | None = None,
        limit: int | None = None,
    ) -> list[Message]:
        """
        List messages synchronously with optional filtering.

        Args:
            config (dict): Configuration dictionary.
            search (str, optional): Search string.
            offset (int, optional): Offset for pagination.
            limit (int, optional): Limit for pagination.

        Returns:
            list[Message]: List of message objects.
        """
        return run_coroutine(self.alist_messages(config, search, offset, limit))

    def delete_message(self, config: dict[str, Any], message_id: str | int) -> Any | None:
        """
        Delete a specific message synchronously.

        Args:
            config (dict): Configuration dictionary.
            message_id (str|int): Message identifier.

        Returns:
            Any | None: Implementation-defined result.
        """
        return run_coroutine(self.adelete_message(config, message_id))

    # -------------------------
    # Thread methods async
    # -------------------------
    @abstractmethod
    async def aput_thread(
        self,
        config: dict[str, Any],
        thread_info: ThreadInfo,
    ) -> Any | None:
        """
        Store thread info asynchronously.

        Args:
            config (dict): Configuration dictionary.
            thread_info (ThreadInfo): Thread information object.

        Returns:
            Any | None: Implementation-defined result.
        """
        raise NotImplementedError

    @abstractmethod
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
        raise NotImplementedError

    @abstractmethod
    async def alist_threads(
        self,
        config: dict[str, Any],
        search: str | None = None,
        offset: int | None = None,
        limit: int | None = None,
    ) -> list[ThreadInfo]:
        """
        List threads asynchronously with optional filtering.

        Args:
            config (dict): Configuration dictionary.
            search (str, optional): Search string.
            offset (int, optional): Offset for pagination.
            limit (int, optional): Limit for pagination.

        Returns:
            list[ThreadInfo]: List of thread information objects.
        """
        raise NotImplementedError

    @abstractmethod
    async def aclean_thread(self, config: dict[str, Any]) -> Any | None:
        """
        Clean/delete thread asynchronously.

        Args:
            config (dict): Configuration dictionary.

        Returns:
            Any | None: Implementation-defined result.
        """
        raise NotImplementedError

    async def aget_thread_owner(self, thread_id: str | int) -> str | int | None:
        """Return the ``user_id`` that owns ``thread_id``, regardless of caller.

        Unlike :meth:`aget_thread` (which is owner-scoped), this resolves ownership
        globally so an authorization layer can decide whether a *different* user may
        act on a thread. It exists to answer three states:

        - returns the owner's ``user_id`` -> the thread exists and is owned by them;
        - returns ``None`` -> the thread does not exist yet (a brand-new session);
        - raises :class:`NotImplementedError` -> this backend cannot resolve ownership.

        The default raises so a backend cannot silently report "no owner" for every
        thread (which would defeat owner-based authorization). Concrete backends that
        persist a thread registry override this.
        """
        raise NotImplementedError

    # -------------------------
    # Thread methods sync
    # -------------------------
    def put_thread(self, config: dict[str, Any], thread_info: ThreadInfo) -> Any | None:
        """
        Store thread info synchronously.

        Args:
            config (dict): Configuration dictionary.
            thread_info (ThreadInfo): Thread information object.

        Returns:
            Any | None: Implementation-defined result.
        """
        return run_coroutine(self.aput_thread(config, thread_info))

    def get_thread(self, config: dict[str, Any]) -> ThreadInfo | None:
        """
        Retrieve thread info synchronously.

        Args:
            config (dict): Configuration dictionary.

        Returns:
            ThreadInfo | None: Thread information object or None.
        """
        return run_coroutine(self.aget_thread(config))

    def list_threads(
        self,
        config: dict[str, Any],
        search: str | None = None,
        offset: int | None = None,
        limit: int | None = None,
    ) -> list[ThreadInfo]:
        """
        List threads synchronously with optional filtering.

        Args:
            config (dict): Configuration dictionary.
            search (str, optional): Search string.
            offset (int, optional): Offset for pagination.
            limit (int, optional): Limit for pagination.

        Returns:
            list[ThreadInfo]: List of thread information objects.
        """
        return run_coroutine(self.alist_threads(config, search, offset, limit))

    def clean_thread(self, config: dict[str, Any]) -> Any | None:
        """
        Clean/delete thread synchronously.

        Args:
            config (dict): Configuration dictionary.

        Returns:
            Any | None: Implementation-defined result.
        """
        return run_coroutine(self.aclean_thread(config))

    # -------------------------
    # Clean Resources
    # -------------------------
    def release(self) -> Any | None:
        """
        Release resources synchronously.

        Returns:
            Any | None: Implementation-defined result.
        """
        return run_coroutine(self.arelease())

    @abstractmethod
    async def arelease(self) -> Any | None:
        """
        Release resources asynchronously.

        Returns:
            Any | None: Implementation-defined result.
        """
        raise NotImplementedError
