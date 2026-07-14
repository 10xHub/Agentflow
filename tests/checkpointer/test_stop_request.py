"""Tests for out-of-band stop requests and per-resource release ownership.

The stop request deliberately lives in its own checkpointer key rather than
inside the cached state blob. A running graph rewrites the state cache after
every node with its own copy of the state, so a flag stored in that blob is
overwritten by the very run it is meant to stop. These tests pin that down.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from agentflow.core.state import AgentState
from agentflow.core.state.execution_state import StopRequestStatus
from agentflow.storage.checkpointer.in_memory_checkpointer import InMemoryCheckpointer
from agentflow.storage.checkpointer.pg_checkpointer import PgCheckpointer


CONFIG = {"thread_id": "t1", "user_id": "u1"}


class TestStopRequest:
    """The stop flag must survive the running graph's own cache writes."""

    @pytest.mark.asyncio
    async def test_stop_survives_state_cache_overwrite(self):
        """Regression: the run loop's state-cache write must not swallow a stop.

        This is the exact race that made stop() unreliable. `stop()` recorded the
        flag inside the cached state; the running loop then wrote its own
        flag-less state back to the same key (at the same durable version, so a
        version-guarded cache accepts it) and the next stop check saw nothing.
        """
        cp = InMemoryCheckpointer()
        await cp.aput_state_cache(CONFIG, AgentState())

        await cp.arequest_stop(CONFIG)

        # The running graph rewrites the state cache with its own copy of the
        # state, which knows nothing about the stop that just arrived.
        await cp.aput_state_cache(CONFIG, AgentState())

        assert await cp.ais_stop_requested(CONFIG) is True

    @pytest.mark.asyncio
    async def test_no_stop_by_default(self):
        cp = InMemoryCheckpointer()
        assert await cp.ais_stop_requested(CONFIG) is False

    @pytest.mark.asyncio
    async def test_stop_is_cleared_once_consumed(self):
        cp = InMemoryCheckpointer()
        await cp.arequest_stop(CONFIG)

        await cp.aclear_stop_request(CONFIG)

        assert await cp.ais_stop_requested(CONFIG) is False

    @pytest.mark.asyncio
    async def test_stop_is_scoped_per_thread(self):
        """A stop on one thread must not stop another user's run."""
        cp = InMemoryCheckpointer()
        await cp.arequest_stop({"thread_id": "t1", "user_id": "u1"})

        assert await cp.ais_stop_requested({"thread_id": "t2", "user_id": "u1"}) is False
        assert await cp.ais_stop_requested({"thread_id": "t1", "user_id": "u2"}) is False
        assert await cp.ais_stop_requested({"thread_id": "t1", "user_id": "u1"}) is True


class TestStopRequestConsumption:
    """check_stop_requested must observe, act on, and consume the flag."""

    @pytest.fixture
    def stop_deps(self):
        """Supply the deps check_stop_requested would otherwise resolve via DI.

        Each test monkeypatches handler_utils.sync_data/reload_state itself, so
        the patches are undone after the test. Assigning them on the module here
        would leak into the rest of the session.
        """
        cp = InMemoryCheckpointer()
        callback_mgr = MagicMock()
        callback_mgr._lifecycle_hooks = None
        return cp, callback_mgr

    @pytest.mark.asyncio
    async def test_flagless_cached_state_still_stops(self, stop_deps, monkeypatch):
        """The stop is honoured even when the cached state has no flag at all.

        This is the post-clobber situation: the loop has already overwritten the
        state cache, so the only surviving record of the stop is the dedicated
        key.
        """
        from agentflow.core.graph.utils import handler_utils
        from agentflow.runtime.publisher.events import ContentType, EventModel

        cp, callback_mgr = stop_deps
        state = AgentState()
        assert state.is_stopped_requested() is False

        # reload_state would hand back the loop's flag-less cached state.
        async def _reload(config, old_state):
            return old_state

        monkeypatch.setattr(handler_utils, "reload_state", _reload)
        monkeypatch.setattr(handler_utils, "sync_data", AsyncMock(return_value=False))
        monkeypatch.setattr(handler_utils, "publish_event", MagicMock())

        await cp.arequest_stop(CONFIG)

        stopped = await handler_utils.check_stop_requested(
            state,
            "agent",
            EventModel.default(CONFIG, data={}, content_type=[ContentType.STATE]),
            [],
            CONFIG,
            callback_mgr=callback_mgr,
            checkpointer=cp,
        )

        assert stopped is True
        assert state.execution_meta.stop_current_execution == StopRequestStatus.STOP_REQUESTED
        assert state.is_interrupted()
        # Consumed, so it cannot leak into the next run on this thread.
        assert await cp.ais_stop_requested(CONFIG) is False

    @pytest.mark.asyncio
    async def test_no_stop_requested_continues(self, stop_deps, monkeypatch):
        from agentflow.core.graph.utils import handler_utils
        from agentflow.runtime.publisher.events import ContentType, EventModel

        cp, callback_mgr = stop_deps
        state = AgentState()

        async def _reload(config, old_state):
            return old_state

        monkeypatch.setattr(handler_utils, "reload_state", _reload)
        monkeypatch.setattr(handler_utils, "sync_data", AsyncMock(return_value=False))
        monkeypatch.setattr(handler_utils, "publish_event", MagicMock())

        stopped = await handler_utils.check_stop_requested(
            state,
            "agent",
            EventModel.default(CONFIG, data={}, content_type=[ContentType.STATE]),
            [],
            CONFIG,
            callback_mgr=callback_mgr,
            checkpointer=cp,
        )

        assert stopped is False
        assert not state.is_interrupted()


class TestReleaseOwnership:
    """arelease() must only close pools the checkpointer itself created."""

    def _pool(self):
        pool = MagicMock()
        pool.is_closing.return_value = False
        pool.close = AsyncMock()
        return pool

    @pytest.mark.asyncio
    async def test_caller_supplied_pg_pool_is_not_closed(self):
        """Regression: building Redis from a URL must not close the caller's pg pool.

        A single shared release flag meant passing your own pg_pool together with
        a redis_url made arelease() close the pool your app was still using.
        """
        pool = self._pool()
        cp = PgCheckpointer(pg_pool=pool, redis_url="redis://localhost:6379/0")
        cp.redis = AsyncMock()

        assert cp._owns_pg_pool is False
        assert cp._owns_redis is True

        await cp.arelease()

        pool.close.assert_not_called()
        cp.redis.aclose.assert_awaited()

    @pytest.mark.asyncio
    async def test_self_created_pg_pool_is_closed(self):
        cp = PgCheckpointer(
            postgres_dsn="postgresql://localhost/test",
            redis_url="redis://localhost:6379/0",
        )
        cp.redis = AsyncMock()
        pool = self._pool()
        cp._pg_pool = pool

        # Ownership is decided from the constructor args (a DSN, not a pool), so
        # it holds even though the pool itself is created lazily.
        assert cp._owns_pg_pool is True

        await cp.arelease()

        pool.close.assert_awaited()

    @pytest.mark.asyncio
    async def test_explicit_release_resources_closes_everything(self):
        """release_resources=True remains an explicit opt-in to close borrowed pools."""
        pool = self._pool()
        cp = PgCheckpointer(
            pg_pool=pool,
            redis=AsyncMock(),
            release_resources=True,
        )

        await cp.arelease()

        pool.close.assert_awaited()

    @pytest.mark.asyncio
    async def test_fully_borrowed_resources_are_left_alone(self):
        pool = self._pool()
        redis = AsyncMock()
        cp = PgCheckpointer(pg_pool=pool, redis=redis)

        await cp.arelease()

        pool.close.assert_not_called()
        redis.aclose.assert_not_called()
