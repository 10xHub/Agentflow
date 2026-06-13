import pytest
from unittest.mock import AsyncMock, MagicMock
from agentflow.storage.checkpointer.base_checkpointer import BaseCheckpointer
from agentflow.core.state import AgentState, Message
from agentflow.utils.thread_info import ThreadInfo


class DummyCheckpointer(BaseCheckpointer):
    async def asetup(self): pass
    async def aput_state(self, config, state): pass
    async def aget_state(self, config): pass
    async def aclear_state(self, config): pass
    async def aput_state_cache(self, config, state): pass
    async def aget_state_cache(self, config): pass
    async def aput_messages(self, config, messages, metadata=None): pass
    async def aget_message(self, config, message_id): pass
    async def alist_messages(self, config, search=None, offset=None, limit=None): pass
    async def adelete_message(self, config, message_id): pass
    async def aput_thread(self, config, thread_info): pass
    async def aget_thread(self, config): pass
    async def alist_threads(self, config, search=None, offset=None, limit=None): pass
    async def aclean_thread(self, config): pass
    async def arelease(self): pass


class MinimalCheckpointer(BaseCheckpointer):
    async def asetup(self): pass
    async def aput_state(self, config, state): pass
    async def aget_state(self, config): pass
    async def aclear_state(self, config): pass
    async def aput_state_cache(self, config, state): pass
    async def aget_state_cache(self, config): pass
    async def aput_messages(self, config, messages, metadata=None): pass
    async def aget_message(self, config, message_id): pass
    async def alist_messages(self, config, search=None, offset=None, limit=None): pass
    async def adelete_message(self, config, message_id): pass
    async def aput_thread(self, config, thread_info): pass
    async def aget_thread(self, config): pass
    async def alist_threads(self, config, search=None, offset=None, limit=None): pass
    async def aclean_thread(self, config): pass
    async def arelease(self): pass


def test_base_checkpointer_sync_wrappers():
    cp = DummyCheckpointer()
    cp.asetup = AsyncMock()
    cp.aput_state = AsyncMock()
    cp.aget_state = AsyncMock()
    cp.aclear_state = AsyncMock()
    cp.aput_state_cache = AsyncMock()
    cp.aget_state_cache = AsyncMock()
    cp.aput_messages = AsyncMock()
    cp.aget_message = AsyncMock()
    cp.alist_messages = AsyncMock()
    cp.adelete_message = AsyncMock()
    cp.aput_thread = AsyncMock()
    cp.aget_thread = AsyncMock()
    cp.alist_threads = AsyncMock()
    cp.aclean_thread = AsyncMock()
    cp.arelease = AsyncMock()

    config = {"thread_id": "test_thread"}
    state = MagicMock(spec=AgentState)
    messages = [MagicMock(spec=Message)]
    thread_info = MagicMock(spec=ThreadInfo)

    # Test setup
    cp.setup()
    cp.asetup.assert_called_once()

    # Test state methods
    cp.put_state(config, state)
    cp.aput_state.assert_called_once_with(config, state)

    cp.get_state(config)
    cp.aget_state.assert_called_once_with(config)

    cp.clear_state(config)
    cp.aclear_state.assert_called_once_with(config)

    cp.put_state_cache(config, state)
    cp.aput_state_cache.assert_called_once_with(config, state)

    cp.get_state_cache(config)
    cp.aget_state_cache.assert_called_once_with(config)

    # Test cache values
    cp.put_cache_value("ns", "k", "v", ttl_seconds=10)
    cp.get_cache_value("ns", "k")
    cp.clear_cache_value("ns", "k")

    # Test message methods
    cp.put_messages(config, messages, metadata={"meta": 1})
    cp.aput_messages.assert_called_once_with(config, messages, {"meta": 1})

    cp.get_message(config, "msg1")
    cp.aget_message.assert_called_once_with(config, "msg1")

    cp.list_messages(config, search="x", offset=1, limit=5)
    cp.alist_messages.assert_called_once_with(config, "x", 1, 5)

    cp.delete_message(config, "msg1")
    cp.adelete_message.assert_called_once_with(config, "msg1")

    # Test thread methods
    cp.put_thread(config, thread_info)
    cp.aput_thread.assert_called_once_with(config, thread_info)

    cp.get_thread(config)
    cp.aget_thread.assert_called_once_with(config)

    cp.list_threads(config, search="x", offset=1, limit=5)
    cp.alist_threads.assert_called_once_with(config, "x", 1, 5)

    cp.clean_thread(config)
    cp.aclean_thread.assert_called_once_with(config)

    # Test release
    cp.release()
    cp.arelease.assert_called_once()


@pytest.mark.asyncio
async def test_base_checkpointer_default_cache_methods():
    cp = MinimalCheckpointer()
    assert await cp.aput_cache_value("ns", "k", "v") is None
    assert await cp.aget_cache_value("ns", "k") is None
    assert await cp.aclear_cache_value("ns", "k") is None
    assert await cp.alist_cache_keys("ns") == []
