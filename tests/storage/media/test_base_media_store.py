import pytest
from typing import Any
from agentflow.storage.media.storage.base import BaseMediaStore
from agentflow.core.state.message_block import MediaRef

class ConcreteMediaStore(BaseMediaStore):
    """A concrete implementation of BaseMediaStore for testing."""
    def __init__(self):
        self.store_dict = {}

    async def store(self, data: bytes, mime_type: str, metadata: dict[str, Any] | None = None) -> str:
        key = f"key_{len(self.store_dict)}"
        self.store_dict[key] = (data, mime_type)
        return key

    async def retrieve(self, storage_key: str) -> tuple[bytes, str]:
        if storage_key not in self.store_dict:
            raise KeyError(f"Key {storage_key} not found")
        return self.store_dict[storage_key]

    async def delete(self, storage_key: str) -> bool:
        if storage_key in self.store_dict:
            del self.store_dict[storage_key]
            return True
        return False

    async def exists(self, storage_key: str) -> bool:
        return storage_key in self.store_dict

@pytest.mark.asyncio
async def test_get_metadata_success():
    store = ConcreteMediaStore()
    key = await store.store(b"hello", "text/plain")
    metadata = await store.get_metadata(key)
    assert metadata == {
        "mime_type": "text/plain",
        "size_bytes": 5
    }

@pytest.mark.asyncio
async def test_get_metadata_key_error():
    store = ConcreteMediaStore()
    metadata = await store.get_metadata("nonexistent")
    assert metadata is None

@pytest.mark.asyncio
async def test_get_direct_url_default():
    store = ConcreteMediaStore()
    url = await store.get_direct_url("key")
    assert url is None

def test_to_media_ref():
    store = ConcreteMediaStore()
    ref = store.to_media_ref("key", "image/png")
    assert isinstance(ref, MediaRef)
    assert ref.kind == "url"
    assert ref.url == "agentflow://media/key"
    assert ref.mime_type == "image/png"
