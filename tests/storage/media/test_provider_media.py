import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from agentflow.storage.media.provider_media import (
    ProviderMediaCache,
    should_use_google_file_api,
    prepare_google_content_part,
    upload_to_google_file_api,
    create_openai_file_search_tool,
    create_openai_file_attachment
)

def test_provider_media_cache():
    cache = ProviderMediaCache(max_entries=2)
    key1 = cache.content_key(b"data1")
    key2 = cache.content_key(b"data2")
    key3 = cache.content_key(b"data3")

    cache.put("google", key1, "ref1")
    cache.put("google", key2, "ref2")
    assert cache.get("google", key1) == "ref1"
    assert cache.get("google", key2) == "ref2"

    # Test eviction: key1 (oldest) should be evicted when key3 is added
    cache.put("google", key3, "ref3")
    assert cache.get("google", key1) is None
    assert cache.get("google", key2) == "ref2"
    assert cache.get("google", key3) == "ref3"

    # Test clear for specific provider
    cache.clear("google")
    assert cache.get("google", key2) is None

    # Test clear for all
    cache.put("google", key2, "ref2")
    cache.clear()
    assert cache.get("google", key2) is None

def test_should_use_google_file_api():
    # threshold is 20MB
    assert should_use_google_file_api(10 * 1024 * 1024) is False
    assert should_use_google_file_api(25 * 1024 * 1024) is True

def test_prepare_google_content_part():
    class _Part:
        @staticmethod
        def from_bytes(data, mime_type):
            return {"data": data, "mime": mime_type}

    with patch("google.genai.types.Part", _Part):
        # Under threshold
        res = prepare_google_content_part(b"abc", "image/png")
        assert res == {"data": b"abc", "mime": "image/png"}

        # Over threshold
        with pytest.raises(ValueError):
            prepare_google_content_part(b"abc" * 10 * 1024 * 1024, "image/png")

@pytest.mark.asyncio
async def test_upload_to_google_file_api():
    class _Part:
        @staticmethod
        def from_uri(file_uri, mime_type):
            return {"uri": file_uri, "mime": mime_type}

    mock_client = MagicMock()
    mock_upload_res = MagicMock()
    mock_upload_res.uri = "gs://test-bucket/file-1"
    mock_upload_res.mime_type = "image/png"
    mock_client.files.upload.return_value = mock_upload_res

    # Mock google.genai.types.Part
    with patch("google.genai.types.Part", _Part):
        # 1. Successful upload without cache
        res = await upload_to_google_file_api(b"data", "image/png", client=mock_client)
        assert res == {"uri": "gs://test-bucket/file-1", "mime": "image/png"}

        # 2. Caching logic (hit and miss)
        cache = ProviderMediaCache()
        res_uncached = await upload_to_google_file_api(b"data", "image/png", cache=cache, client=mock_client)
        assert res_uncached == {"uri": "gs://test-bucket/file-1", "mime": "image/png"}

        # Call again -> should hit cache and not call upload
        mock_client.files.upload.reset_mock()
        res_cached = await upload_to_google_file_api(b"data", "image/png", cache=cache, client=mock_client)
        assert res_cached == {"uri": "gs://test-bucket/file-1", "mime": "image/png"}
        mock_client.files.upload.assert_not_called()

        # 3. Default client creation test (mocking client = None)
        mock_genai_client = MagicMock()
        mock_genai_client.files.upload.return_value = mock_upload_res
        with patch("google.genai.Client", return_value=mock_genai_client):
            res_default_client = await upload_to_google_file_api(b"data_other", "image/png", client=None)
            assert res_default_client == {"uri": "gs://test-bucket/file-1", "mime": "image/png"}

def test_openai_helpers():
    search_tool = create_openai_file_search_tool(["file-1"])
    assert search_tool == {
        "type": "file_search",
        "file_search": {
            "vector_store_ids": []
        }
    }

    attachment = create_openai_file_attachment("file-2", ["file_search"])
    assert attachment == {
        "file_id": "file-2",
        "tools": [{"type": "file_search"}]
    }
