import base64
from unittest.mock import AsyncMock, patch

import pytest

from agentflow.core.exceptions.media_exceptions import UnsupportedMediaInputError
from agentflow.core.state.message_block import MediaRef
from agentflow.storage.media.capabilities import MediaTransportMode
from agentflow.storage.media.media_resolver import MediaResolver, _openai_image_url


class _Store:
    def __init__(self):
        self.retrieve_map = {"k": (b"abc", "image/png")}
        self.url_map = {"k": "https://signed.example/k.png"}

    async def retrieve(self, key):
        return self.retrieve_map[key]

    async def get_direct_url(self, key, mime_type=None):
        return self.url_map.get(key)


class _Cache:
    def __init__(self):
        self.values = {}
        self.put_calls = []

    async def aget_cache_value(self, namespace, key):
        return self.values.get((namespace, key))

    async def aput_cache_value(self, namespace, key, value, ttl_seconds=0):
        self.values[(namespace, key)] = value
        self.put_calls.append((namespace, key, value, ttl_seconds))


@pytest.mark.asyncio
async def test_try_transport_returns_none_for_unsupported_mode():
    resolver = MediaResolver()
    ref = MediaRef(kind="url", url="https://example.com/x.png")
    result = await resolver._try_transport(ref, MediaTransportMode.unsupported, "openai", "gpt-4o", object())
    assert result is None


@pytest.mark.asyncio
async def test_transport_remote_url_returns_none_for_internal_ref_without_store():
    resolver = MediaResolver(media_store=None)
    ref = MediaRef(kind="url", url="agentflow://media/k")
    result = await resolver._transport_remote_url(ref, type("Caps", (), {"accepts_external_urls": True})())
    assert result is None


@pytest.mark.asyncio
async def test_transport_remote_url_respects_external_url_capability_flag():
    resolver = MediaResolver()
    ref = MediaRef(kind="url", url="https://example.com/x.png")
    result = await resolver._transport_remote_url(
        ref,
        type("Caps", (), {"accepts_external_urls": False})(),
    )
    assert result is None


@pytest.mark.asyncio
async def test_retrieve_bytes_from_data_ref():
    payload = base64.b64encode(b"hello").decode()
    resolver = MediaResolver()
    data, mime = await resolver._retrieve_bytes(MediaRef(kind="data", data_base64=payload, mime_type="text/plain"))
    assert data == b"hello"
    assert mime == "text/plain"


@pytest.mark.asyncio
async def test_fetch_external_url_reads_response_bytes_and_mime():
    resolver = MediaResolver()

    class _Resp:
        headers = {"Content-Type": "image/jpeg"}

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def raise_for_status(self):
            return None

        async def read(self):
            return b"jpg"

    class _Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def get(self, url):
            return _Resp()

    with patch("aiohttp.ClientSession", return_value=_Session()):
        data, mime = await resolver._fetch_external_url("https://x")

    assert data == b"jpg"
    assert mime == "image/jpeg"


@pytest.mark.asyncio
async def test_get_direct_url_uses_cache_hit_when_not_expiring(monkeypatch):
    store = _Store()
    cache = _Cache()
    resolver = MediaResolver(media_store=store, cache_backend=cache)

    cache.values[("media:signed-url", "k:application/octet-stream:3600")] = {
        "url": "https://cached.example/k.png",
        "expires_at": 9999999999,
    }

    result = await resolver._get_direct_url(MediaRef(kind="url", url="agentflow://media/k"))
    assert result == "https://cached.example/k.png"


@pytest.mark.asyncio
async def test_get_direct_url_puts_cache_on_miss():
    store = _Store()
    cache = _Cache()
    resolver = MediaResolver(media_store=store, cache_backend=cache)

    result = await resolver._get_direct_url(MediaRef(kind="url", url="agentflow://media/k", mime_type="image/png"))
    assert result == "https://signed.example/k.png"
    assert len(cache.put_calls) == 1


@pytest.mark.asyncio
async def test_resolve_raises_with_attempted_transports_when_all_fail(monkeypatch):
    resolver = MediaResolver()
    ref = MediaRef(kind="url", url="https://bad.example/fail.png")

    async def _always_fail(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(resolver, "_try_transport", _always_fail)

    with pytest.raises(UnsupportedMediaInputError) as exc:
        await resolver.resolve(ref, provider="openai", model="gpt-4o", media_type="image")

    assert "All transports failed" in str(exc.value)


@pytest.mark.asyncio
async def test_transport_provider_file_handles_google_uri_and_data(monkeypatch):
    resolver = MediaResolver(media_store=_Store())

    class _Part:
        @staticmethod
        def from_uri(file_uri, mime_type):
            return {"uri": file_uri, "mime": mime_type}

    async def _upload(data, mime):
        return {"uploaded": True, "mime": mime, "size": len(data)}

    with patch("google.genai.types.Part", _Part), patch(
        "agentflow.storage.media.provider_media.upload_to_google_file_api",
        new=_upload,
    ):
        gs = await resolver._transport_provider_file(
            MediaRef(kind="url", url="gs://bucket/x.jpg", mime_type="image/jpeg"),
            provider="google",
            model="gemini-1.5-pro",
        )
        assert gs["uri"].startswith("gs://")

        data_ref = await resolver._transport_provider_file(
            MediaRef(kind="data", data_base64=base64.b64encode(b"abc").decode(), mime_type="image/png"),
            provider="google",
            model="gemini-1.5-pro",
        )
        assert data_ref["uploaded"] is True


@pytest.mark.asyncio
async def test_transport_provider_file_handles_internal_and_external_url_uploads():
    resolver = MediaResolver(media_store=_Store())

    async def _upload(data, mime):
        return {"uploaded": True, "mime": mime, "size": len(data)}

    with patch("agentflow.storage.media.provider_media.upload_to_google_file_api", new=_upload), patch.object(
        resolver,
        "_retrieve_bytes",
        new=AsyncMock(side_effect=[(b"in", "image/png"), (b"out", "image/jpeg")]),
    ):
        a = await resolver._transport_provider_file(
            MediaRef(kind="url", url="agentflow://media/k"),
            provider="google",
            model="gemini-1.5-pro",
        )
        b = await resolver._transport_provider_file(
            MediaRef(kind="url", url="https://example.com/x.jpg"),
            provider="google",
            model="gemini-1.5-pro",
        )

    assert a["uploaded"] is True
    assert b["uploaded"] is True


@pytest.mark.asyncio
async def test_transport_provider_file_returns_none_on_non_google_and_errors(monkeypatch):
    resolver = MediaResolver(media_store=_Store())
    assert (
        await resolver._transport_provider_file(
            MediaRef(kind="url", url="https://x"),
            provider="openai",
            model="gpt-4o",
        )
        is None
    )

    async def _broken(*args, **kwargs):
        raise RuntimeError("upload failed")

    with patch("agentflow.storage.media.provider_media.upload_to_google_file_api", new=_broken):
        out = await resolver._transport_provider_file(
            MediaRef(kind="data", data_base64=base64.b64encode(b"abc").decode(), mime_type="image/png"),
            provider="google",
            model="gemini-1.5-pro",
        )
    assert out is None


@pytest.mark.asyncio
async def test_transport_provider_file_returns_none_for_unhandled_google_ref_kind():
    resolver = MediaResolver(media_store=_Store())
    out = await resolver._transport_provider_file(
        MediaRef(kind="file_id", file_id="f-1"),
        provider="google",
        model="gemini-1.5-pro",
    )
    assert out is None


@pytest.mark.asyncio
async def test_retrieve_and_get_direct_url_none_paths():
    resolver = MediaResolver(media_store=None, cache_backend=None)

    with pytest.raises(RuntimeError):
        await resolver._retrieve("agentflow://media/missing")

    assert await resolver._get_direct_url(MediaRef(kind="url", url=None)) is None


@pytest.mark.asyncio
async def test_get_direct_url_ignores_invalid_cache_payload_and_falls_back_to_store():
    store = _Store()
    cache = _Cache()
    resolver = MediaResolver(media_store=store, cache_backend=cache)
    cache.values[("media:signed-url", "k:image/png:3600")] = {
        "url": 123,
        "expires_at": "not-a-number",
    }

    out = await resolver._get_direct_url(MediaRef(kind="url", url="agentflow://media/k", mime_type="image/png"))
    assert out == "https://signed.example/k.png"


@pytest.mark.asyncio
async def test_get_direct_url_returns_none_when_store_has_no_url():
    store = _Store()
    store.url_map["k"] = None
    resolver = MediaResolver(media_store=store, cache_backend=None)
    out = await resolver._get_direct_url(MediaRef(kind="url", url="agentflow://media/k", mime_type="image/png"))
    assert out is None


@pytest.mark.asyncio
async def test_retrieve_bytes_invalid_kind_raises():
    resolver = MediaResolver(media_store=_Store())
    with pytest.raises(ValueError):
        await resolver._retrieve_bytes(MediaRef(kind="file_id", file_id="x"))


def test_openai_image_url_helper_shape():
    part = _openai_image_url("https://example.com/x.png")
    assert part == {"type": "image_url", "image_url": {"url": "https://example.com/x.png"}}


def test_with_cache_configures_resolver():
    from agentflow.storage.media.resolver import MediaRefResolver
    resolver = MediaRefResolver()
    cache = object()
    out = resolver.with_cache(cache, expiration_seconds=1800, refresh_buffer_seconds=30)
    assert out is resolver
    assert resolver.cache_backend is cache
    assert resolver.direct_url_expiration_seconds == 1800
    assert resolver.direct_url_refresh_buffer_seconds == 30


@pytest.mark.asyncio
async def test_resolve_openai_legacy_fallback_reftypes_and_empty():
    from agentflow.storage.media.resolver import MediaRefResolver
    resolver = MediaRefResolver()
    ref_empty = MediaRef.model_construct(kind="unknown")
    res = await resolver._resolve_openai_legacy(ref_empty)
    assert res == {"type": "image_url", "image_url": {"url": ""}}


@pytest.mark.asyncio
async def test_resolve_google_legacy_various_refs():
    from agentflow.storage.media.resolver import MediaRefResolver
    resolver = MediaRefResolver(media_store=_Store())
    
    class _Part:
        @staticmethod
        def from_uri(file_uri, mime_type):
            return {"uri": file_uri, "mime": mime_type}
        
        @staticmethod
        def from_bytes(data, mime_type):
            return {"data": data, "mime": mime_type}
            
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    with patch("google.genai.types.Part", _Part):
        ref_internal = MediaRef(kind="url", url="agentflow://media/k")
        part1 = await resolver._resolve_google_legacy(ref_internal)
        assert part1 == {"uri": "https://signed.example/k.png", "mime": "application/octet-stream"}
        
        resolver.media_store.url_map["k"] = None
        part2 = await resolver._resolve_google_legacy(ref_internal)
        assert part2 == {"data": b"abc", "mime": "image/png"}
        
        ref_external = MediaRef(kind="url", url="https://example.com/y.jpg", mime_type="image/jpeg")
        part3 = await resolver._resolve_google_legacy(ref_external)
        assert part3 == {"uri": "https://example.com/y.jpg", "mime": "image/jpeg"}
        
        ref_data = MediaRef(kind="data", data_base64=base64.b64encode(b"hello").decode(), mime_type="text/plain")
        part4 = await resolver._resolve_google_legacy(ref_data)
        assert part4 == {"data": b"hello", "mime": "text/plain"}
        
        ref_file = MediaRef(kind="file_id", file_id="file-123", mime_type="image/png")
        part5 = await resolver._resolve_google_legacy(ref_file)
        assert isinstance(part5, _Part)
        assert part5.kwargs["file_data"].file_uri == "file-123"
        
        ref_unres = MediaRef.model_construct(kind="unknown")
        part6 = await resolver._resolve_google_legacy(ref_unres)
        assert isinstance(part6, _Part)
        assert part6.kwargs["text"] == "[Unresolvable media reference]"


@pytest.mark.asyncio
async def test_try_transport_modes():
    from agentflow.storage.media.resolver import MediaRefResolver
    resolver = MediaRefResolver()
    
    result = await resolver._try_transport(MediaRef(kind="url"), MediaTransportMode.provider_file, "openai", object())
    assert result is None
    
    class _Part:
        @staticmethod
        def from_uri(file_uri, mime_type):
            return {"uri": file_uri, "mime": mime_type}
            
    resolver.media_store = _Store()
    caps = type("Caps", (), {"can_convert_internal_to_remote": True})()
    with patch("google.genai.types.Part", _Part):
        res = await resolver._transport_remote_url(
            MediaRef(kind="url", url="agentflow://media/k", mime_type="image/png"),
            caps,
            provider="google"
        )
        assert res == {"uri": "https://signed.example/k.png", "mime": "image/png"}


@pytest.mark.asyncio
async def test_transport_inline_bytes_url_retrieve_fail():
    from agentflow.storage.media.resolver import MediaRefResolver
    resolver = MediaRefResolver()
    async def _fail(*args):
        raise ValueError("fetch fail")
    resolver._retrieve_bytes = _fail
    res = await resolver._transport_inline_bytes(MediaRef(kind="url", url="https://x"), "openai")
    assert res is None


@pytest.mark.asyncio
async def test_get_cached_signed_url_expired_or_invalid():
    from agentflow.storage.media.resolver import MediaRefResolver
    resolver = MediaRefResolver(cache_backend=_Cache())
    resolver.cache_backend.values[("media:signed-url", "k")] = "not-a-dict"
    res1 = await resolver._get_cached_signed_url("k")
    assert res1 is None
    
    resolver.cache_backend.values[("media:signed-url", "k")] = {"url": "https://x"}
    res2 = await resolver._get_cached_signed_url("k")
    assert res2 is None
    
    resolver.cache_backend.values[("media:signed-url", "k")] = {"url": "https://x", "expires_at": 100}
    resolver.direct_url_refresh_buffer_seconds = 60
    res3 = await resolver._get_cached_signed_url("k")
    assert res3 is None


def test_source_kind_helper_variations():
    from agentflow.storage.media.resolver import _source_kind
    assert _source_kind(MediaRef(kind="url", url="agentflow://media/k")) == "internal_ref"
    assert _source_kind(MediaRef(kind="url", url="https://x")) == "url"
    assert _source_kind(MediaRef(kind="data")) == "data"
    assert _source_kind(MediaRef(kind="file_id")) == "file_id"
    assert _source_kind(MediaRef.model_construct(kind="other")) == "other"


