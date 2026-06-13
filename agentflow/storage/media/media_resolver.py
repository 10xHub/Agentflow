"""Unified media resolver with capability-based fallback chain.

This module provides the generic ``MediaResolver`` entrypoint that selects
the correct transport mode for a given provider/model combination using
the capability matrix defined in ``capabilities.py``.

Instead of hard-coding provider logic, the resolver consults the capability
matrix and tries transports in the declared preference order until one
succeeds.
"""

from __future__ import annotations

import base64
import logging
from typing import Any

from agentflow.core.exceptions.media_exceptions import UnsupportedMediaInputError
from agentflow.core.state.message_block import MediaRef
from agentflow.storage.media.capabilities import (
    MediaTransportMode,
    get_capabilities,
)

from .storage.base import BaseMediaStore


logger = logging.getLogger("agentflow.media.media_resolver")

_AGENTFLOW_SCHEME = "agentflow://media/"


class MediaResolver:
    """Unified media resolver that selects transport by provider/model capability.

    This is the new entrypoint for resolving media references. It consults
    the capability matrix to determine which transport modes are available
    for a given provider/model, then tries them in order until one succeeds.

    Args:
        media_store: Optional media store for internal ``agentflow://media/`` refs.
        cache_backend: Optional cache backend for signed URL caching.
    """

    def __init__(
        self,
        media_store: BaseMediaStore | None = None,
        cache_backend: Any | None = None,
    ) -> None:
        self.media_store = media_store
        self.cache_backend = cache_backend

    async def resolve(
        self,
        ref: MediaRef,
        provider: str,
        model: str,
        media_type: str = "image",
    ) -> Any:
        """Resolve a media reference using capability-based fallback.

        Args:
            ref: The media reference to resolve.
            provider: Target provider (e.g. "openai", "google").
            model: Target model name (e.g. "gpt-4o", "gemini-1.5-pro").
            media_type: Media type (default: "image").

        Returns:
            Provider-specific content part (dict for OpenAI, Part for Google).

        Raises:
            UnsupportedMediaInputError: If the model cannot accept this media type.
        """
        caps = get_capabilities(provider, model)

        if not caps.supports_media_type(media_type):
            raise UnsupportedMediaInputError(
                provider=provider,
                model=model,
                media_type=media_type,
                source_kind=_source_kind(ref),
                transports_attempted=[],
            )

        transport_order = caps.get_transport_order(media_type)
        transports_attempted: list[MediaTransportMode] = []
        last_error: Exception | None = None

        for transport in transport_order:
            transports_attempted.append(transport)
            try:
                result = await self._try_transport(
                    ref,
                    transport,
                    provider,
                    model,
                    caps,
                )
                if result is not None:
                    return result
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Media transport %s failed for %s/%s (%s: %s); trying next fallback",
                    transport.value,
                    provider,
                    model,
                    type(exc).__name__,
                    exc,
                    exc_info=True,
                )
                continue

        raise UnsupportedMediaInputError(
            provider=provider,
            model=model,
            media_type=media_type,
            source_kind=_source_kind(ref),
            transports_attempted=transports_attempted,
            message=(
                f"Model '{model}' from provider '{provider}' could not resolve "
                f"{media_type} input (source: {_source_kind(ref)}). "
                f"All transports failed: "
                f"{', '.join(t.value for t in transports_attempted)}."
            ),
        ) from last_error

    async def _try_transport(
        self,
        ref: MediaRef,
        transport: MediaTransportMode,
        provider: str,
        model: str,
        caps: Any,
    ) -> Any | None:
        """Attempt to resolve using a specific transport mode.

        Returns the resolved part, or None if this transport is not applicable.
        """
        if transport == MediaTransportMode.remote_url:
            return await self._transport_remote_url(ref, caps)

        if transport == MediaTransportMode.inline_bytes:
            return await self._transport_inline_bytes(ref, provider)

        if transport == MediaTransportMode.provider_file:
            return await self._transport_provider_file(ref, provider, model)

        if transport == MediaTransportMode.unsupported:
            return None

        return None

    async def _transport_remote_url(
        self,
        ref: MediaRef,
        caps: Any,
    ) -> dict[str, Any] | None:
        """Resolve to a remote URL (signed or direct)."""
        if ref.kind == "url" and ref.url:
            if ref.url.startswith(_AGENTFLOW_SCHEME):
                url = await self._get_direct_url(ref)
                if url:
                    return _openai_image_url(url)
                return None

            if caps.accepts_external_urls:
                return _openai_image_url(ref.url)

        return None

    async def _transport_inline_bytes(
        self,
        ref: MediaRef,
        provider: str,
    ) -> dict[str, Any] | Any | None:
        """Resolve to inline bytes/data URI."""
        if ref.kind == "data" and ref.data_base64:
            mime = ref.mime_type or "application/octet-stream"
            if provider == "openai":
                return _openai_image_url(f"data:{mime};base64,{ref.data_base64}")
            if provider == "google":
                from google.genai import types

                data = base64.b64decode(ref.data_base64)
                return types.Part.from_bytes(data=data, mime_type=mime)

        if ref.kind == "url" and ref.url:
            try:
                data, mime = await self._retrieve_bytes(ref)
                if provider == "openai":
                    b64 = base64.b64encode(data).decode()
                    return _openai_image_url(f"data:{mime};base64,{b64}")
                if provider == "google":
                    from google.genai import types

                    return types.Part.from_bytes(data=data, mime_type=mime)
            except Exception as exc:
                logger.warning(
                    "inline_bytes transport failed to fetch %s for provider %s (%s: %s)",
                    ref.url,
                    provider,
                    type(exc).__name__,
                    exc,
                    exc_info=True,
                )
                return None

        return None

    async def _transport_provider_file(  # noqa: PLR0911
        self,
        ref: MediaRef,
        provider: str,
        model: str,
    ) -> Any | None:
        """Resolve via provider-native file upload (Google File API)."""
        if provider != "google":
            return None

        try:
            from google.genai import types

            if ref.kind == "url" and ref.url:
                if ref.url.startswith(_AGENTFLOW_SCHEME):
                    data, mime = await self._retrieve_bytes(ref)
                    from .provider_media import upload_to_google_file_api

                    return await upload_to_google_file_api(data, mime)

                if ref.url.startswith("gs://"):
                    mime = ref.mime_type or "image/jpeg"
                    return types.Part.from_uri(file_uri=ref.url, mime_type=mime)

                data, mime = await self._retrieve_bytes(ref)
                from .provider_media import upload_to_google_file_api

                return await upload_to_google_file_api(data, mime)

            if ref.kind == "data" and ref.data_base64:
                data = base64.b64decode(ref.data_base64)
                mime = ref.mime_type or "image/jpeg"
                from .provider_media import upload_to_google_file_api

                return await upload_to_google_file_api(data, mime)

        except Exception as exc:
            logger.warning(
                "provider_file transport failed (Google File API) for ref kind=%s (%s: %s)",
                ref.kind,
                type(exc).__name__,
                exc,
                exc_info=True,
            )
            return None

        return None

    async def _retrieve_bytes(self, ref: MediaRef) -> tuple[bytes, str]:
        """Retrieve bytes for a media reference."""
        if ref.kind == "url" and ref.url:
            if ref.url.startswith(_AGENTFLOW_SCHEME):
                return await self._retrieve(ref.url)
            return await self._fetch_external_url(ref.url)

        if ref.kind == "data" and ref.data_base64:
            data = base64.b64decode(ref.data_base64)
            return data, ref.mime_type or "application/octet-stream"

        raise ValueError(f"Cannot retrieve bytes for ref kind: {ref.kind}")

    async def _fetch_external_url(self, url: str) -> tuple[bytes, str]:
        """Fetch an external URL and return (bytes, mime_type)."""
        import aiohttp

        async with aiohttp.ClientSession() as session, session.get(url) as resp:
            resp.raise_for_status()
            data = await resp.read()
            mime = resp.headers.get("Content-Type", "application/octet-stream")
            return data, mime

    async def _retrieve(self, agentflow_url: str) -> tuple[bytes, str]:
        """Retrieve bytes from the media store for an internal URL."""
        if self.media_store is None:
            raise RuntimeError(
                f"Cannot resolve internal media URL {agentflow_url!r} — "
                "no MediaStore configured."
            )
        key = agentflow_url.removeprefix(_AGENTFLOW_SCHEME)
        return await self.media_store.retrieve(key)

    async def _get_direct_url(self, ref: MediaRef) -> str | None:
        """Return a direct/signed URL for internal media."""
        if self.media_store is None or not ref.url:
            return None

        key = ref.url.removeprefix(_AGENTFLOW_SCHEME)
        mime_type = ref.mime_type or "application/octet-stream"
        cache_key = f"{key}:{mime_type}:3600"

        if self.cache_backend is not None:
            payload = await self.cache_backend.aget_cache_value("media:signed-url", cache_key)
            if isinstance(payload, dict):
                url = payload.get("url")
                expires_at = payload.get("expires_at")
                if isinstance(url, str) and isinstance(expires_at, int | float):
                    import time

                    if expires_at > time.time() + 60:
                        return url

        direct_url = await self.media_store.get_direct_url(key, mime_type=ref.mime_type)
        if direct_url and self.cache_backend is not None:
            import time

            expires_at = int(time.time() + 3600)
            await self.cache_backend.aput_cache_value(
                "media:signed-url",
                cache_key,
                {"url": direct_url, "expires_at": expires_at},
                ttl_seconds=3600,
            )
        return direct_url


def _source_kind(ref: MediaRef) -> str:
    """Determine the source kind for error reporting."""
    if ref.kind == "url":
        if ref.url and ref.url.startswith(_AGENTFLOW_SCHEME):
            return "internal_ref"
        return "url"
    if ref.kind == "data":
        return "data"
    if ref.kind == "file_id":
        return "file_id"
    return ref.kind or "unknown"


def _openai_image_url(url: str) -> dict[str, Any]:
    return {"type": "image_url", "image_url": {"url": url}}
