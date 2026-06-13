"""Resolve ``MediaRef`` objects to provider-specific formats at LLM call time.

The resolver translates lightweight ``MediaRef`` references (which is all
that lives in messages/state) back into actual binary data or URLs that
the OpenAI / Google APIs understand.

``MediaRefResolver`` now acts as a capability-aware wrapper around the
unified ``MediaResolver``.  The legacy ``resolve_for_openai()`` and
``resolve_for_google()`` methods remain for backward compatibility but
now route through the capability-based fallback chain.
"""

from __future__ import annotations

import base64
import logging
import time
from typing import Any

from agentflow.core.exceptions.media_exceptions import UnsupportedMediaInputError
from agentflow.core.state.message_block import MediaRef
from agentflow.storage.media.capabilities import (
    MediaTransportMode,
    get_capabilities,
)

from .storage.base import BaseMediaStore


logger = logging.getLogger("agentflow.media.resolver")

_AGENTFLOW_SCHEME = "agentflow://media/"


class MediaRefResolver:
    """Resolve ``MediaRef`` objects to provider-specific content parts.

    Args:
        media_store: Optional media store for resolving internal
            ``agentflow://media/{key}`` references.  If ``None``,
            internal references will raise.
    """

    def __init__(
        self,
        media_store: BaseMediaStore | None = None,
        cache_backend: Any | None = None,
        direct_url_expiration_seconds: int = 3600,
        direct_url_refresh_buffer_seconds: int = 60,
    ) -> None:
        self.media_store = media_store
        self.cache_backend = cache_backend
        self.direct_url_expiration_seconds = direct_url_expiration_seconds
        self.direct_url_refresh_buffer_seconds = direct_url_refresh_buffer_seconds

    def with_cache(
        self,
        cache_backend: Any,
        expiration_seconds: int = 3600,
        refresh_buffer_seconds: int = 60,
    ) -> MediaRefResolver:
        """Attach a shared cache backend for signed URLs."""
        self.cache_backend = cache_backend
        self.direct_url_expiration_seconds = expiration_seconds
        self.direct_url_refresh_buffer_seconds = refresh_buffer_seconds
        return self

    # ------------------------------------------------------------------
    # OpenAI (capability-aware wrapper)
    # ------------------------------------------------------------------

    async def resolve_for_openai(
        self,
        ref: MediaRef,
        model: str | None = None,
    ) -> dict[str, Any]:
        """Convert a ``MediaRef`` to an OpenAI-style content part dict.

        If *model* is provided, the capability matrix is consulted and
        ``UnsupportedMediaInputError`` is raised for text-only models.
        Otherwise, the legacy ad-hoc resolution path is used.

        Returns:
            A dict like ``{"type": "image_url", "image_url": {"url": ...}}``.
        """
        if model is not None:
            return await self._resolve_with_capabilities(
                ref,
                provider="openai",
                model=model,
                media_type="image",
            )

        return await self._resolve_openai_legacy(ref)

    async def _resolve_openai_legacy(self, ref: MediaRef) -> dict[str, Any]:
        """Legacy OpenAI resolution (no capability check)."""
        if ref.kind == "url" and ref.url and ref.url.startswith(_AGENTFLOW_SCHEME):
            direct_url = await self._get_direct_url(ref)
            if direct_url:
                return _openai_image_url(direct_url)

            data, mime = await self._retrieve(ref.url)
            b64 = base64.b64encode(data).decode()
            return _openai_image_url(f"data:{mime};base64,{b64}")

        if ref.kind == "url" and ref.url:
            return _openai_image_url(ref.url)

        if ref.kind == "data" and ref.data_base64:
            mime = ref.mime_type or "application/octet-stream"
            return _openai_image_url(f"data:{mime};base64,{ref.data_base64}")

        if ref.kind == "file_id":
            url = ref.url or ref.file_id or ""
            return _openai_image_url(url)

        return _openai_image_url("")

    # ------------------------------------------------------------------
    # Google GenAI (capability-aware wrapper)
    # ------------------------------------------------------------------

    async def resolve_for_google(
        self,
        ref: MediaRef,
        model: str | None = None,
    ) -> Any:
        """Convert a ``MediaRef`` to a ``google.genai.types.Part``.

        If *model* is provided, the capability matrix is consulted and
        ``UnsupportedMediaInputError`` is raised for text-only models.
        Otherwise, the legacy ad-hoc resolution path is used.

        Requires ``google-genai`` to be installed.
        """
        if model is not None:
            return await self._resolve_with_capabilities(
                ref,
                provider="google",
                model=model,
                media_type="image",
            )

        return await self._resolve_google_legacy(ref)

    async def _resolve_google_legacy(self, ref: MediaRef) -> Any:
        """Legacy Google resolution (no capability check)."""
        from google.genai import types

        if ref.kind == "url" and ref.url and ref.url.startswith(_AGENTFLOW_SCHEME):
            direct_url = await self._get_direct_url(ref)
            if direct_url:
                mime = ref.mime_type or "application/octet-stream"
                return types.Part.from_uri(file_uri=direct_url, mime_type=mime)

            data, mime = await self._retrieve(ref.url)
            return types.Part.from_bytes(data=data, mime_type=mime)

        if ref.kind == "url" and ref.url:
            mime = ref.mime_type or "image/jpeg"
            return types.Part.from_uri(file_uri=ref.url, mime_type=mime)

        if ref.kind == "data" and ref.data_base64:
            data = base64.b64decode(ref.data_base64)
            mime = ref.mime_type or "application/octet-stream"
            return types.Part.from_bytes(data=data, mime_type=mime)

        if ref.kind == "file_id":
            mime = ref.mime_type or "application/octet-stream"
            return types.Part(
                file_data=types.FileData(
                    file_uri=ref.file_id or "",
                    mime_type=mime,
                )
            )

        return types.Part(text="[Unresolvable media reference]")

    # ------------------------------------------------------------------
    # Capability-based resolution (shared)
    # ------------------------------------------------------------------

    async def _resolve_with_capabilities(
        self,
        ref: MediaRef,
        provider: str,
        model: str,
        media_type: str,
    ) -> Any:
        """Resolve using the capability matrix and fallback chain."""
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
                    caps,
                )
                if result is not None:
                    return result
            except UnsupportedMediaInputError:
                raise
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
        ) from last_error

    async def _try_transport(
        self,
        ref: MediaRef,
        transport: MediaTransportMode,
        provider: str,
        caps: Any,
    ) -> Any | None:
        """Attempt to resolve using a specific transport mode."""
        if transport == MediaTransportMode.remote_url:
            return await self._transport_remote_url(ref, caps, provider)

        if transport == MediaTransportMode.provider_file:
            return await self._transport_provider_file(ref, provider)

        if transport == MediaTransportMode.inline_bytes:
            return await self._transport_inline_bytes(ref, provider)

        return None

    async def _transport_remote_url(
        self,
        ref: MediaRef,
        caps: Any,
        provider: str,
    ) -> Any | None:
        """Resolve to a remote URL (signed or direct)."""
        if ref.kind == "url" and ref.url:
            if ref.url.startswith(_AGENTFLOW_SCHEME):
                if caps.can_convert_internal_to_remote:
                    url = await self._get_direct_url(ref)
                    if url:
                        if provider == "openai":
                            return _openai_image_url(url)
                        if provider == "google":
                            from google.genai import types

                            mime = ref.mime_type or "application/octet-stream"
                            return types.Part.from_uri(file_uri=url, mime_type=mime)
                return None

            if caps.accepts_external_urls and provider == "openai":
                return _openai_image_url(ref.url)

        # file_id references are treated as remote URLs for OpenAI
        if ref.kind == "file_id" and ref.file_id and provider == "openai":
            url = ref.url or ref.file_id or ""
            return _openai_image_url(url)

        return None

    async def _transport_inline_bytes(
        self,
        ref: MediaRef,
        provider: str,
    ) -> Any | None:
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

    async def _transport_provider_file(
        self,
        ref: MediaRef,
        provider: str,
    ) -> Any | None:
        """Resolve via provider-native file upload (Google File API)."""
        if provider != "google":
            return None

        try:
            from google.genai import types

            if ref.kind == "url" and ref.url:
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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _retrieve(self, agentflow_url: str) -> tuple[bytes, str]:
        """Retrieve bytes from the media store for an internal URL."""
        if self.media_store is None:
            raise RuntimeError(
                f"Cannot resolve internal media URL {agentflow_url!r} — "
                "no MediaStore configured. Pass a media_store to the resolver."
            )
        key = agentflow_url.removeprefix(_AGENTFLOW_SCHEME)
        return await self.media_store.retrieve(key)

    async def _get_direct_url(self, ref: MediaRef) -> str | None:
        """Return a direct URL for internal media when the store supports it."""
        if self.media_store is None or not ref.url:
            return None

        key = ref.url.removeprefix(_AGENTFLOW_SCHEME)
        mime_type = ref.mime_type or "application/octet-stream"
        cache_key = f"{key}:{mime_type}:{self.direct_url_expiration_seconds}"
        cached_url = await self._get_cached_signed_url(cache_key)
        if cached_url:
            return cached_url

        direct_url_kwargs: dict[str, Any] = {"mime_type": ref.mime_type}
        if self.cache_backend is not None:
            direct_url_kwargs["expiration"] = self.direct_url_expiration_seconds

        direct_url = await self.media_store.get_direct_url(key, **direct_url_kwargs)
        if direct_url:
            await self._cache_signed_url(cache_key, direct_url)
        return direct_url

    async def _get_cached_signed_url(self, cache_key: str) -> str | None:
        if self.cache_backend is None:
            return None

        payload = await self.cache_backend.aget_cache_value("media:signed-url", cache_key)
        if not isinstance(payload, dict):
            return None

        url = payload.get("url")
        expires_at = payload.get("expires_at")
        if not isinstance(url, str) or not isinstance(expires_at, int | float):
            return None

        if expires_at <= time.time() + self.direct_url_refresh_buffer_seconds:
            return None
        return url

    async def _cache_signed_url(self, cache_key: str, url: str) -> None:
        if self.cache_backend is None:
            return

        expires_at = int(time.time() + self.direct_url_expiration_seconds)
        await self.cache_backend.aput_cache_value(
            "media:signed-url",
            cache_key,
            {
                "url": url,
                "expires_at": expires_at,
            },
            ttl_seconds=self.direct_url_expiration_seconds,
        )


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
