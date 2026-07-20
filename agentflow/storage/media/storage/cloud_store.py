"""Cloud-backed media store — S3 and GCS via ``cloud-storage-manager``.

Uses the `cloud-storage-manager <https://github.com/10xHub/cloud-storage-manager>`_
package to provide unified S3 / GCS blob storage behind the standard
:class:`BaseMediaStore` interface.

Install the optional dependency::

    pip install 10xscale-agentflow[cloud-storage]

Then create a store::

    from cloud_storage_manager import (
        CloudStorageFactory,
        StorageProvider,
        StorageConfig,
        AwsConfig,
    )
    from agentflow.media.storage import CloudMediaStore

    config = StorageConfig(
        aws=AwsConfig(
            bucket_name="my-bucket",
            access_key_id="...",
            secret_access_key="...",
        )
    )
    storage = CloudStorageFactory.get_storage(StorageProvider.AWS, config)
    store = CloudMediaStore(storage)
"""

from __future__ import annotations

import asyncio
import json
import logging
import mimetypes
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

from .base import BaseMediaStore


logger = logging.getLogger("agentflow.storage.media.cloud")

# Extension map for cases where mimetypes doesn't know the type
_FALLBACK_EXT: dict[str, str] = {
    "application/pdf": ".pdf",
    "audio/wav": ".wav",
    "audio/mp3": ".mp3",
    "image/webp": ".webp",
}
_ALLOWED_DOWNLOAD_SCHEMES = {"http", "https"}


def _mime_to_ext(mime_type: str) -> str:
    """Convert MIME type to file extension."""
    ext = mimetypes.guess_extension(mime_type)
    if ext:
        return ext
    return _FALLBACK_EXT.get(mime_type, ".bin")


class CloudMediaStore(BaseMediaStore):
    """S3 / GCS media store backed by ``cloud-storage-manager``.

    Uses :class:`~cloud_storage_manager.base_storage.BaseCloudStorage` for
    the actual cloud operations (upload, delete, get_public_url).

    Layout in the bucket::

        {prefix}/{key[:2]}/{key[2:4]}/{key}{ext}       ← binary blob
        {prefix}/{key[:2]}/{key[2:4]}/{key}.meta.json   ← metadata sidecar

    The sidecar JSON contains ``mime_type``, ``size_bytes``, and any extra
    metadata passed to :meth:`store`.

    Args:
        storage: A ``BaseCloudStorage`` instance (from
            :func:`cloud_storage_manager.CloudStorageFactory.get_storage`).
        prefix: Top-level "directory" in the bucket.  Defaults to
            ``"agentflow-media"``.
    """

    def __init__(
        self,
        storage: Any,
        prefix: str = "agentflow-media",
    ) -> None:
        self._storage = storage
        self._prefix = prefix

    # ------------------------------------------------------------------
    # BaseMediaStore interface
    # ------------------------------------------------------------------

    async def store(
        self,
        data: bytes,
        mime_type: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        key = uuid4().hex
        ext = _mime_to_ext(mime_type)
        blob_path = self._cloud_path(key, ext)
        meta_path = self._meta_cloud_path(key)

        # Upload binary blob via temp file
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name

        try:
            await self._storage.upload(tmp_path, blob_path)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        # Upload sidecar metadata
        meta = {
            "mime_type": mime_type,
            "size_bytes": len(data),
            "ext": ext,
            **(metadata or {}),
        }
        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, mode="w", encoding="utf-8"
        ) as tmp_meta:
            json.dump(meta, tmp_meta)
            tmp_meta_path = tmp_meta.name

        try:
            await self._storage.upload(tmp_meta_path, meta_path)
        finally:
            Path(tmp_meta_path).unlink(missing_ok=True)

        logger.debug("Stored %d bytes as %s at %s", len(data), key, blob_path)
        return key

    async def retrieve(self, storage_key: str) -> tuple[bytes, str]:
        meta = await self._download_meta(storage_key)
        if meta is None:
            raise KeyError(f"Media not found: {storage_key}")

        mime_type = meta["mime_type"]
        ext = meta.get("ext", _mime_to_ext(mime_type))
        blob_path = self._cloud_path(storage_key, ext)

        url = await self._storage.get_public_url(blob_path, expiration=300)
        data = await self._download_from_url(url)

        return data, mime_type

    async def delete(self, storage_key: str) -> bool:
        meta = await self._download_meta(storage_key)
        if meta is None:
            return False

        mime_type = meta["mime_type"]
        ext = meta.get("ext", _mime_to_ext(mime_type))
        blob_path = self._cloud_path(storage_key, ext)
        meta_path = self._meta_cloud_path(storage_key)

        deleted = False
        try:
            result = await self._storage.delete(blob_path)
            deleted = deleted or result
        except Exception:
            logger.warning("Failed to delete blob %s", blob_path)

        try:
            result = await self._storage.delete(meta_path)
            deleted = deleted or result
        except Exception:
            logger.warning("Failed to delete meta %s", meta_path)

        return deleted

    async def exists(self, storage_key: str) -> bool:
        meta = await self._download_meta(storage_key)
        return meta is not None

    async def get_metadata(self, storage_key: str) -> dict[str, Any] | None:
        return await self._download_meta(storage_key)

    # ------------------------------------------------------------------
    # Bonus: direct URL access
    # ------------------------------------------------------------------

    async def get_public_url(
        self,
        storage_key: str,
        expiration: int = 3600,
    ) -> str:
        """Generate a signed URL for direct browser/client access.

        Args:
            storage_key: The key returned by :meth:`store`.
            expiration: Signed URL validity in seconds (default 1 hour).

        Returns:
            A pre-signed URL (S3) or signed URL (GCS).

        Raises:
            KeyError: If the storage key does not exist.
        """
        meta = await self._download_meta(storage_key)
        if meta is None:
            raise KeyError(f"Media not found: {storage_key}")

        ext = meta.get("ext", _mime_to_ext(meta["mime_type"]))
        blob_path = self._cloud_path(storage_key, ext)
        return await self._storage.get_public_url(blob_path, expiration=expiration)

    async def get_direct_url(
        self,
        storage_key: str,
        mime_type: str | None = None,
        expiration: int = 3600,
    ) -> str | None:
        """Return a signed blob URL without downloading the object bytes.

        When ``mime_type`` is known from the existing ``MediaRef``, we can
        derive the object path directly and skip the metadata sidecar lookup.
        That keeps model resolution on a signed-URL path instead of a
        download-then-reupload/base64 path.
        """
        if mime_type:
            ext = _mime_to_ext(mime_type)
            blob_path = self._cloud_path(storage_key, ext)
            return await self._storage.get_public_url(blob_path, expiration=expiration)

        return await self.get_public_url(storage_key, expiration=expiration)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cloud_path(self, key: str, ext: str) -> str:
        return f"{self._prefix}/{key[:2]}/{key[2:4]}/{key}{ext}"

    def _meta_cloud_path(self, key: str) -> str:
        return f"{self._prefix}/{key[:2]}/{key[2:4]}/{key}.meta.json"

    async def _download_meta(self, storage_key: str) -> dict[str, Any] | None:
        """Download and parse the sidecar metadata JSON."""
        meta_path = self._meta_cloud_path(storage_key)
        try:
            url = await self._storage.get_public_url(meta_path, expiration=60)
            raw = await self._download_from_url(url)
            return json.loads(raw)
        except Exception as exc:
            logger.debug(
                "Could not download/parse sidecar metadata at %s (%s: %s)",
                meta_path,
                type(exc).__name__,
                exc,
            )
            return None

    @staticmethod
    async def _download_from_url(url: str) -> bytes:
        """Download bytes from a signed URL.

        Prefers ``httpx`` (async). The stdlib fallback is synchronous, so it is
        run in a worker thread: calling ``urllib.request.urlopen`` directly inside
        an ``async def`` blocked the whole event loop for the duration of the
        download -- stalling every other in-flight run in the process, not just
        this one.
        """
        _validate_download_url(url)
        try:
            import httpx
        except ImportError:
            return await asyncio.to_thread(CloudMediaStore._download_from_url_blocking, url)

        async with httpx.AsyncClient(follow_redirects=True) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            return resp.content

    @staticmethod
    def _download_from_url_blocking(url: str) -> bytes:
        """Synchronous stdlib download. Must only be called off the event loop."""
        import urllib.request

        _validate_download_url(url)
        with urllib.request.urlopen(url) as resp:  # noqa: S310  # nosec B310
            return resp.read()


def _validate_download_url(url: str) -> None:
    """Allow downloads only from explicit HTTP(S) signed URLs."""
    parsed = urlparse(url)
    if parsed.scheme not in _ALLOWED_DOWNLOAD_SCHEMES or not parsed.netloc:
        raise ValueError(f"Unsupported download URL scheme: {parsed.scheme or 'missing'}")
