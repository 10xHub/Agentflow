"""Image validation and processing utilities.

``MediaProcessor`` handles only **images**: validate MIME type, enforce
file-size limits, and optionally resize.  It does NOT perform document
text extraction — that is the API layer's responsibility (see
``agentflow_cli.media.pipeline``).
"""

from __future__ import annotations

import base64
import io
import logging
from typing import Any

from agentflow.core.state.message_block import ImageBlock

from .config import MultimodalConfig


logger = logging.getLogger("agentflow.media.processor")


class MediaProcessor:
    """Validate and optionally resize images before they enter the pipeline.

    Pillow is an *optional* dependency (``pip install 10xscale-agentflow[images]``).
    Without Pillow, validation still works but resizing is skipped.
    """

    def __init__(self, config: MultimodalConfig | None = None):
        self.config = config or MultimodalConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate_image(self, block: ImageBlock) -> None:
        """Validate an ImageBlock against the configured limits.

        Raises:
            ValueError: If MIME type is unsupported or size exceeds limit.
        """
        media = block.media
        mime = media.mime_type or "image/png"

        if mime not in self.config.supported_image_types:
            raise ValueError(
                f"Unsupported image type '{mime}'. "
                f"Allowed: {', '.join(sorted(self.config.supported_image_types))}"
            )

        if media.size_bytes is not None:
            max_bytes = int(self.config.max_image_size_mb * 1_048_576)
            if media.size_bytes > max_bytes:
                raise ValueError(
                    f"Image too large ({media.size_bytes} bytes). "
                    f"Max allowed: {max_bytes} bytes ({self.config.max_image_size_mb} MB)"
                )

        # Check decoded size for inline base64
        if media.kind == "data" and media.data_base64:
            approx_bytes = len(media.data_base64) * 3 // 4
            max_bytes = int(self.config.max_image_size_mb * 1_048_576)
            if approx_bytes > max_bytes:
                raise ValueError(
                    f"Image data too large (~{approx_bytes} bytes). "
                    f"Max allowed: {max_bytes} bytes ({self.config.max_image_size_mb} MB)"
                )

    def resize_image(self, block: ImageBlock) -> ImageBlock:
        """Resize an inline base64 image if it exceeds ``max_image_dimension``.

        Returns the original block unchanged if:
        - Pillow is not installed
        - The image is a URL/file_id reference (cannot resize remotely)
        - Both dimensions are within limits

        Returns:
            A new ImageBlock with resized data, or the original.
        """
        if block.media.kind != "data" or not block.media.data_base64:
            return block  # Can only resize inline data

        try:
            from PIL import Image
        except ImportError:
            logger.debug("Pillow not installed; skipping image resize")
            return block

        raw = base64.b64decode(block.media.data_base64)
        img = Image.open(io.BytesIO(raw))
        max_dim = self.config.max_image_dimension

        if img.width <= max_dim and img.height <= max_dim:
            return block  # Already within limits

        # Resize maintaining aspect ratio
        img.thumbnail((max_dim, max_dim))

        buf = io.BytesIO()
        fmt = _pil_format(block.media.mime_type)
        img.save(buf, format=fmt)
        new_b64 = base64.b64encode(buf.getvalue()).decode()

        new_media = block.media.model_copy(
            update={
                "data_base64": new_b64,
                "width": img.width,
                "height": img.height,
                "size_bytes": len(buf.getvalue()),
            }
        )
        return ImageBlock(media=new_media, alt_text=block.alt_text, bbox=block.bbox)

    def process(self, block: ImageBlock) -> ImageBlock:
        """Validate and optionally resize an image block.

        Convenience method that calls ``validate_image`` then ``resize_image``.
        """
        self.validate_image(block)
        return self.resize_image(block)

    def fix_orientation(self, block: ImageBlock) -> ImageBlock:
        """Apply EXIF orientation to an inline image so it displays correctly.

        JPEG cameras store rotation in EXIF metadata.  This method reads the
        EXIF ``Orientation`` tag and transposes the pixel data accordingly,
        then strips the tag so viewers don't double-rotate.

        Returns the original block if Pillow is unavailable, the image
        is not inline data, or there is no EXIF orientation to fix.
        """
        if not self._needs_orientation_fix(block):
            return block

        try:
            from PIL import ImageOps
        except ImportError:
            return block

        img = self._load_inline_image(block)
        if img is None:
            return block

        if not self._has_exif_rotation(img):
            return block

        corrected = ImageOps.exif_transpose(img)
        buf = io.BytesIO()
        fmt = _pil_format(block.media.mime_type)
        corrected.save(buf, format=fmt)
        new_b64 = base64.b64encode(buf.getvalue()).decode()

        new_media = block.media.model_copy(
            update={
                "data_base64": new_b64,
                "width": corrected.width,
                "height": corrected.height,
                "size_bytes": len(buf.getvalue()),
            }
        )
        return ImageBlock(media=new_media, alt_text=block.alt_text, bbox=block.bbox)

    def _needs_orientation_fix(self, block: ImageBlock) -> bool:
        media = block.media
        if media.kind != "data" or not media.data_base64:
            return False
        mime = (media.mime_type or "").lower()
        return mime in {"image/jpeg", "image/jpg", "image/tiff"}

    def _load_inline_image(self, block: ImageBlock) -> Any | None:
        try:
            from PIL import Image
        except ImportError:
            return None

        raw = base64.b64decode(block.media.data_base64)
        try:
            return Image.open(io.BytesIO(raw))
        except Exception as exc:
            logger.debug(
                "Could not open inline image for orientation fix (%s: %s); skipping",
                type(exc).__name__,
                exc,
            )
            return None

    @staticmethod
    def _has_exif_rotation(img: Any) -> bool:
        exif = img.getexif()
        orientation = exif.get(0x0112)  # 0x0112 = Orientation tag
        return orientation not in (None, 1)

    def generate_thumbnail(self, block: ImageBlock, max_dim: int = 256) -> ImageBlock:
        """Create a small thumbnail version of an inline image.

        Args:
            block: Source image block with inline base64 data.
            max_dim: Maximum width or height for the thumbnail (default 256px).

        Returns:
            A new ImageBlock with thumbnail data, or the original if
            Pillow is missing or the image is not inline.
        """
        if block.media.kind != "data" or not block.media.data_base64:
            return block

        try:
            from PIL import Image
        except ImportError:
            return block

        raw = base64.b64decode(block.media.data_base64)
        img = Image.open(io.BytesIO(raw))

        if img.width <= max_dim and img.height <= max_dim:
            return block

        img.thumbnail((max_dim, max_dim))
        buf = io.BytesIO()
        # Thumbnails are always JPEG for compactness
        if img.mode in ("RGBA", "P", "LA"):
            img = img.convert("RGB")
        img.save(buf, format="JPEG", quality=80)
        new_b64 = base64.b64encode(buf.getvalue()).decode()

        new_media = block.media.model_copy(
            update={
                "data_base64": new_b64,
                "mime_type": "image/jpeg",
                "width": img.width,
                "height": img.height,
                "size_bytes": len(buf.getvalue()),
            }
        )
        return ImageBlock(media=new_media, alt_text=block.alt_text, bbox=block.bbox)

    def optimize_image(
        self,
        block: ImageBlock,
        target_format: str = "JPEG",
        quality: int = 85,
    ) -> ImageBlock:
        """Convert and optimize an inline image for smaller size.

        Converts the image to ``target_format`` (default JPEG) at the given
        ``quality`` level.  Useful for normalising uploads to a single format.

        Args:
            block: Source image block with inline base64 data.
            target_format: PIL format name (``"JPEG"``, ``"PNG"``, ``"WEBP"``).
            quality: Compression quality 1-100 (only affects lossy formats).

        Returns:
            A new ImageBlock, or the original if Pillow is missing.
        """
        if block.media.kind != "data" or not block.media.data_base64:
            return block

        try:
            from PIL import Image
        except ImportError:
            return block

        raw = base64.b64decode(block.media.data_base64)
        img = Image.open(io.BytesIO(raw))

        if target_format.upper() == "JPEG" and img.mode in ("RGBA", "P", "LA"):
            img = img.convert("RGB")

        buf = io.BytesIO()
        save_kwargs: dict[str, Any] = {"format": target_format.upper()}
        if target_format.upper() in ("JPEG", "WEBP"):
            save_kwargs["quality"] = quality
            save_kwargs["optimize"] = True
        img.save(buf, **save_kwargs)

        mime_map = {
            "JPEG": "image/jpeg",
            "PNG": "image/png",
            "WEBP": "image/webp",
            "GIF": "image/gif",
        }
        new_mime = mime_map.get(target_format.upper(), f"image/{target_format.lower()}")
        new_b64 = base64.b64encode(buf.getvalue()).decode()

        new_media = block.media.model_copy(
            update={
                "data_base64": new_b64,
                "mime_type": new_mime,
                "width": img.width,
                "height": img.height,
                "size_bytes": len(buf.getvalue()),
            }
        )
        return ImageBlock(media=new_media, alt_text=block.alt_text, bbox=block.bbox)

    def full_process(self, block: ImageBlock) -> ImageBlock:
        """Full processing pipeline: validate → fix orientation → resize.

        Use this instead of ``process()`` when incoming images may have
        EXIF rotation metadata (common with phone camera uploads).
        """
        self.validate_image(block)
        block = self.fix_orientation(block)
        return self.resize_image(block)


def _pil_format(mime_type: str | None) -> str:
    """Map MIME type to Pillow save format string."""
    mapping: dict[str, str] = {
        "image/jpeg": "JPEG",
        "image/jpg": "JPEG",
        "image/png": "PNG",
        "image/webp": "WEBP",
        "image/gif": "GIF",
    }
    return mapping.get(mime_type or "", "PNG")
