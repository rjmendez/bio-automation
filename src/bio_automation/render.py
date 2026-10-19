from __future__ import annotations

import io
from pathlib import Path

from PIL import Image

from .exif import extract_preview_bytes


def _resize_and_save(img_bytes: bytes, dest_path: Path, max_dim: int | None, quality: int) -> Path:
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    if max_dim is not None:
        img.thumbnail((max_dim, max_dim), Image.LANCZOS)
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(dest_path), "JPEG", quality=quality)
    return dest_path


def render_review_jpeg(raw_path: Path, dest_path: Path, max_dim: int = 2048, quality: int = 85) -> Path:
    """Resize/re-encode the raw's embedded preview (exif.extract_preview_bytes)
    into a review JPEG at dest_path, to drive the keep/reject decision.
    """
    return _resize_and_save(extract_preview_bytes(raw_path), dest_path, max_dim, quality)


def export_inat_jpeg(raw_path: Path, dest_path: Path, max_dim: int = 2048, quality: int = 90) -> Path:
    """Resize/re-encode the raw's embedded preview into an iNat-upload-sized
    JPEG at dest_path.
    """
    return _resize_and_save(extract_preview_bytes(raw_path), dest_path, max_dim, quality)


def export_stock_jpeg(raw_path: Path, dest_path: Path, max_dim: int | None = None, quality: int = 95) -> Path:
    """Re-encode the raw's embedded preview at full size for stock use.

    v1 caveat: sourced from the camera's embedded preview, not a true raw
    development -- resolution/color is capped by whatever the camera
    embedded. Revisit with rawpy/LibRaw if stock buyers need more.
    """
    return _resize_and_save(extract_preview_bytes(raw_path), dest_path, max_dim, quality)
