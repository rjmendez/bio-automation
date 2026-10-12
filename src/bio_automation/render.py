from __future__ import annotations

from pathlib import Path


def render_review_jpeg(raw_path: Path, dest_path: Path, max_dim: int = 2048, quality: int = 85) -> Path:
    """Resize/re-encode the raw's embedded preview (see
    exif.extract_preview_bytes) into a review JPEG at dest_path. Blocked on
    an image-rendering dependency (Pillow) being added to pyproject.toml --
    see docs/folder-metadata-schema.md Implementation notes.
    """
    raise NotImplementedError("render_review_jpeg needs Pillow added to pyproject.toml")


def export_inat_jpeg(raw_path: Path, dest_path: Path, max_dim: int = 2048, quality: int = 90) -> Path:
    """Resize/re-encode the raw's embedded preview into an iNat-upload-sized
    JPEG at dest_path. Same dependency as render_review_jpeg.
    """
    raise NotImplementedError("export_inat_jpeg needs Pillow added to pyproject.toml")


def export_stock_jpeg(raw_path: Path, dest_path: Path, max_dim: int | None = None, quality: int = 95) -> Path:
    """Re-encode the raw's embedded preview at full size for stock use.

    v1 caveat: sourced from the camera's embedded preview, not a true raw
    development -- resolution/color is capped by whatever the camera
    embedded. Revisit with rawpy/LibRaw if stock buyers need more. Same
    dependency as render_review_jpeg.
    """
    raise NotImplementedError("export_stock_jpeg needs Pillow added to pyproject.toml")
