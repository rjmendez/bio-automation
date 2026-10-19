from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import rawpy
from PIL import Image

from .exif import extract_preview_bytes


def _resize_and_save(img_bytes: bytes, dest_path: Path, max_dim: int | None, quality: int) -> Path:
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return _resize_and_save_image(img, dest_path, max_dim, quality)


def _resize_and_save_image(img: Image.Image, dest_path: Path, max_dim: int | None, quality: int) -> Path:
    if max_dim is not None:
        img = img.copy()
        img.thumbnail((max_dim, max_dim), Image.LANCZOS)
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(dest_path), "JPEG", quality=quality)
    return dest_path


def _decode_raw_to_rgb8(raw_path: Path) -> Image.Image:
    """Full raw decode via rawpy at the sensor's native resolution --
    sRGB (output_color, required by Adobe Stock's technical requirements),
    no_auto_bright (a plain, predictable decode, same reasoning as
    preclean.decode_raw_to_bgr16), 16-bit intermediate rounded to 8-bit for
    JPEG. Used for every export meant for an outside service (iNat, Adobe
    Stock) instead of the camera's own embedded JPEG preview
    (exif.extract_preview_bytes, still used by render_review_jpeg, where
    speed for a quick keep/reject call matters more than squeezing out the
    last bit of real sensor detail) -- the embedded preview is whatever the
    camera's own in-body JPEG engine produced, capped by its own tone curve,
    sharpening and compression choices, not a controlled decode.
    """
    with rawpy.imread(str(raw_path)) as raw:
        rgb16 = raw.postprocess(output_bps=16, no_auto_bright=True, use_camera_wb=True,
                                 output_color=rawpy.ColorSpace.sRGB)
    rgb8 = (rgb16.astype(np.float32) / 65535.0 * 255.0).round().clip(0, 255).astype(np.uint8)
    return Image.fromarray(rgb8, mode="RGB")


def render_review_jpeg(raw_path: Path, dest_path: Path, max_dim: int = 2048, quality: int = 85) -> Path:
    """Resize/re-encode the raw's embedded preview (exif.extract_preview_bytes)
    into a review JPEG at dest_path, to drive the keep/reject decision.
    """
    return _resize_and_save(extract_preview_bytes(raw_path), dest_path, max_dim, quality)


def export_inat_jpeg(raw_path: Path, dest_path: Path, max_dim: int = 2048, quality: int = 95) -> Path:
    """Full raw decode (see _decode_raw_to_rgb8) downsized to iNat's own
    server-side ceiling and re-encoded for upload.

    max_dim=2048 matches iNat's own real behavior, not an arbitrary choice:
    iNat resizes every uploaded photo so its longest edge is at most 2048px
    regardless of what's uploaded (forum-confirmed 2026-10-19 -- iNat has no
    public API field for this limit), so uploading larger buys nothing.
    quality=95 (was 90): iNat's real cap is file size, not quality -- 20MB
    per photo (also forum-confirmed) -- and a 2048px sRGB JPEG at quality=95
    lands at a few MB, nowhere near that ceiling, so there is headroom to
    raise quality with no practical cost.
    """
    return _resize_and_save_image(_decode_raw_to_rgb8(raw_path), dest_path, max_dim, quality)


def export_stock_jpeg(raw_path: Path, dest_path: Path, max_dim: int | None = None, quality: int = 95) -> Path:
    """Full raw decode (see _decode_raw_to_rgb8) at full sensor resolution
    for stock use -- never downsized by default (max_dim=None).

    Adobe Stock's current technical requirements (Adobe helpx, checked
    2026-10-19) call for 4-100MP sRGB JPEGs up to 45MB; the Z7II's ~45MP
    sensor is within that range at full resolution, so there is no reason
    to shrink it. This replaces the v1 embedded-preview source (resolution/
    color capped by whatever the camera's own JPEG engine produced) with a
    real demosaic of the sensor data.
    """
    return _resize_and_save_image(_decode_raw_to_rgb8(raw_path), dest_path, max_dim, quality)
