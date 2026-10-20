from __future__ import annotations

import io
import math
from pathlib import Path

import numpy as np
import rawpy
from PIL import Image, ImageCms

from .exif import extract_exif, extract_preview_bytes

INAT_MAX_BYTES = 20 * 1024 * 1024
STOCK_MAX_BYTES = 45 * 1024 * 1024
STOCK_MIN_PIXELS = 4_000_000
STOCK_MAX_PIXELS = 100_000_000

# EXIF tags (all in the Exif/IFD0 space PIL writes) copied from the source raw
# onto every export: capture time is what iNat observation matching keys on.
_TAG_MAKE, _TAG_MODEL = 0x010F, 0x0110
_TAG_DATETIME_ORIGINAL, _TAG_LENS_MODEL = 0x9003, 0xA434
_EXIF_IFD = 0x8769


class ExportBoundsError(Exception):
    """An exported JPEG violates the target service's size/resolution
    bounds (Adobe Stock 4-100 MP and 45 MB, iNat 20 MB). The offending
    file is deleted before this is raised."""


def _validate_crop_numbers(crop: list[float]) -> None:
    """Raise ValueError unless *crop* is four finite real numbers."""
    if not isinstance(crop, (list, tuple)) or len(crop) != 4:
        raise ValueError(f"crop must be four numbers [x0, y0, x1, y1], got {crop!r}")
    for value in crop:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"crop values must be finite numbers, got {crop!r}")


def _apply_crop(img: Image.Image, crop: list[float] | None) -> Image.Image:
    """Crop *img* to the normalized [x0, y0, x1, y1] rect (origin top-left,
    0..1 against the full image -- same convention as models.Asset.crop and
    LightCraft's own mask/crop coordinates), or return it unchanged if *crop*
    is None. Coordinates are sorted and clamped to the image bounds so a
    reversed or out-of-range rect still produces a valid box; a degenerate
    (zero-area) rect is treated as no crop rather than producing an empty
    image.
    """
    if crop is None:
        return img
    _validate_crop_numbers(crop)
    w, h = img.size
    x0, x1 = sorted((crop[0], crop[2]))
    y0, y1 = sorted((crop[1], crop[3]))
    box = (
        max(0, min(w, round(x0 * w))),
        max(0, min(h, round(y0 * h))),
        max(0, min(w, round(x1 * w))),
        max(0, min(h, round(y1 * h))),
    )
    if box[2] <= box[0] or box[3] <= box[1]:
        return img
    return img.crop(box)


def _resize_and_save(img_bytes: bytes, dest_path: Path, max_dim: int | None, quality: int,
                      crop: list[float] | None = None) -> Path:
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = _apply_crop(img, crop)
    return _resize_and_save_image(img, dest_path, max_dim, quality)


def _resize_and_save_image(img: Image.Image, dest_path: Path, max_dim: int | None, quality: int,
                            *, exif: bytes | None = None, icc_profile: bytes | None = None) -> Path:
    if max_dim is not None:
        img = img.copy()
        img.thumbnail((max_dim, max_dim), Image.LANCZOS)
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    extra = {}
    if exif is not None:
        extra["exif"] = exif
    if icc_profile is not None:
        extra["icc_profile"] = icc_profile
    img.save(str(dest_path), "JPEG", quality=quality, **extra)
    return dest_path


def _srgb_icc_bytes() -> bytes:
    return ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()


def _export_exif_bytes(raw_path: Path) -> bytes:
    """EXIF block for an export: DateTimeOriginal, Make, Model, LensModel
    read from the source raw via exiftool (a missing exiftool raises --
    an export without capture time must not be produced silently). GPS is
    deliberately never copied: iNat gets location from the human-approved
    decision row, and Stock files must not leak coordinates."""
    data = extract_exif(raw_path)
    exif = Image.Exif()
    exif_ifd = exif.get_ifd(_EXIF_IFD)
    if data.get("DateTimeOriginal"):
        exif_ifd[_TAG_DATETIME_ORIGINAL] = str(data["DateTimeOriginal"])
    lens = data.get("LensModel") or data.get("Lens")
    if lens:
        exif_ifd[_TAG_LENS_MODEL] = str(lens)
    if data.get("Make"):
        exif[_TAG_MAKE] = str(data["Make"])
    if data.get("Model"):
        exif[_TAG_MODEL] = str(data["Model"])
    return exif.tobytes()


def _check_export_bounds(path: Path, *, max_bytes: int, min_pixels: int | None = None,
                          max_pixels: int | None = None) -> None:
    with Image.open(path) as saved:
        width, height = saved.size
    pixels = width * height
    size = path.stat().st_size
    problem = None
    if min_pixels is not None and pixels < min_pixels:
        problem = f"{width}x{height} = {pixels / 1e6:.2f} MP is below the {min_pixels / 1e6:g} MP minimum"
    elif max_pixels is not None and pixels > max_pixels:
        problem = f"{width}x{height} = {pixels / 1e6:.2f} MP is above the {max_pixels / 1e6:g} MP maximum"
    elif size > max_bytes:
        problem = f"{size} bytes exceeds the {max_bytes} byte maximum"
    if problem:
        path.unlink()
        raise ExportBoundsError(f"{path}: {problem}")


def _export(raw_path: Path, dest_path: Path, max_dim: int | None, quality: int,
             crop: list[float] | None, **bounds) -> Path:
    img = _apply_crop(_decode_raw_to_rgb8(raw_path), crop)
    out = _resize_and_save_image(img, dest_path, max_dim, quality,
                                  exif=_export_exif_bytes(raw_path), icc_profile=_srgb_icc_bytes())
    _check_export_bounds(out, **bounds)
    return out


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


def render_review_jpeg(raw_path: Path, dest_path: Path, max_dim: int = 2048, quality: int = 85,
                        crop: list[float] | None = None) -> Path:
    """Resize/re-encode the raw's embedded preview (exif.extract_preview_bytes)
    into a review JPEG at dest_path, to drive the keep/reject decision.

    *crop* (default None, uncropped) applies the review console's own
    non-destructive crop rect (see _apply_crop) -- so what a reviewer sees
    here matches what export_inat_jpeg/export_stock_jpeg will also crop to.
    """
    return _resize_and_save(extract_preview_bytes(raw_path), dest_path, max_dim, quality, crop=crop)


def export_inat_jpeg(raw_path: Path, dest_path: Path, max_dim: int = 2048, quality: int = 95,
                      crop: list[float] | None = None) -> Path:
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

    *crop* (default None, uncropped) applies the review console's own
    non-destructive crop rect (see _apply_crop) before resizing.

    Raises ExportBoundsError (and deletes the file) if the result exceeds
    20 MB. The JPEG carries the raw's DateTimeOriginal/Make/Model/LensModel
    and an embedded sRGB ICC profile.
    """
    return _export(raw_path, dest_path, max_dim, quality, crop, max_bytes=INAT_MAX_BYTES)


def export_stock_jpeg(raw_path: Path, dest_path: Path, max_dim: int | None = None, quality: int = 95,
                       crop: list[float] | None = None) -> Path:
    """Full raw decode (see _decode_raw_to_rgb8) at full sensor resolution
    for stock use -- never downsized by default (max_dim=None).

    Adobe Stock's current technical requirements (Adobe helpx, checked
    2026-10-19) call for 4-100MP sRGB JPEGs up to 45MB; the Z7II's ~45MP
    sensor is within that range at full resolution, so there is no reason
    to shrink it. This replaces the v1 embedded-preview source (resolution/
    color capped by whatever the camera's own JPEG engine produced) with a
    real demosaic of the sensor data.

    *crop* (default None, uncropped) applies the review console's own
    non-destructive crop rect (see _apply_crop) before resizing.

    Raises ExportBoundsError (and deletes the file) if the result is under
    4 MP, over 100 MP, or over 45 MB -- a tight crop can drop below 4 MP.
    The JPEG carries the raw's DateTimeOriginal/Make/Model/LensModel and an
    embedded sRGB ICC profile.
    """
    return _export(raw_path, dest_path, max_dim, quality, crop, max_bytes=STOCK_MAX_BYTES,
                   min_pixels=STOCK_MIN_PIXELS, max_pixels=STOCK_MAX_PIXELS)
