from __future__ import annotations

from pathlib import Path
import subprocess


def extract_exif(raw_path: Path) -> dict:
    result = subprocess.run(
        ["exiftool", "-j", str(raw_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    import json

    return json.loads(result.stdout)[0]


def extract_preview_bytes(raw_path: Path) -> bytes:
    """Pull the embedded preview/JpgFromRaw out of a raw file via
    `exiftool -b -PreviewImage` (falling back to `-JpgFromRaw`). Intended as
    the source image for render.py's JPEG export stubs, so review/inat/stock
    JPEGs come from the camera's embedded preview rather than a full raw
    development.
    """
    raise NotImplementedError("extract_preview_bytes will be added next.")


def build_exif_sidecar(raw_path: Path, observation_dir: Path) -> dict:
    """Normalize extract_exif()'s vendor-keyed JSON into the
    metadata/<stem>.exif.json schema: file, datetime_original, camera_make,
    camera_model, lens_model, gps.
    """
    raise NotImplementedError("build_exif_sidecar will be added next.")


def render_xmp(data: dict) -> str:
    """Render a minimal XMP sidecar string mirroring build_exif_sidecar()'s
    fields, for metadata/<stem>.xmp. Kept intentionally small for v1.
    """
    raise NotImplementedError("render_xmp will be added next.")

