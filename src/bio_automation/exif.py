from __future__ import annotations

import json
import subprocess
from pathlib import Path


def extract_exif(raw_path: Path) -> dict:
    result = subprocess.run(
        ["exiftool", "-j", str(raw_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)[0]


def extract_preview_bytes(raw_path: Path) -> bytes:
    """Pull the embedded preview/JpgFromRaw out of a raw file via
    `exiftool -b -PreviewImage` (falling back to `-JpgFromRaw`). Intended as
    the source image for render.py's JPEG export functions, so review/inat/
    stock JPEGs come from the camera's embedded preview rather than a full
    raw development.

    Raises RuntimeError if neither tag is present (not every raw embeds a
    full-size preview).
    """
    result = subprocess.run(
        ["exiftool", "-b", "-PreviewImage", str(raw_path)],
        check=True, capture_output=True,
    )
    if result.stdout:
        return result.stdout

    result = subprocess.run(
        ["exiftool", "-b", "-JpgFromRaw", str(raw_path)],
        check=True, capture_output=True,
    )
    if result.stdout:
        return result.stdout

    raise RuntimeError(f"{raw_path}: no embedded PreviewImage or JpgFromRaw tag found")


def _extract_gps_numeric(raw_path: Path) -> dict | None:
    """Read GPSLatitude/GPSLongitude/GPSAltitude as plain decimal numbers.

    extract_exif()'s default `-j` output renders GPS tags as human-readable
    DMS strings (e.g. "41 deg 7' 24.00\" N"), not the decimal degrees the
    metadata/<stem>.exif.json schema wants (docs/folder-metadata-schema.md:
    "gps": {"lat": 41.1234, ...}). exiftool's `#` tag suffix forces numeric
    output for just that tag regardless of other format flags, so this makes
    its own narrow call rather than changing extract_exif()'s global output
    format (used elsewhere, e.g. for human-readable Make/Model strings).
    """
    result = subprocess.run(
        ["exiftool", "-j", "-GPSLatitude#", "-GPSLongitude#", "-GPSAltitude#", str(raw_path)],
        check=True, capture_output=True, text=True,
    )
    data = json.loads(result.stdout)[0]
    lat = data.get("GPSLatitude")
    lon = data.get("GPSLongitude")
    if lat is None or lon is None:
        return None
    return {"lat": lat, "lon": lon, "alt": data.get("GPSAltitude")}


def build_exif_sidecar(raw_path: Path, observation_dir: Path) -> dict:
    """Normalize extract_exif()'s vendor-keyed JSON into the
    metadata/<stem>.exif.json schema: file, datetime_original, camera_make,
    camera_model, lens_model, gps.

    *file* is raw_path relative to observation_dir (forward slashes, matching
    the convention models.Asset.file already uses elsewhere in this
    codebase) so the sidecar is portable if the observation folder moves.
    *gps* is None when the raw has no GPS tags (most cameras without a GPS
    unit or SnapBridge-style phone sync), not a fabricated {0,0,0}.
    """
    data = extract_exif(raw_path)
    return {
        "file": str(Path(raw_path).relative_to(observation_dir)).replace("\\", "/"),
        "datetime_original": data.get("DateTimeOriginal"),
        "camera_make": data.get("Make"),
        "camera_model": data.get("Model"),
        "lens_model": data.get("LensModel") or data.get("Lens"),
        "gps": _extract_gps_numeric(raw_path),
    }


def _xmp_escape(value) -> str:
    if value is None:
        return ""
    return (
        str(value)
        .replace("&", "&amp;")
        .replace('"', "&quot;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def render_xmp(data: dict) -> str:
    """Render a minimal XMP sidecar string mirroring build_exif_sidecar()'s
    fields, for metadata/<stem>.xmp. Kept intentionally small for v1 -- a
    valid XMP packet with exif:/tiff: attributes on one rdf:Description,
    not a full Adobe-schema document.
    """
    gps = data.get("gps") or {}
    gps_attrs = ""
    if gps:
        gps_attrs = (
            f'\n        exif:GPSLatitude="{_xmp_escape(gps.get("lat"))}"'
            f'\n        exif:GPSLongitude="{_xmp_escape(gps.get("lon"))}"'
        )
    return (
        '<?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>\n'
        '<x:xmpmeta xmlns:x="adobe:ns:meta/">\n'
        '  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">\n'
        '    <rdf:Description rdf:about=""\n'
        '        xmlns:exif="http://ns.adobe.com/exif/1.0/"\n'
        '        xmlns:tiff="http://ns.adobe.com/tiff/1.0/"\n'
        f'        exif:DateTimeOriginal="{_xmp_escape(data.get("datetime_original"))}"\n'
        f'        tiff:Make="{_xmp_escape(data.get("camera_make"))}"\n'
        f'        tiff:Model="{_xmp_escape(data.get("camera_model"))}"\n'
        f'        exif:LensModel="{_xmp_escape(data.get("lens_model"))}"'
        f'{gps_attrs}/>\n'
        '  </rdf:RDF>\n'
        '</x:xmpmeta>\n'
        '<?xpacket end="w"?>\n'
    )
