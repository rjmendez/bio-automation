"""One flat zip of export-ready JPEGs for iNaturalist or Adobe Stock, for remote download.

Every file is validated against the owner's export standard BEFORE anything is written; a violation aborts the whole bundle (nothing is
shipped half-checked, nothing is deleted or modified). The zip is flat (basenames only, no folders), stored not compressed (JPEG does not
compress), written to a temp name and renamed into place, with a ``<zip>.sha256`` sidecar in ``sha256sum`` format.

Standard (render.py constants): iNat = long edge <= 2048 px, <= 20 MB. Stock = 4-100 MP, <= 45 MB, no GPS block (Stock files must not leak
coordinates). Both = JPEG, RGB, an sRGB ICC profile, JPEG quality >= 95 (quantisation table no coarser than Pillow's q95 table).
Delivery is not done here: the zip lands in a download folder and is fetched with scp/sftp over the existing sshd.
"""
from __future__ import annotations

import hashlib
import io
import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageCms

from .render import INAT_MAX_BYTES, STOCK_MAX_BYTES, STOCK_MAX_PIXELS, STOCK_MIN_PIXELS

DEFAULT_DOWNLOAD_DIR = "D:/bio-cache/downloads"      # deliverables, not originals; D: has the free space (forward slashes: no escape pitfalls)
INAT_MAX_LONG_EDGE = 2048
KINDS = ("inat", "stock")
_JPEG_SUFFIXES = (".jpg", ".jpeg")
_GPS_IFD = 0x8825
_FREE_SPACE_MARGIN = 1024 ** 3        # keep 1 GiB free beyond the zip itself


class BundleError(Exception):
    """The bundle cannot be built (bad kind, no files, name collision, a standard violation, or not enough disk)."""


@dataclass(frozen=True)
class BundleResult:
    path: Path
    photos: int
    extras: tuple[str, ...]
    size_bytes: int
    sha256: str


@lru_cache(maxsize=1)
def _q95_table() -> tuple[int, ...]:
    buf = io.BytesIO()
    Image.new("RGB", (16, 16)).save(buf, "JPEG", quality=95)
    buf.seek(0)
    return tuple(Image.open(buf).quantization[0])


def validate_jpeg(path: Path, kind: str) -> list[str]:
    """Problems with *path* against the *kind* standard; an empty list means it conforms. Never modifies or deletes the file."""
    if kind not in KINDS:
        raise BundleError(f"kind must be one of {KINDS}, got {kind!r}")
    problems: list[str] = []
    size = Path(path).stat().st_size
    try:
        with Image.open(path) as im:
            if im.format != "JPEG":
                return [f"not a JPEG (format {im.format})"]
            width, height = im.size
            if im.mode != "RGB":
                problems.append(f"mode {im.mode}, not RGB")
            icc = im.info.get("icc_profile")
            if not icc:
                problems.append("no ICC profile (sRGB required)")
            else:
                try:
                    desc = ImageCms.getProfileDescription(ImageCms.ImageCmsProfile(io.BytesIO(icc)))
                except Exception:
                    desc = ""
                if "srgb" not in desc.lower():
                    problems.append(f"ICC profile is {desc.strip()!r}, not sRGB")
            table = tuple(im.quantization[0])
            if len(table) != len(_q95_table()) or any(a > b for a, b in zip(table, _q95_table())):
                problems.append("JPEG quality below 95 (quantisation table coarser than q95)")
            has_gps = bool(im.getexif().get_ifd(_GPS_IFD))
    except OSError as e:
        return [f"cannot be read as an image: {e}"]
    pixels = width * height
    if kind == "inat":
        if max(width, height) > INAT_MAX_LONG_EDGE:
            problems.append(f"long edge {max(width, height)} px exceeds {INAT_MAX_LONG_EDGE}")
        if size > INAT_MAX_BYTES:
            problems.append(f"{size} bytes exceeds the {INAT_MAX_BYTES} byte iNat maximum")
    else:
        if pixels < STOCK_MIN_PIXELS:
            problems.append(f"{width}x{height} = {pixels / 1e6:.2f} MP is below the {STOCK_MIN_PIXELS / 1e6:g} MP minimum")
        if pixels > STOCK_MAX_PIXELS:
            problems.append(f"{width}x{height} = {pixels / 1e6:.2f} MP is above the {STOCK_MAX_PIXELS / 1e6:g} MP maximum")
        if size > STOCK_MAX_BYTES:
            problems.append(f"{size} bytes exceeds the {STOCK_MAX_BYTES} byte Stock maximum")
        if has_gps:
            problems.append("contains a GPS block (Stock files must not carry coordinates)")
    return problems


def _collect(sources: list[Path]) -> tuple[list[Path], list[Path]]:
    photos: list[Path] = []
    extras: list[Path] = []
    for src in sources:
        src = Path(src)
        if src.is_dir():
            for p in sorted(src.iterdir(), key=lambda q: q.name.lower()):
                if p.is_file() and p.suffix.lower() in _JPEG_SUFFIXES:
                    photos.append(p)
                elif p.is_file() and p.suffix.lower() == ".csv":
                    extras.append(p)
        elif src.is_file():
            (photos if src.suffix.lower() in _JPEG_SUFFIXES else extras).append(src)
        else:
            raise BundleError(f"source not found: {src}")
    return photos, extras


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def build_bundle(kind: str, sources: list[Path], out_dir: Path, name: str | None = None) -> BundleResult:
    """Validate every JPEG in *sources* (directories are read non-recursively; ``*.csv`` files come along) and write ONE flat zip into *out_dir*."""
    if kind not in KINDS:
        raise BundleError(f"kind must be one of {KINDS}, got {kind!r}")
    photos, extras = _collect(list(sources))
    if not photos:
        raise BundleError("no JPEG files found in the sources")
    names = [p.name for p in photos + extras]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise BundleError(f"a flat zip needs unique file names; {len(dupes)} collide, e.g. {dupes[:3]}")
    bad = {p.name: validate_jpeg(p, kind) for p in photos}
    bad = {n: v for n, v in bad.items() if v}
    if bad:
        first = "; ".join(f"{n}: {', '.join(v)}" for n, v in list(bad.items())[:5])
        raise BundleError(f"{len(bad)} of {len(photos)} files violate the {kind} standard, nothing written. First: {first}")
    total = sum(p.stat().st_size for p in photos + extras)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(out_dir).free
    if free < total + _FREE_SPACE_MARGIN:
        raise BundleError(f"not enough disk in {out_dir}: {free} bytes free, need {total} + {_FREE_SPACE_MARGIN}")
    stem = name or f"{kind}-{len(photos)}photos-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    final = out_dir / (stem if stem.lower().endswith(".zip") else stem + ".zip")
    if final.exists():
        raise BundleError(f"{final} already exists; refusing to overwrite")
    part = final.with_name(final.name + ".part")
    import zipfile
    try:
        with zipfile.ZipFile(part, "w", zipfile.ZIP_STORED) as z:
            for p in photos + extras:
                z.write(p, arcname=p.name)
        os.replace(part, final)
    finally:
        if part.exists():
            part.unlink()
    digest = _sha256(final)
    # bytes, not write_text: on Windows text mode would write CRLF and `sha256sum -c` elsewhere would then look for a name ending in "\r"
    final.with_name(final.name + ".sha256").write_bytes(f"{digest}  {final.name}\n".encode("ascii"))
    return BundleResult(final, len(photos), tuple(p.name for p in extras), final.stat().st_size, digest)
