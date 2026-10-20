"""Automatic ingest from a camera/memory-card volume (e.g. a card mounted by
a Windows auto-play watcher, out of scope here -- see module docstring
below) into F:\\Bio, with zero manual `bio ingest --obsdate --slug` calls.

This module is the testable logic a volume-insert watcher shells out to as
`bio auto-ingest <volume_path>` (see cli.py); it does not itself detect
volume insertion.

Two real-world correctness requirements drive this module's design, both
confirmed against this repo's actual code before writing a line here:

1. Day grouping is by each raw file's REAL EXIF capture date, not by
   today's wall-clock date and not one folder for the whole card -- a
   memory card commonly holds more than one shooting day. Files are
   grouped by the (capture date, camera model) pair found in their own
   EXIF (see read_capture_info()/group_by_capture()), one observation
   folder per group.

   EXIF source: this repo's one other real EXIF reader (exif.py) shells
   out to `exiftool`, which is confirmed NOT installed in this dev
   environment (`exiftool -ver` -> command not found here). rawpy (already
   a real dependency, used in preclean.py) exposes `RawPy.other.timestamp`
   as a real datetime -- confirmed live against tests/fixtures/sample_raw.NEF
   -- but, also confirmed live (`dir(rawpy.RawPy)` has no model/make/idata
   attribute of any kind), rawpy's Python bindings expose no camera
   make/model at all, so rawpy alone cannot satisfy the (date, camera
   model) grouping key this module needs. What does work for both fields
   at once, confirmed live against both committed NEF fixtures: Nikon NEF
   (like most raw formats) is itself a valid TIFF container, so Pillow's
   own `Image.getexif()` -- Pillow is already a real, committed dependency
   (pyproject.toml, used elsewhere in render.py/preclean.py) -- reads the
   DateTimeOriginal/Make/Model IFD0 tags directly off the raw file no
   differently than off a JPEG. That is what read_capture_info() below
   uses, rather than mixing rawpy for one field and a second new
   dependency for the other.

2. Re-inserting the same card (or a card with some already-ingested files
   plus new ones) must be idempotent and must NEVER reset a curation
   decision. pipeline.save_assets() fully overwrites assets.jsonl
   (write_jsonl, not append/merge), and the existing cmd_ingest() CLI
   handler calls it unconditionally with a fresh "ingested" Asset for
   every copied file on every run -- confirmed by reading cli.py: re-running
   `bio ingest` against an observation folder that already has kept/
   rejected assets would silently reset them all back to "ingested". This
   module's path never does that: upsert_assets() merges by relative file
   path, leaving every already-known asset's status/selected exactly as
   found and only appending genuinely new files as "ingested". Likewise
   upsert_observation() never resets a field (observation_id most of all)
   that something else -- `bio fetch-inat`, or a prior auto-ingest run --
   already set away from Observation's own dataclass default; see its
   docstring. cmd_ingest()/copy_into_raw()'s own behavior for the existing
   manual `bio ingest` command is untouched by this module.
"""
from __future__ import annotations

import datetime as dt
import re
from dataclasses import MISSING, dataclass, fields, replace
from pathlib import Path

from PIL import Image

from .models import Asset, Observation
from .paths import make_observation_path, metadata_dir
from .pipeline import (
    copy_into_raw,
    discover_source_files,
    ensure_layout,
    load_observation,
    save_assets,
    save_observation,
    scan_assets,
)

# TIFF-based raw formats Pillow can open directly (see module docstring);
# CR3 (ISO-BMFF, not TIFF) and other non-TIFF raw containers are out of
# scope -- read_capture_info() will raise ExifReadError for one of those
# rather than silently mis-reading it.
RAW_EXTENSIONS = {".nef", ".cr2", ".arw", ".dng", ".orf", ".rw2", ".raf", ".pef", ".srw"}
IMAGE_EXTENSIONS = RAW_EXTENSIONS | {".jpg", ".jpeg"}

# EXIF IFD0/EXIF-IFD tag ids (same numeric tags exif.py's own schema is
# built from, just read via Pillow instead of exiftool here).
_TAG_DATETIME_ORIGINAL = 36867
_TAG_DATETIME = 306
_TAG_MODEL = 272

UNKNOWN_CAMERA_SLUG = "unknown-camera"

# The only Observation fields an auto-ingest proposes a value for; every
# other field (observation_id, taxon, license, quality_grade, ...) is never
# touched by upsert_observation(), by construction (it is not in this
# tuple), independent of whatever "is this at its default" check runs.
AUTO_INGEST_OWNED_FIELDS = ("obsdate", "source", "place_guess")

_OBSERVATION_FIELD_DEFAULTS = {
    f.name: (f.default_factory() if f.default_factory is not MISSING else f.default)
    for f in fields(Observation)
}


class ExifReadError(RuntimeError):
    """read_capture_info() could not find/parse a usable capture date."""


class NotACameraVolumeError(RuntimeError):
    """auto_ingest_volume() was pointed at a path that does not look like a
    camera/memory-card source -- see is_camera_volume()."""


@dataclass(frozen=True)
class CaptureInfo:
    capture_date: dt.date
    camera_model: str | None


def read_capture_info(raw_path: Path) -> CaptureInfo:
    """Read the real EXIF capture date (and camera model, if present) out of
    *raw_path* via Pillow's Image.getexif() -- see module docstring for why
    Pillow, not exiftool or rawpy, is the real method used here.

    Prefers DateTimeOriginal (tag 36867, when the camera wrote it); falls
    back to DateTime (tag 306) otherwise -- exif.py's own build_exif_sidecar()
    reads the same DateTimeOriginal tag as its primary field, so this mirrors
    that preference rather than inventing a new one. Raises ExifReadError
    (not a fabricated/guessed date) if neither tag is present or the value
    does not parse as a real EXIF datetime -- a card with an unreadable file
    must fail loud, not get silently mis-grouped by today's date.

    camera_model is None (not "", not a fabricated string) when tag 272 is
    absent or blank -- camera_model_slug() is what turns that into a stable
    folder-name fallback, kept as a separate, explicit step.
    """
    image = Image.open(raw_path)
    try:
        exif = image.getexif()
    finally:
        image.close()

    raw_value = exif.get(_TAG_DATETIME_ORIGINAL) or exif.get(_TAG_DATETIME)
    if not raw_value:
        raise ExifReadError(
            f"{raw_path}: no DateTimeOriginal/DateTime EXIF tag found -- cannot determine "
            "its real capture date"
        )
    try:
        parsed = dt.datetime.strptime(str(raw_value).strip(), "%Y:%m:%d %H:%M:%S")
    except ValueError as exc:
        raise ExifReadError(
            f"{raw_path}: EXIF datetime {raw_value!r} does not parse as a real capture date"
        ) from exc

    model = exif.get(_TAG_MODEL)
    model = model.strip() if isinstance(model, str) and model.strip() else None

    return CaptureInfo(capture_date=parsed.date(), camera_model=model)


def camera_model_slug(camera_model: str | None) -> str:
    """Filesystem-safe, deterministic slug for *camera_model*, used as the
    observation folder's per-group slug (see make_observation_path()).
    Falls back to UNKNOWN_CAMERA_SLUG for None/blank -- a real, named
    fallback rather than an empty or guessed slug, so files whose camera
    model could not be read still land in one consistent, reproducible
    folder instead of a different one on every run.
    """
    if not camera_model or not camera_model.strip():
        return UNKNOWN_CAMERA_SLUG
    slug = re.sub(r"[^a-z0-9]+", "-", camera_model.strip().lower()).strip("-")
    return slug or UNKNOWN_CAMERA_SLUG


def is_camera_volume(volume_path: Path) -> bool:
    """Real heuristic for "does this path look like a camera/card source":
    true if it has a top-level DCIM folder (any case -- the DCF/DCIM
    standard real cameras/cards use) OR it directly contains at least one
    raw/JPEG file anywhere under it. False for a path that does not exist,
    is not a directory, or holds no DCIM folder and no image files at all
    (an unrelated drive).
    """
    volume_path = Path(volume_path)
    if not volume_path.is_dir():
        return False
    if any(child.is_dir() and child.name.upper() == "DCIM" for child in volume_path.iterdir()):
        return True
    return any(
        path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        for path in volume_path.rglob("*")
    )


def group_by_capture(raw_paths: list[Path]) -> dict[tuple[dt.date, str | None], list[Path]]:
    """Group *raw_paths* by the (EXIF capture date, camera model) pair read
    from each file via read_capture_info() -- real-world day grouping, not
    one group for the whole card and not dated by today. Returns an
    insertion-ordered dict (first-seen order of each new key); within a
    group, paths keep the order they were given in.
    """
    groups: dict[tuple[dt.date, str | None], list[Path]] = {}
    for raw_path in raw_paths:
        info = read_capture_info(raw_path)
        key = (info.capture_date, info.camera_model)
        groups.setdefault(key, []).append(raw_path)
    return groups


def upsert_assets(folder: Path, candidates: list[Asset]) -> dict[str, list[str]]:
    """Merge-safe alternative to pipeline.save_assets() for re-running
    ingest against a folder that may already have curated (kept/rejected)
    assets -- see module docstring, requirement 2.

    *candidates* is the full proposed set for this run (e.g. one "ingested"
    Asset per raw file currently on the card, mirroring cmd_ingest()'s own
    Asset-building convention). Any candidate whose `file` already appears
    in the folder's existing assets.jsonl is left alone -- its on-disk
    status/selected is NEVER touched, regardless of what status the
    candidate itself carries. Only candidates whose `file` is not already
    present are appended as new rows. assets.jsonl is left untouched (no
    write at all) when there is nothing new to add.

    Returns {"added": [file, ...], "skipped": [file, ...]} (both relative
    file paths, in *candidates* order) so a caller can report real counts.
    """
    existing = scan_assets(folder)
    existing_files = {asset.file for asset in existing}

    added: list[Asset] = []
    added_files: list[str] = []
    skipped_files: list[str] = []
    for candidate in candidates:
        if candidate.file in existing_files:
            skipped_files.append(candidate.file)
        else:
            added.append(candidate)
            added_files.append(candidate.file)
            existing_files.add(candidate.file)  # a card can't list the same relpath twice anyway

    if added:
        save_assets(folder, existing + added)

    return {"added": added_files, "skipped": skipped_files}


def upsert_observation(folder: Path, proposed: Observation) -> Observation:
    """Merge-safe alternative to pipeline.save_observation() for re-running
    ingest against a folder that may already have an observation.json --
    see module docstring, requirement 2.

    If metadata/observation.json does not exist yet, *proposed* is written
    as-is (first ingest of this group). If it already exists, only the
    fields auto-ingest ever proposes a value for (AUTO_INGEST_OWNED_FIELDS)
    are candidates to change, and even then ONLY when the existing record
    still holds that field's Observation dataclass default (i.e. nothing
    has set it yet) -- mirrors pipeline.merge_inat_fields()'s own
    INAT_OWNED_FIELDS convention of "the dataclass default means not yet
    set". Every other field -- observation_id (a `bio fetch-inat` link)
    above all -- is never part of *proposed* and is therefore never
    touched, no matter what it currently holds.
    """
    observation_path = metadata_dir(folder) / "observation.json"
    if not observation_path.exists():
        save_observation(folder, proposed)
        return proposed

    existing = load_observation(folder)
    overrides = {
        field_name: getattr(proposed, field_name)
        for field_name in AUTO_INGEST_OWNED_FIELDS
        if getattr(existing, field_name) == _OBSERVATION_FIELD_DEFAULTS[field_name]
    }
    if not overrides:
        return existing

    merged = replace(existing, **overrides)
    save_observation(folder, merged)
    return merged


def ingest_capture_group(base_dir: Path, capture_date: dt.date, camera_model: str | None,
                          raw_paths: list[Path]) -> dict:
    """Ingest one (capture_date, camera_model) group's *raw_paths* into its
    own observation folder under *base_dir*: ensure_layout + copy_into_raw
    (per file, reusing pipeline's own single-source copy function -- see
    auto_ingest_volume()) + upsert_observation + upsert_assets.

    Returns {"folder": str, "obsdate": str, "camera_model": str | None,
    "folder_is_new": bool, "new_files": [str, ...], "already_present": [str, ...]}.
    """
    obsdate = capture_date.isoformat()
    slug = camera_model_slug(camera_model)
    observation_dir = make_observation_path(Path(base_dir), obsdate, slug)
    folder_is_new = not observation_dir.exists()

    ensure_layout(observation_dir)

    copied: list[Path] = []
    for raw_path in raw_paths:
        copied.extend(copy_into_raw(observation_dir, raw_path))

    upsert_observation(
        observation_dir,
        Observation(
            obsdate=obsdate,
            source="auto-ingest",
        ),
    )

    candidates = [
        Asset(
            file=str(path.relative_to(observation_dir)).replace("\\", "/"),
            role="original",
            status="ingested",
            selected=False,
        )
        for path in copied
    ]
    asset_result = upsert_assets(observation_dir, candidates)

    return {
        "folder": str(observation_dir),
        "obsdate": obsdate,
        "camera_model": camera_model,
        "folder_is_new": folder_is_new,
        "new_files": asset_result["added"],
        "already_present": asset_result["skipped"],
    }


def auto_ingest_volume(volume_path: Path, base_dir: Path) -> dict:
    """Entry point for `bio auto-ingest <volume_path>`: detect whether
    *volume_path* looks like a camera/card source, group its raw files by
    real EXIF (capture date, camera model), and ingest each group into its
    own observation folder under *base_dir* -- idempotently (see module
    docstring).

    Raises NotACameraVolumeError if *volume_path* does not pass
    is_camera_volume() -- refuses to treat an unrelated drive as a photo
    source rather than silently doing nothing or creating empty folders.

    Returns {"volume": str, "base_dir": str,
    "groups": [ingest_capture_group()'s dict, ...] (one per group, sorted by
    (obsdate, camera_model) for a stable report order),
    "observation_folders_created": int, "observation_folders_updated": int,
    "files_ingested": int, "files_already_present": int}.
    """
    volume_path = Path(volume_path)
    base_dir = Path(base_dir)

    if not is_camera_volume(volume_path):
        raise NotACameraVolumeError(
            f"{volume_path} does not look like a camera/card source (no DCIM folder and no "
            "raw/JPEG files found) -- refusing to auto-ingest"
        )

    raw_paths = [
        path for path in discover_source_files(volume_path)
        if path.suffix.lower() in RAW_EXTENSIONS
    ]

    groups = group_by_capture(raw_paths)

    group_results = [
        ingest_capture_group(base_dir, capture_date, camera_model, paths)
        for (capture_date, camera_model), paths in sorted(
            groups.items(), key=lambda item: (item[0][0].isoformat(), item[0][1] or "")
        )
    ]

    return {
        "volume": str(volume_path),
        "base_dir": str(base_dir),
        "groups": group_results,
        "observation_folders_created": sum(1 for g in group_results if g["folder_is_new"]),
        "observation_folders_updated": sum(1 for g in group_results if not g["folder_is_new"]),
        "files_ingested": sum(len(g["new_files"]) for g in group_results),
        "files_already_present": sum(len(g["already_present"]) for g in group_results),
    }
