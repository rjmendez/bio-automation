from __future__ import annotations

import shutil
from dataclasses import asdict
from pathlib import Path

from .io import read_json, read_jsonl, write_json, write_jsonl
from .models import Asset, Observation
from .paths import export_dir, metadata_dir, review_dir, source_raw_dir


def load_observation(folder: Path) -> Observation:
    return Observation(**read_json(metadata_dir(folder) / "observation.json"))


def save_observation(folder: Path, data: Observation) -> None:
    write_json(metadata_dir(folder) / "observation.json", asdict(data))


def scan_assets(folder: Path) -> list[Asset]:
    return [Asset(**row) for row in read_jsonl(metadata_dir(folder) / "assets.jsonl")]


def save_assets(folder: Path, assets: list[Asset]) -> None:
    write_jsonl(metadata_dir(folder) / "assets.jsonl", [asdict(asset) for asset in assets])


def set_asset_status(folder: Path, file_names: list[str], status: str, selected: bool | None) -> list[Asset]:
    assets = scan_assets(folder)
    wanted = {name.replace("\\", "/") for name in file_names}
    updated: list[Asset] = []
    for asset in assets:
        if asset.file in wanted or asset.file.split("/")[-1] in wanted:
            asset.status = status
            asset.selected = selected
        updated.append(asset)
    save_assets(folder, updated)
    return updated


def discover_source_files(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    files = [path for path in source.rglob("*") if path.is_file()]
    return sorted(files)


def copy_into_raw(folder: Path, source: Path) -> list[Path]:
    raw_dir = source_raw_dir(folder)
    raw_dir.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for file_path in discover_source_files(source):
        destination = raw_dir / file_path.name
        if not destination.exists():
            shutil.copy2(file_path, destination)
        copied.append(destination)
    return copied


def delete_rejected_raws(folder: Path) -> list[Path]:
    assets = scan_assets(folder)
    deleted: list[Path] = []
    for asset in assets:
        if asset.role == "original" and asset.status == "rejected":
            file_path = folder / asset.file
            if file_path.exists():
                file_path.unlink()
                deleted.append(file_path)
    return deleted


def ensure_layout(folder: Path) -> None:
    for subdir in [
        source_raw_dir(folder),
        review_dir(folder),
        export_dir(folder, "inat"),
        export_dir(folder, "stock"),
        metadata_dir(folder),
    ]:
        subdir.mkdir(parents=True, exist_ok=True)


INAT_OWNED_FIELDS = ("observation_id", "source", "taxon", "place_guess", "license", "quality_grade")


def merge_inat_fields(existing: Observation, fetched: dict) -> Observation:
    """Merge a fetch_inat_observation() payload into an already-ingested
    Observation. Overwrites only INAT_OWNED_FIELDS; unions keywords_seed
    instead of replacing it; leaves every other local field (obsdate,
    locality, ...) untouched -- obsdate in particular must never be
    silently rewritten from iNat's observed_on, since it is already baked
    into the folder path via make_observation_path(). Pure, no I/O.
    """
    raise NotImplementedError("fetch-inat reconciliation merge will be added next.")


def write_sidecars(folder: Path) -> list[Path]:
    """Write metadata/<stem>.exif.json and metadata/<stem>.xmp for every
    kept raw in source/raw/, one pair per raw (an observation folder can
    hold multiple kept raws, so a single shared sidecar file would collide).
    """
    raise NotImplementedError("write_sidecars will be added next.")


def render_review(folder: Path) -> list[Path]:
    """Render review/<stem>.jpg proxies for every kept raw, to drive the
    keep/reject decision. Decoupled from where the raw itself lives -- see
    Workflow step 2 (cull). Needs render.render_review_jpeg, which is
    blocked on an image-rendering dependency (see Implementation notes).
    """
    raise NotImplementedError("render_review will be added next.")


def export_inat(folder: Path) -> list[Path]:
    """Render export/inat/<stem>.jpg upload-ready copies for every kept
    raw. Needs render.export_inat_jpeg (same dependency as render_review).
    """
    raise NotImplementedError("export_inat will be added next.")


def export_stock(folder: Path) -> list[Path]:
    """Render export/stock/<stem>.jpg stock-ready derivatives for every
    kept raw. Needs render.export_stock_jpeg (same dependency as
    render_review).
    """
    raise NotImplementedError("export_stock will be added next.")


def build_and_save_keywords(folder: Path) -> dict:
    """Build metadata/keywords.json from the observation record via
    stock.build_keywords() and write it to disk. One record per observation
    folder, not per asset.
    """
    raise NotImplementedError("build_and_save_keywords will be added next.")
