from __future__ import annotations

import shutil
from pathlib import Path

from .io import read_json, read_jsonl, write_json, write_jsonl
from .paths import export_dir, metadata_dir, review_dir, source_raw_dir


def load_observation(folder: Path) -> dict:
    return read_json(metadata_dir(folder) / "observation.json")


def save_observation(folder: Path, data: dict) -> None:
    write_json(metadata_dir(folder) / "observation.json", data)


def scan_assets(folder: Path) -> list[dict]:
    return read_jsonl(metadata_dir(folder) / "assets.jsonl")


def save_assets(folder: Path, assets: list[dict]) -> None:
    write_jsonl(metadata_dir(folder) / "assets.jsonl", assets)


def set_asset_status(folder: Path, file_names: list[str], status: str, selected: bool | None) -> list[dict]:
    assets = scan_assets(folder)
    wanted = {name.replace("\\", "/") for name in file_names}
    updated: list[dict] = []
    for asset in assets:
        if asset.get("file") in wanted or asset.get("file", "").split("/")[-1] in wanted:
            asset["status"] = status
            asset["selected"] = selected
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
        if asset.get("role") == "original" and asset.get("status") == "rejected":
            file_path = folder / asset["file"]
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
