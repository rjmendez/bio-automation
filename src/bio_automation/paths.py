from __future__ import annotations

from pathlib import Path


def make_observation_path(base_dir: Path, obsdate: str, slug: str) -> Path:
    year = obsdate[:4]
    return base_dir / year / obsdate / f"obs-{obsdate.replace('-', '')}-{slug}"


def source_raw_dir(observation_dir: Path) -> Path:
    return observation_dir / "source" / "raw"


def metadata_dir(observation_dir: Path) -> Path:
    return observation_dir / "metadata"


def work_dir(observation_dir: Path) -> Path:
    return observation_dir / "work"


def review_dir(observation_dir: Path) -> Path:
    return observation_dir / "review"


def export_dir(observation_dir: Path, kind: str) -> Path:
    return observation_dir / "export" / kind

