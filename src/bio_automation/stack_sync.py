"""stack_sync.py -- ingest focus-stacks synced in from the NikonStacker Pi
(see nikon-z7ii-tools/stack_buffer.py) and, per the owner's explicit
decision this session, auto-process them on arrival.

Scope, deliberately narrow, mirroring auto_ingest.py's own discipline: a
"synced stack" directory under SYNC_DIR already represents ONE deliberate
focus-stack sequence that lens_stack.py fired and stack_buffer.py tagged
with a manifest on the Pi -- this module never infers groupings or decides
what counts as a stack, it only reads what the Pi already decided. The Pi's
sync step lands each stack atomically (rsync into a hidden ".incoming-*"
staging name, then an on-box rename to the real name) specifically so this
module never has to guess whether a visible directory is still receiving
files -- any directory here with a manifest.json is complete by construction.

Unlike auto_ingest.py (card dumps: ingest only, curation and stacking stay
manual), the owner explicitly chose auto-stack-on-arrival for this path,
since these are already-deliberate stack sequences, not raw uncurated
frames needing a human keep/reject decision.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from .models import Asset, Observation
from .paths import make_observation_path, metadata_dir, source_raw_dir
from .pipeline import ensure_layout, process_observation
from .auto_ingest import upsert_assets, upsert_observation

SYNC_DIR = Path(r"D:\nikon_stack_sync")
BASE_DIR = Path(r"F:\Bio")


class ManifestError(RuntimeError):
    """A stack directory's manifest.json is missing or unreadable."""


def find_new_synced_stacks(sync_dir: Path = SYNC_DIR) -> list[Path]:
    """Complete, not-yet-ingested stack directories under *sync_dir*.

    A ".incoming-*" name means the Pi's rsync hasn't finished staging it
    yet (never visible under its real name until the atomic rename
    completes, so this check is a belt-and-suspenders skip, not the
    primary safety mechanism). A directory without manifest.json isn't a
    real stack. A directory with ".ingested" was already processed by a
    prior call -- skipped so repeated polling is idempotent.
    """
    if not sync_dir.exists():
        return []
    return sorted(
        d for d in sync_dir.iterdir()
        if d.is_dir()
        and not d.name.startswith(".incoming-")
        and (d / "manifest.json").exists()
        and not (d / ".ingested").exists()
    )


def ingest_synced_stack(stack_dir: Path, base_dir: Path = BASE_DIR) -> Path:
    """Copy one synced stack's raw files into its own F:\\Bio observation
    folder and record it (merge-safe: reruns on an already-ingested folder
    never touch a curated asset or an already-linked observation field --
    see auto_ingest.upsert_assets/upsert_observation). Returns the real
    observation directory."""
    manifest_path = stack_dir / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        raise ManifestError(f"{manifest_path}: {e}") from e

    obsdate = manifest["captured_at"][:10]
    slug = f"nikon-stack-{manifest['stack_id'][-8:]}"
    observation_dir = make_observation_path(base_dir, obsdate, slug)
    ensure_layout(observation_dir)

    raw_dir = source_raw_dir(observation_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name in manifest["files"]:
        dest = raw_dir / name
        if not dest.exists():
            shutil.copy2(stack_dir / name, dest)

    upsert_observation(observation_dir, Observation(
        obsdate=obsdate, source="nikon-stack-sync", place_guess="",
    ))
    upsert_assets(observation_dir, [
        Asset(file=f"source/raw/{name}", role="original", status="ingested", selected=False)
        for name in manifest["files"]
    ])
    return observation_dir


def process_synced_stacks(sync_dir: Path = SYNC_DIR, base_dir: Path = BASE_DIR, *,
                           processor=process_observation, cleanup: bool = True,
                           **process_kwargs) -> list[dict]:
    """Ingest and auto-process every new synced stack. For each: ingest,
    then run *processor* (defaults to the real pipeline.process_observation
    -- stack/sidecars/review/export, per the owner's auto-stack-on-arrival
    decision). Only marks a stack ".ingested" and deletes its D:\\ landing
    copy (scratch space, not permanent retention -- the durable copy is now
    under F:\\Bio) after BOTH ingest and processing succeed; a failure in
    either step leaves the synced stack exactly as it arrived, to retry on
    the next call rather than silently lose it.

    Returns one result dict per stack: {"stack_dir", "observation_dir",
    "status": "ok"|"error", "error"?: str}.
    """
    results = []
    for stack_dir in find_new_synced_stacks(sync_dir):
        try:
            observation_dir = ingest_synced_stack(stack_dir, base_dir)
            processor(observation_dir, **process_kwargs)
        except Exception as e:
            results.append({"stack_dir": str(stack_dir), "status": "error", "error": str(e)})
            continue

        if cleanup:
            shutil.rmtree(stack_dir)
        else:
            (stack_dir / ".ingested").write_text("")
        results.append({
            "stack_dir": str(stack_dir),
            "observation_dir": str(observation_dir),
            "status": "ok",
        })
    return results
