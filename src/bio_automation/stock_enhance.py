"""Sharpen the Stock-ready backlog into a separate output directory (never in
place) via bio_automation.topaz, stock mode, GPU.

settings.STOCK_DIR / "ready" holds already-rendered Adobe Stock exports at
full sensor resolution (measured 7360x4912 on a sample, matching the Stock
export standard of 4-100 MP) -- no upscale belongs here: these are already at
or above the resolution Stock requires, so only the lens-blur sharpen gate
(topaz.enhance(mode="stock")) is in scope.

Non-destructive by design: the 9,802 files under "ready" are the only copies
of these originals (no backup), so this NEVER overwrites a source file --
every result is written to *out_dir* (default settings.PHOTOS_CACHE /
"stock-enhanced", on D:, matching the owner's "new results/caches on D:,
leave F: alone" rule) under the same filename. Resumable: a file already
present in *out_dir* is skipped on the next run -- no separate sidecar is
needed, since the output file's own existence is the resume marker.
"""
from __future__ import annotations

import os
import shutil
from collections.abc import Callable
from pathlib import Path

from . import disk_guard, settings, topaz

READY_DIR = settings.STOCK_DIR / "ready"
DEFAULT_OUT_DIR = settings.PHOTOS_CACHE / "stock-enhanced"
DEFAULT_REVIEW_NAME = "stock-enhanced"


def run(
    src_dir: Path = READY_DIR,
    *,
    out_dir: Path = DEFAULT_OUT_DIR,
    min_free_gb: float = 10.0,
    limit: int | None = None,
    cache: topaz.ModelCache | None = None,
    log: Callable[[str], None] = print,
) -> dict:
    """Sharpen every not-yet-enhanced JPEG in *src_dir* into *out_dir*.

    Returns {"done": int, "skipped": int, "failed": int, "total": int}.
    """
    out_dir = Path(out_dir)
    files = sorted(Path(src_dir).glob("*.jpg"))
    if limit is not None:
        files = files[:limit]

    done = skipped = failed = 0

    def _process(active_cache: topaz.ModelCache) -> None:
        nonlocal done, skipped, failed
        for jpg in files:
            dst = out_dir / jpg.name
            if dst.exists():
                skipped += 1
                continue

            if not disk_guard.wait_for_output(out_dir, min_free_gb, who="stock_enhance", say=log):
                free = shutil.disk_usage(out_dir.parent if not out_dir.exists() else out_dir).free / 1e9
                log(f"STOP: only {free:.0f} GB free (limit {min_free_gb:.0f} GB); "
                    f"{len(files) - done - skipped - failed} left")
                return

            try:
                topaz.enhance(jpg, dst, mode="stock", cache=active_cache, min_free_gb=min_free_gb)
                done += 1
            except Exception as e:
                failed += 1
                log(f"ERROR {jpg.name}: {type(e).__name__} {str(e)[:120]}")
                if dst.exists():  # never leave a partial/corrupt file counted as "done" on resume
                    dst.unlink()

    if cache is None:
        with topaz.ModelCache() as owned_cache:
            _process(owned_cache)
    else:
        _process(cache)

    return {"done": done, "skipped": skipped, "failed": failed, "total": len(files)}


def publish_review(
    out_dir: Path = DEFAULT_OUT_DIR,
    *,
    gallery_dir: Path | None = None,
    review_name: str = DEFAULT_REVIEW_NAME,
    queue=None,
) -> dict:
    """Hardlink every enhanced file in *out_dir* into the gallery under
    _review/<review_name>/, so it shows up for a visual compare, without
    touching the originals in READY_DIR or anything already in the gallery's
    dated <year>/<year-month>/ trees. Safe to call again later as more files
    finish: already-linked files are left alone.

    Publishes one gallery.reindex message (via *queue*, if given) when any
    new file was linked.
    """
    gallery_dir = Path(gallery_dir) if gallery_dir is not None else settings.GALLERY_DIR
    rel_folder = f"_review/{review_name}"
    dst_dir = gallery_dir / rel_folder
    dst_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(Path(out_dir).glob("*.jpg"))
    linked: list[str] = []
    for f in files:
        dst = dst_dir / f.name
        if dst.exists():
            continue
        os.link(f, dst)
        linked.append(f.name)

    published = None
    if queue is not None and linked:
        published = queue.publish("gallery.reindex", {"gallery_dir": str(gallery_dir), "folders": [rel_folder]})

    return {"linked": len(linked), "already_there": len(files) - len(linked),
            "total": len(files), "reindex_message": published}
