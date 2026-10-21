"""Auto-adjusted review thumbnails via lightcraft-cli, for the review
console's /img endpoint -- replaces the flat, unprocessed embedded camera
preview (exif.extract_preview_bytes) that turned out to be useless for
culling (no visible exposure/contrast/detail to judge sharpness by, see
the 2026-10-19 LightCraft-capabilities workflow finding).

Each render runs `lightcraft-cli run --import <raw> develop.auto
app.export ...` as its own ephemeral headless process -- NOT
`--connect 127.0.0.1:7980` to the live GUI-attached instance. This is
deliberate: the live GUI instance holds the owner's real 16,620-photo
library and whatever they're interactively editing; a background cache
build sharing that session would risk fighting real edit-in-progress
state (see that same workflow's "real blocker" finding). A fresh headless
process per file has no such shared state -- it only ever touches the one
raw file it's given -- at the cost of paying its own ~3-4s process-startup
+ decode overhead per call instead of amortizing it across a batch; this
module cuts that cost down to roughly 1/max_workers wall-clock time by
rendering several files in parallel (they are, by construction,
independent processes, so parallelizing them is simply concurrent
subprocess execution, not a design risk).

The rendered cache lives in work/review_cache/ (a disposable intermediate,
same convention as work/denoised/ and work/stack_decoded/) -- never in
source/raw/, and never substitutes for a real export; it exists purely to
make the review console's thumbnails show real, judgeable detail.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from . import settings
from .paths import work_dir

LIGHTCRAFT_CLI_ENV = settings.LIGHTCRAFT_CLI_ENV
DEFAULT_LIGHTCRAFT_CLI = str(settings.LIGHTCRAFT_CLI_DEFAULT)

DEFAULT_LONG_EDGE = 1024
DEFAULT_QUALITY = 70
DEFAULT_MAX_WORKERS = 4


class LightCraftCliError(RuntimeError):
    """Raised when lightcraft-cli is missing, or a render/export step it
    ran reports ok: false in its own JSON-lines output."""


def _lightcraft_cli_binary() -> str:
    override = os.environ.get(LIGHTCRAFT_CLI_ENV)
    if override:
        return override
    on_path = shutil.which("lightcraft-cli")
    if on_path:
        return on_path
    if Path(DEFAULT_LIGHTCRAFT_CLI).exists():
        return DEFAULT_LIGHTCRAFT_CLI
    raise LightCraftCliError(
        f"lightcraft-cli not found on PATH or at {DEFAULT_LIGHTCRAFT_CLI} -- "
        f"install it or set {LIGHTCRAFT_CLI_ENV}"
    )


def review_cache_dir(folder: Path) -> Path:
    return work_dir(folder) / "review_cache"


def cache_path_for(folder: Path, raw_name: str) -> Path:
    return review_cache_dir(folder) / f"{Path(raw_name).stem}.jpg"


def render_preview_via_lightcraft(raw_path: Path, dest_path: Path, *,
                                   long_edge: int = DEFAULT_LONG_EDGE,
                                   quality: int = DEFAULT_QUALITY,
                                   timeout: float = 60.0,
                                   popen=subprocess.Popen) -> Path:
    """Auto-develop and export *raw_path* to *dest_path* via lightcraft-cli's
    own ephemeral headless `run` subcommand -- one process per call, no
    shared state with the live GUI instance (see module docstring).

    Raises LightCraftCliError if the binary is missing, either step's own
    JSON-lines output reports ok: false, the process times out, or it
    exits non-zero.
    """
    binary = _lightcraft_cli_binary()
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        binary, "run", "--import", str(raw_path),
        "develop.auto",
        "app.export", f"path={dest_path}", f"longEdge={long_edge}", f"quality={quality}",
    ]
    proc = popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        output_text, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate()
        raise LightCraftCliError(f"lightcraft-cli timed out after {timeout}s on {raw_path}")

    if proc.returncode != 0:
        raise LightCraftCliError(f"lightcraft-cli exited {proc.returncode} on {raw_path}:\n{output_text}")

    for line in output_text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if not record.get("ok", False):
            raise LightCraftCliError(
                f"lightcraft-cli step {record.get('command')!r} failed on {raw_path}: "
                f"{record.get('error')}"
            )

    if not dest_path.exists():
        raise LightCraftCliError(f"lightcraft-cli reported success but did not write {dest_path}")
    return dest_path


def build_folder_cache(folder: Path, raw_names: list[str], *,
                        cache_dir: Path | None = None,
                        max_workers: int = DEFAULT_MAX_WORKERS,
                        long_edge: int = DEFAULT_LONG_EDGE,
                        quality: int = DEFAULT_QUALITY,
                        render_fn=render_preview_via_lightcraft) -> dict:
    """Render an auto-adjusted cache thumbnail for every raw in *raw_names*
    (basenames, matching folder_assets()'s own identifier) that doesn't
    already have one, in parallel -- each render is an independent
    subprocess (see render_preview_via_lightcraft's docstring), so
    parallelizing them is just concurrent subprocess execution, not a
    shared-state risk.

    One failing raw does not abort the batch: its error is collected, the
    rest still render. Returns {rendered: [...], skipped: [...],
    failed: [{file, error}]} -- file names in each list, never Path
    objects, so this is directly JSON-serializable for the console's API.
    """
    folder = Path(folder)
    cache_dir = Path(cache_dir) if cache_dir is not None else review_cache_dir(folder)
    cache_dir.mkdir(parents=True, exist_ok=True)

    to_render = []
    skipped = []
    for name in raw_names:
        dest = cache_dir / f"{Path(name).stem}.jpg"
        if dest.exists():
            skipped.append(name)
        else:
            to_render.append(name)

    rendered: list[str] = []
    failed: list[dict] = []
    if to_render:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(render_fn, folder / "source" / "raw" / name, cache_dir / f"{Path(name).stem}.jpg",
                            long_edge=long_edge, quality=quality): name
                for name in to_render
            }
            for future in as_completed(futures):
                name = futures[future]
                try:
                    future.result()
                except LightCraftCliError as exc:
                    failed.append({"file": name, "error": str(exc)})
                else:
                    rendered.append(name)

    return {"rendered": rendered, "skipped": skipped, "failed": failed}
