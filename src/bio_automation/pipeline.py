from __future__ import annotations

import contextlib
import shutil
from dataclasses import asdict, replace
from pathlib import Path

from .daclip import load_session
from .exif import build_exif_sidecar, render_xmp
from .focus_stack import run_focus_stack as focus_stack_run_batch
from .grouping import DEFAULT_THRESHOLD, SubjectMismatchError, check_subject_consistency
from .io import read_json, read_jsonl, write_json, write_jsonl
from .models import Asset, Observation
from .paths import export_dir, metadata_dir, review_dir, source_raw_dir, work_dir
from .preclean import decode_raw_frames_to_tiff, denoise_stack_frames
from .render import _validate_crop_numbers, export_inat_jpeg, export_stock_jpeg, render_review_jpeg
from .stock import build_keywords
from .topaz import ModelCache
from .zerene import run_batch as zerene_run_batch

# Stacking-engine dispatch (see stack_focus()). "focus-stack" is the default
# engine (2026-10-19): Zerene's own multi-file batch loader is confirmed
# (decompiled, see zerene.py's module docstring) to never get past the first
# frame on this machine, so PMax never actually runs there. "zerene" stays
# reachable as an explicit, opt-in alternate -- it is not deleted, just no
# longer the default. This is a dispatch choice between two whole engines,
# distinct from (and unrelated to) zerene.py's own METHOD_TASK_CODES, which
# selects a task *within* Zerene's batch API (currently only "pmax" exists
# there) -- that internal choice is now fixed to "pmax" by this module
# whenever the "zerene" engine is selected, since it is the only documented
# code zerene.py supports.
STACK_ENGINES = ("focus-stack", "zerene")


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


def validate_crop(crop: list[float]) -> None:
    """Raise ValueError unless *crop* is [x0, y0, x1, y1]: four finite
    numbers in [0, 1] with x0 < x1 and y0 < y1."""
    _validate_crop_numbers(crop)
    x0, y0, x1, y1 = crop
    if not all(0 <= v <= 1 for v in crop):
        raise ValueError(f"crop values must be within [0, 1], got {crop!r}")
    if not (x0 < x1 and y0 < y1):
        raise ValueError(f"crop must satisfy x0 < x1 and y0 < y1, got {crop!r}")


def set_asset_crop(folder: Path, file_name: str, crop: list[float] | None) -> Asset:
    """Set (or clear, when *crop* is None) the non-destructive crop rect
    (see models.Asset.crop) on exactly one asset in *folder*, identified by
    its file path relative to *folder* or just its basename -- same
    name-matching convention as set_asset_status(). Unlike set_asset_status
    (which silently no-ops on a name that matches nothing, since it can
    target several files at once), this raises ValueError when *file_name*
    matches no asset: the review console calls this for one specific file a
    user is looking at, so a typo'd/stale name must fail loud, not silently
    do nothing.
    """
    if crop is not None:
        validate_crop(crop)
    assets = scan_assets(folder)
    wanted = file_name.replace("\\", "/")
    matched: Asset | None = None
    for asset in assets:
        if asset.file == wanted or asset.file.split("/")[-1] == wanted:
            asset.crop = crop
            matched = asset
    if matched is None:
        raise ValueError(f"no asset matches {file_name!r} in {folder}")
    save_assets(folder, assets)
    return matched


def discover_source_files(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    files = [path for path in source.rglob("*") if path.is_file()]
    return sorted(files)


def discover_observation_folders(base_dir: Path) -> list[Path]:
    """Find every observation folder under *base_dir*.

    An observation folder is, by this repo's layout convention (see
    make_observation_path()/metadata_dir() in paths.py), any directory that
    holds a metadata/assets.jsonl -- that file's parent's parent, found by
    walking the whole tree rather than assuming a fixed nesting depth, since
    make_observation_path() nests by base_dir/<year>/<obsdate>/obs-....
    """
    base_dir = Path(base_dir)
    return sorted(
        {
            assets_file.parent.parent
            for assets_file in base_dir.rglob("assets.jsonl")
            if assets_file.parent.name == "metadata"
        }
    )


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
        work_dir(folder),
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
    overrides = {field: fetched[field] for field in INAT_OWNED_FIELDS if field in fetched}
    merged = replace(existing, **overrides)
    merged.keywords_seed = sorted(set(existing.keywords_seed) | set(fetched.get("keywords_seed", [])))
    return merged


def _kept_original_raws(folder: Path) -> list[Asset]:
    """Original raws marked 'kept' by `bio keep` -- the only assets
    write_sidecars/render_review/export_inat/export_stock operate on, per
    docs/folder-metadata-schema.md's workflow ordering (cull, THEN sidecar
    write/export). A raw still sitting at 'ingested' (cull never ran on it)
    is deliberately excluded, not treated as implicitly kept.
    """
    return [a for a in scan_assets(folder) if a.role == "original" and a.status == "kept"]


def write_sidecars(folder: Path) -> list[Path]:
    """Write metadata/<stem>.exif.json and metadata/<stem>.xmp for every
    kept raw in source/raw/, one pair per raw (an observation folder can
    hold multiple kept raws, so a single shared sidecar file would collide).
    """
    written: list[Path] = []
    for asset in _kept_original_raws(folder):
        raw_path = folder / asset.file
        stem = raw_path.stem
        sidecar = build_exif_sidecar(raw_path, folder)

        exif_path = metadata_dir(folder) / f"{stem}.exif.json"
        write_json(exif_path, sidecar)
        written.append(exif_path)

        xmp_path = metadata_dir(folder) / f"{stem}.xmp"
        xmp_path.write_text(render_xmp(sidecar), encoding="utf-8")
        written.append(xmp_path)
    return written


def _render_derivatives(folder: Path, render_fn, dest_dir: Path, role: str,
                         status: str) -> list[Path]:
    """Shared body for render_review/export_inat/export_stock: run
    *render_fn* over every kept raw into *dest_dir*, upsert a matching
    Asset row (role, status) for each (saved per file), return the written paths.

    Each kept raw's own Asset.crop (set by the review console, see
    review_console.py; None means uncropped) is forwarded to *render_fn* as
    its crop= kwarg, so a reviewer's non-destructive crop is reflected in
    every derivative without ever touching the original raw file.
    """
    written: list[Path] = []
    for asset in _kept_original_raws(folder):
        raw_path = folder / asset.file
        dest_path = dest_dir / f"{raw_path.stem}.jpg"
        render_fn(raw_path, dest_path, crop=asset.crop)
        written.append(dest_path)
        # Upsert by (file, role) and save per file: a re-run never duplicates
        # a row, and a failure on a later file keeps the rows already written.
        rel = str(dest_path.relative_to(folder)).replace("\\", "/")
        assets = [a for a in scan_assets(folder) if not (a.file == rel and a.role == role)]
        assets.append(Asset(file=rel, role=role, status=status, selected=None))
        save_assets(folder, assets)
    return written


def render_review(folder: Path) -> list[Path]:
    """Render review/<stem>.jpg proxies for every kept raw, to drive the
    keep/reject decision. Decoupled from where the raw itself lives -- see
    Workflow step 2 (cull).
    """
    return _render_derivatives(folder, render_review_jpeg, review_dir(folder),
                                role="review", status="generated")


def export_inat(folder: Path) -> list[Path]:
    """Render export/inat/<stem>.jpg upload-ready copies for every kept raw."""
    return _render_derivatives(folder, export_inat_jpeg, export_dir(folder, "inat"),
                                role="upload", status="ready")


def export_stock(folder: Path) -> list[Path]:
    """Render export/stock/<stem>.jpg stock-ready derivatives for every
    kept raw.
    """
    return _render_derivatives(folder, export_stock_jpeg, export_dir(folder, "stock"),
                                role="stock", status="ready")


def _run_subject_consistency_gate(raw_paths: list[Path], threshold: float = DEFAULT_THRESHOLD) -> None:
    """Hard safety gate: refuse to hand *raw_paths* to Zerene unless every
    frame's DA-CLIP content embedding is close enough to the batch's
    reference frame (see grouping.check_subject_consistency). Called from
    stack_focus() when check_subjects=True (the default); see its
    docstring for the incident that motivated this.

    Design decision: if the DA-CLIP model itself is missing,
    daclip.load_session() raises FileNotFoundError -- a real possibility
    on a machine that has not downloaded the model. This re-raises that as
    a clearer, loud failure instead of silently skipping the safety check:
    a machine that cannot run the gate must not launch an expensive,
    possibly disk-filling Zerene run as if the gate had passed.
    check_subjects=False on stack_focus()/clean_and_stack() is the
    explicit, opt-in-to-disable override for a deliberate bypass of this
    whole gate (including the "model missing" case).
    """
    try:
        session = load_session()
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Subject-consistency check unavailable: {exc}. Refusing to proceed "
            "to Zerene without it -- pass check_subjects=False to explicitly "
            "skip this safety gate."
        ) from exc
    outliers = check_subject_consistency(raw_paths, session, threshold=threshold)
    if outliers:
        raise SubjectMismatchError(outliers, threshold)


def stack_focus(folder: Path, method: str = "focus-stack", timeout: float = 1800.0,
                 source_dir: Path | None = None, min_free_gb: float = 5.0,
                 check_subjects: bool = True) -> Asset:
    """Run a focus-stacking engine over *source_dir* (default: source/raw/),
    move the merged output into work/, and append a stack_output Asset row.

    *method* picks the stacking ENGINE, not a task within it: "focus-stack"
    (default, see STACK_ENGINES above) runs the real focus-stack CLI binary
    via focus_stack.run_focus_stack(); "zerene" runs Zerene Stacker's
    headless batch API via zerene.run_batch() with its only documented task
    code (PMax). Any other value raises ValueError.

    *source_dir* lets clean_and_stack() point the engine at a denoised
    intermediate instead of the raw NEFs directly; callers that pass it
    are responsible for that directory already holding only the kept
    bracket frames for one focus stack. With the default, this assumes
    cull/delete_rejected_raws already ran. Both engines write their merged
    image into *source_dir* itself (see each engine module's docstring for
    why); it is moved into work/ immediately after so raw/ keeps holding
    only camera originals, per the "originals stay immutable" principle in
    docs/folder-metadata-schema.md. A batch script Zerene reads from (and
    leaves behind in) *source_dir* is removed once the run succeeds -- a
    focus-stack run never writes one, so this cleanup is a no-op for that
    engine.

    *check_subjects* (default True) runs grouping.check_subject_consistency()
    over the files in *source_dir* before handing off to the stacking
    engine, and raises SubjectMismatchError if any frame looks like a
    different subject -- see _run_subject_consistency_gate for why this
    defaults on and fails loud rather than open. Pass False only to
    explicitly bypass it for a deliberate override; there is no separate
    flag to silently skip just the "model missing" case.

    With the default *source_dir* (None, i.e. source/raw/ itself), the raw
    frames are decoded to 16-bit TIFFs first (preclean.decode_raw_frames_to_tiff,
    into a disposable work/stack_decoded/) before either engine ever sees
    them -- neither focus-stack nor Zerene is a raw decoder, and handing one
    a .NEF directly was confirmed (2026-10-19) to silently produce a tiny,
    wrong output instead of erroring. When *source_dir* is passed explicitly
    (clean_and_stack()'s denoised intermediate), it is assumed already
    decoded and is used as-is, same as before.
    """
    folder = Path(folder)
    using_raw_source = source_dir is None
    raw_dir = Path(source_dir) if source_dir is not None else source_raw_dir(folder)

    if method not in STACK_ENGINES:
        raise ValueError(f"Unknown stacking method {method!r}; choose from {STACK_ENGINES}")

    # Excludes a leftover ZereneBatch.xml from a prior crashed Zerene run
    # (see the cleanup below) -- it is that engine's own bookkeeping file,
    # never a frame, so it must not be fed to the DA-CLIP embedder or the
    # raw decoder.
    raw_paths = sorted(
        p for p in raw_dir.iterdir() if p.is_file() and p.name != "ZereneBatch.xml"
    )
    if check_subjects:
        _run_subject_consistency_gate(raw_paths)

    decoded_dir = work_dir(folder) / "stack_decoded" if using_raw_source else None
    engine_dir = raw_dir
    if decoded_dir is not None:
        decode_raw_frames_to_tiff(raw_paths, decoded_dir)
        engine_dir = decoded_dir

    try:
        if method == "focus-stack":
            output_path = focus_stack_run_batch(engine_dir, timeout=timeout, min_free_gb=min_free_gb)
        else:
            output_path = zerene_run_batch(engine_dir, method="pmax", timeout=timeout, min_free_gb=min_free_gb)
    except Exception:
        if decoded_dir is not None and decoded_dir.exists():
            shutil.rmtree(decoded_dir)
        raise

    dest_dir = work_dir(folder)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / output_path.name
    output_path.replace(dest_path)

    if decoded_dir is not None and decoded_dir.exists():
        shutil.rmtree(decoded_dir)

    batch_script = raw_dir / "ZereneBatch.xml"
    if batch_script.exists():
        batch_script.unlink()

    asset = Asset(
        file=str(dest_path.relative_to(folder)).replace("\\", "/"),
        role="stack_output",
        status="ready",
        selected=True,
    )
    assets = scan_assets(folder)
    assets.append(asset)
    save_assets(folder, assets)
    return asset


def clean_and_stack(folder: Path, noise: float = 0.0, exposure_ev: float = 0.0,
                     subject_boost_ev: float = 0.0,
                     method: str = "focus-stack", timeout: float = 1800.0,
                     min_free_gb: float = 5.0, check_subjects: bool = True,
                     cache: ModelCache | None = None) -> Asset:
    """Denoise every raw in source/raw/ with one shared parameter set (see
    preclean.denoise_stack_frames: raw denoise, then levels, in that fixed
    order) into work/denoised/, then run the stacking engine named by
    *method* (see stack_focus()'s docstring; default "focus-stack") against
    that denoised set instead of the raw NEFs. Fixed params across the whole
    stack keep the engine's cross-frame contrast comparison reading real
    focus differences, not per-frame cleanup variance.

    Assumes cull/delete_rejected_raws already ran, like stack_focus().

    *check_subjects*: the subject-consistency gate runs FIRST, against the
    raw files in source/raw/, before denoise_stack_frames() is even called
    -- denoising does not change which subject is in frame, so checking the
    cheap-to-read raws catches a mismatch before paying the real cost of
    denoising every frame (per-frame OpenVINO inference, large TIFF writes
    to work/denoised/). stack_focus() is then called with
    check_subjects=False, since the batch has already been verified
    consistent; it does not re-check the denoised intermediates.

    *subject_boost_ev* is forwarded to preclean.denoise_stack_frames() as-is
    (default 0.0, no mask model call -- see its docstring for the fixed-
    across-the-stack masking it does when non-zero).

    *cache* is forwarded to preclean.denoise_stack_frames() as-is. Left at
    the default None, behavior is unchanged from before this parameter
    existed: denoise_stack_frames() builds its own ModelCache for this call
    and tears it down when done. Passing a ModelCache built by the caller
    is purely additive -- it lets a future batch orchestrator share one
    already-loaded ModelCache across many clean_and_stack() calls (e.g. one
    per observation folder) instead of reloading the denoise models from
    disk every time.

    work/denoised/ holds the per-frame denoised TIFFs fed to the stacking
    engine -- large, disposable intermediates once stacking is done. On a
    successful stack (stack_focus() returns its Asset without raising), that
    directory and its contents are deleted; only the final stack_output
    asset in work/ is kept. On failure -- an exception from
    denoise_stack_frames() or from stack_focus() (whichever engine's error,
    FocusStackError or ZereneError) -- any partial work/denoised/ output is
    cleaned up the same way before the original exception propagates
    unchanged, so a failed batch item does not leave large orphaned files
    behind either.
    """
    folder = Path(folder)
    raw_dir = source_raw_dir(folder)
    raw_paths = sorted(p for p in raw_dir.iterdir() if p.is_file())

    if check_subjects:
        _run_subject_consistency_gate(raw_paths)

    denoised_dir = work_dir(folder) / "denoised"
    try:
        denoise_stack_frames(raw_paths, denoised_dir, noise=noise, exposure_ev=exposure_ev,
                              subject_boost_ev=subject_boost_ev,
                              cache=cache, min_free_gb=min_free_gb)
        asset = stack_focus(folder, method=method, timeout=timeout, source_dir=denoised_dir,
                             min_free_gb=min_free_gb, check_subjects=False)
    except Exception:
        if denoised_dir.exists():
            shutil.rmtree(denoised_dir)
        raise
    if denoised_dir.exists():
        shutil.rmtree(denoised_dir)
    return asset


def stack_all(base_dir: Path, noise: float = 0.0, exposure_ev: float = 0.0,
              subject_boost_ev: float = 0.0,
              method: str = "focus-stack", timeout: float = 1800.0,
              min_free_gb: float = 5.0, check_subjects: bool = True) -> list[dict]:
    """Run cull + clean_and_stack() over every observation folder under
    *base_dir* (see discover_observation_folders()), sharing one ModelCache
    across the whole batch so denoise models load once instead of once per
    folder. *method* picks the stacking engine ("focus-stack" default,
    "zerene" alternate -- see stack_focus()'s docstring) used for every
    folder in the batch.

    One bad folder (a stacking-engine failure, a subject-consistency
    mismatch, a disk-space guard tripping, or anything else
    clean_and_stack() can raise) does not abort the batch: the exception is
    caught per folder and recorded, and the next folder still runs.

    Returns a list of one dict per folder --
    {folder, status: "ok"|"failed", error: str|None, asset: str|None} -- in
    the same order discover_observation_folders() returned them. The same
    list is also written as indented, human-readable JSON to
    base_dir/"stack_batch_manifest.json" so a batch run leaves a durable,
    inspectable record of which folders failed without the caller having to
    track that itself.
    """
    base_dir = Path(base_dir)
    folders = discover_observation_folders(base_dir)
    results: list[dict] = []
    with ModelCache() as cache:
        for folder in folders:
            try:
                delete_rejected_raws(folder)
                asset = clean_and_stack(
                    folder, noise=noise, exposure_ev=exposure_ev,
                    subject_boost_ev=subject_boost_ev, method=method,
                    timeout=timeout, min_free_gb=min_free_gb,
                    check_subjects=check_subjects, cache=cache,
                )
            except Exception as exc:
                results.append({
                    "folder": str(folder),
                    "status": "failed",
                    "error": str(exc),
                    "asset": None,
                })
            else:
                results.append({
                    "folder": str(folder),
                    "status": "ok",
                    "error": None,
                    "asset": asset.file,
                })
    write_json(base_dir / "stack_batch_manifest.json", results)
    return results


def build_and_save_keywords(folder: Path) -> dict:
    """Build metadata/keywords.json from the observation record via
    stock.build_keywords() and write it to disk. One record per observation
    folder, not per asset.
    """
    observation = load_observation(folder)
    keywords = build_keywords(observation)
    write_json(metadata_dir(folder) / "keywords.json", keywords)
    return keywords


def process_observation(folder: Path, *, method: str = "focus-stack", timeout: float = 1800.0,
                         min_free_gb: float = 5.0, check_subjects: bool = True,
                         denoise: bool = False, noise: float = 0.0, exposure_ev: float = 0.0,
                         subject_boost_ev: float = 0.0, cache: ModelCache | None = None) -> dict:
    """Run the mechanical, no-judgment-required chain for one observation
    folder that a human has already ingested and curated (kept/rejected via
    `bio keep`/`bio reject`, cull via `bio cull` already applied) and
    optionally linked to iNat (via `bio fetch-inat`):

        stack -> write_sidecars -> render_review -> export_inat (if linked)
        -> export_stock -> build_and_save_keywords

    This is the ONLY new auto-chaining this orchestrator adds. It
    deliberately does NOT call ingest/copy_into_raw, delete_rejected_raws
    (cull), set_asset_status (reject/keep), or fetch_inat_observation --
    those stay separate, deliberate, human-triggered steps (see cli.py);
    folding them in here would mean this function was silently making a
    culling or iNat-linking decision for the user. Callers are responsible
    for having already run `bio cull`/`bio reject`/`bio keep` (and, if
    applicable, `bio fetch-inat`) on *folder* before calling this.

    Stacking dispatch mirrors `bio stack`'s own --denoise switch: with
    denoise=False (the default), stack_focus() runs directly against
    source/raw/; with denoise=True, clean_and_stack() runs first (shared
    noise/exposure_ev/subject_boost_ev/cache params, same meaning as
    clean_and_stack()'s own). method/timeout/min_free_gb/check_subjects are
    forwarded to whichever one runs, same as cmd_stack().

    export_inat is skipped -- not treated as an error, not run with zero
    files -- when the observation has no linked iNat observation. "Linked"
    is exactly the signal `bio fetch-inat` writes and cmd_fetch_inat reads:
    Observation.observation_id is not None (see models.Observation and
    merge_inat_fields(), which only ever sets it from a real
    fetch_inat_observation() payload). A folder that was ingested but never
    fetch-inat'd has observation_id=None from Observation's own default, so
    this is a real, already-existing business rule, not a new inference.

    Any stage raising -- FocusStackError/ZereneError/SubjectMismatchError
    from stacking, or anything else -- propagates immediately and
    unchanged: no later stage in the chain runs, and this function returns
    nothing (an exception, not a partial dict) on that path, so a caller
    can never mistake a partial run for success.

    Returns a dict on success: {folder: str, stack_asset: str,
    sidecars: list[str], review: list[str],
    inat: list[str] | None (None means "skipped, not iNat-linked"),
    stock: list[str], keywords: dict}.
    """
    folder = Path(folder)

    if denoise:
        stack_asset = clean_and_stack(
            folder, noise=noise, exposure_ev=exposure_ev, subject_boost_ev=subject_boost_ev,
            method=method, timeout=timeout, min_free_gb=min_free_gb,
            check_subjects=check_subjects, cache=cache,
        )
    else:
        stack_asset = stack_focus(
            folder, method=method, timeout=timeout, min_free_gb=min_free_gb,
            check_subjects=check_subjects,
        )

    sidecars = write_sidecars(folder)
    review = render_review(folder)

    observation = load_observation(folder)
    inat = export_inat(folder) if observation.observation_id is not None else None

    stock = export_stock(folder)
    keywords = build_and_save_keywords(folder)

    return {
        "folder": str(folder),
        "stack_asset": stack_asset.file,
        "sidecars": [str(p) for p in sidecars],
        "review": [str(p) for p in review],
        "inat": None if inat is None else [str(p) for p in inat],
        "stock": [str(p) for p in stock],
        "keywords": keywords,
    }


def process_all(base_dir: Path, *, method: str = "focus-stack", timeout: float = 1800.0,
                 min_free_gb: float = 5.0, check_subjects: bool = True,
                 denoise: bool = False, noise: float = 0.0, exposure_ev: float = 0.0,
                 subject_boost_ev: float = 0.0) -> list[dict]:
    """Run process_observation() over every observation folder under
    *base_dir*, reusing discover_observation_folders() -- the same
    folder-discovery stack_all() already uses -- rather than a second
    implementation of it.

    Mirrors stack_all()'s own batch contract: one bad folder (any exception
    from process_observation(), including a mid-chain stage failure) does
    not abort the batch -- it is caught per folder and recorded, and the
    next folder still runs. Results are a list of
    {folder, status: "ok"|"failed", error: str|None, result: dict|None} in
    discover_observation_folders() order, also written as indented JSON to
    base_dir/"process_batch_manifest.json" (a distinct filename from
    stack_all()'s own "stack_batch_manifest.json" so the two batch
    commands' manifests never collide on disk).

    When denoise=True, one ModelCache is built and shared across the whole
    batch (same reuse stack_all() does for clean_and_stack(), so the
    denoise models load once instead of once per folder); when denoise is
    False (the default -- stack_focus() does not use a ModelCache at all),
    no ModelCache is built.
    """
    base_dir = Path(base_dir)
    folders = discover_observation_folders(base_dir)
    results: list[dict] = []
    cache_scope = ModelCache() if denoise else contextlib.nullcontext(None)
    with cache_scope as cache:
        for folder in folders:
            try:
                result = process_observation(
                    folder, method=method, timeout=timeout, min_free_gb=min_free_gb,
                    check_subjects=check_subjects, denoise=denoise, noise=noise,
                    exposure_ev=exposure_ev, subject_boost_ev=subject_boost_ev, cache=cache,
                )
            except Exception as exc:
                results.append({
                    "folder": str(folder),
                    "status": "failed",
                    "error": str(exc),
                    "result": None,
                })
            else:
                results.append({
                    "folder": str(folder),
                    "status": "ok",
                    "error": None,
                    "result": result,
                })
    write_json(base_dir / "process_batch_manifest.json", results)
    return results
