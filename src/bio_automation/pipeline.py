from __future__ import annotations

import shutil
from dataclasses import asdict
from pathlib import Path

from .daclip import load_session
from .grouping import DEFAULT_THRESHOLD, SubjectMismatchError, check_subject_consistency
from .io import read_json, read_jsonl, write_json, write_jsonl
from .models import Asset, Observation
from .paths import export_dir, metadata_dir, review_dir, source_raw_dir, work_dir
from .preclean import denoise_stack_frames
from .topaz import ModelCache
from .zerene import run_batch as zerene_run_batch


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


def stack_focus(folder: Path, method: str = "pmax", timeout: float = 1800.0,
                 source_dir: Path | None = None, min_free_gb: float = 5.0,
                 check_subjects: bool = True) -> Asset:
    """Run Zerene Stacker (headless batch) over *source_dir* (default:
    source/raw/), move the merged output into work/, and append a
    stack_output Asset row.

    *source_dir* lets clean_and_stack() point Zerene at a denoised
    intermediate instead of the raw NEFs directly; callers that pass it
    are responsible for that directory already holding only the kept
    bracket frames for one focus stack. With the default, this assumes
    cull/delete_rejected_raws already ran. Zerene's
    OutputImageDispositionCode=2 writes the merged image into *source_dir*
    itself (the only documented "save in source folder" option); it is
    moved into work/ immediately after so raw/ keeps holding only camera
    originals, per the "originals stay immutable" principle in
    docs/folder-metadata-schema.md. The batch script Zerene reads from (and
    leaves behind in) *source_dir* is removed once the run succeeds.

    *check_subjects* (default True) runs grouping.check_subject_consistency()
    over the files in *source_dir* before handing off to Zerene, and raises
    SubjectMismatchError if any frame looks like a different subject --
    see _run_subject_consistency_gate for why this defaults on and fails
    loud rather than open. Pass False only to explicitly bypass it for a
    deliberate override; there is no separate flag to silently skip just
    the "model missing" case.
    """
    folder = Path(folder)
    raw_dir = Path(source_dir) if source_dir is not None else source_raw_dir(folder)

    if check_subjects:
        # Excludes a leftover ZereneBatch.xml from a prior crashed run (see
        # the cleanup below) -- it is this module's own bookkeeping file,
        # never a frame, so it must not be fed to the DA-CLIP embedder.
        raw_paths = sorted(
            p for p in raw_dir.iterdir() if p.is_file() and p.name != "ZereneBatch.xml"
        )
        _run_subject_consistency_gate(raw_paths)

    output_path = zerene_run_batch(raw_dir, method=method, timeout=timeout, min_free_gb=min_free_gb)

    dest_dir = work_dir(folder)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / output_path.name
    output_path.replace(dest_path)

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


def clean_and_stack(folder: Path, noise: float = 1.0, exposure_ev: float = 0.0,
                     method: str = "pmax", timeout: float = 1800.0,
                     min_free_gb: float = 5.0, check_subjects: bool = True,
                     cache: ModelCache | None = None) -> Asset:
    """Denoise every raw in source/raw/ with one shared parameter set (see
    preclean.denoise_stack_frames: raw denoise, then levels, in that fixed
    order) into work/denoised/, then run Zerene against that denoised set
    instead of the raw NEFs. Fixed params across the whole stack keep
    Zerene's cross-frame contrast comparison reading real focus
    differences, not per-frame cleanup variance.

    Assumes cull/delete_rejected_raws already ran, like stack_focus().

    *check_subjects* is forwarded to stack_focus(): the subject-consistency
    gate runs against work/denoised/ (the files actually handed to Zerene),
    not the pre-denoise raws, since denoising does not change which
    subject is in frame. See stack_focus()'s docstring.

    *cache* is forwarded to preclean.denoise_stack_frames() as-is. Left at
    the default None, behavior is unchanged from before this parameter
    existed: denoise_stack_frames() builds its own ModelCache for this call
    and tears it down when done. Passing a ModelCache built by the caller
    is purely additive -- it lets a future batch orchestrator share one
    already-loaded ModelCache across many clean_and_stack() calls (e.g. one
    per observation folder) instead of reloading the denoise models from
    disk every time.

    work/denoised/ holds the per-frame denoised TIFFs fed to Zerene -- large,
    disposable intermediates once stacking is done. On a successful stack
    (stack_focus() returns its Asset without raising), that directory and
    its contents are deleted; only the final stack_output asset in work/ is
    kept. On failure -- an exception from denoise_stack_frames() or from
    stack_focus()/zerene_run_batch() -- any partial work/denoised/ output is
    cleaned up the same way before the original exception propagates
    unchanged, so a failed batch item does not leave large orphaned files
    behind either.
    """
    folder = Path(folder)
    raw_dir = source_raw_dir(folder)
    raw_paths = sorted(p for p in raw_dir.iterdir() if p.is_file())
    denoised_dir = work_dir(folder) / "denoised"
    try:
        denoise_stack_frames(raw_paths, denoised_dir, noise=noise, exposure_ev=exposure_ev,
                              cache=cache, min_free_gb=min_free_gb)
        asset = stack_focus(folder, method=method, timeout=timeout, source_dir=denoised_dir,
                             min_free_gb=min_free_gb, check_subjects=check_subjects)
    except Exception:
        if denoised_dir.exists():
            shutil.rmtree(denoised_dir)
        raise
    if denoised_dir.exists():
        shutil.rmtree(denoised_dir)
    return asset


def stack_all(base_dir: Path, noise: float = 1.0, exposure_ev: float = 0.0,
              method: str = "pmax", timeout: float = 1800.0,
              min_free_gb: float = 5.0, check_subjects: bool = True) -> list[dict]:
    """Run cull + clean_and_stack() over every observation folder under
    *base_dir* (see discover_observation_folders()), sharing one ModelCache
    across the whole batch so denoise models load once instead of once per
    folder.

    One bad folder (a Zerene failure, a subject-consistency mismatch, a
    disk-space guard tripping, or anything else clean_and_stack() can raise)
    does not abort the batch: the exception is caught per folder and
    recorded, and the next folder still runs.

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
                    folder, noise=noise, exposure_ev=exposure_ev, method=method,
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
    raise NotImplementedError("build_and_save_keywords will be added next.")
