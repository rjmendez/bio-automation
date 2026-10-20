"""Pre-cull quality triage.

Reads a folder's still-``ingested`` original raws (the ones `bio cull` has not
yet decided), scores each for sharpness, exposure clipping, and near-duplicate
frames within the same folder, and writes a proposal --
``presort_suggestions.json`` in the observation folder -- for a human to
review. It never decides anything itself.

This is a hard rule, not a style preference: the owner's instruction was
"pre-sort them, do not cull yourself." Concretely, this module (and the
`bio presort` CLI command built on it):

- never imports or calls `pipeline.set_asset_status` or
  `pipeline.delete_rejected_raws` (see the import block below -- neither name
  appears in it);
- never calls `pipeline.save_assets` or otherwise writes to `assets.jsonl`;
- only ever reads assets via `pipeline.scan_assets` and writes its own,
  separate `presort_suggestions.json`.

A human applies the actual decision afterwards via the existing `bio reject`/
`bio keep` CLI commands -- see tests/test_quality.py's
TestNeverMutatesAssetStatus for the recording-stub proof.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import rawpy
from PIL import Image

from .io import write_json
from .pipeline import scan_assets

PRESORT_FILENAME = "presort_suggestions.json"

# Downsample target (long edge, pixels) for quality signals. Full-resolution
# decode is unnecessary for blur/exposure/dup-hash heuristics and would make
# `bio presort` slow over a whole folder; half_size=True in _decode_preview_rgb
# plus this further PIL downsample keeps it fast.
_WORK_MAX_DIM = 768

# dHash: resize to (_DHASH_N + 1) x _DHASH_N, compare each row's adjacent
# columns left->right -> _DHASH_N * _DHASH_N bits (64 for the default 8).
_DHASH_N = 8

# Hamming distance (out of 64 bits) at or below which two frames in the same
# folder count as near-duplicates. Byte-identical frames hash to distance 0;
# see tests/test_quality.py for the measured distance on genuinely different
# synthetic frames (far above this), which is what grounds the cutoff.
_NEAR_DUP_HAMMING_MAX = 6

# A frame scores as a blur cull-candidate when its sharpness falls below this
# fraction of the folder's own median sharpness -- relative to the folder, not
# an absolute threshold, since "sharp" depends on the subject/lens/lighting of
# the specific shoot.
_BLUR_RATIO_THRESHOLD = 0.35

# A frame scores as an exposure cull-candidate when at least this fraction of
# its pixels sit at/near the shadow or highlight clip point.
_CLIPPED_FRACTION_THRESHOLD = 0.15


def sharpness_score(gray: np.ndarray) -> float:
    """Laplacian-variance-style sharpness proxy, numpy only (no opencv): the
    variance of a discrete 2D Laplacian response. A perfectly flat region
    (no edges anywhere) has a zero Laplacian at every pixel and so a variance
    of exactly 0.0; real detail/edges push individual Laplacian values away
    from 0, raising the variance. *gray* is a single-channel array (any real
    dtype); edge pixels are replicated (``mode="edge"``) so the kernel is
    well-defined at the border without inventing new values.
    """
    gray = gray.astype(np.float32, copy=False)
    padded = np.pad(gray, 1, mode="edge")
    laplacian = (
        padded[0:-2, 1:-1]   # up
        + padded[2:, 1:-1]   # down
        + padded[1:-1, 0:-2]  # left
        + padded[1:-1, 2:]    # right
        - 4.0 * padded[1:-1, 1:-1]  # center
    )
    return float(laplacian.var())


def exposure_clipping(gray: np.ndarray, shadow_thresh: int = 5,
                       highlight_thresh: int = 250) -> dict:
    """Fraction of pixels at/near the shadow clip point (<= *shadow_thresh*)
    and at/near the highlight clip point (>= *highlight_thresh*), out of an
    8-bit [0,255] *gray* array. Returns exact fractions (count / total), not
    a model estimate.
    """
    total = gray.size
    shadow_frac = float(np.count_nonzero(gray <= shadow_thresh)) / total
    highlight_frac = float(np.count_nonzero(gray >= highlight_thresh)) / total
    return {
        "shadow_clipped_fraction": shadow_frac,
        "highlight_clipped_fraction": highlight_frac,
    }


def difference_hash(gray_img: Image.Image, n: int = _DHASH_N) -> int:
    """Standard dHash: resize *gray_img* (any mode/size) to ((n+1) x n), then
    for each row set one bit per column pair (1 if the pixel to the right is
    brighter than the one to its left, else 0), giving an n*n-bit integer.
    Two frames with the same hash (distance 0 via hamming_distance) are
    pixel-pattern-identical at this resolution; nearby hashes are
    near-duplicates (e.g. consecutive burst frames)."""
    small = gray_img.convert("L").resize((n + 1, n), Image.BILINEAR)
    arr = np.asarray(small, dtype=np.int16)
    bits = arr[:, 1:] > arr[:, :-1]
    value = 0
    for bit in bits.flatten():
        value = (value << 1) | int(bit)
    return value


def hamming_distance(a: int, b: int) -> int:
    """Number of differing bits between two dHash integers."""
    return bin(a ^ b).count("1")


def _decode_preview_rgb(raw_path: Path) -> np.ndarray:
    """Fast, small RGB decode of *raw_path* for quality triage only --
    half-size 8-bit demosaic via rawpy (not preclean.py's full-resolution
    16-bit decode, which is tuned as denoise-model input, not a quick
    per-folder heuristic), downsampled further to _WORK_MAX_DIM on the long
    edge with PIL. no_auto_bright=True for the same reason preclean.py uses
    it: comparing sharpness/exposure across frames must not be skewed by
    LibRaw's own per-image auto-brightness.
    """
    with rawpy.imread(str(raw_path)) as raw:
        rgb8 = raw.postprocess(half_size=True, output_bps=8, use_camera_wb=True,
                                no_auto_bright=True)
    img = Image.fromarray(rgb8)
    img.thumbnail((_WORK_MAX_DIM, _WORK_MAX_DIM), Image.BILINEAR)
    return np.array(img)


def _compute_asset_metrics(raw_path: Path) -> dict:
    """Decode *raw_path* and compute its sharpness/exposure/dHash signals.
    The only function in this module that touches disk for a raw's pixels --
    isolated so suggest_culling()'s verdict assembly (_build_verdicts) can be
    tested with controlled numbers, and so tests can stub this one function
    instead of faking raw files."""
    rgb = _decode_preview_rgb(raw_path)
    gray_img = Image.fromarray(rgb).convert("L")
    gray = np.asarray(gray_img, dtype=np.float32)
    metrics = {"sharpness": sharpness_score(gray), "dhash": difference_hash(gray_img)}
    metrics.update(exposure_clipping(gray.astype(np.uint8)))
    return metrics


def _build_verdicts(file_metrics: dict[str, dict]) -> list[dict]:
    """Pure verdict assembly: given ``{file: {sharpness, dhash,
    shadow_clipped_fraction, highlight_clipped_fraction}}`` for every
    candidate (already decoded, in assets.jsonl order), return one verdict
    dict per file. No I/O.

    Near-duplicate pairing walks *file_metrics* in its given (dict
    insertion/assets.jsonl) order: the first frame in a near-duplicate run is
    always the one named by a later frame's "near-duplicate of" reason, never
    the reverse, and a frame already paired is not re-paired against a third.
    """
    files = list(file_metrics)
    if not files:
        return []

    median_sharp = float(np.median([file_metrics[f]["sharpness"] for f in files]))

    dup_of: dict[str, str] = {}
    for i, f_i in enumerate(files):
        if f_i in dup_of:
            continue
        for f_j in files[i + 1:]:
            if f_j in dup_of:
                continue
            if hamming_distance(file_metrics[f_i]["dhash"], file_metrics[f_j]["dhash"]) <= _NEAR_DUP_HAMMING_MAX:
                dup_of[f_j] = f_i

    verdicts = []
    for f in files:
        m = file_metrics[f]
        reasons: list[str] = []
        cull = False

        if median_sharp > 0 and m["sharpness"] < median_sharp * _BLUR_RATIO_THRESHOLD:
            reasons.append(
                f"blurry: sharpness {m['sharpness']:.2f} vs folder median {median_sharp:.2f}"
            )
            cull = True
        if m["shadow_clipped_fraction"] >= _CLIPPED_FRACTION_THRESHOLD:
            reasons.append(
                f"underexposed: {m['shadow_clipped_fraction'] * 100:.0f}% clipped shadows"
            )
            cull = True
        if m["highlight_clipped_fraction"] >= _CLIPPED_FRACTION_THRESHOLD:
            reasons.append(
                f"overexposed: {m['highlight_clipped_fraction'] * 100:.0f}% clipped highlights"
            )
            cull = True
        if f in dup_of:
            reasons.append(f"near-duplicate of {dup_of[f]}")
            cull = True
        if not reasons:
            reasons = ["sharp, well-exposed, and not a near-duplicate"]

        verdicts.append({
            "file": f,
            "verdict": "cull_candidate" if cull else "keep_candidate",
            "reasons": reasons,
            "sharpness": m["sharpness"],
            "shadow_clipped_fraction": m["shadow_clipped_fraction"],
            "highlight_clipped_fraction": m["highlight_clipped_fraction"],
            "near_duplicate_of": dup_of.get(f),
        })
    return verdicts


def suggest_culling(folder: Path) -> dict:
    """Read *folder*'s still-``ingested`` original raws, score each, and
    write the resulting proposal to ``folder/presort_suggestions.json`` --
    never assets.jsonl, and never via set_asset_status/delete_rejected_raws
    (see module docstring). Returns the same dict that was written.

    Only ``role == "original"`` assets at ``status == "ingested"`` are
    considered -- a raw cull has already decided (``kept``/``rejected``) is
    left alone, matching pipeline.py's own "ingested is not implicitly kept"
    convention (see pipeline._kept_original_raws).
    """
    folder = Path(folder)
    candidates = [a for a in scan_assets(folder) if a.role == "original" and a.status == "ingested"]

    file_metrics = {asset.file: _compute_asset_metrics(folder / asset.file) for asset in candidates}
    verdicts = _build_verdicts(file_metrics)

    keep_n = sum(1 for v in verdicts if v["verdict"] == "keep_candidate")
    result = {
        "folder": str(folder),
        "assets": verdicts,
        "summary": {
            "total": len(verdicts),
            "keep_candidates": keep_n,
            "cull_candidates": len(verdicts) - keep_n,
        },
    }
    write_json(folder / PRESORT_FILENAME, result)
    return result
