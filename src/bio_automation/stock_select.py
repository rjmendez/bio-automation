"""Adobe Stock candidate selection: measure each raw, tag the ones worth a Stock submission, publish them for review.

Stock files must come from the raws (the iNat set is 2048 px). Every frame is measured, nothing assumed:
  technical  sensor MP inside 4-100 MP, ISO <= max_iso, no location attached;
  sharpness  half-size demosaic, 8x8 tiles, Laplacian variance per tile, score = 90th percentile of the tiles (a sharp subject anywhere);
  clipping   fraction of near-white pixels;
  floor      the 10th percentile of the same sharpness measured on a folder of already-approved exports (calibrate_floor), so the bar is
             "as sharp as the weakest tenth of what was already judged good". A heuristic, NOT a quality verdict: exports are processed,
             raws are not. The owner's review of the published candidates is the real gate;
  burst      at most top_per_group frames per group (Stock rejects near-duplicates), sharpest first;
  release    people/property are never checked here: every row is 'unscreened'.
A missing floor, an unreadable raw or an unknown ISO is reported as a reason on the row, never silently passed.
"""
from __future__ import annotations

import contextlib
import csv
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .io import read_json, write_json
from .paths import metadata_dir

TILES = 8
TILE_PERCENTILE = 90
FLOOR_PERCENTILE = 10
DEFAULT_MAX_ISO = 3200
DEFAULT_TOP_PER_GROUP = 2
MAX_CLIPPED_FRACTION = 0.02
STOCK_MIN_MP, STOCK_MAX_MP = 4.0, 100.0
DEFAULT_FLOOR_PATH = Path(os.environ.get("BIO_STOCK_FLOOR", r"D:\bio-cache\photos\stock_floor.json"))
TAGS_FILENAME = "stock_tags.json"
RAW_CACHE_FIELDS = ["lc_id", "sharp", "clip", "iso", "error"]
RELEASE_UNSCREENED = "unscreened"


class StockFloorMissing(Exception):
    """No sharpness floor has been calibrated, so nothing can be judged against it."""


@dataclass(frozen=True)
class FrameScore:
    sharp: float
    clip: float


def tile_sharpness(rgb: np.ndarray) -> FrameScore:
    """rgb: HxWx3 uint8/float. Sharpness = 90th percentile of per-tile Laplacian variance over an 8x8 grid; clip = share of pixels with max channel >= 250."""
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"rgb must be HxWx3, got shape {rgb.shape}")
    if min(rgb.shape[:2]) < TILES * 3:
        raise ValueError(f"image {rgb.shape[1]}x{rgb.shape[0]} is too small for an {TILES}x{TILES} tile grid")
    rgb = rgb.astype(np.float32, copy=False)
    gray = rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114
    lap = gray[1:-1, 1:-1] * 4 - gray[:-2, 1:-1] - gray[2:, 1:-1] - gray[1:-1, :-2] - gray[1:-1, 2:]
    h, w = lap.shape
    th, tw = h // TILES, w // TILES
    tiles = lap[: th * TILES, : tw * TILES].reshape(TILES, th, TILES, tw).transpose(0, 2, 1, 3).reshape(TILES * TILES, -1)
    return FrameScore(float(np.percentile(tiles.var(axis=1), TILE_PERCENTILE)), float((rgb.max(axis=2) >= 250).mean()))


def calibrate_floor(sharp_values: list[float]) -> dict:
    if not sharp_values:
        raise ValueError("cannot calibrate a floor from zero measurements")
    v = np.asarray(sharp_values, dtype=np.float64)
    return {"n": int(v.size), "p10": float(np.percentile(v, FLOOR_PERCENTILE)),
            "p25": float(np.percentile(v, 25)), "median": float(np.median(v))}


def load_floor(path: Path | None = None) -> dict:
    path = Path(path or DEFAULT_FLOOR_PATH)
    if not path.is_file():
        raise StockFloorMissing(f"no sharpness floor at {path}; run `bio stock-calibrate <folder of approved exports>` first")
    return read_json(path)


def calibrate_from_folder(folder: Path, out: Path | None = None, sample: int = 300) -> dict:
    """Measure up to *sample* evenly spaced JPEGs under *folder* (halved by a 2x box filter so they match a half-size demosaic) and save the floor."""
    from PIL import Image

    files = sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in (".jpg", ".jpeg"))
    if not files:
        raise ValueError(f"no JPEGs under {folder}")
    step = max(1, len(files) // sample)
    vals = []
    for p in files[::step][:sample]:
        with Image.open(p) as im:
            im = im.convert("RGB")
            im = im.resize((im.width // 2, im.height // 2), Image.BOX)
            vals.append(tile_sharpness(np.asarray(im)).sharp)
    floor = calibrate_floor(vals)
    write_json(Path(out or DEFAULT_FLOOR_PATH), floor)
    return floor


def score_raw(raw_path: Path) -> FrameScore:
    import rawpy

    with rawpy.imread(str(raw_path)) as r:
        rgb = r.postprocess(half_size=True, use_camera_wb=True, output_bps=8)
    return tile_sharpness(rgb)


def raw_megapixels(raw_path: Path) -> float:
    import rawpy

    with rawpy.imread(str(raw_path)) as r:
        return r.sizes.height * r.sizes.width / 1e6


def raw_iso(raw_path: Path) -> int | None:
    from .exif import extract_exif

    iso = extract_exif(Path(raw_path)).get("ISO")
    return int(iso) if iso not in (None, "") else None


def technical_reasons(*, mp: float, iso: int | None, score: FrameScore | None, floor: dict,
                      max_iso: int = DEFAULT_MAX_ISO, located: bool = False) -> list[str]:
    """Every reason this frame fails the technical bar; an empty list means it passes."""
    reasons = []
    if score is None:
        return ["unscored"]
    if not (STOCK_MIN_MP <= mp <= STOCK_MAX_MP):
        reasons.append(f"{mp:.1f} MP outside {STOCK_MIN_MP:g}-{STOCK_MAX_MP:g}")
    if iso is None:
        reasons.append("ISO unknown")
    elif iso > max_iso:
        reasons.append(f"ISO {iso} > {max_iso}")
    if score.sharp < floor["p10"]:
        reasons.append(f"sharpness {score.sharp:.1f} < floor {floor['p10']:.1f}")
    if score.clip > MAX_CLIPPED_FRACTION:
        reasons.append(f"{score.clip * 100:.1f}% clipped highlights")
    if located:
        reasons.append("carries a location")
    return reasons


def pick_top_per_group(rows: list[dict], top: int = DEFAULT_TOP_PER_GROUP) -> set:
    """ids of the *top* sharpest technically passing rows of each group. rows need id, group, sharp, tech_pass."""
    by_group: dict = {}
    for r in rows:
        if r["tech_pass"]:
            by_group.setdefault(r["group"], []).append(r)
    chosen = set()
    for items in by_group.values():
        items.sort(key=lambda r: -r["sharp"])
        chosen.update(r["id"] for r in items[:top])
    return chosen


def decide(frames: list[dict], scores: dict, floor: dict, *, max_iso: int = DEFAULT_MAX_ISO,
           top_per_group: int = DEFAULT_TOP_PER_GROUP) -> list[dict]:
    """frames: dicts with id, group, mp, iso, located. scores: id -> FrameScore | None. Returns one row per frame with tech_pass, stock_candidate, reasons."""
    rows = []
    for f in frames:
        s = scores.get(f["id"])
        reasons = technical_reasons(mp=f["mp"], iso=f["iso"], score=s, floor=floor, max_iso=max_iso, located=f.get("located", False))
        rows.append({"id": f["id"], "group": f["group"], "mp": round(f["mp"], 1), "iso": f["iso"],
                     "sharp": round(s.sharp, 2) if s else None, "clip": round(s.clip, 5) if s else None,
                     "tech_pass": not reasons, "reasons": reasons, "release": RELEASE_UNSCREENED})
    chosen = pick_top_per_group(rows, top_per_group)
    for r in rows:
        r["stock_candidate"] = r["id"] in chosen
        if r["tech_pass"] and not r["stock_candidate"]:
            r["reasons"] = [f"burst: not among the {top_per_group} sharpest of its group"]
    return rows


# ---- pipeline step: one observation folder (called from pipeline.process_observation) ----

def tag_observation(folder: Path, floor_path: Path | None = None, *, max_iso: int = DEFAULT_MAX_ISO,
                    top_per_group: int = DEFAULT_TOP_PER_GROUP) -> dict:
    """Score the kept raws of one observation folder, write metadata/stock_tags.json, return it. One folder = one group."""
    from .pipeline import _kept_original_raws

    floor = load_floor(floor_path)
    frames, scores = [], {}
    for asset in _kept_original_raws(folder):
        raw = Path(folder) / asset.file
        try:
            scores[asset.file] = score_raw(raw)
        except Exception:
            scores[asset.file] = None
        frames.append({"id": asset.file, "group": 0, "mp": raw_megapixels(raw), "iso": raw_iso(raw), "located": False})
    rows = decide(frames, scores, floor, max_iso=max_iso, top_per_group=top_per_group)
    result = {"floor": floor, "max_iso": max_iso, "top_per_group": top_per_group, "frames": rows}
    write_json(metadata_dir(Path(folder)) / TAGS_FILENAME, result)
    return result


# ---- batch step: the LightCraft iNat backlog queue ----

def _score_item(item: dict) -> dict:
    try:
        s = score_raw(Path(item["raw_abs"]))
        return {"lc_id": item["lc_id"], "sharp": round(s.sharp, 2), "clip": round(s.clip, 5), "iso": item.get("iso", ""), "error": ""}
    except Exception as e:
        return {"lc_id": item["lc_id"], "sharp": "", "clip": "", "iso": "", "error": f"{type(e).__name__}: {e}"}


def tag_queue(queue_json: Path, raw_root: Path, out_csv: Path, *, floor_path: Path | None = None, workers: int = 3,
              max_iso: int = DEFAULT_MAX_ISO, top_per_group: int = DEFAULT_TOP_PER_GROUP, limit: int | None = None) -> list[dict]:
    """Score every frame of a LightCraft queue (items with lc_id, raw, group, w, h, copy, loc_source). Resumable via out_csv + '.raw'."""
    from PIL import Image

    floor = load_floor(floor_path)
    queue = read_json(Path(queue_json))
    cache_path = Path(str(out_csv) + ".raw")
    cached: dict = {}
    if cache_path.exists():
        with open(cache_path, newline="") as f:
            cached = {int(r["lc_id"]): r for r in csv.DictReader(f)}
    todo = [dict(x, raw_abs=str(Path(raw_root) / x["raw"])) for x in queue if x["lc_id"] not in cached][:limit]
    for x in todo:                                   # ISO is read from the item's own render, which carries the raw's EXIF
        try:
            with Image.open(x["copy"]) as im:
                x["iso"] = im.getexif().get_ifd(0x8769).get(0x8827, "")
        except Exception:
            x["iso"] = ""
    new = not cache_path.exists()
    with open(cache_path, "a", newline="") as f, contextlib.ExitStack() as stack:
        if workers > 1:
            results = stack.enter_context(ProcessPoolExecutor(workers)).map(_score_item, todo, chunksize=4)
        else:
            results = map(_score_item, todo)
        w = csv.DictWriter(f, RAW_CACHE_FIELDS)
        if new:
            w.writeheader()
        for r in results:
            w.writerow(r)
            f.flush()
            cached[int(r["lc_id"])] = {k: str(v) for k, v in r.items()}
    frames, scores = [], {}
    for x in queue:
        c = cached.get(x["lc_id"])
        scored = c is not None and not c["error"] and c["sharp"] != ""
        scores[x["lc_id"]] = FrameScore(float(c["sharp"]), float(c["clip"])) if scored else None
        iso = c["iso"] if c else ""
        frames.append({"id": x["lc_id"], "group": x["group"], "mp": x["w"] * x["h"] / 1e6,
                       "iso": int(float(iso)) if iso not in ("", "None", None) else None, "located": x.get("loc_source", "none") != "none"})
    rows = decide(frames, scores, floor, max_iso=max_iso, top_per_group=top_per_group)
    by_id = {x["lc_id"]: x for x in queue}
    for r in rows:
        r["day"], r["raw"] = by_id[r["id"]]["day"], by_id[r["id"]]["raw"]
        r["reasons"] = " | ".join(r["reasons"])
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return rows


# ---- publish: full-size q95 renders of the candidates, hardlinked into the review gallery ----

def publish_candidates(rows_csv: Path, raw_root: Path, render_dir: Path, gallery_dir: Path) -> dict:
    """Render every stock_candidate row from its raw (render.export_stock_jpeg: full size, q95, sRGB, no GPS) into
    render_dir/<year>/<day>/<stem>.jpg and hardlink it into gallery_dir (same volume). Existing renders are kept; a render that
    violates the Stock bounds is reported, not shipped."""
    from .render import ExportBoundsError, export_stock_jpeg

    with open(rows_csv, newline="", encoding="utf-8") as f:
        picks = [r for r in csv.DictReader(f) if r["stock_candidate"] == "True"]
    made = linked = 0
    failed = []
    for r in picks:
        raw = Path(raw_root) / r["raw"]
        dest = Path(render_dir) / r["day"][:4] / r["day"] / f"{raw.stem}.jpg"
        link = Path(gallery_dir) / r["day"][:4] / r["day"] / dest.name
        try:
            if not (dest.is_file() and dest.stat().st_size):
                dest.parent.mkdir(parents=True, exist_ok=True)
                export_stock_jpeg(raw, dest)
                made += 1
            link.parent.mkdir(parents=True, exist_ok=True)
            if not link.exists():
                os.link(dest, link)
                linked += 1
        except (ExportBoundsError, OSError, ValueError) as e:
            failed.append({"raw": r["raw"], "error": f"{type(e).__name__}: {e}"})
    return {"candidates": len(picks), "rendered": made, "linked": linked, "failed": failed}
