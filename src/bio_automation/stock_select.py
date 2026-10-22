"""Adobe Stock candidate selection: measure each raw, tag the ones worth a Stock submission, publish them for review.

Stock files must come from the raws (the iNat set is 2048 px). Every frame is measured, nothing assumed:
  technical  sensor MP inside 4-100 MP, ISO <= max_iso, no location attached;
  sharpness  half-size demosaic, 8x8 tiles, Laplacian variance per tile, score = 90th percentile of the tiles (a sharp subject anywhere);
  clipping   fraction of near-white pixels;
  floor      the 10th percentile of the same sharpness measured on a folder of already-approved exports (calibrate_floor), so the bar is
             "as sharp as the weakest tenth of what was already judged good". A heuristic, NOT a quality verdict: exports are processed,
             raws are not. The owner's review of the published candidates is the real gate;
  burst      optional: at most top_per_group frames per group (Stock rejects near-duplicates), sharpest first; default is every frame that passes;
  release    people/property are never checked here: every row is 'unscreened'.
A missing floor, an unreadable raw or an unknown ISO is reported as a reason on the row, never silently passed.
"""
from __future__ import annotations

import contextlib
import csv
import os
import re
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import settings, topics
from .io import read_json, write_json
from .paths import metadata_dir

TILES = 8
TILE_PERCENTILE = 90
FLOOR_PERCENTILE = 10
CLIP_PERCENTILE = 95          # the share of near-white pixels that only 5% of the approved exports exceed
DEFAULT_MAX_ISO = 3200
DEFAULT_TOP_PER_GROUP = None          # None = every frame that passes; a number keeps only that many sharpest per group (Stock rejects near-duplicates)
STOCK_MIN_MP, STOCK_MAX_MP = 4.0, 100.0
DEFAULT_FLOOR_PATH = settings.STOCK_FLOOR
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


def median_luma(rgb: np.ndarray) -> float:
    """Median of the 0-255 luma (Rec.601) of an HxWx3 image: a frame whose median sits far below every approved export is under-exposed."""
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"rgb must be HxWx3, got shape {rgb.shape}")
    rgb = rgb.astype(np.float32, copy=False)
    return float(np.median(rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114))


def calibrate_floor(sharp_values: list[float], luma_values: list[float] | None = None, clip_values: list[float] | None = None) -> dict:
    if not sharp_values:
        raise ValueError("cannot calibrate a floor from zero measurements")
    v = np.asarray(sharp_values, dtype=np.float64)
    floor = {"n": int(v.size), "p10": float(np.percentile(v, FLOOR_PERCENTILE)),
             "p25": float(np.percentile(v, 25)), "median": float(np.median(v))}
    if luma_values:
        floor["luma_p10"] = float(np.percentile(np.asarray(luma_values, dtype=np.float64), FLOOR_PERCENTILE))
    if clip_values:
        floor["clip_p95"] = float(np.percentile(np.asarray(clip_values, dtype=np.float64), CLIP_PERCENTILE))
    return floor


def require_complete_floor(floor: dict) -> dict:
    """The floor must carry every measured limit (sharpness p10, luma_p10, clip_p95); an older file is refused, never silently half-applied."""
    missing = [k for k in ("luma_p10", "clip_p95") if k not in floor]
    if missing:
        raise StockFloorMissing(f"the sharpness floor has no {', '.join(missing)}; re-run `bio stock-calibrate <folder of approved exports>`")
    return floor


def load_floor(path: Path | None = None) -> dict:
    path = Path(path or DEFAULT_FLOOR_PATH)
    if not path.is_file():
        raise StockFloorMissing(f"no sharpness floor at {path}; run `bio stock-calibrate <folder of approved exports>` first")
    return read_json(path)


def _small(im, edge: int = 512):
    """Copy of a PIL image limited to *edge* px on its long side (luma is a global statistic, no need for the full frame)."""
    out = im.copy()
    out.thumbnail((edge, edge))
    return out


def calibrate_from_folder(folder: Path, out: Path | None = None, sample: int = 300, exclude_names: set[str] | None = None) -> dict:
    """Measure up to *sample* evenly spaced JPEGs under *folder* (halved by a 2x box filter so they match a half-size demosaic) and save the floor.
    exclude_names: file names to leave out (the candidates this tool published into the same folder must not calibrate their own floor)."""
    from PIL import Image

    files = sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in (".jpg", ".jpeg") and p.name not in (exclude_names or set()))
    if not files:
        raise ValueError(f"no JPEGs under {folder}")
    step = max(1, len(files) // sample)
    vals, lumas, clips = [], [], []
    for p in files[::step][:sample]:
        with Image.open(p) as im:
            im = im.convert("RGB")
            half = im.resize((im.width // 2, im.height // 2), Image.BOX)
            score = tile_sharpness(np.asarray(half))
            vals.append(score.sharp)
            clips.append(score.clip)
            lumas.append(median_luma(np.asarray(_small(im))))
    floor = calibrate_floor(vals, lumas, clips)
    write_json(Path(out or DEFAULT_FLOOR_PATH), floor)
    return floor


IMAGE_SOURCE_EXTENSIONS = (".jpg", ".jpeg", ".tif", ".tiff")      # exported sources (a day with no raw): LightCraft develops them too


def score_raw(raw_path: Path) -> FrameScore:
    """Sharpness of a source at half size: a raw's half-size demosaic, or an exported JPEG/TIFF reduced by 2 (same scale as the floor)."""
    if Path(raw_path).suffix.lower() in IMAGE_SOURCE_EXTENSIONS:
        from PIL import Image

        with Image.open(raw_path) as im:
            im = im.convert("RGB")
            rgb = np.asarray(im.reduce(2))
        return tile_sharpness(rgb)
    import rawpy

    with rawpy.imread(str(raw_path)) as r:
        rgb = r.postprocess(half_size=True, use_camera_wb=True, output_bps=8)
    return tile_sharpness(rgb)


def raw_megapixels(raw_path: Path) -> float:
    if Path(raw_path).suffix.lower() in IMAGE_SOURCE_EXTENSIONS:
        from PIL import Image

        with Image.open(raw_path) as im:
            return im.width * im.height / 1e6
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
    if score.clip > floor["clip_p95"]:
        reasons.append(f"{score.clip * 100:.1f}% clipped highlights")
    if located:
        reasons.append("carries a location")
    return reasons


def pick_top_per_group(rows: list[dict], top: int | None = DEFAULT_TOP_PER_GROUP) -> set:
    """ids of the *top* sharpest technically passing rows of each group. rows need id, group, sharp, tech_pass."""
    by_group: dict = {}
    for r in rows:
        if r["tech_pass"]:
            by_group.setdefault(r["group"], []).append(r)
    chosen = set()
    for items in by_group.values():
        items.sort(key=lambda r: -r["sharp"])
        chosen.update(r["id"] for r in (items if top is None else items[:top]))
    return chosen


def decide(frames: list[dict], scores: dict, floor: dict, *, max_iso: int = DEFAULT_MAX_ISO,
           top_per_group: int | None = DEFAULT_TOP_PER_GROUP, excluded_groups: dict | None = None) -> list[dict]:
    """frames: dicts with id, group, mp, iso, located. scores: id -> FrameScore | None. Returns one row per frame with tech_pass, stock_candidate, reasons.

    excluded_groups maps a group to the reason a person flagged it (people, signs, not wildlife, ...): none of its frames can be a candidate,
    however sharp -- Stock needs releases for identifiable people and property, and the measurements cannot see either.
    """
    excluded_groups = excluded_groups or {}
    rows = []
    for f in frames:
        s = scores.get(f["id"])
        if f["group"] in excluded_groups:
            reasons = [f"group excluded: {excluded_groups[f['group']]}"]          # the exclusion is the reason; its frames are not measured for nothing
        else:
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
                    top_per_group: int | None = DEFAULT_TOP_PER_GROUP) -> dict:
    """Score the kept raws of one observation folder, write metadata/stock_tags.json, return it. One folder = one group.

    The folder's observation must be linked to iNat (bio fetch-inat) and pass the label policy: its taxon, place and date become the Stock label
    (inat_stock.label_from_observation). Otherwise no frame of the folder can be a candidate and the reason is in every row and in `label_error`.
    """
    from .inat_stock import LabelError, label_from_observation
    from .pipeline import _kept_original_raws, load_observation

    floor = require_complete_floor(load_floor(floor_path))
    obs = load_observation(folder)
    label, label_error = None, None
    if obs.observation_id is None:
        label_error = "no iNat observation linked (run bio fetch-inat first)"
    else:
        try:
            label = label_from_observation(obs)
        except LabelError as e:
            label_error = str(e)
    frames, scores = [], {}
    for asset in _kept_original_raws(folder):
        raw = Path(folder) / asset.file
        try:
            scores[asset.file] = score_raw(raw)
        except Exception:
            scores[asset.file] = None
        frames.append({"id": asset.file, "group": 0, "mp": raw_megapixels(raw), "iso": raw_iso(raw), "located": False})
    rows = decide(frames, scores, floor, max_iso=max_iso, top_per_group=top_per_group,
                  excluded_groups={0: label_error} if label_error else None)
    result = {"floor": floor, "max_iso": max_iso, "top_per_group": top_per_group, "label": label, "label_error": label_error, "frames": rows}
    write_json(metadata_dir(Path(folder)) / TAGS_FILENAME, result)
    return result


# ---- batch step: the LightCraft iNat backlog queue ----

QUEUE_ITEM_FIELDS = ("lc_id", "raw", "day", "t", "w", "h")


class QueueItemError(ValueError):
    """A tag_queue item breaks the contract: lc_id must be a LightCraft photo id (int) and the fields tag_queue reads must be present."""


def validate_queue_items(items: list[dict]) -> None:
    """Reject a queue up front, naming every bad item, instead of failing deep inside matching/caching on the first one.
    lc_id is the LightCraft catalog id the render step drives (photo.inspect/export): a file name or any non-int cannot be rendered."""
    if not items:
        raise QueueItemError("the queue is empty: nothing to tag")
    bad = []
    for i, x in enumerate(items):
        missing = [k for k in QUEUE_ITEM_FIELDS if k not in x]
        if missing:
            bad.append(f"item {i}: missing {', '.join(missing)}")
        elif type(x["lc_id"]) is not int:
            bad.append(f"item {i}: lc_id {x['lc_id']!r} is {type(x['lc_id']).__name__}, not a LightCraft photo id (int)")
    if bad:
        shown = "; ".join(bad[:5]) + (f"; ... {len(bad) - 5} more" if len(bad) > 5 else "")
        raise QueueItemError(f"{len(bad)} of {len(items)} queue items are invalid: {shown}")

def _score_item(item: dict) -> dict:
    try:
        s = score_raw(Path(item["raw_abs"]))
        return {"lc_id": item["lc_id"], "sharp": round(s.sharp, 2), "clip": round(s.clip, 5), "iso": item.get("iso", ""), "error": ""}
    except Exception as e:
        return {"lc_id": item["lc_id"], "sharp": "", "clip": "", "iso": "", "error": f"{type(e).__name__}: {e}"}


LABEL_COLUMNS = ("obs_id", "taxon", "grade", "rank", "title", "keywords", "caption", "category", "category_id")
KEYWORD_SEP = "|"


def tag_queue(queue_json: Path, raw_root: Path, out_csv: Path, *, observations: list[dict], fetch=None, floor_path: Path | None = None,
              workers: int = 3, max_iso: int = DEFAULT_MAX_ISO, top_per_group: int | None = DEFAULT_TOP_PER_GROUP, limit: int | None = None,
              excluded_groups: dict | None = None, embedder=None, photo_cache: Path | None = None, get_bytes=None,
              min_similarity: float | None = None, window: float | None = None, render_dir: Path | None = None) -> list[dict]:
    """Score every frame of a LightCraft queue (items with lc_id, raw, day, t = capture epoch, w, h, copy, loc_source). Resumable via out_csv + '.raw'.

    `observations` (the owner's iNat observations, inat_stock.load_snapshot) is REQUIRED, and the iNat photos decide which frames count: each photo of
    each observation takes the frame that looks most like it (inat_stock.match_photos_to_frames, DA-CLIP on the GPU; embedder/photo_cache/get_bytes are
    the injectable dependencies), so exactly the frames the owner uploaded are the candidates -- not their unuploaded siblings, and not frames shot near
    an unrelated observation. Each matched observation is fetched live (fetch, default inat.fetch_inat_observation) and labelled; a frame that is not an
    uploaded photo, or whose observation cannot be labelled (no taxon, casual), is excluded with the reason. top_per_group (default: none) can still cap
    how many frames of one observation are kept.
    excluded_groups (keyed by observation id here): see decide().

    The Stock size bar (4-100 MP) is judged on what will ship, not on the uncropped sensor: the frame's retouched render when one exists at
    render_dir/<year>/<day>/<render_name> (its pixel size), else the item's out_w/out_h (the export size after the crop, when the producer
    knows it), else the sensor w/h. Each row says which in `mp_source` ('render' | 'export' | 'sensor').

    A NEF and its Lightroom '-Enhanced-NR' DNG are the same frame: when both are candidates for one observation, only one stays a candidate
    (collapse_raw_variants: the Enhanced-NR one) and the other gets the reason. Transient iNat errors (inat.InatTransientError, or any error
    other than a deleted observation) propagate: no csv is written and the caller's message fails, so it can be retried.
    """
    from PIL import Image

    from .inat import fetch_inat_observation
    from .inat_stock import (PHOTO_CACHE_DEFAULT, PHOTO_MATCH_MIN_SIMILARITY, PHOTO_MATCH_WINDOW_S, LabelError, cached_photo, default_embedder,
                             label_observation, label_problems, match_photos_to_frames)

    queue = read_json(Path(queue_json))
    validate_queue_items(queue)
    lineage_cache: dict = {}
    fetch = fetch or (lambda i: fetch_inat_observation(i, with_lineage=True, lineage_cache=lineage_cache))
    floor = require_complete_floor(load_floor(floor_path))
    observed_on = {o["id"]: o.get("observed_on") for o in observations}
    photo_kw = {"get_bytes": get_bytes} if get_bytes else {}
    claims = match_photos_to_frames(
        [{"id": x["lc_id"], "t": x["t"], "image": x.get("copy") or ""} for x in queue], observations, embedder or default_embedder(),
        lambda pid, url: cached_photo(pid, url, photo_cache or PHOTO_CACHE_DEFAULT, **photo_kw),
        window=PHOTO_MATCH_WINDOW_S if window is None else window,
        min_similarity=PHOTO_MATCH_MIN_SIMILARITY if min_similarity is None else min_similarity)
    matches = {lc_id: obs_id for lc_id, (obs_id, _) in claims.items()}
    similarity = {lc_id: sim for lc_id, (_, sim) in claims.items()}
    for x in queue:              # the original raw of a matched crop is the same exposure: it inherits the match DA-CLIP made for its variant
        v = x.get("variant_of")
        if v is not None and x["lc_id"] not in matches and v in matches:
            matches[x["lc_id"]], similarity[x["lc_id"]] = matches[v], similarity[v]
    labels, exclusions = {}, dict(excluded_groups or {})
    for obs_id in sorted(set(matches.values())):
        try:
            labels[obs_id] = label_observation(obs_id, observed_on.get(obs_id), fetch=lambda i: fetch(i))
        except LabelError as e:
            exclusions.setdefault(obs_id, str(e))
            continue
        problems = label_problems("x.jpg", labels[obs_id])
        if problems:
            exclusions.setdefault(obs_id, "label: " + "; ".join(problems))
    for x in queue:
        if x["lc_id"] not in matches:
            exclusions.setdefault(-x["lc_id"], "not one of the photos uploaded to an iNat observation")
    cache_path = Path(str(out_csv) + ".raw")
    cached: dict = {}
    if cache_path.exists():
        with open(cache_path, newline="") as f:
            cached = {int(r["lc_id"]): r for r in csv.DictReader(f)}
    todo = [dict(x, raw_abs=str(Path(raw_root) / x["raw"])) for x in queue
            if x["lc_id"] not in cached and x["lc_id"] in matches and matches[x["lc_id"]] not in exclusions][:limit]       # only frames that can still be candidates are measured
    for x in todo:                                   # ISO is read from the item's own picture when it carries EXIF, else from the raw itself
        try:
            with Image.open(x["copy"]) as im:
                x["iso"] = im.getexif().get_ifd(0x8769).get(0x8827, "")
        except Exception:
            x["iso"] = ""
        if x["iso"] == "":
            try:
                x["iso"] = raw_iso(Path(x["raw_abs"])) or ""
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
    frames, scores, mp_source = [], {}, {}
    for x in queue:
        c = cached.get(x["lc_id"])
        scored = c is not None and not c["error"] and c["sharp"] != ""
        scores[x["lc_id"]] = FrameScore(float(c["sharp"]), float(c["clip"])) if scored else None
        iso = c["iso"] if c else ""
        mp, mp_source[x["lc_id"]] = shipped_megapixels(x, render_dir)
        frames.append({"id": x["lc_id"], "group": matches.get(x["lc_id"], -x["lc_id"]), "mp": mp,
                       "iso": int(float(iso)) if iso not in ("", "None", None) else None, "located": False})          # the retouched export removes location (and publish checks the file), so a queue location is irrelevant
    rows = decide(frames, scores, floor, max_iso=max_iso, top_per_group=top_per_group, excluded_groups=exclusions)
    by_id = {x["lc_id"]: x for x in queue}
    for r in rows:
        r["day"], r["raw"] = by_id[r["id"]]["day"], by_id[r["id"]]["raw"]
        r["mp_source"] = mp_source[r["id"]]
        r["reasons"] = " | ".join(r["reasons"])
        obs_id = matches.get(r["id"])
        lab = labels.get(obs_id) if obs_id else None
        r.update({c: "" for c in LABEL_COLUMNS})
        sim = similarity.get(r["id"])
        r["similarity"] = "" if sim is None else round(sim, 3)
        if obs_id:
            r["obs_id"] = obs_id
        if lab:
            r.update({"taxon": lab["taxon"], "grade": lab["grade"], "rank": lab["rank"], "title": lab["title"],
                      "keywords": KEYWORD_SEP.join(lab["keywords"]), "caption": lab["caption"], "category": lab["category"],
                      "category_id": lab["category_id"]})
    for dropped_id, kept in collapse_raw_variants([r for r in rows if r["stock_candidate"]]).items():
        r = next(r for r in rows if r["id"] == dropped_id)
        r["stock_candidate"] = False
        r["reasons"] = f"same frame as {kept['raw']} (id {kept['id']}), which is the candidate"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return rows


# ---- publish: retouched full-size renders, validated against the Stock standard, hardlinked into the review gallery ----

def shipped_megapixels(item: dict, render_dir: Path | None = None) -> tuple[float, str]:
    """(MP, source) of what a queue item will ship as: its render under render_dir if present ('render'), else out_w x out_h ('export'),
    else the sensor w x h ('sensor')."""
    if render_dir is not None:
        render = Path(render_dir) / item["day"][:4] / item["day"] / render_name({"raw": item["raw"], "day": item["day"], "id": item["lc_id"]})
        if render.is_file():
            from PIL import Image

            with Image.open(render) as im:
                return im.width * im.height / 1e6, "render"
    if item.get("out_w") and item.get("out_h"):
        return item["out_w"] * item["out_h"] / 1e6, "export"
    return item["w"] * item["h"] / 1e6, "sensor"


ENHANCED_SUFFIX = "-Enhanced-NR"


def frame_stem(raw: str) -> str:
    """The camera frame a raw belongs to: its stem without a Lightroom export stamp '(...)' or the '-Enhanced-NR' suffix, so a NEF and the
    Enhanced-NR DNG made from it share one stem ('DSC_4296 (2025_06_08 22_03_07 UTC).nef' and 'DSC_4296-Enhanced-NR (...).dng': DSC_4296)."""
    return re.sub(r"\s*\(.*?\)", "", Path(raw).stem).replace(ENHANCED_SUFFIX, "").strip()


def collapse_raw_variants(candidates: list[dict]) -> dict:
    """{dropped row id: kept row} for candidates that are one frame twice: same day, same frame_stem, same obs_id. The kept one is the
    Enhanced-NR variant (the owner's own denoised version), then the sharper, then the lower id; every other is dropped. A variant below the
    Stock size minimum (an Enhanced-NR export can be a small crop) is never a candidate in the first place, so its full-size original wins."""

    def sharp(r):
        try:
            return float(r.get("sharp") or 0.0)
        except ValueError:
            return 0.0

    groups: dict = {}
    for r in candidates:
        groups.setdefault((r["day"], frame_stem(r["raw"]), str(r.get("obs_id", ""))), []).append(r)
    dropped = {}
    for rows in groups.values():
        if len(rows) < 2:
            continue
        rows = sorted(rows, key=lambda r: (ENHANCED_SUFFIX not in Path(r["raw"]).stem, -sharp(r), int(r["id"])))
        for r in rows[1:]:
            dropped[r["id"]] = rows[0]
    return dropped


def render_name(row: dict) -> str:
    """<day>_<raw stem>.jpg -- camera counters wrap, so the day keeps names unique (same convention as the existing gallery files).

    A raw that is itself a Lightroom 'Enhanced-NR' export has a long name ("DSC_3770 (2025_06_03 19_23_22 UTC)-Enhanced-NR"), over Adobe's 30-character
    limit and not unique per frame: the parenthesised export stamp and the suffix are dropped and the frame id is appended instead
    ("2025-04-25_DSC_3770_14081.jpg"). Anything else keeps its plain name."""
    stem = Path(row["raw"]).stem
    clean = re.sub(r"\s*\(.*?\)", "", stem).replace("-Enhanced-NR", "").strip()
    if clean != stem:
        return f"{row['day']}_{clean}_{row['id']}.jpg"
    return f"{row['day']}_{stem}.jpg"


def row_label(row: dict) -> dict:
    """The Stock label stored in a candidates-csv row (tag_queue's label columns)."""
    return {"title": row.get("title", ""), "keywords": [k for k in row.get("keywords", "").split(KEYWORD_SEP) if k], "caption": row.get("caption", ""),
            "category": row.get("category", ""), "category_id": int(row["category_id"]) if row.get("category_id") else None}


NOT_PUBLISHED_FIELDS = ("raw", "name", "status", "reason")


def _describe(path: Path) -> str:
    st = path.stat()
    import datetime as dt

    when = dt.datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds")
    return f"{path} ({st.st_size} B, modified {when}, {st.st_nlink} link{'s' if st.st_nlink != 1 else ''})"


def _same_frame_files(folder: Path, day: str, stem: str) -> list[Path]:
    """Gallery files in `folder` that are this frame under any of the names render_name gives it: <day>_<stem>.jpg, <day>_<stem>_<id>.jpg."""
    if not folder.is_dir():
        return []
    pattern = re.compile(rf"{re.escape(day)}_{re.escape(stem)}(_\d+)?\.jpg", re.IGNORECASE)
    return sorted(p for p in folder.iterdir() if pattern.fullmatch(p.name))


def _default_other_galleries(gallery_dir: Path) -> list[Path]:
    """The gallery (settings.GALLERY_DIR, read at call time) when gallery_dir is somewhere else, so a frame is never published twice."""
    mine = os.path.normcase(os.path.realpath(gallery_dir))
    return [Path(g) for g in (settings.GALLERY_DIR,) if os.path.normcase(os.path.realpath(g)) != mine]


def _same_drive(a: Path, b: Path) -> bool:
    return os.path.splitdrive(os.path.abspath(a))[0].lower() == os.path.splitdrive(os.path.abspath(b))[0].lower()


def link_into_gallery(render: Path, link: Path, same_drive=_same_drive) -> str:
    """Put `render` into the gallery at `link` without a second copy: a hardlink when both are on one drive, else a file symlink (the gallery
    keeps ONE <year>/<year-month> tree whichever drive a render lives on). Returns 'hardlink' or 'symlink'. A symlink needs the
    SeCreateSymbolicLinkPrivilege (admin, or Windows Developer Mode); without it the OSError says so and the candidate fails visibly."""
    if same_drive(render, link):
        os.link(render, link)
        return "hardlink"
    os.symlink(os.path.abspath(render), link)
    return "symlink"


def publish_candidates(rows_csv: Path, raw_root: Path, render_dir: Path, gallery_dir: Path, *, plain_decode: bool = False,
                       floor_path: Path | None = None, adobe_csv: Path | None = None, queue: object | None = None,
                       other_galleries: list[Path] | None = None, not_published_csv: Path | None = None) -> dict:
    """Link every stock_candidate row's RETOUCHED render into gallery_dir/<year>/<year-month>/ (the layout of the existing Stock gallery).

    The retouched render (LightCraft develop + export, full size, q95, sRGB, no location) is produced outside this package and expected at
    render_dir/<year>/<day>/<render_name>. A candidate without one is reported in 'missing' and NOT shipped: a plain raw decode is flat and
    un-retouched, so it is only produced when plain_decode=True is passed explicitly.

    Nothing is linked without its Stock label: each render must pass bundle.validate_jpeg('stock'), the exposure floor (median luma >= luma_p10 of
    the approved exports) and inat_stock.label_problems; its title/caption/keywords are then written INTO the file (XMP + IPTC), read back and
    compared, and only then is it linked (link_into_gallery). violations go to 'failed'. The Adobe Stock CSV of every passing file is written to adobe_csv
    (default render_dir/adobe_stock_metadata.csv).

    A gallery file is never replaced or duplicated from another source: 'refused' lists every candidate whose gallery name already holds a
    file that is not this render's own earlier link (another pipeline's file, or an older render of it), and every candidate whose frame is
    already in this gallery or in one of other_galleries (default: settings.GALLERY_DIR when gallery_dir is another folder) under
    another of its names. Only this render's own link is refreshed (re-embedding replaces the render's inode). A NEF and its Enhanced-NR DNG
    of one frame in the same csv are published once ('skipped' names the other: collapse_raw_variants).

    Every candidate that is not linked (missing, failed, refused, skipped) is written with its reason to not_published_csv (default
    <adobe_csv folder>/not_published/<adobe_csv name>), rewritten on every run, so the record exists outside the log.

    `queue` (anything with a `.publish(topic, payload)`, e.g. bio_automation.queue.Queue) turns "someone has to remember to tell the gallery"
    into a real handoff: when this call links at least one new file, it publishes ONE 'gallery.reindex' message naming every <year>/<year-month>
    folder it touched, in this same call, before returning -- not a separate manual or scheduled step. queue=None (the default) is the exact
    old behavior: nothing is published, nothing else changes.
    """
    from PIL import Image

    from .bundle import validate_jpeg
    from .inat_stock import LabelError, embed_label, label_problems, read_label, write_adobe_csv
    from .render import ExportBoundsError, export_stock_jpeg

    floor = require_complete_floor(load_floor(floor_path))
    gallery_dir = Path(gallery_dir)
    others = _default_other_galleries(gallery_dir) if other_galleries is None else [Path(g) for g in other_galleries]
    adobe_csv = Path(adobe_csv or Path(render_dir) / "adobe_stock_metadata.csv")
    not_published_csv = Path(not_published_csv or adobe_csv.parent / "not_published" / adobe_csv.name)

    with open(rows_csv, newline="", encoding="utf-8") as f:
        picks = [r for r in csv.DictReader(f) if r["stock_candidate"] == "True"]
    decoded = linked = 0
    missing, failed, refused, skipped, shipped = [], [], [], [], []
    record: list[dict] = []
    touched_folders: set[str] = set()
    twins = collapse_raw_variants([r for r in picks if "id" in r])
    for r in picks:
        name = render_name(r)
        if r.get("id") in twins:
            kept = twins[r["id"]]
            skipped.append({"raw": r["raw"], "error": f"same frame as {kept['raw']} (id {kept['id']}), which is published instead"})
            record.append({"raw": r["raw"], "name": name, "status": "skipped", "reason": skipped[-1]["error"]})
            continue
        dest = Path(render_dir) / r["day"][:4] / r["day"] / name
        sub = Path(r["day"][:4]) / r["day"][:7]
        link = gallery_dir / sub / name
        label = row_label(r)
        try:
            own = link.exists() and dest.is_file() and os.path.samefile(dest, link)
            if link.exists() and not own:
                refused.append({"raw": r["raw"], "error": f"the gallery already holds a different file under this name: {_describe(link)}; "
                                                          "not replaced (delete it first if this render should replace it)"})
                record.append({"raw": r["raw"], "name": name, "status": "refused", "reason": refused[-1]["error"]})
                continue
            stem = frame_stem(r["raw"])
            elsewhere = [p for p in _same_frame_files(gallery_dir / sub, r["day"], stem) if p != link]
            elsewhere += [p for g in others for p in _same_frame_files(g / sub, r["day"], stem)]
            if elsewhere:
                refused.append({"raw": r["raw"], "error": "this frame is already published as " + ", ".join(_describe(p) for p in elsewhere)})
                record.append({"raw": r["raw"], "name": name, "status": "refused", "reason": refused[-1]["error"]})
                continue
            if not (dest.is_file() and dest.stat().st_size):
                if not plain_decode:
                    missing.append(r["raw"])
                    record.append({"raw": r["raw"], "name": name, "status": "missing", "reason": f"no retouched render at {dest}"})
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                export_stock_jpeg(Path(raw_root) / r["raw"], dest)
                decoded += 1
            problems = validate_jpeg(dest, "stock") + label_problems(name, label)
            with Image.open(dest) as im:
                im.draft("RGB", (1024, 1024))
                luma = median_luma(np.asarray(_small(im.convert("RGB"))))
            if luma < floor["luma_p10"]:
                problems.append(f"under-exposed: median luma {luma:.1f} < floor {floor['luma_p10']:.1f}")
            if problems:
                failed.append({"raw": r["raw"], "error": "; ".join(problems)})
                record.append({"raw": r["raw"], "name": name, "status": "failed", "reason": failed[-1]["error"]})
                continue
            embed_label(dest, label)
            back = read_label(dest)
            if (back["title"], back["caption"], back["keywords"]) != (label["title"], label["caption"], label["keywords"]):
                failed.append({"raw": r["raw"], "error": "embedded metadata does not read back as written"})
                record.append({"raw": r["raw"], "name": name, "status": "failed", "reason": failed[-1]["error"]})
                continue
            link.parent.mkdir(parents=True, exist_ok=True)
            if own and not os.path.samefile(dest, link):         # this render's own earlier link: re-embedding gave the render a new inode
                link.unlink()
            if not link.exists():
                link_into_gallery(dest, link)
                linked += 1
                touched_folders.add(f"{r['day'][:4]}/{r['day'][:7]}")
            shipped.append((name, label))
        except (ExportBoundsError, LabelError, OSError, ValueError) as e:
            failed.append({"raw": r["raw"], "error": f"{type(e).__name__}: {e}"})
            record.append({"raw": r["raw"], "name": name, "status": "failed", "reason": failed[-1]["error"]})
    if shipped:
        write_adobe_csv(shipped, adobe_csv)
    not_published_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(not_published_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, NOT_PUBLISHED_FIELDS)
        w.writeheader()
        w.writerows(record)
    queued = False
    if queue is not None and touched_folders:
        queue.publish(topics.GALLERY_REINDEX, {"gallery_dir": str(gallery_dir), "folders": sorted(touched_folders)})
        queued = True
    return {"candidates": len(picks), "plain_decoded": decoded, "linked": linked, "missing": missing, "failed": failed,
            "refused": refused, "skipped": skipped, "adobe_csv_rows": len(shipped), "queued_reindex": queued,
            "not_published_csv": str(not_published_csv)}
