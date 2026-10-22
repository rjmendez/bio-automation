"""Stock labels from the owner's iNaturalist observations: the design is fetch-inat -> build-keywords -> export-stock.

A frame is only a Stock candidate when it matches one of the owner's LIVE iNat observations by capture time (AGENTS.md: check what is on iNat before
processing photos). Each matched observation is fetched live (inat.fetch_inat_observation, read-only, 1 req/s), turned into title/keywords/caption/
category by stock.build_metadata, written INTO the JPEG (XMP + IPTC, which is what Adobe Stock reads) and into an Adobe Stock CSV. Nothing in this
package publishes a Stock file without those labels (stock_select.publish_candidates refuses).

Adobe Stock limits (helpx.adobe.com/stock/contributor/manage-your-portfolio/csv-requirements-content.html): filename <= 30 characters, title <= 70
characters without commas, <= 50 keywords by relevance. Category numbers are from helpx.adobe.com/stock/contributor/help/categories.html (the page
lists 1 Animals, 14 Plants and Flowers, 16 Science; entries 20-21 of the list were not read and are not used).
"""
from __future__ import annotations

import bisect
import csv
import datetime as dt
import json
import os
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Callable

from . import settings

import numpy as np

from .exif import _exiftool_binary
from .inat import InatNotFoundError, fetch_inat_observation
from .models import Observation
from .pipeline import merge_inat_fields
from .stock import MIN_KEYWORDS, build_metadata

MAX_SNAPSHOT_AGE_DAYS = 3
MAX_FILENAME, MAX_TITLE, MAX_KEYWORDS = 30, 70, 50
ADOBE_CATEGORY_IDS = {"Animals": 1, "Plants and Flowers": 14, "Science": 16}
ADOBE_CSV_FIELDS = ("Filename", "Title", "Keywords", "Category", "Releases")


# Which frames belong to an observation: the observation's own iNat photos decide, not timestamps. Each iNat photo takes the frame, among those captured
# within PHOTO_MATCH_WINDOW_S of the observation's time, whose DA-CLIP content vector is closest (argmax); the match counts at >= 0.97. Measured on
# observation 409468156 (13 photos): every photo had one frame at 0.993-1.000 and the nearest sibling shot at <= 0.987, whereas time matching (90 s) used two
# frames that were never uploaded and missed three that were. Time-only matching also put a species on frames shot near an unrelated observation.
PHOTO_MATCH_MIN_SIMILARITY = settings.EXACT_FRAME_SIMILARITY
PHOTO_MATCH_WINDOW_S = settings.PHOTO_MATCH_WINDOW_S
PHOTO_CACHE_DEFAULT = settings.INAT_PHOTO_CACHE
PHOTO_USER_AGENT = "bio-automation-owner-sync/1.0 (own account rjmendez; read-only)"


class SnapshotStale(Exception):
    """The local copy of the owner's iNat observations is too old to decide what is already on iNat."""


class LabelError(Exception):
    """An observation could not be labelled for Stock: it no longer exists on iNat, has no taxon, or the label policy refuses it. A refusal
    of the observation itself -- never a transient fetch problem (rate limit, server or network error), which propagates instead."""


def load_snapshot(path: Path, *, max_age_days: float = MAX_SNAPSHOT_AGE_DAYS, now: float | None = None) -> list[dict]:
    """The owner's observations (id, time, observed_on, ...). Refuses a snapshot older than max_age_days: local copies go stale."""
    path = Path(path)
    age = ((time.time() if now is None else now) - path.stat().st_mtime) / 86400
    if age > max_age_days:
        raise SnapshotStale(f"{path} is {age:.1f} days old (limit {max_age_days:g}); refresh it from the live iNat account first")
    return json.loads(path.read_text(encoding="utf-8"))["observations"]


def clean_title(title: str) -> str:
    """Adobe: no commas or special characters in the title. Parentheses go, the em dash becomes a hyphen, commas go; truncated to 70."""
    t = title.replace("—", "-").replace("(", "").replace(")", "").replace(",", "")
    t = " ".join(t.split())
    return t if len(t) <= MAX_TITLE else t[: MAX_TITLE - 3].rstrip() + "..."


def label_policy(quality_grade: str | None, rank: str | None) -> str | None:
    """None when the observation may be labelled for Stock, else the reason it may not. Only casual observations (no wild/verifiable organism) are
    refused. A genus- or family-level identification is labelled AT that rank (title and keywords name the genus, not a guessed species); grade and
    rank travel in the candidates csv so the owner can veto the 'needs ID' ones. Rank is accepted for symmetry with the callers."""
    if quality_grade == "casual":
        return "casual observation"
    return None


def label_observation(obs_id: int, observed_on: str | None, *, fetch: Callable[[int], dict] = fetch_inat_observation) -> dict:
    """Fetch one observation live and build its Stock label {title, keywords, caption, category, category_id, taxon, grade, rank}.
    Raises LabelError when it cannot be labelled (the reason is the message). Only a missing observation (InatNotFoundError) is such a
    refusal; every other fetch error (InatTransientError after its retries, any other API or network error) propagates unchanged, so the
    caller fails or retries its work instead of recording a throttle as a permanent exclusion."""
    try:
        fetched = fetch(obs_id)
    except InatNotFoundError as e:               # the observation is gone: a real refusal, reported with its id
        raise LabelError(f"observation {obs_id}: fetch failed: {type(e).__name__}: {e}") from e
    return label_from_observation(merge_inat_fields(Observation(obsdate=observed_on), fetched), obs_id)


def label_from_observation(obs: Observation, obs_id: int | None = None) -> dict:
    """The Stock label of an Observation that already carries its iNat fields (what bio fetch-inat merges into observation.json).
    Raises LabelError when it has no taxon or the label policy refuses it."""
    obs_id = obs_id if obs_id is not None else obs.observation_id
    taxon = obs.taxon or {}
    if not taxon.get("name"):
        raise LabelError(f"observation {obs_id}: no taxon on iNat")
    refusal = label_policy(obs.quality_grade, taxon.get("rank"))
    if refusal:
        raise LabelError(f"observation {obs_id}: {refusal}")
    meta = build_metadata(obs)
    category = meta.categories[0] if meta.categories else ""
    return {"title": clean_title(meta.title), "keywords": meta.keywords[:MAX_KEYWORDS], "caption": meta.caption, "category": category,
            "category_id": ADOBE_CATEGORY_IDS.get(category), "taxon": taxon["name"], "grade": obs.quality_grade, "rank": taxon.get("rank")}


def label_problems(filename: str, label: dict) -> list[str]:
    """Everything wrong with a label for Adobe Stock; an empty list means it can ship."""
    p = []
    if len(filename) > MAX_FILENAME:
        p.append(f"filename is {len(filename)} characters (limit {MAX_FILENAME})")
    title = label.get("title") or ""
    if not title:
        p.append("no title")
    elif len(title) > MAX_TITLE or "," in title:
        p.append("title over 70 characters or contains a comma")
    kws = label.get("keywords") or []
    if len(kws) < MIN_KEYWORDS:
        p.append(f"{len(kws)} keywords (need at least {MIN_KEYWORDS})")
    if len(kws) > MAX_KEYWORDS:
        p.append(f"{len(kws)} keywords (limit {MAX_KEYWORDS})")
    if not label.get("caption"):
        p.append("no caption")
    if label.get("category_id") is None:
        p.append(f"no Adobe category number for {label.get('category')!r}")
    return p


def embed_label(jpeg: Path, label: dict) -> None:
    """Write title/caption/keywords into the JPEG as XMP-dc and IPTC (UTF-8), replacing any earlier ones. Never touches GPS or the pixels."""
    exiftool = _exiftool_binary()
    # exiftool applies "tag=" and "tag+=" in one command in an order that leaves earlier list items behind, so clear first, then add.
    steps = [["-IPTC:Keywords=", "-XMP-dc:Subject="],
             ["-IPTC:CodedCharacterSet=UTF8", "-XMP-dc:Title=" + label["title"], "-IPTC:ObjectName=" + label["title"],
              "-XMP-dc:Description=" + label["caption"], "-IPTC:Caption-Abstract=" + label["caption"]]
             + [a for kw in label["keywords"] for a in ("-IPTC:Keywords+=" + kw, "-XMP-dc:Subject+=" + kw)]]
    for step in steps:
        # Arguments go through a UTF-8 argfile: on Windows the command line is the ANSI codepage and turns "café" into "caf?".
        with tempfile.NamedTemporaryFile("w", suffix=".args", delete=False, encoding="utf-8", newline="\n") as af:
            af.write("\n".join(step) + "\n")
        try:
            r = subprocess.run([exiftool, "-overwrite_original", "-m", "-charset", "utf8", "-@", af.name, str(jpeg)], capture_output=True, text=True)
        finally:
            os.unlink(af.name)
        if r.returncode != 0 or not ("1 image files updated" in r.stdout or "1 image files unchanged" in r.stdout):
            raise LabelError(f"exiftool failed on {jpeg}: {(r.stderr or r.stdout).strip()[:200]}")


def read_label(jpeg: Path) -> dict:
    """What is actually embedded in the file: {title, caption, keywords} (read back, not assumed)."""
    r = subprocess.run([_exiftool_binary(), "-j", "-charset", "utf8", "-charset", "iptc=UTF8", "-XMP-dc:Title", "-IPTC:Caption-Abstract",
                        "-IPTC:Keywords", str(jpeg)], capture_output=True, check=True)
    d = json.loads(r.stdout.decode("utf-8"))[0]      # bytes decoded explicitly: text=True would use the Windows codepage and mangle accents
    kw = d.get("Keywords") or []
    kw = [kw] if not isinstance(kw, list) else kw
    # exiftool's JSON turns the strings "true"/"false" and digit strings into booleans/numbers: read every value back as text.
    text = lambda v: ("true" if v is True else "false" if v is False else str(v))
    return {"title": text(d.get("Title") or ""), "caption": text(d.get("Caption-Abstract") or ""), "keywords": [text(k) for k in kw]}


def write_adobe_csv(rows: list[tuple[str, dict]], path: Path) -> int:
    """Adobe Stock CSV (Filename, Title, Keywords, Category, Releases) for (filename, label) pairs. Releases is empty: none have been screened."""
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, ADOBE_CSV_FIELDS)
        w.writeheader()
        for name, label in rows:
            w.writerow({"Filename": name, "Title": label["title"], "Keywords": ",".join(label["keywords"]),
                        "Category": label["category_id"], "Releases": ""})
    return len(rows)


# ---- visual verification of time matches ----

def observation_photo_urls(record: dict, size: str = "medium") -> list[tuple[str, str]]:
    """[(photo id, url)] of an observation snapshot record's photos at the given iNat size (the snapshot stores 'square' urls)."""
    out = []
    for ph in record.get("photos") or []:
        url = (ph.get("url") or "").replace("/square.", f"/{size}.")
        if url and ph.get("id") is not None:
            out.append((str(ph["id"]), url))
    return out


def _http_get(url: str) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": PHOTO_USER_AGENT}), timeout=60) as r:
        return r.read()


def cached_photo(photo_id: str, url: str, cache_dir: Path, *, get_bytes: Callable[[str], bytes] = _http_get,
                 sleep: Callable[[float], None] = time.sleep, delay: float = 0.4) -> Path:
    """The photo from the cache, downloading it (politely: one request per `delay` seconds) when it is not there yet."""
    path = Path(cache_dir) / f"{photo_id}.jpg"
    if not (path.is_file() and path.stat().st_size):
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        sleep(delay)
        path.write_bytes(get_bytes(url))
    return path


def default_embedder() -> Callable[[Path], np.ndarray]:
    """DA-CLIP content-vector embedder on the GPU (daclip.load_session fails loudly without CUDA: no CPU fallback). Unit vectors."""
    from . import daclip

    session = daclip.load_session()

    def embed(path: Path) -> np.ndarray:
        v, _ = daclip.embed_path(Path(path), session)
        return v / np.linalg.norm(v)

    return embed


def photo_matches(frames: list[dict], observations: list[dict], embed: Callable[[Path], np.ndarray],
                  photo_for: Callable[[str, str], Path], *, window: float = PHOTO_MATCH_WINDOW_S) -> dict:
    """{(observation id, photo id): (frame id, similarity) | None} for EVERY photo of every observation that has frames nearby.

    frames: dicts with id, t (capture epoch) and image (a Path to a picture of the frame). For each observation with frames captured within `window`
    seconds of its time, each of its iNat photos is paired with the one frame that looks most like it (argmax of the DA-CLIP cosine), whatever the
    similarity -- the caller applies a threshold. A photo whose observation has no frame nearby maps to None. Photos of observations that are not
    near any frame are not fetched at all."""
    ordered = sorted(frames, key=lambda f: f["t"])
    times = [f["t"] for f in ordered]
    vectors: dict = {}

    def vec(f):
        if f["id"] not in vectors:
            vectors[f["id"]] = embed(Path(f["image"]))
        return vectors[f["id"]]

    out: dict = {}
    for o in observations:
        if not o.get("time"):
            continue
        t0 = dt.datetime.fromisoformat(o["time"]).timestamp()
        near = [f for f in ordered[bisect.bisect_left(times, t0 - window):bisect.bisect_right(times, t0 + window)] if Path(f["image"]).is_file()]
        if not near:
            continue
        for pid, url in observation_photo_urls(o):
            pv = embed(photo_for(pid, url))
            sim, best = max(((float(vec(f) @ pv), f) for f in near), key=lambda x: x[0])
            out[(o["id"], pid)] = (best["id"], sim)
    return out


def match_photos_to_frames(frames: list[dict], observations: list[dict], embed: Callable[[Path], np.ndarray],
                           photo_for: Callable[[str, str], Path], *, window: float = PHOTO_MATCH_WINDOW_S,
                           min_similarity: float = PHOTO_MATCH_MIN_SIMILARITY) -> dict:
    """{frame id: (observation id, similarity)} for the frames that ARE photos of the owner's observations.

    Built on photo_matches(): a pair counts when the similarity is at least min_similarity; a frame claimed by two observations keeps the closer one.
    Frames that are not an uploaded photo (siblings of a burst, or shots from near an unrelated observation) are not in the result."""
    claims: dict = {}
    for (oid, _pid), hit in photo_matches(frames, observations, embed, photo_for, window=window).items():
        if hit is None:
            continue
        fid, sim = hit
        if sim >= min_similarity and (fid not in claims or sim > claims[fid][1]):
            claims[fid] = (oid, sim)
    return claims
