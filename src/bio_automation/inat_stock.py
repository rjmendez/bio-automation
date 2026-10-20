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
from pathlib import Path
from typing import Callable

from .exif import _exiftool_binary
from .inat import fetch_inat_observation
from .models import Observation
from .pipeline import merge_inat_fields
from .stock import MIN_KEYWORDS, build_metadata

MATCH_TOLERANCE_S = 90.0
MAX_SNAPSHOT_AGE_DAYS = 3
MAX_FILENAME, MAX_TITLE, MAX_KEYWORDS = 30, 70, 50
ADOBE_CATEGORY_IDS = {"Animals": 1, "Plants and Flowers": 14, "Science": 16}
ADOBE_CSV_FIELDS = ("Filename", "Title", "Keywords", "Category", "Releases")


class SnapshotStale(Exception):
    """The local copy of the owner's iNat observations is too old to decide what is already on iNat."""


class LabelError(Exception):
    """An observation could not be labelled for Stock (fetch failed, no taxon, ...)."""


def load_snapshot(path: Path, *, max_age_days: float = MAX_SNAPSHOT_AGE_DAYS, now: float | None = None) -> list[dict]:
    """The owner's observations (id, time, observed_on, ...). Refuses a snapshot older than max_age_days: local copies go stale."""
    path = Path(path)
    age = ((time.time() if now is None else now) - path.stat().st_mtime) / 86400
    if age > max_age_days:
        raise SnapshotStale(f"{path} is {age:.1f} days old (limit {max_age_days:g}); refresh it from the live iNat account first")
    return json.loads(path.read_text(encoding="utf-8"))["observations"]


def match_frames(frames: list[dict], observations: list[dict], tol: float = MATCH_TOLERANCE_S) -> dict:
    """{frame id: observation id} for every frame (needs id, t = capture epoch) within tol seconds of an observation's time; nearest wins."""
    times = sorted((dt.datetime.fromisoformat(o["time"]).timestamp(), o["id"]) for o in observations if o.get("time"))
    ts = [a for a, _ in times]
    out = {}
    for f in frames:
        i = bisect.bisect_left(ts, f["t"])
        best = None
        for j in (i - 1, i):
            if 0 <= j < len(ts) and abs(ts[j] - f["t"]) <= tol and (best is None or abs(ts[j] - f["t"]) < abs(ts[best] - f["t"])):
                best = j
        if best is not None:
            out[f["id"]] = times[best][1]
    return out


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
    Raises LabelError when it cannot be labelled (the reason is the message)."""
    try:
        fetched = fetch(obs_id)
    except Exception as e:                       # network / 404: report which observation, never label from nothing
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
    return {"title": d.get("Title") or "", "caption": d.get("Caption-Abstract") or "", "keywords": [kw] if isinstance(kw, str) else list(kw)}


def write_adobe_csv(rows: list[tuple[str, dict]], path: Path) -> int:
    """Adobe Stock CSV (Filename, Title, Keywords, Category, Releases) for (filename, label) pairs. Releases is empty: none have been screened."""
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, ADOBE_CSV_FIELDS)
        w.writeheader()
        for name, label in rows:
            w.writerow({"Filename": name, "Title": label["title"], "Keywords": ",".join(label["keywords"]),
                        "Category": label["category_id"], "Releases": ""})
    return len(rows)
