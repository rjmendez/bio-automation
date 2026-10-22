"""The owner's Darkroom storefront -> the social posts' prints.csv.

Darkroom has no API. The storefront's HTML lists every product with its image, so `list_products` reads the home page, each collection it
links to, and their ?page=N pages (one request per page, paused). Each product image is matched to the gallery by DA-CLIP content-vector
cosine similarity (owner rule: similarity decides): a print belongs to the observation of the gallery file it looks most like when that
similarity reaches EXACT_FRAME_SIMILARITY and no frame of another observation comes within DARKROOM_MARGIN of it. One observation gets
one link, the best product's. The owner's prints_manual.csv rows win over everything. Products that did not match (too different or
ambiguous) are written to a review csv for a look by eye, never linked.

Everything that touches the network or the GPU is injected, so the sync is testable."""
from __future__ import annotations

import csv
import json
import os
import re
import time
import urllib.request
from pathlib import Path
from typing import Callable, NamedTuple

import numpy as np

from . import settings

USER_AGENT = "bio-automation-darkroom/0.1 (owner storefront sync)"
MAX_PAGES = 200                       # a guard against a listing that never runs out
_ANCHOR = re.compile(r'<a\b[^>]*?href="(/products/(?:([a-z0-9_-]+)/)?(\d+))"[^>]*>(.*?)</a>', re.S)
_IMG = re.compile(r'src="(https://s\d+\.darkroom\.com/[a-z0-9]{20,40})"')
_COLLECTION = re.compile(r'href="/collections/([a-z0-9_-]+)"')


class DarkroomError(Exception):
    pass


class Product(NamedTuple):
    product_id: int
    collection: str | None
    image_url: str
    url: str


class Match(NamedTuple):
    product_id: int
    status: str                       # "match" | "below_threshold" | "ambiguous"
    obs_id: int
    gallery_file: str
    score: float
    rival_obs: int | None             # the closest frame of another observation
    rival_score: float | None


def http_text(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode("utf-8", "replace")


def http_bytes(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


# --------------------------------------------------------------------------------------------------- the storefront

def parse_products(page_html: str, base: str) -> list[Product]:
    """The products of one page, in page order, once each: a product link (/products/<id> or /products/<collection>/<id>) with an image."""
    out: dict[int, Product] = {}
    for m in _ANCHOR.finditer(page_html):
        path, collection, pid = m.group(1), m.group(2), int(m.group(3))
        img = _IMG.search(m.group(4))
        if img and pid not in out:
            out[pid] = Product(pid, collection, img.group(1), base.rstrip("/") + path)
    return list(out.values())


def list_products(fetch: Callable[[str], str], *, base: str, sleep: Callable[[float], None] = time.sleep,
                  pause_s: float = 1.0) -> list[Product]:
    """Every product on the home page and in every collection it links to, paging until a page adds nothing. Raises DarkroomError when
    nothing at all was found: that is a changed or unreachable storefront, not a storefront without products."""
    base = base.rstrip("/")
    home = fetch(base + "/")
    sources = [base] + [f"{base}/collections/{c}" for c in sorted(set(_COLLECTION.findall(home)))]
    found: dict[int, Product] = {}
    for src in sources:
        for page in range(1, MAX_PAGES + 1):
            sleep(pause_s)
            new = [p for p in parse_products(fetch(f"{src}/?page={page}" if src == base else f"{src}?page={page}"), base)
                   if p.product_id not in found]
            if not new:
                break
            found.update({p.product_id: p for p in new})
    if not found:
        raise DarkroomError(f"no products found on {base} (checked {len(sources)} listing(s)): the storefront changed or is unreachable")
    return sorted(found.values(), key=lambda p: p.product_id)


# --------------------------------------------------------------------------------------------------- vectors and matching

def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def gallery_vectors(files: dict[str, int], gallery_dir: Path, cache_file: Path, embed_file: Callable[[Path], np.ndarray],
                    ) -> tuple[list[str], np.ndarray, list[str], int]:
    """(relpaths, unit vectors, unreadable relpaths, newly embedded) for the indexed gallery files. A file is embedded again only when its
    size or mtime changed since the cache was written."""
    cached: dict[str, tuple[str, np.ndarray]] = {}
    if cache_file.is_file():
        with np.load(cache_file, allow_pickle=False) as z:
            cached = {r: (s, v) for r, s, v in zip(z["rels"].tolist(), z["sigs"].tolist(), z["vecs"])}
    rels: list[str] = []
    vecs: list[np.ndarray] = []
    bad: list[str] = []
    fresh = 0
    for rel in sorted(files):
        path = Path(gallery_dir) / rel
        try:
            st = path.stat()
            sig = f"{st.st_size}:{st.st_mtime_ns}"
            if rel in cached and cached[rel][0] == sig:
                v = cached[rel][1]
            else:
                v = _unit(embed_file(path))
                cached[rel] = (sig, v)
                fresh += 1
        except (OSError, ValueError):
            bad.append(rel)
            continue
        rels.append(rel)
        vecs.append(v)
    if not vecs:
        raise DarkroomError("no gallery file could be embedded: nothing to match the prints against")
    if fresh:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_file.with_name(cache_file.name + ".tmp.npz")
        np.savez(tmp, rels=np.array(rels), sigs=np.array([cached[r][0] for r in rels]), vecs=np.stack([cached[r][1] for r in rels]))
        os.replace(tmp, cache_file)
    return rels, np.stack(vecs), bad, fresh


def match_product(pid: int, vec: np.ndarray, rels: list[str], matrix: np.ndarray, files: dict[str, int], *, threshold: float,
                  margin: float) -> Match:
    """The gallery frame `vec` looks most like, and whether that is a match: reaches `threshold`, and no frame of another observation is
    within `margin` of it (frames of the same observation may be: a burst of one subject)."""
    scores = matrix @ vec
    order = np.argsort(-scores)
    best = int(order[0])
    obs = int(files[rels[best]])
    rival = next((int(i) for i in order[1:] if int(files[rels[int(i)]]) != obs), None)
    r_obs = int(files[rels[rival]]) if rival is not None else None
    r_score = float(scores[rival]) if rival is not None else None
    score = float(scores[best])
    if score < threshold:
        status = "below_threshold"
    elif r_score is not None and score - r_score < margin:
        status = "ambiguous"
    else:
        status = "match"
    return Match(pid, status, obs, rels[best], score, r_obs, r_score)


# --------------------------------------------------------------------------------------------------- the sync

def read_rows(path: Path) -> dict[int, str]:
    """{obs_id: url} of an obs_id,url csv; no file -> {}."""
    if not path.is_file():
        return {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not {"obs_id", "url"} <= set(reader.fieldnames or ()):
            raise DarkroomError(f"{path}: needs the header obs_id,url")
        return {int(r["obs_id"]): r["url"].strip() for r in reader if (r["obs_id"] or "").strip().isdigit()}


def _write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\r\n")
        w.writerow(header)
        w.writerows(rows)
    os.replace(tmp, path)


def sync(*, index: dict, gallery_dir: Path, work_dir: Path, out_file: Path, manual_file: Path, base: str,
         embed_file: Callable[[Path], np.ndarray], fetch: Callable[[str], str] | None = None,
         download: Callable[[str], bytes] | None = None, sleep: Callable[[float], None] = time.sleep, pause_s: float = 1.0,
         threshold: float | None = None, margin: float | None = None, write: bool = True) -> dict:
    """Read the storefront, fetch and embed the products not seen before, match every listed product to the gallery and write `out_file`
    (obs_id,url) and <work_dir>/darkroom_review.csv. Re-running costs the listing requests and the new products only."""
    threshold = settings.EXACT_FRAME_SIMILARITY if threshold is None else threshold
    margin = settings.DARKROOM_MARGIN if margin is None else margin
    fetch, download = fetch or http_text, download or http_bytes
    work_dir = Path(work_dir)
    files = {rel: int(obs) for rel, obs in index["files"].items()}
    products = list_products(fetch, base=base, sleep=sleep, pause_s=pause_s)

    state_file, vec_file = work_dir / "products.json", work_dir / "product_vecs.npz"
    state = json.loads(state_file.read_text(encoding="utf-8")) if state_file.is_file() else {}
    pvecs: dict[str, np.ndarray] = {}
    if vec_file.is_file():
        with np.load(vec_file, allow_pickle=False) as z:
            pvecs = dict(zip(z["ids"].tolist(), z["vecs"]))
    downloaded = 0
    (work_dir / "img").mkdir(parents=True, exist_ok=True)
    for p in products:
        key = str(p.product_id)
        if key in pvecs and state.get(key, {}).get("image_url") == p.image_url:
            continue
        img = work_dir / "img" / f"{p.product_id}.jpg"
        img.write_bytes(download(p.image_url))
        downloaded += 1
        pvecs[key] = _unit(embed_file(img))
        state[key] = {"image_url": p.image_url, "url": p.url, "collection": p.collection}
    state_file.write_text(json.dumps(state, indent=0, sort_keys=True), encoding="utf-8")
    np.savez(vec_file, ids=np.array(list(pvecs)), vecs=np.stack(list(pvecs.values())))

    rels, matrix, bad, fresh = gallery_vectors(files, gallery_dir, work_dir / "gallery_vecs.npz", embed_file)
    matches = [match_product(p.product_id, pvecs[str(p.product_id)], rels, matrix, files, threshold=threshold, margin=margin)
               for p in products]
    url_of = {p.product_id: p.url for p in products}
    best: dict[int, Match] = {}
    for m in matches:
        if m.status == "match" and (m.obs_id not in best or m.score > best[m.obs_id].score):
            best[m.obs_id] = m
    manual = read_rows(manual_file)
    final = {**{obs: url_of[m.product_id] for obs, m in best.items()}, **manual}
    review = sorted((m for m in matches if m.status != "match"), key=lambda m: -m.score)
    if write:
        _write_csv(out_file, ["obs_id", "url"], [[o, final[o]] for o in sorted(final)])
        _write_csv(work_dir / "darkroom_review.csv", ["product_id", "url", "status", "score", "obs_id", "gallery_file", "rival_obs", "rival_score"],
                   [[m.product_id, url_of[m.product_id], m.status, f"{m.score:.4f}", m.obs_id, m.gallery_file, m.rival_obs or "",
                     "" if m.rival_score is None else f"{m.rival_score:.4f}"] for m in review])
    return {"products": len(products), "downloaded": downloaded, "gallery_embedded": fresh, "gallery_unreadable": len(bad),
            "matched_products": sum(m.status == "match" for m in matches), "observations": len(final), "from_manual": len(manual),
            "review": {s: sum(m.status == s for m in review) for s in ("below_threshold", "ambiguous")}, "written": write}
