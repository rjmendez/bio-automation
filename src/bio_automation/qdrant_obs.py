"""The gallery's images, with their iNat and Stock data, in Qdrant: one point per gallery image in its own collection.

Each point has two named vectors -- `image` (the DA-CLIP content vector, 512-d, the same one the Stock matching and darkroom-sync use) and
`text` (an embedding of the point's metadata text with the pool's "embed" model) -- and a payload with the observation's identity,
taxonomy (flattened lineage ranks for filtering), iNat grade/date/license/tags, the Stock label embedded in the file (title, caption,
keywords) and category, the Darkroom print links, the social post state, and the image's size.

Safety: the collection is created by this module and carries metadata that says so. A name that is an alias, or that exists without that
metadata (somebody else's collection), is refused before anything is written; nothing else in Qdrant is touched. Point ids are
deterministic (uuid5 of the gallery path), so a re-run updates the same points.

The network, the GPU, exiftool and iNat are injected, so the whole thing runs in tests without them."""
from __future__ import annotations

import csv
import glob
import json
import os
import subprocess
import time
import tomllib
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Callable

import numpy as np

from . import settings

IMAGE_DIM = 512
OWNER_META = {"owner": "bio-automation", "kind": "observations"}
ID_NAMESPACE = uuid.UUID("5d0c9e0e-8d9a-4c2e-9d66-0b10b5e5f0a1")          # fixed: ids must be the same on every machine and run
BATCH = 128
EXIFTOOL_CHUNK = 400                                                      # files per exiftool call
KEYWORD_FIELDS = ("kingdom", "phylum", "class", "order", "family", "genus", "species", "quality_grade")
INTEGER_FIELDS = ("year", "month", "obs_id")
RANKS = ("kingdom", "phylum", "class", "order", "family", "genus", "species")
STATE_ORDER = ("posted", "posting", "approved", "drafted", "failed", "rejected")      # the most advanced state of an observation wins


class QdrantError(Exception):
    pass


# --------------------------------------------------------------------------------------------------- the client

def qdrant_config() -> tuple[str, str]:
    """(url, api key): BIO_QDRANT_URL / BIO_QDRANT_API_KEY, else the [qdrant] section of Loci's backends.toml (the one Qdrant on this
    machine). The key is never printed or stored."""
    url, key = os.environ.get("BIO_QDRANT_URL"), os.environ.get("BIO_QDRANT_API_KEY")
    if not (url and key):
        cfg = Path.home() / ".loci" / "backends.toml"
        if cfg.is_file():
            q = tomllib.loads(cfg.read_text(encoding="utf-8")).get("qdrant", {})
            url, key = url or q.get("url"), key or q.get("api_key")
    if not url:
        raise QdrantError("no Qdrant url: set BIO_QDRANT_URL (and BIO_QDRANT_API_KEY)")
    return url.rstrip("/"), key or ""


def _http(method: str, url: str, headers: dict, body: bytes | None) -> tuple[int, bytes]:
    req = urllib.request.Request(url, data=body, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


class Qdrant:
    """Just enough of Qdrant's REST API. `http(method, url, headers, body) -> (status, bytes)` is injectable."""

    def __init__(self, url: str, api_key: str, *, http: Callable = _http):
        self.url, self._key, self._http = url.rstrip("/"), api_key, http

    def call(self, method: str, path: str, body: dict | None = None) -> dict:
        headers = {"Content-Type": "application/json", **({"api-key": self._key} if self._key else {})}
        status, raw = self._http(method, self.url + path, headers, None if body is None else json.dumps(body).encode("utf-8"))
        if status >= 300:
            raise QdrantError(f"{method} {path}: HTTP {status}: {raw[:300].decode('utf-8', 'replace')}")
        return json.loads(raw.decode("utf-8")) if raw else {}


def collection_state(client: Qdrant, name: str) -> str:
    """'absent', or 'ours' (it exists and carries OWNER_META); an alias or a collection without the metadata raises QdrantError."""
    if any(a.get("alias_name") == name for a in client.call("GET", "/aliases")["result"]["aliases"]):
        raise QdrantError(f"{name!r} is an alias in Qdrant: refusing to write through it")
    if name not in {c["name"] for c in client.call("GET", "/collections")["result"]["collections"]}:
        return "absent"
    meta = (client.call("GET", f"/collections/{name}")["result"].get("config") or {}).get("metadata") or {}
    if any(meta.get(k) != v for k, v in OWNER_META.items()):
        raise QdrantError(f"collection {name!r} exists but was not created by bio-automation (metadata {meta}): refusing to touch it")
    return "ours"


def ensure_collection(client: Qdrant, name: str, text_dim: int) -> str:
    """Create the collection (named vectors image + text, payload indexes) when it is absent; return 'created' or 'existing'."""
    if collection_state(client, name) == "ours":
        have = client.call("GET", f"/collections/{name}")["result"]["config"]["params"]["vectors"]
        if have.get("text", {}).get("size") != text_dim or have.get("image", {}).get("size") != IMAGE_DIM:
            raise QdrantError(f"collection {name!r} has vector sizes {({k: v.get('size') for k, v in have.items()})}, "
                              f"this run needs image {IMAGE_DIM} and text {text_dim}")
        return "existing"
    client.call("PUT", f"/collections/{name}", {
        "vectors": {"image": {"size": IMAGE_DIM, "distance": "Cosine"}, "text": {"size": text_dim, "distance": "Cosine"}},
        "metadata": OWNER_META})
    for field in KEYWORD_FIELDS:
        client.call("PUT", f"/collections/{name}/index?wait=true", {"field_name": field, "field_schema": "keyword"})
    for field in INTEGER_FIELDS:
        client.call("PUT", f"/collections/{name}/index?wait=true", {"field_name": field, "field_schema": "integer"})
    return "created"


def point_id(rel: str) -> str:
    return str(uuid.uuid5(ID_NAMESPACE, rel))


# --------------------------------------------------------------------------------------------------- the data

def read_labels(paths: list[Path], *, run: Callable[[list[str]], bytes] | None = None, exiftool: str | None = None,
                chunk: int = EXIFTOOL_CHUNK) -> dict[Path, dict]:
    """{path: {title, caption, keywords}} read back from the files (XMP-dc Title, IPTC Caption-Abstract and Keywords), `chunk` files per
    exiftool call instead of one process per file."""
    if run is None:
        from .inat_stock import _exiftool_binary

        exe = exiftool or _exiftool_binary()
        run = lambda args: subprocess.run([exe, *args], capture_output=True, check=True).stdout
    text = lambda v: "true" if v is True else "false" if v is False else str(v)        # exiftool's json turns "true" and digits into non-text
    out: dict[Path, dict] = {}
    for i in range(0, len(paths), chunk):
        part = paths[i:i + chunk]
        rows = json.loads(run(["-j", "-charset", "utf8", "-charset", "iptc=UTF8", "-XMP-dc:Title", "-IPTC:Caption-Abstract", "-IPTC:Keywords",
                               *map(str, part)]).decode("utf-8"))
        by_file = {os.path.normcase(os.path.normpath(r["SourceFile"])): r for r in rows}
        for p in part:
            d = by_file.get(os.path.normcase(os.path.normpath(str(p)))) or {}
            kw = d.get("Keywords") or []
            out[p] = {"title": text(d.get("Title") or ""), "caption": text(d.get("Caption-Abstract") or ""),
                      "keywords": [text(k) for k in (kw if isinstance(kw, list) else [kw])]}
    return out


def load_inat(obs_ids, cache_file: Path, *, fetch: Callable[..., dict], lineage_cache=None, max_age_s: float | None = None,
              now: Callable[[], float] = time.time, chunk: int = 1000) -> tuple[dict[int, dict], dict]:
    """({observation id: entry}, stats): taxon (id, name, common_name, rank, iconic, lineage), grade, observed_on, geoprivacy, place_guess,
    license and the observation's tags, from inat.fetch_inat_observations in batches of `chunk` (cache saved after each); entries younger
    than max_age_s are not fetched again. An observation iNat does not return is {"missing": True}."""
    max_age_s = settings.INAT_TAXON_TTL_S if max_age_s is None else max_age_s
    cache_file = Path(cache_file)
    cached = json.loads(cache_file.read_text(encoding="utf-8")) if cache_file.is_file() else {}
    t = now()
    wanted = sorted({int(o) for o in obs_ids})
    todo = [o for o in wanted if str(o) not in cached or t - cached[str(o)].get("fetched", 0) > max_age_s]
    stats = {"wanted": len(wanted), "fetched": 0, "missing": 0, "cached": len(wanted) - len(todo)}
    for i in range(0, len(todo), chunk):
        ids = todo[i:i + chunk]
        got = fetch(ids, with_lineage=True, lineage_cache=lineage_cache)
        for o in ids:
            r = got.get(o)
            if not isinstance(r, dict):
                cached[str(o)] = {"missing": True, "fetched": t}
                stats["missing"] += 1
                continue
            tx = r.get("taxon") or {}
            cached[str(o)] = {"taxon": {"id": tx.get("id"), "name": tx.get("name"), "common_name": tx.get("common_name"),
                                        "rank": tx.get("rank"), "iconic": tx.get("iconic_taxon_name"), "lineage": tx.get("lineage") or []},
                              "grade": r.get("quality_grade"), "observed_on": r.get("observed_on"), "geoprivacy": r.get("geoprivacy"),
                              "place_guess": r.get("place_guess"), "license": r.get("license"), "tags": r.get("keywords_seed") or [],
                              "fetched": t}
            stats["fetched"] += 1
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_file.with_name(cache_file.name + ".tmp")
        tmp.write_text(json.dumps(cached, sort_keys=True), encoding="utf-8")
        os.replace(tmp, cache_file)
    return {o: cached[str(o)] for o in wanted if str(o) in cached}, stats


def categories_from_candidates(patterns: list[str]) -> dict[int, tuple[str, str]]:
    """{obs_id: (Adobe category, category id)} from the Stock candidate csvs (rows that carry an observation and a category)."""
    out: dict[int, tuple[str, str]] = {}
    for pat in patterns:
        for f in sorted(glob.glob(pat)):
            with open(f, newline="", encoding="utf-8-sig") as fh:
                for row in csv.DictReader(fh):
                    obs, cat = (row.get("obs_id") or "").strip(), (row.get("category") or "").strip()
                    if obs.isdigit() and cat:
                        out[int(obs)] = (cat, (row.get("category_id") or "").strip())
    return out


def social_states(ledger_drafts: dict[str, dict]) -> dict[int, dict]:
    """{obs_id: {"state", "draft_id"}} from the social ledger's drafts: the most advanced state of each observation."""
    out: dict[int, dict] = {}
    for draft_id, v in ledger_drafts.items():
        if v.get("obs_id") is None or v.get("state") not in STATE_ORDER:
            continue
        o = int(v["obs_id"])
        if o not in out or STATE_ORDER.index(v["state"]) < STATE_ORDER.index(out[o]["state"]):
            out[o] = {"state": v["state"], "draft_id": draft_id}
    return out


def darkroom_links(work_dir: Path, files: dict[str, int], rels: list[str], matrix: np.ndarray, *, threshold: float, margin: float
                   ) -> dict[int, list[str]]:
    """{obs_id: [Darkroom product urls]} for every listed product that darkroom.match_product matches to the gallery (same rule as
    darkroom-sync, but all products of an observation, not only the best one)."""
    from . import darkroom

    work_dir = Path(work_dir)
    state_file, vec_file = work_dir / "products.json", work_dir / "product_vecs.npz"
    if not (state_file.is_file() and vec_file.is_file()):
        return {}
    state = json.loads(state_file.read_text(encoding="utf-8"))
    with np.load(vec_file, allow_pickle=False) as z:
        pvecs = dict(zip(z["ids"].tolist(), z["vecs"]))
    out: dict[int, list[str]] = {}
    for key, v in pvecs.items():
        if key not in state:
            continue
        m = darkroom.match_product(int(key), v, rels, matrix, files, threshold=threshold, margin=margin)
        if m.status == "match":
            out.setdefault(m.obs_id, []).append(state[key]["url"])
    return {o: sorted(u) for o, u in out.items()}


def lineage_ranks(taxon: dict) -> dict[str, str]:
    """{rank: name} over RANKS from a taxon entry (its ancestors and itself; a subspecies or variety also fills species)."""
    out = {a["rank"]: a["name"] for a in taxon.get("lineage") or [] if a.get("rank") in RANKS and a.get("name")}
    if taxon.get("rank") in RANKS and taxon.get("name"):
        out[taxon["rank"]] = taxon["name"]
    elif taxon.get("rank") in ("subspecies", "variety", "form", "hybrid") and taxon.get("name"):
        out.setdefault("species", " ".join(taxon["name"].split()[:2]))
    return out


def build_payload(rel: str, obs_id: int, inat: dict, label: dict, category: tuple[str, str] | None, prints: list[str], status: dict | None,
                  size: dict) -> dict:
    """The payload of one gallery image. The place is public iNat text only: omitted when the observation's geoprivacy is private."""
    tx = inat.get("taxon") or {}
    ranks = lineage_ranks(tx)
    parts = rel.split("/")
    folder = parts[-2] if len(parts) >= 2 else ""
    year, month = (int(folder[:4]), int(folder[5:7])) if len(folder) >= 7 and folder[:4].isdigit() and folder[5:7].isdigit() else (None, None)
    place = None if inat.get("geoprivacy") == "private" else ((inat.get("place_guess") or "").strip() or None)
    tags = list(dict.fromkeys([*inat.get("tags", []), *label.get("keywords", [])]))
    payload = {"obs_id": obs_id, "inat_url": f"https://www.inaturalist.org/observations/{obs_id}", "file": rel, "year": year, "month": month,
               "observed_on": inat.get("observed_on"), "taxon_id": tx.get("id"), "taxon_name": tx.get("name"),
               "common_name": tx.get("common_name"), "rank": tx.get("rank"), "iconic": tx.get("iconic"),
               **{r: ranks[r] for r in RANKS if r in ranks}, "quality_grade": inat.get("grade"), "license": inat.get("license"),
               "geoprivacy": inat.get("geoprivacy"), "place": place, "tags": tags,
               "stock_title": label.get("title") or None, "stock_caption": label.get("caption") or None,
               "stock_keywords": label.get("keywords") or [], "stock_category": category[0] if category else None,
               "stock_category_id": category[1] if category else None, "prints": prints,
               "social_state": status["state"] if status else None, "social_draft": status["draft_id"] if status else None, **size}
    return {k: v for k, v in payload.items() if v not in (None, [], "")}


def record_text(p: dict) -> str:
    """The text that is embedded for a point: what a person would search by."""
    lineage = " > ".join(p[r] for r in RANKS if r in p)
    lines = [p.get("stock_title") or "", " / ".join(x for x in (p.get("common_name"), p.get("taxon_name")) if x),
             f"Lineage: {lineage}" if lineage else "", f"Grade: {p['quality_grade']}" if p.get("quality_grade") else "",
             f"Place: {p['place']}" if p.get("place") else "", f"Date: {p['observed_on']}" if p.get("observed_on") else "",
             f"Tags: {', '.join(p['tags'])}" if p.get("tags") else "", p.get("stock_caption") or ""]
    return "\n".join(x for x in lines if x)


# --------------------------------------------------------------------------------------------------- the sync

def sync(*, index: dict, gallery_dir: Path, client: Qdrant | None, collection: str, work_dir: Path, embed_image: Callable[[Path], np.ndarray],
         embed_text: Callable[[list[str]], list[list[float]]], fetch: Callable[..., dict], lineage_cache=None,
         candidates: list[str], prints: dict[int, str], ledger_drafts: dict[str, dict], inat_cache: Path, label_run=None,
         batch: int = BATCH, limit: int | None = None, write: bool = True, threshold: float | None = None, margin: float | None = None,
         sizes: Callable[[Path], dict] | None = None) -> dict:
    """Upsert one point per indexed gallery image (`limit` for a trial run). write=False builds everything and reports, sending nothing."""
    from . import darkroom

    threshold = settings.EXACT_FRAME_SIMILARITY if threshold is None else threshold
    margin = settings.DARKROOM_MARGIN if margin is None else margin
    files = {rel: int(obs) for rel, obs in index["files"].items()}
    if limit:
        files = dict(sorted(files.items())[:limit])
    gallery_dir = Path(gallery_dir)
    all_files = {rel: int(obs) for rel, obs in index["files"].items()}
    rels_all, matrix_all, bad, fresh = darkroom.gallery_vectors(all_files, gallery_dir, Path(work_dir) / "gallery_vecs.npz", embed_image)
    row = {r: i for i, r in enumerate(rels_all)}
    rels = [r for r in sorted(files) if r in row]
    inat, inat_stats = load_inat(set(files.values()), inat_cache, fetch=fetch, lineage_cache=lineage_cache)
    labels = read_labels([gallery_dir / r for r in rels], run=label_run)
    cats = categories_from_candidates(candidates)
    links = darkroom_links(Path(work_dir), all_files, rels_all, matrix_all, threshold=threshold, margin=margin)
    status = social_states(ledger_drafts)
    sizes = sizes or _image_size

    payloads = []
    for rel in rels:
        o = files[rel]
        links_o = list(dict.fromkeys([*([prints[o]] if o in prints else []), *links.get(o, [])]))
        payloads.append(build_payload(rel, o, inat.get(o) or {}, labels.get(gallery_dir / rel) or {}, cats.get(o), links_o, status.get(o),
                                      sizes(gallery_dir / rel)))
    texts = [record_text(p) for p in payloads]
    vectors_text: list[list[float]] = []
    for i in range(0, len(texts), 256):
        vectors_text += embed_text(texts[i:i + 256])
    if len(vectors_text) != len(texts):
        raise QdrantError(f"the embedder returned {len(vectors_text)} vectors for {len(texts)} texts")
    summary = {"images": len(rels), "gallery_embedded": fresh, "gallery_unreadable": len(bad), "inat": inat_stats,
               "with": {k: sum(1 for p in payloads if k in p) for k in ("taxon_name", "kingdom", "stock_title", "stock_keywords", "stock_category",
                                                                       "place", "prints", "social_state", "tags")},
               "text_dim": len(vectors_text[0]) if vectors_text else 0, "written": False}
    if not write:
        return summary
    summary["collection_action"] = ensure_collection(client, collection, summary["text_dim"])
    for i in range(0, len(rels), batch):
        client.call("PUT", f"/collections/{collection}/points?wait=true", {"points": [
            {"id": point_id(rels[j]), "vector": {"image": matrix_all[row[rels[j]]].tolist(), "text": vectors_text[j]}, "payload": payloads[j]}
            for j in range(i, min(i + batch, len(rels)))]})
    summary["count"] = client.call("POST", f"/collections/{collection}/points/count", {"exact": True})["result"]["count"]
    summary["written"] = True
    return summary


def _image_size(path: Path) -> dict:
    from PIL import Image

    with Image.open(path) as im:
        return {"width": im.width, "height": im.height, "bytes": path.stat().st_size}


def query(client: Qdrant, collection: str, *, using: str, vector: list[float], limit: int = 5, where: dict | None = None) -> list[dict]:
    """[{score, payload}] nearest points to `vector` in the named vector `using`, optionally filtered."""
    body = {"query": vector, "using": using, "limit": limit, "with_payload": True, **({"filter": where} if where else {})}
    return [{"score": p["score"], "payload": p["payload"]} for p in client.call("POST", f"/collections/{collection}/points/query", body)
            ["result"]["points"]]
