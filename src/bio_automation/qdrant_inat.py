"""All of the owner's iNaturalist data (no photo files) in Qdrant, structured so photos can be organised by taxonomy and by embedding.

Two collections of their own, each created by this module and carrying metadata that says so (qdrant_obs.OWNER_META, kind-specific):
  * `bio_inat_observations` -- one point per iNat observation: payload = identity, dates, quality grade, identifications, annotations, tags,
    description, observation fields, place names, PUBLIC coordinates, license, counts, project ids, photo metadata, what is stored locally
    (gallery files, photos with a local source), social post state, Darkroom product links; vectors `text` (the pool's embed model, 768-d)
    and `image` (the mean DA-CLIP vector of the observation's local photos, 512-d; absent when it has none).
  * `bio_inat_taxa` -- one point per taxon in the owner's observed tree (the observed taxa and every ancestor): counts below it, first and
    last observed, species below; vector `text`.
Both payloads carry `text`, `title` and `source`, which Loci's rag_context_search reads, and the `text` vector is the 768-d one Loci's
retrieval picks, so `rag_context_search(query, collections=[...])` works on them without any Loci configuration.

Private data stays out: the data comes from iNat's unauthenticated public API (it never returns private coordinates) and the payload
drops coordinates and place names whenever geoprivacy or taxon_geoprivacy is private. Obscured coordinates are kept as iNat returns them.

Incremental: the observation cache is topped up with id_above (new) and updated_since (changed), deletions are found by comparing the id
list when the count differs, and only points whose content changed are embedded and written. Network, GPU, Qdrant and the embedder are
injected."""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import os
import urllib.parse
import uuid
from pathlib import Path
from typing import Callable

import numpy as np

from . import qdrant_obs, settings
from .qdrant_obs import BATCH, QdrantError, RANKS

INAT_API_BASE = "https://api.inaturalist.org/v1"
PER_PAGE = 200
PLACES_PER_REQUEST = 50
ID_NAMESPACE = uuid.UUID("7a3f1c52-2b86-4f0e-8a5d-6c9e4b0d1f27")          # fixed: point ids are the same on every machine and run
KIND_OBS = {**qdrant_obs.OWNER_META, "kind": "inat-observations"}
KIND_TAXA = {**qdrant_obs.OWNER_META, "kind": "inat-taxa"}
LOCAL_STATUSES = ("adobe_ready_only", "both", "library_raw_only")        # inat_coverage.csv: the photo has a local source file
OBS_KEYWORDS = (*RANKS, "iconic", "quality_grade", "geoprivacy", "taxon_geoprivacy", "license", "place_country", "place_state", "place_county",
                "ann_life_stage", "ann_sex", "ann_alive_or_dead", "ann_flowers_and_fruits", "ann_leaf_phenology", "social_state")
OBS_INTEGERS = ("obs_id", "taxon_id", "year", "month", "identifications_count", "faves_count", "comments_count", "photo_count", "local_files")
OBS_BOOLS = ("has_local_photo", "in_gallery", "obscured", "captive")
TAXA_KEYWORDS = (*RANKS, "rank", "iconic")
TAXA_INTEGERS = ("taxon_id", "parent_id", "obs_count", "direct_count", "research_count", "species_count", "local_obs_count")
SKIP_RANKS = ("stateofmatter",)


# --------------------------------------------------------------------------------------------------- the iNat records

def shape_observation(r: dict) -> dict:
    """The fields of one iNat API observation worth keeping (no private location: the API we call never returns one, and none is read)."""
    t = r.get("taxon") or {}
    ids = r.get("identifications") or []
    annotations: dict[str, list[str]] = {}
    for a in r.get("annotations") or []:
        label, value = (a.get("controlled_attribute") or {}).get("label"), (a.get("controlled_value") or {}).get("label")
        if label and value:
            annotations.setdefault(label, []).append(value)
    lat = lng = None
    loc = r.get("location")
    if loc and r.get("geoprivacy") != "private" and r.get("taxon_geoprivacy") != "private":
        try:
            lat, lng = (float(x) for x in str(loc).split(","))
        except ValueError:
            lat = lng = None
    return {
        "id": r["id"], "uuid": r.get("uuid"), "uri": r.get("uri"), "observed_on": r.get("observed_on"), "time_observed_at": r.get("time_observed_at"),
        "tz": r.get("observed_time_zone") or r.get("time_zone"), "created_at": r.get("created_at"), "updated_at": r.get("updated_at"),
        "quality_grade": r.get("quality_grade"), "community_taxon_id": r.get("community_taxon_id"), "captive": r.get("captive"),
        "description": (r.get("description") or "").strip() or None, "tags": [x for x in (r.get("tags") or []) if isinstance(x, str) and x],
        "place_guess": r.get("place_guess"), "place_ids": r.get("place_ids") or [], "geoprivacy": r.get("geoprivacy"),
        "taxon_geoprivacy": r.get("taxon_geoprivacy"), "obscured": bool(r.get("obscured")), "lat": lat, "lng": lng,
        "positional_accuracy": r.get("positional_accuracy"), "public_positional_accuracy": r.get("public_positional_accuracy"),
        "license": r.get("license_code"), "identifications_count": len(ids), "identifications_current": sum(1 for i in ids if i.get("current")),
        "agreements": r.get("num_identification_agreements"), "disagreements": r.get("num_identification_disagreements"),
        "comments_count": r.get("comments_count") if r.get("comments_count") is not None else len(r.get("comments") or []),
        "faves_count": r.get("faves_count") if r.get("faves_count") is not None else len(r.get("faves") or []),
        "annotations": annotations,
        "ofvs": [{"name": o.get("name"), "value": o.get("value")} for o in r.get("ofvs") or [] if o.get("name") and o.get("value") not in (None, "")],
        "project_ids": sorted({po["project"]["id"] for po in r.get("project_observations") or [] if (po.get("project") or {}).get("id")}
                              | set(r.get("project_ids") or [])),
        "photos": [{"id": p.get("id"), "license": p.get("license_code"), "w": (p.get("original_dimensions") or {}).get("width"),
                    "h": (p.get("original_dimensions") or {}).get("height")} for p in r.get("photos") or []],
        "sounds_count": len(r.get("sounds") or []),
        "taxon": {"id": t.get("id"), "name": t.get("name"), "rank": t.get("rank"), "common": t.get("preferred_common_name"),
                  "iconic": t.get("iconic_taxon_name"), "ancestor_ids": t.get("ancestor_ids") or []} if t.get("id") else None,
    }


class InatCache:
    """{observation id: shaped record} plus the sync state, one JSON file written atomically."""

    def __init__(self, path: Path):
        self.path = Path(path)
        doc = json.loads(self.path.read_text(encoding="utf-8")) if self.path.is_file() else {}
        self.login: str | None = doc.get("login")
        self.last_sync: str | None = doc.get("last_sync")
        self.obs: dict[int, dict] = {int(k): v for k, v in (doc.get("observations") or {}).items()}
        self.places: dict[int, dict] = {int(k): v for k, v in (doc.get("places") or {}).items()}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps({"login": self.login, "last_sync": self.last_sync, "observations": self.obs, "places": self.places},
                                  sort_keys=True), encoding="utf-8")
        os.replace(tmp, self.path)


def _page(get: Callable[[str], dict], wait: Callable[[], None], login: str, *, id_above: int, updated_since: str | None = None,
          only_id: bool = False) -> dict:
    q = {"user_login": login, "per_page": PER_PAGE, "order_by": "id", "order": "asc", "id_above": id_above}
    if updated_since:
        q["updated_since"] = updated_since
    if only_id:
        q["only_id"] = "true"
    wait()
    return get(f"{INAT_API_BASE}/observations?{urllib.parse.urlencode(q)}")


def _pull(get, wait, login, *, updated_since: str | None = None, id_above: int = 0) -> list[dict]:
    out: list[dict] = []
    while True:
        results = _page(get, wait, login, id_above=id_above, updated_since=updated_since)["results"]
        if not results:
            return out
        out += results
        id_above = results[-1]["id"]


def sync_records(cache: InatCache, login: str, *, get: Callable[[str], dict], wait: Callable[[], None] = lambda: None,
                 now: Callable[[], dt.datetime] = lambda: dt.datetime.now(dt.timezone.utc), full: bool = False,
                 limit: int | None = None) -> dict:
    """Bring `cache` up to date with the live account: everything on a first (or `full`) run, else the new observations (id_above the
    highest cached id) and the changed ones (updated_since the last sync). A count that still differs from iNat's total triggers an id-only
    listing: observations that are gone are dropped, ones that are missing are fetched. Raises QdrantInatError when the count still
    disagrees. `limit` keeps only the lowest N ids (a trial run; the count check is skipped)."""
    started = now()
    total = get(f"{INAT_API_BASE}/observations?{urllib.parse.urlencode({'user_login': login, 'per_page': 1})}")["total_results"]
    stats = {"live_total": total, "new": 0, "changed": 0, "dropped": 0, "refetched": 0}
    if cache.login not in (None, login):
        raise QdrantInatError(f"the cache {cache.path} belongs to iNat user {cache.login!r}, not {login!r}")
    cache.login = login
    first = full or not cache.obs
    if first:
        cache.obs.clear()
        for r in _pull(get, wait, login):
            cache.obs[r["id"]] = shape_observation(r)
        stats["new"] = len(cache.obs)
    else:
        added = set()
        for r in _pull(get, wait, login, id_above=max(cache.obs)):
            cache.obs[r["id"]] = shape_observation(r)
            added.add(r["id"])
        stats["new"] = len(added)
        if cache.last_sync:
            for r in _pull(get, wait, login, updated_since=cache.last_sync):
                if r["id"] not in added and r["id"] in cache.obs:
                    stats["changed"] += 1
                cache.obs[r["id"]] = shape_observation(r)
    if len(cache.obs) != total and not limit:
        live: set[int] = set()
        above = 0
        while True:
            res = _page(get, wait, login, id_above=above, only_id=True)["results"]
            if not res:
                break
            live |= {r["id"] for r in res}
            above = res[-1]["id"]
        for gone in sorted(set(cache.obs) - live):
            del cache.obs[gone]
            stats["dropped"] += 1
        missing = sorted(live - set(cache.obs))
        for i in range(0, len(missing), PER_PAGE):
            chunk = missing[i:i + PER_PAGE]
            wait()
            for r in get(f"{INAT_API_BASE}/observations?id={','.join(map(str, chunk))}&per_page={len(chunk)}")["results"]:
                cache.obs[r["id"]] = shape_observation(r)
                stats["refetched"] += 1
        if len(cache.obs) != total:
            raise QdrantInatError(f"the cache holds {len(cache.obs)} observations but iNat reports {total} for {login}")
    cache.last_sync = started.isoformat(timespec="seconds")
    cache.save()
    stats["cached"] = len(cache.obs)
    return stats


class QdrantInatError(QdrantError):
    pass


# --------------------------------------------------------------------------------------------------- places and taxa

def resolve_places(cache: InatCache, *, get: Callable[[str], dict], wait: Callable[[], None] = lambda: None) -> int:
    """Fill cache.places ({place id: {name, admin_level}}) for the public observations' place ids; returns how many were fetched."""
    want = sorted({p for r in cache.obs.values() if r["geoprivacy"] != "private" and r["taxon_geoprivacy"] != "private"
                   for p in r["place_ids"]} - set(cache.places))
    for i in range(0, len(want), PLACES_PER_REQUEST):
        chunk = want[i:i + PLACES_PER_REQUEST]
        wait()
        for p in get(f"{INAT_API_BASE}/places/{','.join(map(str, chunk))}")["results"]:
            cache.places[p["id"]] = {"name": p.get("name"), "display": p.get("display_name"), "admin_level": p.get("admin_level")}
        for p in chunk:
            cache.places.setdefault(p, {"name": None, "display": None, "admin_level": None})      # asked once: not asked again
    if want:
        cache.save()
    return len(want)


def place_fields(r: dict, places: dict[int, dict]) -> dict:
    """{place_country, place_state, place_county, place_names} from the observation's place ids; nothing for a private one."""
    if r["geoprivacy"] == "private" or r["taxon_geoprivacy"] == "private":
        return {}
    named = sorted(((places.get(p) or {}) for p in r["place_ids"]), key=lambda x: (x.get("admin_level") is None, x.get("admin_level") or 0))
    by_level = {0: "place_country", 10: "place_state", 20: "place_county"}
    out = {by_level[p["admin_level"]]: p["name"] for p in named if p.get("admin_level") in by_level and p.get("name")}
    names = [p["name"] for p in named if p.get("admin_level") is not None and p.get("name")]
    return {**out, **({"place_names": names} if names else {})}


def taxon_lineages(cache: InatCache, lineage_cache, fetch_lineages: Callable[[list[int]], dict[int, list[dict]]]) -> dict[int, list[dict]]:
    """{taxon id: ancestors root first, each {id, rank, name, common}} for every taxon the owner observed. The names come from
    inat.fetch_taxon_lineages (cached in `lineage_cache`), the ids from the observation's own ancestor_ids; a taxon whose two lists do not
    line up gets its ancestors by name only (id None)."""
    wanted = sorted({r["taxon"]["id"] for r in cache.obs.values() if r["taxon"]})
    have = {t: lineage_cache.get(t) for t in wanted if lineage_cache.get(t) is not None}
    for t, lin in fetch_lineages([t for t in wanted if t not in have]).items():
        lineage_cache[t] = lin
        have[t] = lin
    ancestor_ids = {r["taxon"]["id"]: r["taxon"]["ancestor_ids"] for r in cache.obs.values() if r["taxon"]}
    out: dict[int, list[dict]] = {}
    for t, lin in have.items():
        ids = list(ancestor_ids.get(t) or [])
        if ids and ids[-1] == t:
            ids = ids[:-1]
        aligned = len(ids) == len(lin)
        out[t] = [{"id": ids[i] if aligned else None, "rank": a.get("rank"), "name": a.get("name"), "common": a.get("common_name")}
                  for i, a in enumerate(lin)]
    return out


def lineage_fields(taxon: dict, ancestors: list[dict]) -> dict[str, str]:
    out = {a["rank"]: a["name"] for a in ancestors if a["rank"] in RANKS and a["name"]}
    if taxon.get("rank") in RANKS and taxon.get("name"):
        out[taxon["rank"]] = taxon["name"]
    elif taxon.get("rank") in ("subspecies", "variety", "form", "hybrid") and taxon.get("name"):
        out.setdefault("species", " ".join(taxon["name"].split()[:2]))
    return out


# --------------------------------------------------------------------------------------------------- payloads

def _slug(label: str) -> str:
    return "ann_" + "".join(c if c.isalnum() else "_" for c in label.lower()).strip("_")


def point_id(kind: str, key: int) -> str:
    return str(uuid.uuid5(ID_NAMESPACE, f"{kind}:{key}"))


def obs_text(p: dict) -> str:
    """What is embedded for an observation: what a person would search by."""
    lineage = " > ".join(p[r] for r in RANKS if r in p)
    notes = "; ".join(f"{k}: {', '.join(v)}" for k, v in sorted((p.get("annotations") or {}).items()))
    fields = "; ".join(f"{o['name']}: {o['value']}" for o in p.get("ofvs") or [])
    lines = [" / ".join(x for x in (p.get("common_name"), p.get("taxon_name")) if x) or "Unidentified observation",
             f"Lineage: {lineage}" if lineage else "", f"Grade: {p['quality_grade']}" if p.get("quality_grade") else "",
             f"Date: {p['observed_on']}" if p.get("observed_on") else "",
             f"Place: {', '.join(p['place_names'])}" if p.get("place_names") else (f"Place: {p['place']}" if p.get("place") else ""),
             f"Annotations: {notes}" if notes else "", f"Tags: {', '.join(p['tags'])}" if p.get("tags") else "",
             f"Observation fields: {fields}" if fields else "", (p.get("description") or "")[:600]]
    return "\n".join(x for x in lines if x)


def build_obs_payload(r: dict, ancestors: list[dict], places: dict[int, dict], *, local_files: int, local_photos: int, prints: list[str],
                      status: dict | None) -> dict:
    t = r["taxon"] or {}
    ranks = lineage_fields(t, ancestors) if t else {}
    public = r["geoprivacy"] != "private" and r["taxon_geoprivacy"] != "private"
    od = r.get("observed_on") or ""
    ann = {_slug(k): v[0] for k, v in (r.get("annotations") or {}).items() if v}
    p = {"obs_id": r["id"], "inat_url": r.get("uri") or f"https://www.inaturalist.org/observations/{r['id']}", "uuid": r.get("uuid"),
         "observed_on": r.get("observed_on"), "year": int(od[:4]) if od[:4].isdigit() else None, "month": int(od[5:7]) if od[5:7].isdigit() else None,
         "created_at": r.get("created_at"), "updated_at": r.get("updated_at"), "taxon_id": t.get("id"), "taxon_name": t.get("name"),
         "common_name": t.get("common"), "rank": t.get("rank"), "iconic": t.get("iconic"), **{k: ranks[k] for k in RANKS if k in ranks},
         "community_taxon_id": r.get("community_taxon_id"), "quality_grade": r.get("quality_grade"),
         "identifications_count": r["identifications_count"], "identifications_current": r["identifications_current"],
         "agreements": r.get("agreements"), "disagreements": r.get("disagreements"), "faves_count": r["faves_count"],
         "comments_count": r["comments_count"], "annotations": r.get("annotations") or {}, **ann, "tags": r.get("tags") or [],
         "description": r.get("description"), "ofvs": r.get("ofvs") or [], "project_ids": r.get("project_ids") or [],
         "geoprivacy": r.get("geoprivacy"), "taxon_geoprivacy": r.get("taxon_geoprivacy"), "obscured": r.get("obscured"),
         "captive": r.get("captive"), "license": r.get("license"), "positional_accuracy": r.get("positional_accuracy") if public else None,
         "lat": r.get("lat") if public else None, "lng": r.get("lng") if public else None,
         "place": (r.get("place_guess") or "").strip() or None if public else None, **place_fields(r, places),
         "photo_count": len(r.get("photos") or []), "photos": r.get("photos") or [], "sounds_count": r.get("sounds_count"),
         "local_files": local_files, "in_gallery": local_files > 0, "inat_photos_with_local_source": local_photos,
         "has_local_photo": local_files > 0 or local_photos > 0, "prints": prints,
         "social_state": status["state"] if status else None, "social_draft": status["draft_id"] if status else None}
    p = {k: v for k, v in p.items() if v not in (None, [], "", {})}
    p["text"] = obs_text(p)
    p["title"] = " / ".join(x for x in (p.get("common_name"), p.get("taxon_name")) if x) or f"Observation {r['id']}"
    p["source"] = p["inat_url"]
    return p


def taxa_text(p: dict) -> str:
    lineage = " > ".join(p[r] for r in RANKS if r in p)
    lines = [" / ".join(x for x in (p.get("common_name"), p.get("name")) if x), f"Rank: {p['rank']}", f"Lineage: {lineage}" if lineage else "",
             f"{p['obs_count']} observations ({p['research_count']} research grade), {p['species_count']} species",
             f"Observed {p['first_observed']} to {p['last_observed']}" if p.get("first_observed") else ""]
    return "\n".join(x for x in lines if x)


def build_taxa(cache: InatCache, lineages: dict[int, list[dict]], local_obs: set[int]) -> dict[int, dict]:
    """{taxon id: payload} for every taxon of the owner's observed tree: the observed taxa and all their ancestors, with the counts of the
    observations at or below each, the observations exactly at it, research-grade ones, first and last observed, species below it and
    how many of its observations have a local photo."""
    nodes: dict[int, dict] = {}
    species_below: dict[int, set] = {}
    for r in cache.obs.values():
        t = r["taxon"]
        if not t:
            continue
        ancestors = lineages.get(t["id"], [])
        chain = [*(a for a in ancestors if a["id"] and a["rank"] not in SKIP_RANKS), {"id": t["id"], "rank": t["rank"], "name": t["name"], "common": t["common"]}]
        ranks = lineage_fields(t, ancestors)
        species = ranks.get("species")
        parent = None
        for a in chain:
            n = nodes.setdefault(a["id"], {"taxon_id": a["id"], "name": a["name"], "rank": a["rank"], "common_name": a.get("common"), "obs_count": 0,
                                           "direct_count": 0, "research_count": 0, "local_obs_count": 0, "first_observed": None,
                                           "last_observed": None, "parent_id": parent, "_anc": [x for x in chain[:chain.index(a)]]})
            n["obs_count"] += 1
            n["direct_count"] += a["id"] == t["id"]
            n["research_count"] += r["quality_grade"] == "research"
            n["local_obs_count"] += r["id"] in local_obs
            d = r.get("observed_on")
            if d:
                n["first_observed"] = d if not n["first_observed"] or d < n["first_observed"] else n["first_observed"]
                n["last_observed"] = d if not n["last_observed"] or d > n["last_observed"] else n["last_observed"]
            if species and a["rank"] != "species" and a["rank"] in RANKS:
                species_below.setdefault(a["id"], set()).add(species)
            if t.get("iconic") and a["id"] == t["id"] and not n.get("iconic"):          # only a taxon's own observations say which iconic group it is
                n["iconic"] = t["iconic"]
            parent = a["id"]
    out = {}
    for tid, n in nodes.items():
        anc = n.pop("_anc")
        n.update(lineage_fields({"rank": n["rank"], "name": n["name"]}, anc))
        n["species_count"] = len(species_below.get(tid, ())) or (1 if n["rank"] in ("species", "subspecies", "variety", "form", "hybrid") else 0)
        n = {k: v for k, v in n.items() if v not in (None, "")}
        n["text"] = taxa_text(n)
        n["title"] = " / ".join(x for x in (n.get("common_name"), n["name"]) if x)
        n["source"] = f"https://www.inaturalist.org/taxa/{tid}"
        out[tid] = n
    return out


# --------------------------------------------------------------------------------------------------- local data

def local_photo_counts(coverage_csv: Path | None) -> dict[int, int]:
    """{obs id: number of its iNat photos that have a local source file} from inat_coverage.csv."""
    out: dict[int, int] = {}
    if coverage_csv and Path(coverage_csv).is_file():
        with open(coverage_csv, newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                if (row.get("status") or "") in LOCAL_STATUSES and (row.get("obs_id") or "").isdigit():
                    out[int(row["obs_id"])] = out.get(int(row["obs_id"]), 0) + 1
    return out


def mean_image_vectors(rels: list[str], matrix: np.ndarray, files: dict[str, int]) -> dict[int, list[float]]:
    """{obs id: the normalised mean of the DA-CLIP vectors of its local photos}."""
    acc: dict[int, list[np.ndarray]] = {}
    for i, rel in enumerate(rels):
        acc.setdefault(files[rel], []).append(matrix[i])
    out = {}
    for o, vs in acc.items():
        m = np.mean(vs, axis=0)
        n = float(np.linalg.norm(m))
        if n > 0:
            out[o] = (m / n).tolist()
    return out


# --------------------------------------------------------------------------------------------------- collections and sync

def ensure(client: qdrant_obs.Qdrant, name: str, meta: dict, vectors: dict, keywords, integers, bools=()) -> str:
    """Create our collection (named vectors, owner metadata, payload indexes) when absent; an alias or somebody else's is refused."""
    if any(a.get("alias_name") == name for a in client.call("GET", "/aliases")["result"]["aliases"]):
        raise QdrantInatError(f"{name!r} is an alias in Qdrant: refusing to write through it")
    if name in {c["name"] for c in client.call("GET", "/collections")["result"]["collections"]}:
        have = client.call("GET", f"/collections/{name}")["result"]["config"]
        if any((have.get("metadata") or {}).get(k) != v for k, v in meta.items()):
            raise QdrantInatError(f"collection {name!r} exists but was not created by bio-automation (metadata {have.get('metadata') or {}}): "
                                  "refusing to touch it")
        sizes = {k: v.get("size") for k, v in have["params"]["vectors"].items()}
        if sizes != {k: v["size"] for k, v in vectors.items()}:
            raise QdrantInatError(f"collection {name!r} has vector sizes {sizes}, this run needs {({k: v['size'] for k, v in vectors.items()})}")
        return "existing"
    client.call("PUT", f"/collections/{name}", {"vectors": vectors, "metadata": meta})
    for schema, fields in (("keyword", keywords), ("integer", integers), ("bool", bools)):
        for f in fields:
            client.call("PUT", f"/collections/{name}/index?wait=true", {"field_name": f, "field_schema": schema})
    return "created"


def _sig(*parts) -> str:
    return hashlib.sha1(json.dumps(parts, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def _upsert(client, name: str, items: list[dict], embed_text, state: dict, batch: int, save: Callable[[], None]) -> dict:
    """Embed and write the items (id, payload, optional image vector) whose content differs from `state`; `save` after every batch, so an
    interrupted run resumes where it stopped."""
    todo = [i for i in items if state.get(i["id"]) != i["sig"]]
    for s in range(0, len(todo), batch):
        part = todo[s:s + batch]
        vecs = embed_text([i["payload"]["text"] for i in part])
        if len(vecs) != len(part):
            raise QdrantInatError(f"the embedder returned {len(vecs)} vectors for {len(part)} texts")
        client.call("PUT", f"/collections/{name}/points?wait=true", {"points": [
            {"id": i["id"], "vector": {"text": v, **({"image": i["image"]} if i.get("image") else {})}, "payload": i["payload"]}
            for i, v in zip(part, vecs)]})
        for i in part:
            state[i["id"]] = i["sig"]
        save()
    return {"written": len(todo), "unchanged": len(items) - len(todo)}


def sync(*, cache_file: Path, state_file: Path, client: qdrant_obs.Qdrant | None, prefix: str = "bio_inat_", login: str,
         get: Callable[[str], dict], wait: Callable[[], None], lineage_cache, fetch_lineages: Callable[[list[int]], dict[int, list[dict]]],
         embed_text: Callable[[list[str]], list[list[float]]], index: dict, gallery_dir: Path, work_dir: Path,
         embed_image: Callable[[Path], np.ndarray], prints: dict[int, str], ledger_drafts: dict[str, dict], coverage_csv: Path | None = None,
         batch: int = BATCH, limit: int | None = None, write: bool = True, full: bool = False,
         now: Callable[[], dt.datetime] = lambda: dt.datetime.now(dt.timezone.utc)) -> dict:
    """The observation and taxa collections `<prefix>observations` and `<prefix>taxa` brought in line with the live iNat account."""
    from . import darkroom

    cache = InatCache(cache_file)
    summary: dict = {"inat": sync_records(cache, login, get=get, wait=wait, now=now, full=full, limit=limit)}
    summary["places_fetched"] = resolve_places(cache, get=get, wait=wait)                 # saves the whole cache: truncate for a trial only afterwards
    if limit:
        cache.obs = dict(sorted(cache.obs.items())[:limit])
    lineages = taxon_lineages(cache, lineage_cache, fetch_lineages)

    files = {rel: int(obs) for rel, obs in index["files"].items()}
    rels, matrix, bad, fresh = darkroom.gallery_vectors(files, Path(gallery_dir), Path(work_dir) / "gallery_vecs.npz", embed_image)
    means = mean_image_vectors(rels, matrix, {r: files[r] for r in rels})
    n_files = {}
    for rel in rels:
        n_files[files[rel]] = n_files.get(files[rel], 0) + 1
    local_photos = local_photo_counts(coverage_csv)
    links = qdrant_obs.darkroom_links(Path(work_dir), files, rels, matrix, threshold=settings.EXACT_FRAME_SIMILARITY, margin=settings.DARKROOM_MARGIN)
    status = qdrant_obs.social_states(ledger_drafts)
    local_obs = {o for o in cache.obs if n_files.get(o) or local_photos.get(o)}

    obs_items = []
    for o, r in sorted(cache.obs.items()):
        anc = lineages.get((r["taxon"] or {}).get("id"), [])
        pr = list(dict.fromkeys([*([prints[o]] if o in prints else []), *links.get(o, [])]))
        payload = build_obs_payload(r, anc, cache.places, local_files=n_files.get(o, 0), local_photos=local_photos.get(o, 0), prints=pr, status=status.get(o))
        obs_items.append({"id": point_id("obs", o), "payload": payload, "image": means.get(o),
                          "sig": _sig(payload, [round(x, 5) for x in means[o]] if o in means else None)})
    taxa = build_taxa(cache, lineages, local_obs)
    taxa_items = [{"id": point_id("taxon", t), "payload": p, "sig": _sig(p)} for t, p in sorted(taxa.items())]
    summary.update({"observations": len(obs_items), "taxa": len(taxa_items), "gallery_embedded": fresh, "gallery_unreadable": len(bad),
                    "with_image_vector": sum(1 for i in obs_items if i["image"]), "with_local_photo": len(local_obs), "written": False,
                    "with": {k: sum(1 for i in obs_items if k in i["payload"]) for k in
                             ("taxon_name", "kingdom", "annotations", "tags", "description", "place_names", "lat", "prints", "social_state")},
                    "private": sum(1 for r in cache.obs.values() if r["geoprivacy"] == "private" or r["taxon_geoprivacy"] == "private")})
    if not write:
        return summary
    text_dim = len(embed_text(["dimension probe"])[0])
    names = {"observations": f"{prefix}observations", "taxa": f"{prefix}taxa"}
    state_file = Path(state_file)
    state = json.loads(state_file.read_text(encoding="utf-8")) if state_file.is_file() else {}

    def save_state() -> None:
        state_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = state_file.with_name(state_file.name + ".tmp")
        tmp.write_text(json.dumps(state, sort_keys=True), encoding="utf-8")
        os.replace(tmp, state_file)

    for key, items, meta, kw, ints, bools, with_image in (
            ("observations", obs_items, KIND_OBS, OBS_KEYWORDS, OBS_INTEGERS, OBS_BOOLS, True),
            ("taxa", taxa_items, KIND_TAXA, TAXA_KEYWORDS, TAXA_INTEGERS, (), False)):
        vectors = {"text": {"size": text_dim, "distance": "Cosine"}, **({"image": {"size": qdrant_obs.IMAGE_DIM, "distance": "Cosine"}} if with_image else {})}
        name = names[key]
        summary[f"{key}_collection"] = ensure(client, name, meta, vectors, kw, ints, bools)
        st = state.setdefault(name, {})
        summary[f"{key}_write"] = _upsert(client, name, items, embed_text, st, batch, save_state)
        gone = sorted(set(st) - {i["id"] for i in items})
        for s in range(0, len(gone), batch):
            client.call("POST", f"/collections/{name}/points/delete?wait=true", {"points": gone[s:s + batch]})
        for g in gone:
            del st[g]
        summary[f"{key}_deleted"] = len(gone)
        summary[f"{key}_count"] = client.call("POST", f"/collections/{name}/points/count", {"exact": True})["result"]["count"]
        save_state()
        if summary[f"{key}_count"] != len(items):
            raise QdrantInatError(f"{name} holds {summary[f'{key}_count']} points after the sync, expected {len(items)}")
    summary["written"] = True
    return summary
