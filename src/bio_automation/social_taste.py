"""Taste for the Bluesky posts: variety, a soft seasonal lean, and a cap on the downloaded photo cache (rules variety, seasonal_boost,
photo_cache_cap).

  variety          (order, priority 60, LAST)  candidates of the same genus as any of the last N posted drafts move to the END of the list.
  seasonal_boost   (order, priority 40, first)  candidates photographed within +-SEASON_DAYS of today's day of the year (any year) move to the FRONT.
  photo_cache_cap  (maintenance, 80)     settings.INAT_BEST_PHOTO_DIR is kept under <value> GB: least recently used files first, never one
                                         a pending (drafted/approved) draft still names, never anything outside that directory.

The two order hooks only reorder (stable, nothing is removed), so a short list is never made shorter. The taxonomy comes from the hook
context when it has one, else from the cache file settings.OBS_TAXONOMY_FILE, read tolerantly: a missing or broken file is no change."""
from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path
from typing import Iterable

from . import settings, social_hooks
from .social_rules import Audit

SEASON_DAYS = 15
BYTES_PER_GB = 1024 ** 3
DEFAULT_LAST_N = 3
DEFAULT_CAP_GB = 10


# --------------------------------------------------------------------------------------------------- taxonomy

def _load_taxonomy_file() -> dict:
    try:
        raw = json.loads(Path(settings.OBS_TAXONOMY_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return raw if isinstance(raw, dict) else {}


def _entry(taxonomy: dict, obs_id) -> dict:
    for key in (obs_id, str(obs_id)):
        e = taxonomy.get(key)
        if isinstance(e, dict):
            return e
    try:
        e = taxonomy.get(int(obs_id))
    except (TypeError, ValueError):
        return {}
    return e if isinstance(e, dict) else {}


def genus_of(entry: dict) -> str | None:
    """The genus of a taxonomy entry: its lineage's genus, else the first word of a species-or-lower name; None for a coarser taxon."""
    tx = entry.get("taxon") if isinstance(entry, dict) else None
    if not isinstance(tx, dict):
        return None
    for a in tx.get("lineage") or []:
        if isinstance(a, dict) and a.get("rank") == "genus" and a.get("name"):
            return a["name"]
    if tx.get("rank") == "genus" and tx.get("name"):
        return tx["name"]
    if tx.get("rank") in ("species", "subspecies", "variety", "form", "hybrid") and tx.get("name"):
        return str(tx["name"]).split()[0]
    return None


def _taxonomy(ctx: dict) -> dict:
    tax = ctx.get("taxonomy")
    return tax if isinstance(tax, dict) and tax else _load_taxonomy_file()


def _stable_partition(items: list, first) -> None:
    """Reorder `items` in place: those for which first(item) is true come first, both groups in their old order."""
    items[:] = [x for x in items if first(x)] + [x for x in items if not first(x)]


# --------------------------------------------------------------------------------------------------- variety

def recent_posted_obs(ledger, n: int) -> list:
    """The observation ids of the last `n` posted drafts (latest post last in ledger order)."""
    posted = [(v["seq"], v) for v in ledger.drafts().values() if v["state"] == "posted"]
    posted.sort(key=lambda p: p[0])
    out: list = []
    for _, v in posted[-n:] if n > 0 else []:
        out += list(v.get("obs_ids") or ([v["obs_id"]] if v.get("obs_id") is not None else []))
    return out


def variety(ctx: dict):
    taxonomy = _taxonomy(ctx)
    if not taxonomy:
        return None
    n = ctx["rules"].value("variety") or DEFAULT_LAST_N
    recent = {genus_of(_entry(taxonomy, o)) for o in recent_posted_obs(ctx["ledger"], n)} - {None}
    if not recent:
        return None
    _stable_partition(ctx["candidates"], lambda o: genus_of(_entry(taxonomy, o)) not in recent)
    return None


# --------------------------------------------------------------------------------------------------- seasonal boost

def day_distance(observed_on: str | None, today: dt.date) -> int | None:
    """Days between the observation's calendar day and `today`'s calendar day, ignoring years (the nearer way round the year end);
    None when the date is missing or unreadable."""
    try:
        d = dt.date.fromisoformat(str(observed_on)[:10])
    except ValueError:
        return None
    best = None
    for year in (today.year - 1, today.year, today.year + 1):
        try:
            same = d.replace(year=year)
        except ValueError:                               # 29 February in a year without one
            same = dt.date(year, 3, 1)
        gap = abs((same - today).days)
        best = gap if best is None else min(best, gap)
    return best


def seasonal_boost(ctx: dict):
    taxonomy = _taxonomy(ctx)
    if not taxonomy:
        return None
    today = ctx["today"]

    def in_season(o) -> bool:
        gap = day_distance(_entry(taxonomy, o).get("observed_on"), today)
        return gap is not None and gap <= SEASON_DAYS

    _stable_partition(ctx["candidates"], in_season)
    return None


# --------------------------------------------------------------------------------------------------- photo cache cap

def _is_link(p: Path) -> bool:
    return p.is_symlink() or os.path.isjunction(p)


def _referenced(ledger, drafts_dir: Path) -> set[Path] | None:
    """Resolved paths the pending drafts name as image sources; None when a pending draft cannot be read (then nothing may be deleted)."""
    out: set[Path] = set()
    for did, v in ledger.drafts().items():
        if v["state"] not in ("drafted", "approved"):
            continue
        try:
            doc = json.loads((Path(drafts_dir) / did / "draft.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        for img in doc.get("images") or []:
            src = img.get("source") if isinstance(img, dict) else None
            if src:
                try:
                    out.add(Path(src).resolve())
                except (OSError, ValueError):
                    pass
    return out


def _cache_files(root: Path) -> list[tuple[Path, int, float]]:
    """(path, bytes, last use) of every regular file under `root`; links, and anything resolving outside `root`, are left out."""
    rooted = root.resolve()
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        here = Path(dirpath)
        dirnames[:] = [d for d in dirnames if not _is_link(here / d)]
        for name in filenames:
            p = here / name
            if _is_link(p):
                continue
            try:
                if not p.resolve().is_relative_to(rooted):
                    continue
                st = p.stat()
            except OSError:
                continue
            out.append((p, st.st_size, max(st.st_atime, st.st_mtime)))
    return out


def evict(root: Path, cap_bytes: int, keep: Iterable[Path]) -> dict:
    """Delete least recently used files under `root` until it holds at most `cap_bytes`, never one in `keep` (resolved paths).
    Returns {"before", "after", "deleted", "freed", "kept_referenced"}; a `root` that is missing or is a link is left alone."""
    root = Path(root)
    keep = set(keep)
    result = {"before": 0, "after": 0, "deleted": 0, "freed": 0, "kept_referenced": 0}
    if not root.is_dir() or _is_link(root):
        return result
    files = _cache_files(root)
    total = sum(s for _, s, _ in files)
    result["before"] = result["after"] = total
    if total <= cap_bytes:
        return result
    for p, size, _ in sorted(files, key=lambda f: (f[2], str(f[0]))):
        if result["after"] <= cap_bytes:
            break
        if p.resolve() in keep:
            result["kept_referenced"] += 1
            continue
        try:
            p.unlink()
        except OSError:
            continue
        result["deleted"] += 1
        result["freed"] += size
        result["after"] -= size
    return result


def photo_cache_cap(ctx: dict):
    root = Path(settings.INAT_BEST_PHOTO_DIR)
    gb = ctx["rules"].value("photo_cache_cap") or DEFAULT_CAP_GB
    if not root.is_dir() or _is_link(root):
        return None
    keep = _referenced(ctx["ledger"], ctx["drafts_dir"])
    if keep is None:
        return None                                    # a pending draft whose photos are unknown: delete nothing
    res = evict(root, gb * BYTES_PER_GB, keep)
    if res["deleted"]:
        Audit().log("cache_evicted", directory=str(root), cap_gb=gb, files=res["deleted"], bytes=res["freed"], bytes_before=res["before"],
                    bytes_after=res["after"], kept_referenced=res["kept_referenced"])
    return None


def register_hooks() -> None:
    # seasonal_boost first, variety LAST: a stable partition run later wins, and a recently posted genus must stay demoted even in season
    social_hooks.register("seasonal_boost", "order", seasonal_boost, rule="seasonal_boost", priority=40)
    social_hooks.register("variety", "order", variety, rule="variety", priority=60)
    social_hooks.register("photo_cache_cap", "maintenance", photo_cache_cap, rule="photo_cache_cap", priority=80)
