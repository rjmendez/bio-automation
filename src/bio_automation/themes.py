"""Themed Bluesky posts: a weekday series ("Mushroom Monday") made of several photos of one kind of organism in one post.

Three pieces, each testable without the network:
  * taxonomy  -- {observation id: taxon + lineage}, fetched from iNat in batches and cached (the pool is thousands of observations);
  * themes    -- the series: a weekday, a name, hashtags and a rule that says which taxa belong (kingdom, class, order, family, genus,
                 species or iconic taxon), from a TOML file with built-in defaults;
  * grouping  -- up to 4 observations of the theme (a Bluesky post carries at most 4 images), different species first, observations with a
                 print first, not yet taken, in an order that is stable for one day.

The default calendar was checked against Bluesky itself on 2026-10-22 (public search, last 35 days, up to 300 posts per tag): every tag
below except WildflowerSaturday is an established recurring series; `note` says how big."""
from __future__ import annotations

import datetime as dt
import json
import os
import random
import time
import tomllib
from pathlib import Path
from typing import Callable, NamedTuple

from . import settings

RANKS = ("kingdom", "phylum", "class", "order", "family", "genus", "species")
RULE_KEYS = RANKS + ("iconic",)
WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
FETCH_CHUNK = 1000                    # observations per cache save; the API itself takes 200 per request


class ThemeError(Exception):
    pass


# --------------------------------------------------------------------------------------------------- taxonomy

def load_taxonomy(obs_ids, cache_file: Path, *, fetch: Callable[..., dict], lineage_cache=None, max_age_s: float | None = None,
                  now: Callable[[], float] = time.time, chunk: int = FETCH_CHUNK) -> tuple[dict[int, dict], dict]:
    """({observation id: entry}, stats) for `obs_ids`. An entry is {"taxon": {id, name, common_name, rank, iconic, lineage}, "grade",
    "observed_on", "geoprivacy", "fetched"}; an observation iNat no longer returns is {"missing": True, "fetched"}. Entries younger than
    `max_age_s` are used as they are; the rest are fetched with `fetch` (inat.fetch_inat_observations) `chunk` ids at a time and the
    cache is written after every chunk, so an interrupted run keeps what it got."""
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
                              "fetched": t}
            stats["fetched"] += 1
        _save(cache_file, cached)
    return {o: cached[str(o)] for o in wanted if str(o) in cached}, stats


def _save(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def ranks_of(entry: dict) -> dict[str, str]:
    """{rank: name} for an observation's taxon: its ancestors, and the taxon itself at its own rank (species, genus ... ; a subspecies or
    variety also fills 'species' with its first two words)."""
    tx = entry.get("taxon") or {}
    out = {a["rank"]: a["name"] for a in tx.get("lineage") or [] if a.get("rank") and a.get("name")}
    if tx.get("rank") and tx.get("name"):
        out[tx["rank"]] = tx["name"]
        if tx["rank"] in ("subspecies", "variety", "form", "hybrid"):
            out.setdefault("species", " ".join(tx["name"].split()[:2]))
    return out


# --------------------------------------------------------------------------------------------------- themes

class Theme(NamedTuple):
    key: str
    name: str
    weekday: int                      # 0 = Monday
    tags: tuple[str, ...]             # hashtags without '#', first ones first
    emoji: str
    include: dict                     # {rule key: [names]}: a taxon belongs when ANY listed name matches
    exclude: dict                     # the same shape: a taxon matching any of these never belongs
    min_photos: int = 2
    max_photos: int = 4
    enabled: bool = True
    note: str = ""


_MUSHROOM_ORDERS = ["Agaricales", "Boletales", "Russulales", "Cantharellales", "Amanitales", "Pluteales", "Gomphales", "Phallales"]

DEFAULT_THEMES = [
    Theme("mushroom-monday", "Mushroom Monday", 0, ("MushroomMonday", "mushrooms", "fungi"), "🍄",
          {"order": _MUSHROOM_ORDERS}, {}, note="Bluesky 2026-10-22: #MushroomMonday 99 posts, 75 authors, 94% on Mondays"),
    Theme("texture-tuesday", "Texture Tuesday", 1, ("TextureTuesday", "lichen", "moss"), "🌿",
          {"class": ["Lecanoromycetes", "Bryopsida", "Polytrichopsida", "Sphagnopsida"], "phylum": ["Bryophyta", "Marchantiophyta"]}, {},
          note="Bluesky 2026-10-22: #TextureTuesday 99 posts, 46 authors, 95% on Tuesdays; lichens, mosses and liverworts"),
    Theme("wildlife-wednesday", "Wildlife Wednesday", 2, ("WildlifeWednesday", "wildlife"), "🐾",
          {"kingdom": ["Animalia"]}, {"class": ["Insecta"]},
          note="Bluesky 2026-10-22: #WildlifeWednesday 100 posts, 45 authors, 94% on Wednesdays; animals other than insects"),
    Theme("insect-thursday", "Insect Thursday", 3, ("InsectThursday", "insects", "bugs"), "🐞",
          {"class": ["Insecta"]}, {}, note="Bluesky 2026-10-22: #InsectThursday 100 posts, 70 authors, 90% on Thursdays"),
    Theme("fungi-friday", "Fungi Friday", 4, ("FungiFriday", "fungi", "mycology"), "🍄",
          {"kingdom": ["Fungi"]}, {"order": _MUSHROOM_ORDERS, "class": ["Lecanoromycetes"]},
          note="Bluesky 2026-10-22: #FungiFriday 99 posts, 78 authors, 88% on Fridays (CSIRO runs it too); fungi that are not Monday's mushrooms or Tuesday's lichens"),
    Theme("wildflower-saturday", "Wildflower Saturday", 5, ("WildflowerSaturday", "wildflowers", "botany"), "🌼",
          {"kingdom": ["Plantae"]}, {"phylum": ["Bryophyta", "Marchantiophyta"]},
          note="own series (#FlowerFriday is the established tag, 100 posts, 49 authors, but Friday belongs to the fungi)"),
    Theme("slime-mold-sunday", "Slime Mold Sunday", 6, ("SlimeMoldSunday", "slimemold", "mycetozoa"), "🟡",
          {"phylum": ["Mycetozoa"]}, {}, note="Bluesky 2026-10-22: #SlimeMoldSunday 17 posts, 8 authors, 100% on Sundays"),
]


def _check_rules(rules: dict, where: str) -> dict:
    out = {}
    for k, v in (rules or {}).items():
        if k not in RULE_KEYS:
            raise ThemeError(f"{where}: unknown rule {k!r} (use one of {', '.join(RULE_KEYS)})")
        if not isinstance(v, list) or not v or not all(isinstance(x, str) and x for x in v):
            raise ThemeError(f"{where}: {k} must be a non-empty list of names")
        out[k] = list(v)
    return out


def load_themes(path: Path | None = None) -> list[Theme]:
    """The themes of the owner's TOML file (settings.SOCIAL_THEMES_FILE), which REPLACES the defaults; the defaults when there is no file.
    A bad file raises ThemeError naming the theme and the field."""
    path = Path(path or settings.SOCIAL_THEMES_FILE)
    if not path.is_file():
        return list(DEFAULT_THEMES)
    try:
        doc = tomllib.loads(path.read_text(encoding="utf-8-sig"))
    except tomllib.TOMLDecodeError as e:
        raise ThemeError(f"{path}: {e}") from e
    out: list[Theme] = []
    for i, t in enumerate(doc.get("theme") or [], 1):
        key = t.get("key") or f"theme #{i}"
        where = f"{path}: theme {key!r}"
        for need in ("key", "name", "weekday", "include"):
            if need not in t:
                raise ThemeError(f"{where}: missing {need}")
        wd = t["weekday"]
        if isinstance(wd, str):
            if wd.lower() not in WEEKDAYS:
                raise ThemeError(f"{where}: weekday must be one of {', '.join(WEEKDAYS)}, got {wd!r}")
            wd = WEEKDAYS.index(wd.lower())
        if not isinstance(wd, int) or not 0 <= wd <= 6:
            raise ThemeError(f"{where}: weekday must be a day name or 0-6")
        lo, hi = int(t.get("min_photos", 2)), int(t.get("max_photos", 4))
        if not 1 <= lo <= hi <= 4:
            raise ThemeError(f"{where}: need 1 <= min_photos <= max_photos <= 4")
        if key in {x.key for x in out}:
            raise ThemeError(f"{where}: key used twice")
        out.append(Theme(key, t["name"], wd, tuple(t.get("tags") or ()), t.get("emoji", ""), _check_rules(t["include"], where),
                         _check_rules(t.get("exclude") or {}, where), lo, hi, bool(t.get("enabled", True)), t.get("note", "")))
    return out


def themes_toml(themes: list[Theme] | None = None) -> str:
    """The themes as a TOML document load_themes() reads back: the starting point for the owner's own file."""
    def arr(xs):
        return "[" + ", ".join(json.dumps(x, ensure_ascii=False) for x in xs) + "]"
    out = ["# Bluesky theme series (bio-automation social). This file REPLACES the built-in calendar when it exists.",
           "# A taxon belongs to a theme when ANY include name matches (rank names: kingdom phylum class order family genus species, or iconic)",
           "# and NONE of the exclude names does. weekday: monday..sunday. A day with no enabled theme posts the usual single observation.", ""]
    for t in themes or DEFAULT_THEMES:
        out += ["[[theme]]", f"key = {json.dumps(t.key)}", f"name = {json.dumps(t.name, ensure_ascii=False)}",
                f"weekday = {json.dumps(WEEKDAYS[t.weekday])}", f"tags = {arr(t.tags)}", f"emoji = {json.dumps(t.emoji, ensure_ascii=False)}",
                f"min_photos = {t.min_photos}", f"max_photos = {t.max_photos}", f"enabled = {str(t.enabled).lower()}",
                f"note = {json.dumps(t.note, ensure_ascii=False)}", "[theme.include]"]
        out += [f"{k} = {arr(v)}" for k, v in t.include.items()]
        if t.exclude:
            out.append("[theme.exclude]")
            out += [f"{k} = {arr(v)}" for k, v in t.exclude.items()]
        out.append("")
    return "\n".join(out)


def _hit(rules: dict, ranks: dict[str, str], iconic: str | None) -> bool:
    for k, names in rules.items():
        have = iconic if k == "iconic" else ranks.get(k)
        if have and have.lower() in {n.lower() for n in names}:
            return True
    return False


def matches(theme: Theme, entry: dict) -> bool:
    """Does the observation's taxon belong to the theme: some include rule matches and no exclude rule does."""
    if not entry or entry.get("missing"):
        return False
    ranks, iconic = ranks_of(entry), (entry.get("taxon") or {}).get("iconic")
    return _hit(theme.include, ranks, iconic) and not _hit(theme.exclude, ranks, iconic)


def theme_for(day: dt.date, themes: list[Theme]) -> Theme | None:
    """The first enabled theme of the weekday of `day`."""
    return next((t for t in themes if t.enabled and t.weekday == day.weekday()), None)


# --------------------------------------------------------------------------------------------------- grouping

def pick_group(theme: Theme, candidates: dict[int, dict], *, day: dt.date, taken: set[int], reserve: int = 3,
               species_first: bool = True) -> list[int]:
    """Up to max_photos + `reserve` observation ids for the theme on `day`, best first, or [] when fewer than min_photos qualify.

    candidates: {obs id: {"entry": taxonomy entry, "print": url or None}}. Not taken, not casual, matching the theme. One observation per
    species first (the species taking turns in an order that is the same all day), observations with a print before those without; the
    reserve is there for an observation that turns out unpostable when it is fetched live."""
    rng = random.Random(f"{day.isoformat()}:{theme.key}")
    by_species: dict[str, list[int]] = {}
    for o in sorted(candidates):
        c = candidates[o]
        e = c["entry"]
        if o in taken or e.get("missing") or e.get("grade") == "casual" or not matches(theme, e):
            continue
        sp = (ranks_of(e).get("species") or (e.get("taxon") or {}).get("name") or str(o)) if species_first else str(o)
        by_species.setdefault(sp, []).append(o)
    if not species_first:                                                 # rule off: one shuffled pool, a print first, no species preference
        pool = [o for obs in by_species.values() for o in obs]
        rng.shuffle(pool)
        pool.sort(key=lambda o: candidates[o].get("print") is None)
        out = pool[:theme.max_photos + reserve]
        return out if len(out) >= theme.min_photos else []
    def fine(o: int) -> bool:                                             # identified to a species (or below), not just a genus or family
        return (candidates[o]["entry"].get("taxon") or {}).get("rank") in ("species", "subspecies", "variety", "form", "hybrid")

    for obs in by_species.values():
        rng.shuffle(obs)
        obs.sort(key=lambda o: (not fine(o), candidates[o].get("print") is None))     # stable: species-level, then a print; the shuffle kept inside
    species = sorted(by_species)
    rng.shuffle(species)
    species.sort(key=lambda s: (not fine(by_species[s][0]), candidates[by_species[s][0]].get("print") is None))
    want = theme.max_photos + reserve
    out: list[int] = []
    depth = 0
    while len(out) < want and any(len(by_species[s]) > depth for s in species):
        out += [by_species[s][depth] for s in species if len(by_species[s]) > depth][:want - len(out)]
        depth += 1
    return out if len(out) >= theme.min_photos else []
