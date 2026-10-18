from __future__ import annotations

from dataclasses import dataclass

from .models import Observation


@dataclass
class StockMetadata:
    title: str          # ≤70 chars
    keywords: list[str] # 15–35 items, singular, ordered by relevance
    caption: str        # narrative ≤500 chars
    categories: list[str]  # Adobe Stock category names (1–2)


_MONTH_SEASON: dict[int, str] = {
    1: "winter", 2: "winter", 12: "winter",
    3: "spring", 4: "spring", 5: "spring",
    6: "summer", 7: "summer", 8: "summer",
    9: "autumn", 10: "autumn", 11: "autumn",
}

_HABITAT_WORDS: set[str] = {
    "national park", "state park", "forest", "meadow", "prairie", "wetland",
    "mountain", "alpine", "desert", "coastal", "riparian", "woodland",
    "grassland", "canyon", "valley", "lake", "river", "creek", "marsh",
    "bog", "tundra",
}

# Maps iconic_taxon_name (lowercase) to Adobe Stock category
_CATEGORY_MAP: dict[str, str] = {
    "insecta": "Animals",
    "arachnida": "Animals",
    "aves": "Animals",
    "mammalia": "Animals",
    "reptilia": "Animals",
    "amphibia": "Animals",
    "actinopterygii": "Animals",
    "animalia": "Animals",
    "plantae": "Plants and Flowers",
    "fungi": "Science",
    "chromista": "Science",
    "protozoa": "Science",
}

# Taxa where macro/close-up keywords apply
_MACRO_TAXA: set[str] = {"insecta", "arachnida", "plantae", "fungi"}


def _season_from_obsdate(obsdate: str | None) -> str | None:
    if not obsdate:
        return None
    try:
        month = int(obsdate.split("-")[1])
        return _MONTH_SEASON.get(month)
    except (IndexError, ValueError):
        return None


def _extract_geo_keywords(place_guess: str | None) -> list[str]:
    """Return geographic keywords from a comma-separated place string.

    Single-word comma parts (state, country) come first (higher priority),
    then individual words from multi-word parts, then habitat-type words.
    """
    if not place_guess:
        return []
    lower = place_guess.lower()
    parts = [p.strip().lower() for p in place_guess.split(",")]

    result: list[str] = []
    seen: set[str] = set()

    # Single-word parts first (state/country names)
    for part in parts:
        words = part.split()
        if len(words) == 1 and words[0] not in seen:
            seen.add(words[0])
            result.append(words[0])

    # Words from multi-word parts
    for part in parts:
        words = part.split()
        if len(words) > 1:
            for word in words:
                if len(word) > 2 and word not in seen:
                    seen.add(word)
                    result.append(word)

    # Habitat-type words found anywhere in the string
    for habitat in _HABITAT_WORDS:
        if habitat in lower:
            for w in habitat.split():
                if len(w) > 2 and w not in seen:
                    seen.add(w)
                    result.append(w)

    return result


def _category_from_taxon(taxon: dict) -> list[str]:
    iconic = (taxon.get("iconic_taxon_name") or "").lower()
    cat = _CATEGORY_MAP.get(iconic)
    if cat:
        return [cat]
    # Fallback: check name for known taxa strings
    name_lower = (taxon.get("name") or "").lower()
    for key, val in _CATEGORY_MAP.items():
        if key in name_lower:
            return [val]
    return ["Animals"]


def _build_keywords(observation: Observation) -> list[str]:
    taxon = observation.taxon or {}
    scientific = (taxon.get("name") or "").strip()
    common = (taxon.get("common_name") or "").strip()
    rank = (taxon.get("rank") or "").lower()
    iconic = (taxon.get("iconic_taxon_name") or "").lower()

    priority: list[str] = []
    rest: list[str] = []
    seen: set[str] = set()

    def add(kw: str, is_priority: bool = False) -> None:
        k = kw.strip().lower()
        if not k or k in seen:
            return
        seen.add(k)
        (priority if is_priority else rest).append(k)

    # 1. Common name words → priority
    for word in common.lower().split():
        add(word, is_priority=True)

    # 2. Scientific name parts → priority
    for part in scientific.lower().split():
        add(part, is_priority=True)

    # 3. Season → priority
    season = _season_from_obsdate(observation.obsdate)
    if season:
        add(season, is_priority=True)

    # 4. Geographic keywords: first 2 parts (state/country) → priority
    geo = _extract_geo_keywords(observation.place_guess)
    for i, g in enumerate(geo):
        add(g, is_priority=(i < 2))

    # 5. Manual seed keywords → priority
    for kw in (observation.keywords_seed or []):
        add(kw.lower().strip(), is_priority=True)

    # 6. Always-include photography/nature terms → rest
    add("wildlife")
    add("nature")
    add("outdoor")
    add("photography")

    # 7. Macro terms for insects/arachnids/plants
    if iconic in _MACRO_TAXA:
        add("macro")
        add("close-up")

    # 8. Genus alone (for species-rank taxa)
    if rank == "species" and len(scientific.split()) >= 2:
        add(scientific.split()[0].lower())

    # 9. Generic biology terms → rest
    for kw in ("wild", "habitat", "biology", "natural", "species",
               "organism", "taxonomy", "field", "observation", "ecosystem",
               "environment", "conservation", "endemic", "terrestrial"):
        add(kw)

    # Combine: first 10 from priority, overflow priority, then rest
    combined = priority[:10]
    for k in priority[10:]:
        if k not in combined:
            combined.append(k)
    for k in rest:
        if k not in combined:
            combined.append(k)

    return combined[:35]


def _build_title(observation: Observation) -> str:
    taxon = observation.taxon or {}
    common = (taxon.get("common_name") or taxon.get("name") or "Unknown").strip()
    scientific = (taxon.get("name") or "").strip()

    base = f"{common} ({scientific})" if scientific else common

    context = ""
    if observation.place_guess:
        context = observation.place_guess.split(",")[0].strip()
    elif observation.obsdate:
        season = _season_from_obsdate(observation.obsdate)
        if season:
            context = season.capitalize()

    title = f"{base} — {context}" if context else base
    if len(title) > 70:
        title = title[:67] + "..."
    return title


def _build_caption(observation: Observation) -> str:
    taxon = observation.taxon or {}
    common = (taxon.get("common_name") or taxon.get("name") or "Unknown").strip()
    scientific = (taxon.get("name") or "").strip()
    place = observation.place_guess or "an unknown location"
    date = observation.obsdate or "an unknown date"

    label = f"{common} ({scientific})" if scientific else common
    caption = f"{label} photographed at {place} on {date}."
    if len(caption) > 500:
        caption = caption[:497] + "..."
    return caption


def build_metadata(observation: Observation) -> StockMetadata:
    """Build StockMetadata from an Observation using taxonomy + place + date.

    No LLM/NLP — pure rule-based from taxon fields, place_guess, and obsdate.
    """
    return StockMetadata(
        title=_build_title(observation),
        keywords=_build_keywords(observation),
        caption=_build_caption(observation),
        categories=_category_from_taxon(observation.taxon or {}),
    )


def build_keywords(observation: Observation) -> dict:
    """Build a template-only {title, caption, keywords, source_taxon,
    source_observation_id} record from the observation's taxon,
    keywords_seed, and locality.

    One record per observation folder, not per asset -- metadata/keywords.json
    is a single shared file, so this takes the folder-level Observation, not
    a per-file Asset. No NLP/LLM generation in v1.
    """
    meta = build_metadata(observation)
    taxon = observation.taxon or {}
    return {
        "title": meta.title,
        "caption": meta.caption,
        "keywords": meta.keywords,
        "source_taxon": taxon.get("name"),
        "source_observation_id": observation.observation_id,
    }
