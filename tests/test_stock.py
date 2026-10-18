"""Unit tests for bio_automation.stock — rule-based metadata generation."""
from __future__ import annotations

import pytest

from bio_automation.models import Observation
from bio_automation.stock import StockMetadata, build_keywords, build_metadata


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def monarch_obs() -> Observation:
    return Observation(
        obsdate="2026-08-15",
        taxon={
            "name": "Danaus plexippus",
            "common_name": "Monarch Butterfly",
            "rank": "species",
            "iconic_taxon_name": "Insecta",
        },
        place_guess="Rocky Mountain National Park, Colorado, USA",
        keywords_seed=["migration", "milkweed"],
    )


# ---------------------------------------------------------------------------
# 1. build_metadata returns correct type
# ---------------------------------------------------------------------------

def test_build_metadata_returns_stock_metadata(monarch_obs):
    meta = build_metadata(monarch_obs)
    assert isinstance(meta, StockMetadata)


# ---------------------------------------------------------------------------
# 2. Title checks
# ---------------------------------------------------------------------------

def test_title_contains_common_name(monarch_obs):
    meta = build_metadata(monarch_obs)
    assert "Monarch" in meta.title


def test_title_contains_scientific_name(monarch_obs):
    meta = build_metadata(monarch_obs)
    assert "Danaus plexippus" in meta.title


def test_title_length_at_most_70(monarch_obs):
    meta = build_metadata(monarch_obs)
    assert len(meta.title) <= 70


def test_title_over_70_is_truncated():
    obs = Observation(
        taxon={
            "name": "Hypothetical extremelylongscientificname elongatissimus",
            "common_name": "A Very Long Common Name That Would Exceed The Limit Easily",
            "rank": "species",
        },
        place_guess="A Very Long Place Name That Makes The Title Way Too Long, Somewhere, Earth",
    )
    meta = build_metadata(obs)
    assert len(meta.title) <= 70


# ---------------------------------------------------------------------------
# 3. Keyword content checks (with positive twins)
# ---------------------------------------------------------------------------

def test_keywords_contains_monarch_singular(monarch_obs):
    meta = build_metadata(monarch_obs)
    assert "monarch" in meta.keywords
    # Positive twin: plural must not appear
    assert "monarchs" not in meta.keywords


def test_keywords_contains_butterfly_singular(monarch_obs):
    meta = build_metadata(monarch_obs)
    assert "butterfly" in meta.keywords
    # Positive twin: plural must not appear
    assert "butterflies" not in meta.keywords


def test_keywords_contains_genus_danaus(monarch_obs):
    meta = build_metadata(monarch_obs)
    assert "danaus" in meta.keywords


def test_keywords_contains_state_colorado(monarch_obs):
    meta = build_metadata(monarch_obs)
    assert "colorado" in meta.keywords
    # Positive twin: wrong state must not appear
    assert "california" not in meta.keywords


def test_keywords_contains_summer_season(monarch_obs):
    meta = build_metadata(monarch_obs)
    assert "summer" in meta.keywords
    # Positive twin: wrong season must not appear
    assert "winter" not in meta.keywords


def test_keywords_contains_seed_migration(monarch_obs):
    meta = build_metadata(monarch_obs)
    assert "migration" in meta.keywords


def test_keywords_contains_wildlife(monarch_obs):
    meta = build_metadata(monarch_obs)
    assert "wildlife" in meta.keywords
    # Positive twin: "captive" should not appear
    assert "captive" not in meta.keywords


def test_keywords_contains_macro_for_insect(monarch_obs):
    meta = build_metadata(monarch_obs)
    # Insects get macro keyword
    assert "macro" in meta.keywords
    # Plants also get macro; mammals should not
    mammal_obs = Observation(
        taxon={"name": "Ursus arctos", "common_name": "Brown Bear",
               "rank": "species", "iconic_taxon_name": "Mammalia"},
        place_guess="Yellowstone, Wyoming",
    )
    mammal_meta = build_metadata(mammal_obs)
    assert "macro" not in mammal_meta.keywords


# ---------------------------------------------------------------------------
# 4. Keyword list integrity
# ---------------------------------------------------------------------------

def test_keywords_count_in_range(monarch_obs):
    meta = build_metadata(monarch_obs)
    assert 15 <= len(meta.keywords) <= 35


def test_keywords_no_duplicates(monarch_obs):
    meta = build_metadata(monarch_obs)
    assert len(meta.keywords) == len(set(meta.keywords))


def test_keywords_all_lowercase(monarch_obs):
    meta = build_metadata(monarch_obs)
    for kw in meta.keywords:
        assert kw == kw.lower(), f"keyword not lowercase: {kw!r}"


def test_keywords_first_10_most_relevant(monarch_obs):
    meta = build_metadata(monarch_obs)
    first_ten = meta.keywords[:10]
    # Species name, common name, and season must be in the first 10 slots
    assert "monarch" in first_ten
    assert "danaus" in first_ten
    assert "summer" in first_ten


# ---------------------------------------------------------------------------
# 5. No brand or people names
# ---------------------------------------------------------------------------

_BANNED_TERMS = {
    "nikon", "canon", "sony", "fujifilm", "leica", "olympus",
    "adobe", "lightroom", "photoshop",
    "inat", "inaturalist",
}


def test_keywords_no_brand_names(monarch_obs):
    meta = build_metadata(monarch_obs)
    for kw in meta.keywords:
        assert kw not in _BANNED_TERMS, f"brand keyword found: {kw!r}"


def test_keywords_no_observation_id(monarch_obs):
    obs = Observation(
        observation_id=12345678,
        obsdate="2026-08-15",
        taxon={"name": "Danaus plexippus", "common_name": "Monarch Butterfly",
               "rank": "species", "iconic_taxon_name": "Insecta"},
        place_guess="Colorado, USA",
    )
    meta = build_metadata(obs)
    assert "12345678" not in meta.keywords
    assert str(obs.observation_id) not in meta.keywords


# ---------------------------------------------------------------------------
# 6. Caption checks
# ---------------------------------------------------------------------------

def test_caption_starts_with_common_name(monarch_obs):
    meta = build_metadata(monarch_obs)
    assert meta.caption.startswith("Monarch Butterfly")


def test_caption_contains_scientific_name(monarch_obs):
    meta = build_metadata(monarch_obs)
    assert "Danaus plexippus" in meta.caption


def test_caption_contains_place(monarch_obs):
    meta = build_metadata(monarch_obs)
    assert "Rocky Mountain National Park" in meta.caption


def test_caption_contains_date(monarch_obs):
    meta = build_metadata(monarch_obs)
    assert "2026-08-15" in meta.caption


def test_caption_length_at_most_500(monarch_obs):
    meta = build_metadata(monarch_obs)
    assert len(meta.caption) <= 500


# ---------------------------------------------------------------------------
# 7. Categories
# ---------------------------------------------------------------------------

def test_categories_insect_is_animals(monarch_obs):
    meta = build_metadata(monarch_obs)
    assert "Animals" in meta.categories


def test_categories_plant_is_plants_and_flowers():
    obs = Observation(
        taxon={"name": "Rosa canina", "common_name": "Dog Rose",
               "rank": "species", "iconic_taxon_name": "Plantae"},
    )
    meta = build_metadata(obs)
    assert "Plants and Flowers" in meta.categories
    assert "Animals" not in meta.categories


def test_categories_count_in_range(monarch_obs):
    meta = build_metadata(monarch_obs)
    assert 1 <= len(meta.categories) <= 2


# ---------------------------------------------------------------------------
# 8. Seasonal mapping
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("obsdate,expected_season", [
    ("2026-01-10", "winter"),
    ("2026-04-22", "spring"),
    ("2026-07-04", "summer"),
    ("2026-10-31", "autumn"),
    ("2026-12-25", "winter"),
])
def test_season_keyword_matches_month(obsdate, expected_season):
    obs = Observation(
        obsdate=obsdate,
        taxon={"name": "Danaus plexippus", "common_name": "Monarch Butterfly",
               "rank": "species", "iconic_taxon_name": "Insecta"},
    )
    meta = build_metadata(obs)
    assert expected_season in meta.keywords
    # Positive twin: no other season keyword present
    other_seasons = {"winter", "spring", "summer", "autumn"} - {expected_season}
    for other in other_seasons:
        assert other not in meta.keywords, f"unexpected season {other!r} for {obsdate}"


# ---------------------------------------------------------------------------
# 9. build_keywords backward-compat wrapper
# ---------------------------------------------------------------------------

def test_build_keywords_returns_dict_with_required_keys(monarch_obs):
    result = build_keywords(monarch_obs)
    assert isinstance(result, dict)
    for key in ("title", "caption", "keywords", "source_taxon", "source_observation_id"):
        assert key in result, f"missing key {key!r}"


def test_build_keywords_source_taxon(monarch_obs):
    result = build_keywords(monarch_obs)
    assert result["source_taxon"] == "Danaus plexippus"
    assert result["source_taxon"] != "Unknown"


def test_build_keywords_keywords_is_list(monarch_obs):
    result = build_keywords(monarch_obs)
    assert isinstance(result["keywords"], list)
    assert len(result["keywords"]) >= 15


# ---------------------------------------------------------------------------
# 10. Observation with no optional fields (graceful degradation)
# ---------------------------------------------------------------------------

def test_minimal_observation_no_error():
    obs = Observation(
        taxon={"name": "Quercus robur", "common_name": "English Oak",
               "rank": "species", "iconic_taxon_name": "Plantae"},
    )
    meta = build_metadata(obs)
    assert "english" in meta.keywords
    assert "oak" in meta.keywords
    # No season without obsdate
    for season in ("winter", "spring", "summer", "autumn"):
        assert season not in meta.keywords


def test_empty_taxon_does_not_raise():
    obs = Observation(taxon={})
    meta = build_metadata(obs)
    assert isinstance(meta, StockMetadata)
    assert 15 <= len(meta.keywords) <= 35
