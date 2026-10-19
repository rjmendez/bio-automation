"""Unit tests for pipeline.merge_inat_fields. Pure function, no I/O, no
stubbing needed -- exercises the real merge logic directly."""
from __future__ import annotations

from bio_automation.models import Observation
from bio_automation.pipeline import merge_inat_fields


def _fetched(**overrides) -> dict:
    base = {
        "observation_id": 999,
        "source": "inaturalist",
        "taxon": {"id": 1, "name": "Araneae"},
        "place_guess": "Marin County, CA",
        "license": "cc-by-nc",
        "quality_grade": "research",
        "keywords_seed": ["spider", "nocturnal"],
    }
    base.update(overrides)
    return base


class TestMergeInatFields:
    def test_overwrites_only_inat_owned_fields(self):
        existing = Observation(
            observation_id=None, obsdate="2026-09-01", source="local-ingest",
            taxon={}, place_guess=None, locality="backyard, under the porch light",
            license=None, quality_grade=None, keywords_seed=["porch-light"],
        )

        merged = merge_inat_fields(existing, _fetched())

        assert merged.observation_id == 999
        assert merged.source == "inaturalist"
        assert merged.taxon == {"id": 1, "name": "Araneae"}
        assert merged.place_guess == "Marin County, CA"
        assert merged.license == "cc-by-nc"
        assert merged.quality_grade == "research"

    def test_never_rewrites_obsdate_or_locality(self):
        existing = Observation(obsdate="2026-09-01", locality="backyard, under the porch light")

        merged = merge_inat_fields(existing, _fetched())

        assert merged.obsdate == "2026-09-01"
        assert merged.locality == "backyard, under the porch light"

    def test_unions_keywords_seed_instead_of_replacing_it(self):
        existing = Observation(keywords_seed=["porch-light", "spider"])

        merged = merge_inat_fields(existing, _fetched(keywords_seed=["spider", "nocturnal"]))

        assert merged.keywords_seed == ["nocturnal", "porch-light", "spider"]

    def test_missing_fetched_field_leaves_existing_value_untouched(self):
        existing = Observation(license="cc-by")
        fetched = _fetched()
        del fetched["license"]

        merged = merge_inat_fields(existing, fetched)

        assert merged.license == "cc-by"

    def test_does_not_mutate_the_existing_observation_in_place(self):
        existing = Observation(observation_id=None, keywords_seed=["porch-light"])

        merge_inat_fields(existing, _fetched())

        assert existing.observation_id is None
        assert existing.keywords_seed == ["porch-light"]
