from __future__ import annotations

INAT_API_BASE = "https://api.inaturalist.org/v1"


def fetch_inat_observation(observation_id: int) -> dict:
    """Fetch one observation from the public iNaturalist v1 API (read-only,
    unauthenticated GET to ``{INAT_API_BASE}/observations/{id}``) and
    reshape it into the iNat-owned subset of observation.json fields:
    observation_id, source, taxon, place_guess, license, quality_grade,
    keywords_seed. Returns bio-automation's own field names, not iNat's raw
    JSON shape. Performs no writes -- callers merge the result into an
    existing Observation via pipeline.merge_inat_fields().
    """
    raise NotImplementedError("Read-only iNaturalist fetch will be added next.")

