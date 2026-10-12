from __future__ import annotations

from .models import Observation


def build_keywords(observation: Observation) -> dict:
    """Build a template-only {title, caption, keywords, source_taxon,
    source_observation_id} record from the observation's taxon,
    keywords_seed, and locality.

    One record per observation folder, not per asset -- metadata/keywords.json
    is a single shared file, so this takes the folder-level Observation, not
    a per-file Asset. No NLP/LLM generation in v1.
    """
    raise NotImplementedError("build_keywords will be added next.")
