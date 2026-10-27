"""Thin settings-bound wrapper around inat_client (src/inat_client), a self-contained, settings-free iNaturalist v1 API client
module that ships in this same repo but has no dependency on bio_automation itself -- it could be lifted into its own package
unchanged if ever needed.

The generic client logic (retries, reshaping, rate limiting, the taxon-lineage cache) lives in inat_client; this module exists
only to bind its defaults to bio-automation's own settings (file locations, retry/backoff tuning), so every existing caller in
this package keeps working unchanged. New code with no need for that binding can use inat_client directly."""
from __future__ import annotations

import urllib.request  # noqa: F401 -- re-exported so tests can patch urllib.request.urlopen via this module's attribute path
from typing import Callable

from inat_client import (  # noqa: F401 -- re-exported for existing importers of bio_automation.inat
    INAT_API_BASE,
    InatApiError,
    InatNotFoundError,
    InatTransientError,
    RateLimiter,
    SharedRateLimiter,
    TaxonLineageCache,
    USER_AGENT_DEFAULT,
    fetch_inat_observation as _lib_fetch_inat_observation,
    fetch_inat_observations as _lib_fetch_inat_observations,
    fetch_taxon_lineage as _lib_fetch_taxon_lineage,
    fetch_taxon_lineages as _lib_fetch_taxon_lineages,
)
from inat_client.client import _reshape  # noqa: F401 -- re-exported: tests/test_stock.py builds fixtures with it directly
from inat_client.client import default_get as _lib_default_get

from . import settings

OBSERVATIONS_PER_REQUEST = 200      # kept here too: pipeline.py and tests reference bio_automation.inat.OBSERVATIONS_PER_REQUEST directly
TAXA_PER_REQUEST = 30

# Module-level limiter shared by the default transport below: one pace for every process on this machine (settings.INAT_RATE_DB), so
# fetch calls stay paced even when callers don't pass their own limiter, and parallel workers do not add up to a burst.
_default_rate_limiter = SharedRateLimiter(settings.INAT_RATE_DB, settings.INAT_MIN_INTERVAL_S)


def _default_get(url: str, user_agent: str, *, urlopen: Callable | None = None, sleep: Callable[[float], None] | None = None,
                 retries: int | None = None, backoff_s: float | None = None, backoff_max_s: float | None = None,
                 hold_off: Callable[[float], None] | None = None) -> dict:
    """inat_client.client.default_get, with bio-automation's own settings.INAT_RETRIES/BACKOFF_S/BACKOFF_MAX_S as the
    fallback defaults (never inat_client's own)."""
    return _lib_default_get(
        url, user_agent, urlopen=urlopen, sleep=sleep,
        retries=settings.INAT_RETRIES if retries is None else retries,
        backoff_s=settings.INAT_BACKOFF_S if backoff_s is None else backoff_s,
        backoff_max_s=settings.INAT_BACKOFF_MAX_S if backoff_max_s is None else backoff_max_s,
        hold_off=hold_off or _default_rate_limiter.hold_off,
    )


def fetch_taxon_lineage(taxon_id: int, *, get_fn: Callable[[str, str], dict] | None = None, user_agent: str | None = None,
                        rate_limiter: RateLimiter | None = None) -> list[dict]:
    """See inat_client.fetch_taxon_lineage. Defaults to this module's own settings-bound transport/limiter."""
    return _lib_fetch_taxon_lineage(taxon_id, get_fn=get_fn or _default_get, user_agent=user_agent,
                                    rate_limiter=rate_limiter if rate_limiter is not None else _default_rate_limiter)


def fetch_inat_observation(observation_id: int, *, get_fn: Callable[[str, str], dict] | None = None,
                            user_agent: str | None = None,
                            rate_limiter: RateLimiter | None = None,
                            with_lineage: bool = False, lineage_cache: dict | None = None) -> dict:
    """See inat_client.fetch_inat_observation. Defaults to this module's own settings-bound transport/limiter."""
    return _lib_fetch_inat_observation(observation_id, get_fn=get_fn or _default_get, user_agent=user_agent,
                                       rate_limiter=rate_limiter if rate_limiter is not None else _default_rate_limiter,
                                       with_lineage=with_lineage, lineage_cache=lineage_cache)


def fetch_taxon_lineages(taxon_ids, *, get_fn: Callable[[str, str], dict] | None = None, user_agent: str | None = None,
                         rate_limiter=None) -> dict[int, list[dict]]:
    """See inat_client.fetch_taxon_lineages. Defaults to this module's own settings-bound transport/limiter."""
    return _lib_fetch_taxon_lineages(taxon_ids, get_fn=get_fn or _default_get, user_agent=user_agent,
                                     rate_limiter=rate_limiter if rate_limiter is not None else _default_rate_limiter)


def fetch_inat_observations(observation_ids, *, get_fn: Callable[[str, str], dict] | None = None, user_agent: str | None = None,
                            rate_limiter=None, with_lineage: bool = False, lineage_cache=None) -> dict[int, dict | InatNotFoundError]:
    """See inat_client.fetch_inat_observations. Defaults to this module's own settings-bound transport/limiter.

    OBSERVATIONS_PER_REQUEST here is a plain re-export for introspection/tests: changing it does not affect the library's own
    chunking (it chunks by inat_client.client.OBSERVATIONS_PER_REQUEST, monkeypatch that instead to change real batch size)."""
    return _lib_fetch_inat_observations(observation_ids, get_fn=get_fn or _default_get, user_agent=user_agent,
                                        rate_limiter=rate_limiter if rate_limiter is not None else _default_rate_limiter,
                                        with_lineage=with_lineage, lineage_cache=lineage_cache)
