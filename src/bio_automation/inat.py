from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Callable

INAT_API_BASE = "https://api.inaturalist.org/v1"

# No contact address is baked in here on purpose -- set INAT_USER_AGENT in the
# environment to add one. iNaturalist only requires an identified client, not
# a specific contact string.
USER_AGENT_DEFAULT = "bio-automation/0.1 (+https://github.com/; read-only iNaturalist client)"


class InatApiError(Exception):
    """A non-2xx response from the iNaturalist API that isn't a 404."""


class InatNotFoundError(InatApiError):
    """The requested observation does not exist (404), or the API returned
    an empty results list for it."""


class RateLimiter:
    """Blocking rate limiter: wait() sleeps (via *sleep_fn*) just long enough
    that consecutive wait() calls are never closer together than
    *min_interval* seconds. Legacy pipeline/stages/inat_console.py's
    taxa_search() and inat_post.py's pace() both use `1.1 - elapsed` against
    the public iNaturalist API; the default here matches that exactly
    (min_interval=1.1).

    *time_fn*/*sleep_fn* are injected (default time.monotonic/time.sleep) so
    tests can run this with a fake clock instead of a real sleep.
    """

    def __init__(self, min_interval: float = 1.0, *,
                 time_fn: Callable[[], float] = time.monotonic,
                 sleep_fn: Callable[[float], None] = time.sleep):
        if min_interval < 0:
            raise ValueError(f"min_interval must be >= 0, got {min_interval!r}")
        self.min_interval = min_interval
        self._time_fn = time_fn
        self._sleep_fn = sleep_fn
        self._last_call: float | None = None

    def wait(self) -> float:
        """Sleep as needed, then record this call's time. Returns the number
        of seconds actually slept (0.0 for the first call, or any call that
        is already far enough after the previous one)."""
        now = self._time_fn()
        slept = 0.0
        if self._last_call is not None:
            remaining = self.min_interval - (now - self._last_call)
            if remaining > 0:
                self._sleep_fn(remaining)
                slept = remaining
                now = now + remaining
        self._last_call = now
        return slept


# Module-level limiter shared by the default transport below, so repeated
# fetch_inat_observation() calls in-process stay paced even when callers
# don't pass their own RateLimiter.
_default_rate_limiter = RateLimiter(min_interval=1.1)


def _default_get(url: str, user_agent: str) -> dict:
    """Real transport: one unauthenticated GET. This is the network boundary
    -- tests pass their own get_fn instead of exercising this against the
    live API."""
    request = urllib.request.Request(url, headers={"User-Agent": user_agent})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise InatNotFoundError(f"observation not found (404): {url}") from exc
        raise InatApiError(f"iNaturalist API error {exc.code} for {url}") from exc


def _reshape(observation: dict, requested_id: int) -> dict:
    """Reshape one raw iNat API observation object into bio-automation's own
    field names (see fetch_inat_observation()'s docstring for the exact set).
    """
    taxon = observation.get("taxon") or {}
    tags = observation.get("tags") or []
    keywords_seed = sorted({tag for tag in tags if isinstance(tag, str) and tag})
    return {
        "observation_id": observation.get("id", requested_id),
        "source": "inaturalist",
        "taxon": {
            "id": taxon.get("id"),
            "name": taxon.get("name"),
            "preferred_common_name": taxon.get("preferred_common_name"),
            "rank": taxon.get("rank"),
        },
        "place_guess": observation.get("place_guess"),
        "license": observation.get("license_code"),
        "quality_grade": observation.get("quality_grade"),
        "keywords_seed": keywords_seed,
    }


def fetch_inat_observation(observation_id: int, *, get_fn: Callable[[str, str], dict] | None = None,
                            user_agent: str | None = None,
                            rate_limiter: RateLimiter | None = None) -> dict:
    """Fetch one observation from the public iNaturalist v1 API (read-only,
    unauthenticated GET to ``{INAT_API_BASE}/observations/{id}``) and
    reshape it into the iNat-owned subset of observation.json fields:
    observation_id, source, taxon, place_guess, license, quality_grade,
    keywords_seed. Returns bio-automation's own field names, not iNat's raw
    JSON shape. Performs no writes -- callers merge the result into an
    existing Observation via pipeline.merge_inat_fields().

    *get_fn* is the network boundary (url, user_agent) -> parsed JSON dict;
    defaults to a real urllib GET. *rate_limiter* defaults to a shared,
    module-level RateLimiter paced at <=1 req/s, matching the legacy
    scripts' own pacing convention for this API.

    Raises InatNotFoundError if the id doesn't exist (404, or a 200 with an
    empty results list) and InatApiError for any other non-2xx response.
    """
    get_fn = get_fn or _default_get
    limiter = rate_limiter if rate_limiter is not None else _default_rate_limiter
    limiter.wait()

    url = f"{INAT_API_BASE}/observations/{observation_id}"
    payload = get_fn(url, user_agent or USER_AGENT_DEFAULT)

    results = payload.get("results") or []
    if not results:
        raise InatNotFoundError(f"observation {observation_id}: no results in API response")
    return _reshape(results[0], observation_id)
