from __future__ import annotations

import json
import socket
import sqlite3
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

from . import settings

INAT_API_BASE = "https://api.inaturalist.org/v1"

# No contact address is baked in here on purpose -- set INAT_USER_AGENT in the
# environment to add one. iNaturalist only requires an identified client, not
# a specific contact string.
USER_AGENT_DEFAULT = "bio-automation/0.1 (+https://github.com/; read-only iNaturalist client)"


class InatApiError(Exception):
    """A non-2xx response from the iNaturalist API that isn't a 404."""


class InatTransientError(InatApiError):
    """A rate limit (429), a server error (5xx) or a network failure that was still there after every retry. It says nothing about the
    observation itself: a caller must fail or retry its work, never record it as a refusal of that observation."""


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


class SharedRateLimiter:
    """RateLimiter's contract across processes: every process (and thread) using the same *path* reserves its own slot in one SQLite row,
    so all of them together make no more than one request per *min_interval* seconds. wait() reserves the next free slot inside a short
    transaction and sleeps outside it, so no lock is held while waiting. hold_off(s) pushes the next slot of every process at least *s*
    seconds out (a 429 slows the whole machine down, not only the worker that got it).

    The reservation uses wall-clock time (time_fn, default time.time): a monotonic clock is not comparable between processes. A slot further
    out than *max_wait_s* (a clock that jumped back, a stale file) is not waited for; the wait is cut to *max_wait_s*."""

    def __init__(self, path: Path | str, min_interval: float, *, max_wait_s: float | None = None,
                 time_fn: Callable[[], float] = time.time, sleep_fn: Callable[[float], None] = time.sleep):
        if min_interval < 0:
            raise ValueError(f"min_interval must be >= 0, got {min_interval!r}")
        self.path = Path(path)
        self.min_interval = min_interval
        self.max_wait_s = settings.INAT_BACKOFF_MAX_S if max_wait_s is None else max_wait_s
        self._time_fn = time_fn
        self._sleep_fn = sleep_fn
        self._lock = threading.Lock()
        self._db: sqlite3.Connection | None = None

    def _conn(self) -> sqlite3.Connection:
        if self._db is None:                    # opened on first use: importing this module never creates the file
            self.path.parent.mkdir(parents=True, exist_ok=True)
            db = sqlite3.connect(str(self.path), timeout=30, isolation_level=None, check_same_thread=False)
            db.execute("CREATE TABLE IF NOT EXISTS pace (k INTEGER PRIMARY KEY CHECK (k = 0), next_slot REAL NOT NULL)")
            self._db = db
        return self._db

    def _update(self, step: Callable[[float, float], tuple[float, float]]) -> float:
        """In one write transaction: (now, stored next slot or now) -> (new next slot, value to return)."""
        with self._lock:
            db = self._conn()
            db.execute("BEGIN IMMEDIATE")
            try:
                now = self._time_fn()
                row = db.execute("SELECT next_slot FROM pace WHERE k = 0").fetchone()
                next_slot, result = step(now, now if row is None else min(row[0], now + self.max_wait_s))
                db.execute("INSERT INTO pace (k, next_slot) VALUES (0, ?) ON CONFLICT (k) DO UPDATE SET next_slot = excluded.next_slot",
                           (next_slot,))
                db.execute("COMMIT")
            except BaseException:
                db.execute("ROLLBACK")
                raise
        return result

    def wait(self) -> float:
        """Reserve the next free slot, sleep until it, return the seconds slept (0.0 when the slot is now)."""
        def step(now, stored):
            slot = max(now, stored)
            return slot + self.min_interval, slot - now
        slept = self._update(step)
        if slept > 0:
            self._sleep_fn(slept)
        return slept

    def hold_off(self, seconds: float) -> None:
        """No process gets a slot sooner than *seconds* from now."""
        self._update(lambda now, stored: (max(stored, now + seconds), 0.0))


class TaxonLineageCache:
    """{taxon id: lineage} kept in SQLite (settings.INAT_TAXON_CACHE, on D:) so a taxon's ancestors are fetched once per *ttl_s*, not once
    per process. An entry older than *ttl_s* counts as absent (taxonomy changes, slowly). Supports `in`, `[]`, `[] =` and get(), which is
    all fetch_inat_observation()/fetch_inat_observations() use, so a plain dict works in its place."""

    def __init__(self, path: Path | str, ttl_s: float, *, time_fn: Callable[[], float] = time.time):
        self.path = Path(path)
        self.ttl_s = ttl_s
        self._time_fn = time_fn
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(self.path), timeout=30, isolation_level=None, check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("CREATE TABLE IF NOT EXISTS lineage (taxon_id INTEGER PRIMARY KEY, lineage TEXT NOT NULL, fetched REAL NOT NULL)")

    def get(self, taxon_id: int, default=None):
        row = self._db.execute("SELECT lineage FROM lineage WHERE taxon_id = ? AND fetched > ?",
                               (taxon_id, self._time_fn() - self.ttl_s)).fetchone()
        return default if row is None else json.loads(row[0])

    def __contains__(self, taxon_id) -> bool:
        return self.get(taxon_id) is not None

    def __getitem__(self, taxon_id: int) -> list[dict]:
        found = self.get(taxon_id)
        if found is None:
            raise KeyError(taxon_id)
        return found

    def __setitem__(self, taxon_id: int, lineage: list[dict]) -> None:
        self._db.execute("INSERT OR REPLACE INTO lineage (taxon_id, lineage, fetched) VALUES (?, ?, ?)",
                         (taxon_id, json.dumps(lineage), self._time_fn()))


# Module-level limiter shared by the default transport below: one pace for every process on this machine (settings.INAT_RATE_DB), so
# fetch calls stay paced even when callers don't pass their own limiter, and parallel workers do not add up to a burst.
_default_rate_limiter = SharedRateLimiter(settings.INAT_RATE_DB, settings.INAT_MIN_INTERVAL_S)


RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


def _retry_after(exc: urllib.error.HTTPError) -> float | None:
    """The Retry-After header in seconds (the API sends a number), or None when it is absent or not a number."""
    value = (exc.headers or {}).get("Retry-After")
    try:
        return max(0.0, float(value)) if value is not None else None
    except ValueError:
        return None


def _default_get(url: str, user_agent: str, *, urlopen: Callable | None = None, sleep: Callable[[float], None] | None = None,
                 retries: int | None = None, backoff_s: float | None = None, backoff_max_s: float | None = None,
                 hold_off: Callable[[float], None] | None = None) -> dict:
    """Real transport: one unauthenticated GET. This is the network boundary
    -- tests pass their own get_fn instead of exercising this against the
    live API.

    404 raises InatNotFoundError at once. 429, 5xx and network errors (URLError, timeouts) are retried up to settings.INAT_RETRIES times,
    waiting settings.INAT_BACKOFF_S doubling each time (or the response's Retry-After, if longer), never more than settings.INAT_BACKOFF_MAX_S
    per wait; when they persist, InatTransientError. Any other HTTP status raises InatApiError at once. A 429 also calls hold_off(that
    wait) -- default: the machine-wide limiter's, so every other worker backs off too."""
    urlopen = urlopen or urllib.request.urlopen
    sleep = sleep or time.sleep
    hold_off = hold_off or _default_rate_limiter.hold_off
    retries = settings.INAT_RETRIES if retries is None else retries
    wait = settings.INAT_BACKOFF_S if backoff_s is None else backoff_s
    cap = settings.INAT_BACKOFF_MAX_S if backoff_max_s is None else backoff_max_s
    request = urllib.request.Request(url, headers={"User-Agent": user_agent})
    for attempt in range(retries + 1):
        try:
            with urlopen(request, timeout=30) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise InatNotFoundError(f"observation not found (404): {url}") from exc
            if exc.code not in RETRYABLE_STATUS:
                raise InatApiError(f"iNaturalist API error {exc.code} for {url}") from exc
            last, hinted = exc, _retry_after(exc)
            what = f"iNaturalist API error {exc.code}"
        except (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError) as exc:
            last, hinted = exc, None
            what = f"network error {type(exc).__name__}: {exc}"
        if attempt == retries:
            raise InatTransientError(f"{what} for {url}, still failing after {retries + 1} attempts") from last
        delay = min(cap, max(wait, hinted or 0.0))
        if isinstance(last, urllib.error.HTTPError) and last.code == 429:
            hold_off(delay)
        sleep(delay)
        wait *= 2
    raise AssertionError("unreachable")


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
            # stock.py reads these two names (not the iNat API's own).
            "common_name": taxon.get("preferred_common_name"),
            "iconic_taxon_name": taxon.get("iconic_taxon_name"),
            "rank": taxon.get("rank"),
        },
        "place_guess": observation.get("place_guess"),
        "license": observation.get("license_code"),
        "quality_grade": observation.get("quality_grade"),
        "keywords_seed": keywords_seed,
    }


def fetch_taxon_lineage(taxon_id: int, *, get_fn: Callable[[str, str], dict] | None = None, user_agent: str | None = None,
                        rate_limiter: RateLimiter | None = None) -> list[dict]:
    """The taxonomic ancestors of a taxon, root first, from the public iNaturalist v1 API (``{INAT_API_BASE}/taxa/{id}``, read-only):
    [{"rank": "family", "name": "Gomphaceae", "common_name": None | "..."}, ...]. The taxon itself is not included.
    Raises InatNotFoundError when the taxon does not exist and InatApiError for any other non-2xx response."""
    get_fn = get_fn or _default_get
    limiter = rate_limiter if rate_limiter is not None else _default_rate_limiter
    limiter.wait()
    payload = get_fn(f"{INAT_API_BASE}/taxa/{taxon_id}", user_agent or USER_AGENT_DEFAULT)
    results = payload.get("results") or []
    if not results:
        raise InatNotFoundError(f"taxon {taxon_id}: no results in API response")
    return [{"rank": a.get("rank"), "name": a.get("name"), "common_name": a.get("preferred_common_name")}
            for a in (results[0].get("ancestors") or []) if a.get("name")]


def fetch_inat_observation(observation_id: int, *, get_fn: Callable[[str, str], dict] | None = None,
                            user_agent: str | None = None,
                            rate_limiter: RateLimiter | None = None,
                            with_lineage: bool = False, lineage_cache: dict | None = None) -> dict:
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

    *with_lineage* adds one more request (fetch_taxon_lineage) and stores the ancestors under taxon["lineage"], which stock keywords use
    (family/order/class names and their common names). *lineage_cache* ({taxon id: lineage}) shares those lookups across calls.

    Raises InatNotFoundError if the id doesn't exist (404, or a 200 with an
    empty results list), InatTransientError when a rate limit / server / network error outlasts the transport's retries, and InatApiError
    for any other non-2xx response.
    """
    get_fn = get_fn or _default_get
    limiter = rate_limiter if rate_limiter is not None else _default_rate_limiter
    limiter.wait()

    url = f"{INAT_API_BASE}/observations/{observation_id}"
    payload = get_fn(url, user_agent or USER_AGENT_DEFAULT)

    results = payload.get("results") or []
    if not results:
        raise InatNotFoundError(f"observation {observation_id}: no results in API response")
    fetched = _reshape(results[0], observation_id)
    taxon_id = fetched["taxon"].get("id")
    if with_lineage and taxon_id is not None:
        cache = lineage_cache if lineage_cache is not None else {}
        if taxon_id not in cache:
            cache[taxon_id] = fetch_taxon_lineage(taxon_id, get_fn=get_fn, user_agent=user_agent, rate_limiter=limiter)
        fetched["taxon"]["lineage"] = cache[taxon_id]
    return fetched


OBSERVATIONS_PER_REQUEST = 200      # the API's per_page maximum
TAXA_PER_REQUEST = 30               # /taxa/{ids} takes at most 30 ids


def _chunks(ids: list[int], size: int) -> list[list[int]]:
    return [ids[i:i + size] for i in range(0, len(ids), size)]


def fetch_taxon_lineages(taxon_ids, *, get_fn: Callable[[str, str], dict] | None = None, user_agent: str | None = None,
                         rate_limiter=None) -> dict[int, list[dict]]:
    """fetch_taxon_lineage() for many taxa, TAXA_PER_REQUEST per request (``{INAT_API_BASE}/taxa/{a,b,...}``): {taxon id: lineage}.
    A taxon the API does not return is absent from the result."""
    get_fn = get_fn or _default_get
    limiter = rate_limiter if rate_limiter is not None else _default_rate_limiter
    out = {}
    for chunk in _chunks(sorted(set(taxon_ids)), TAXA_PER_REQUEST):
        limiter.wait()
        payload = get_fn(f"{INAT_API_BASE}/taxa/{','.join(map(str, chunk))}", user_agent or USER_AGENT_DEFAULT)
        for t in payload.get("results") or []:
            if t.get("id") in chunk:
                out[t["id"]] = [{"rank": a.get("rank"), "name": a.get("name"), "common_name": a.get("preferred_common_name")}
                                for a in (t.get("ancestors") or []) if a.get("name")]
    return out


def fetch_inat_observations(observation_ids, *, get_fn: Callable[[str, str], dict] | None = None, user_agent: str | None = None,
                            rate_limiter=None, with_lineage: bool = False, lineage_cache=None) -> dict[int, dict | InatNotFoundError]:
    """fetch_inat_observation() for many observations: OBSERVATIONS_PER_REQUEST per request (``{INAT_API_BASE}/observations?id=a,b,...``),
    then one /taxa request per TAXA_PER_REQUEST taxa that *lineage_cache* does not already hold. Same reshaped fields, still fetched live.

    Returns {observation id: reshaped observation, or the InatNotFoundError fetch_inat_observation() would have raised for it} for every
    requested id: an observation the API does not return (deleted, or not visible) or whose taxon it does not return is that error, so a
    caller refuses only that observation. Transient and other API errors raise, as in fetch_inat_observation()."""
    get_fn = get_fn or _default_get
    limiter = rate_limiter if rate_limiter is not None else _default_rate_limiter
    ua = user_agent or USER_AGENT_DEFAULT
    ids = sorted(set(observation_ids))
    found: dict[int, dict] = {}
    for chunk in _chunks(ids, OBSERVATIONS_PER_REQUEST):
        limiter.wait()
        payload = get_fn(f"{INAT_API_BASE}/observations?id={','.join(map(str, chunk))}&per_page={len(chunk)}", ua)
        for o in payload.get("results") or []:
            if o.get("id") in chunk:
                found[o["id"]] = _reshape(o, o["id"])
    out: dict[int, dict | InatNotFoundError] = {
        i: found.get(i) or InatNotFoundError(f"observation {i}: no results in API response") for i in ids}
    if with_lineage:
        cache = lineage_cache if lineage_cache is not None else {}
        wanted = {o["taxon"]["id"] for o in found.values() if o["taxon"].get("id") is not None}
        fetched = fetch_taxon_lineages([t for t in wanted if t not in cache], get_fn=get_fn, user_agent=user_agent, rate_limiter=limiter)
        for t, lineage in fetched.items():
            cache[t] = lineage
        for i, o in found.items():
            t = o["taxon"].get("id")
            if t is None:
                continue
            lineage = cache.get(t)
            if lineage is None:
                out[i] = InatNotFoundError(f"taxon {t}: no results in API response")
            else:
                o["taxon"]["lineage"] = lineage
    return out
