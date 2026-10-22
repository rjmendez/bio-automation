"""Unit tests for bio_automation.inat (hermetic -- the urllib network
boundary is always stubbed via get_fn/monkeypatch, never hit for real)."""
from __future__ import annotations

import urllib.error

import pytest

from bio_automation.inat import (
    InatApiError,
    InatNotFoundError,
    InatTransientError,
    RateLimiter,
    fetch_inat_observation,
    fetch_taxon_lineage,
)


RAW_OBSERVATION = {
    "id": 123456,
    "place_guess": "Marin County, CA",
    "license_code": "cc-by-nc",
    "quality_grade": "research",
    "tags": ["burrowing", "nocturnal"],
    "taxon": {
        "id": 48484,
        "name": "Araneae",
        "preferred_common_name": "spiders",
        "iconic_taxon_name": "Arachnida",
        "rank": "order",
    },
}


class RecordingGet:
    """Stands in for inat._default_get: records every call, returns a
    configured payload (or raises a configured exception)."""

    def __init__(self, payload=None, to_raise=None):
        self.calls: list[dict] = []
        self.payload = payload
        self.to_raise = to_raise

    def __call__(self, url, user_agent):
        self.calls.append({"url": url, "user_agent": user_agent})
        if self.to_raise is not None:
            raise self.to_raise
        return self.payload


class TestFetchInatObservation:
    def test_reshapes_a_found_observation_into_owned_fields(self):
        get_fn = RecordingGet(payload={"results": [RAW_OBSERVATION]})

        result = fetch_inat_observation(123456, get_fn=get_fn, rate_limiter=RateLimiter(min_interval=0))

        assert result == {
            "observation_id": 123456,
            "source": "inaturalist",
            "taxon": {
                "id": 48484,
                "name": "Araneae",
                "preferred_common_name": "spiders",
                "common_name": "spiders",
                "iconic_taxon_name": "Arachnida",
                "rank": "order",
            },
            "place_guess": "Marin County, CA",
            "license": "cc-by-nc",
            "quality_grade": "research",
            "keywords_seed": ["burrowing", "nocturnal"],
        }
        assert len(get_fn.calls) == 1
        assert get_fn.calls[0]["url"] == "https://api.inaturalist.org/v1/observations/123456"

    def test_raises_not_found_on_empty_results(self):
        get_fn = RecordingGet(payload={"results": []})

        with pytest.raises(InatNotFoundError, match="123456"):
            fetch_inat_observation(123456, get_fn=get_fn, rate_limiter=RateLimiter(min_interval=0))

    def test_passes_through_get_fn_api_error(self):
        get_fn = RecordingGet(to_raise=InatApiError("iNaturalist API error 500 for ..."))

        with pytest.raises(InatApiError, match="500"):
            fetch_inat_observation(123456, get_fn=get_fn, rate_limiter=RateLimiter(min_interval=0))

    def test_uses_a_default_user_agent_when_none_given(self):
        get_fn = RecordingGet(payload={"results": [RAW_OBSERVATION]})

        fetch_inat_observation(123456, get_fn=get_fn, rate_limiter=RateLimiter(min_interval=0))

        assert get_fn.calls[0]["user_agent"] == "bio-automation/0.1 (+https://github.com/; read-only iNaturalist client)"

    def test_honors_a_custom_user_agent(self):
        get_fn = RecordingGet(payload={"results": [RAW_OBSERVATION]})

        fetch_inat_observation(123456, get_fn=get_fn, user_agent="custom-ua/1.0",
                                rate_limiter=RateLimiter(min_interval=0))

        assert get_fn.calls[0]["user_agent"] == "custom-ua/1.0"

    def test_paces_consecutive_calls_through_the_shared_rate_limiter(self):
        get_fn = RecordingGet(payload={"results": [RAW_OBSERVATION]})
        clock = iter([100.0, 100.0])  # same instant both times -> second call must sleep
        slept: list[float] = []
        limiter = RateLimiter(min_interval=1.0, time_fn=lambda: next(clock), sleep_fn=slept.append)

        fetch_inat_observation(1, get_fn=get_fn, rate_limiter=limiter)
        fetch_inat_observation(2, get_fn=get_fn, rate_limiter=limiter)

        assert slept == [1.0]


class TestRateLimiter:
    def test_first_wait_never_sleeps(self):
        slept: list[float] = []
        limiter = RateLimiter(min_interval=1.0, time_fn=lambda: 50.0, sleep_fn=slept.append)

        result = limiter.wait()

        assert result == 0.0
        assert slept == []

    def test_second_wait_sleeps_the_remaining_interval(self):
        slept: list[float] = []
        times = iter([10.0, 10.4])
        limiter = RateLimiter(min_interval=1.0, time_fn=lambda: next(times), sleep_fn=slept.append)

        limiter.wait()
        second = limiter.wait()

        assert second == pytest.approx(0.6)
        assert slept == [pytest.approx(0.6)]

    def test_wait_does_not_sleep_once_enough_time_has_passed(self):
        slept: list[float] = []
        times = iter([10.0, 12.0])
        limiter = RateLimiter(min_interval=1.0, time_fn=lambda: next(times), sleep_fn=slept.append)

        limiter.wait()
        second = limiter.wait()

        assert second == 0.0
        assert slept == []

    def test_rejects_a_negative_min_interval(self):
        with pytest.raises(ValueError, match="min_interval"):
            RateLimiter(min_interval=-1.0)


class TestDefaultGetTransport:
    """The real urllib transport, with only urllib.request.urlopen itself
    (the actual dependency making the network call) stubbed -- not the
    function under test."""

    def test_404_raises_not_found(self, monkeypatch):
        import bio_automation.inat as inat_module

        def fake_urlopen(request, timeout=30):
            raise urllib.error.HTTPError(request.full_url, 404, "Not Found", {}, None)

        monkeypatch.setattr(inat_module.urllib.request, "urlopen", fake_urlopen)

        with pytest.raises(InatNotFoundError, match="404"):
            inat_module._default_get("https://api.inaturalist.org/v1/observations/999", "ua/1.0")

    def test_500_raises_api_error(self, monkeypatch):
        import bio_automation.inat as inat_module

        def fake_urlopen(request, timeout=30):
            raise urllib.error.HTTPError(request.full_url, 500, "Server Error", {}, None)

        monkeypatch.setattr(inat_module.urllib.request, "urlopen", fake_urlopen)
        sleeps = []

        with pytest.raises(InatApiError, match="500"):
            inat_module._default_get("https://api.inaturalist.org/v1/observations/1", "ua/1.0", sleep=sleeps.append, retries=1)
        assert sleeps == [2.0]


class ScriptedUrlopen:
    """urllib.request.urlopen (the network dependency) replaying a script: an exception instance is raised, a dict is the JSON body."""

    def __init__(self, *script):
        self.script, self.urls = list(script), []

    def __call__(self, request, timeout=30):
        self.urls.append(request.full_url)
        step = self.script.pop(0)
        if isinstance(step, BaseException):
            raise step
        import io
        import json

        return io.BytesIO(json.dumps(step).encode("utf-8"))


def http_error(code, retry_after=None):
    headers = {"Retry-After": retry_after} if retry_after is not None else {}
    return urllib.error.HTTPError("https://api.inaturalist.org/v1/observations/7", code, "x", headers, None)


URL = "https://api.inaturalist.org/v1/observations/7"


class TestDefaultGetRetries:
    """A 429 / 5xx / network error is transient: retried with backoff, and only when it outlasts every retry is it InatTransientError."""

    @pytest.fixture(autouse=True)
    def retry_settings(self, monkeypatch):
        from bio_automation import settings

        monkeypatch.setattr(settings, "INAT_RETRIES", 3)
        monkeypatch.setattr(settings, "INAT_BACKOFF_S", 2.0)
        monkeypatch.setattr(settings, "INAT_BACKOFF_MAX_S", 60.0)

    def test_a_429_is_retried_with_doubling_backoff_until_it_succeeds(self):
        import bio_automation.inat as inat_module

        opener, sleeps = ScriptedUrlopen(http_error(429), http_error(429), {"results": [{"id": 7}]}), []
        assert inat_module._default_get(URL, "ua", urlopen=opener, sleep=sleeps.append) == {"results": [{"id": 7}]}
        assert sleeps == [2.0, 4.0]
        assert opener.urls == [URL, URL, URL]

    def test_retry_after_is_honoured_when_longer_and_capped(self):
        import bio_automation.inat as inat_module

        opener, sleeps = ScriptedUrlopen(http_error(429, "7"), http_error(503, "1"), http_error(429, "900"), {"results": []}), []
        assert inat_module._default_get(URL, "ua", urlopen=opener, sleep=sleeps.append) == {"results": []}
        assert sleeps == [7.0, 4.0, 60.0]                       # 7 > 2; backoff 4 > hint 1; hint 900 capped at 60

    def test_a_persistent_429_is_a_transient_error_after_every_retry(self):
        import bio_automation.inat as inat_module

        opener, sleeps = ScriptedUrlopen(*[http_error(429)] * 4), []
        with pytest.raises(InatTransientError, match=r"iNaturalist API error 429 for https://api\.inaturalist\.org/v1/observations/7, "
                                                      r"still failing after 4 attempts"):
            inat_module._default_get(URL, "ua", urlopen=opener, sleep=sleeps.append)
        assert sleeps == [2.0, 4.0, 8.0]
        assert len(opener.urls) == 4

    def test_a_network_error_is_retried_too(self):
        import bio_automation.inat as inat_module

        opener, sleeps = ScriptedUrlopen(urllib.error.URLError("getaddrinfo failed"), TimeoutError("timed out"), {"results": [{"id": 7}]}), []
        assert inat_module._default_get(URL, "ua", urlopen=opener, sleep=sleeps.append) == {"results": [{"id": 7}]}
        assert sleeps == [2.0, 4.0]

    def test_a_client_error_other_than_429_is_not_retried(self):
        import bio_automation.inat as inat_module

        opener, sleeps = ScriptedUrlopen(http_error(403), {"results": []}), []
        with pytest.raises(InatApiError, match=r"iNaturalist API error 403 for "):
            inat_module._default_get(URL, "ua", urlopen=opener, sleep=sleeps.append)
        assert sleeps == [] and len(opener.urls) == 1

    def test_a_403_is_not_reported_as_transient(self):
        import bio_automation.inat as inat_module

        with pytest.raises(InatApiError) as info:
            inat_module._default_get(URL, "ua", urlopen=ScriptedUrlopen(http_error(403)), sleep=lambda s: None)
        assert type(info.value) is InatApiError
        assert str(info.value) == f"iNaturalist API error 403 for {URL}"


class TestFetchTaxonLineage:
    PAYLOAD = {"results": [{"id": 48707, "ancestors": [
        {"rank": "kingdom", "name": "Fungi", "preferred_common_name": "Fungi Including Lichens"},
        {"rank": "order", "name": "Gomphales"},
        {"rank": "family", "name": "Gomphaceae", "preferred_common_name": None},
        {"rank": "genus", "name": None}]}]}

    def test_returns_ancestors_root_first_with_common_names(self):
        get_fn = RecordingGet(payload=self.PAYLOAD)
        out = fetch_taxon_lineage(48707, get_fn=get_fn, rate_limiter=RateLimiter(min_interval=0))
        assert out == [{"rank": "kingdom", "name": "Fungi", "common_name": "Fungi Including Lichens"},
                       {"rank": "order", "name": "Gomphales", "common_name": None},
                       {"rank": "family", "name": "Gomphaceae", "common_name": None}]                  # the nameless entry is dropped
        assert [c["url"] for c in get_fn.calls] == ["https://api.inaturalist.org/v1/taxa/48707"]

    def test_unknown_taxon_raises_not_found(self):
        with pytest.raises(InatNotFoundError, match="taxon 5: no results in API response"):
            fetch_taxon_lineage(5, get_fn=RecordingGet(payload={"results": []}), rate_limiter=RateLimiter(min_interval=0))

    def test_taxon_with_no_ancestors_gives_an_empty_lineage(self):
        assert fetch_taxon_lineage(1, get_fn=RecordingGet(payload={"results": [{"id": 1}]}), rate_limiter=RateLimiter(min_interval=0)) == []


class TestFetchObservationWithLineage:
    def getter(self):
        calls = []

        def get(url, ua):
            calls.append(url)
            if "/taxa/" in url:
                return {"results": [{"ancestors": [{"rank": "family", "name": "Fam", "preferred_common_name": "Fams"}]}]}
            return {"results": [RAW_OBSERVATION]}

        return get, calls

    def test_lineage_is_stored_under_the_taxon_with_one_extra_request(self):
        get, calls = self.getter()
        out = fetch_inat_observation(123456, get_fn=get, rate_limiter=RateLimiter(min_interval=0), with_lineage=True)
        assert out["taxon"]["lineage"] == [{"rank": "family", "name": "Fam", "common_name": "Fams"}]
        assert calls == ["https://api.inaturalist.org/v1/observations/123456", f"https://api.inaturalist.org/v1/taxa/{out['taxon']['id']}"]

    def test_default_makes_no_taxon_request_and_stores_no_lineage(self):
        get, calls = self.getter()
        out = fetch_inat_observation(123456, get_fn=get, rate_limiter=RateLimiter(min_interval=0))
        assert "lineage" not in out["taxon"] and len(calls) == 1

    def test_cache_shares_one_taxon_request_across_observations(self):
        get, calls = self.getter()
        cache = {}
        for oid in (1, 2, 3):
            fetch_inat_observation(oid, get_fn=get, rate_limiter=RateLimiter(min_interval=0), with_lineage=True, lineage_cache=cache)
        assert sum("/taxa/" in c for c in calls) == 1 and sum("/observations/" in c for c in calls) == 3


class FakeClock:
    def __init__(self, now=1000.0):
        self.now = now

    def __call__(self):
        return self.now


class TestSharedRateLimiter:
    """Two limiters on one file stand in for two worker processes: together they keep one pace."""

    def pair(self, tmp_path, interval=1.1, **kw):
        from bio_automation.inat import SharedRateLimiter

        clock, slept = FakeClock(), []
        make = lambda: SharedRateLimiter(tmp_path / "pace.sqlite3", interval, time_fn=clock, sleep_fn=slept.append, **kw)
        return make(), make(), clock, slept

    def test_a_second_process_waits_for_the_first_ones_slot(self, tmp_path):
        a, b, clock, slept = self.pair(tmp_path)
        assert a.wait() == 0.0
        assert b.wait() == pytest.approx(1.1)
        assert a.wait() == pytest.approx(2.2)                # slots are reserved, not raced: the third request is two intervals out
        assert slept == [pytest.approx(1.1), pytest.approx(2.2)]

    def test_a_request_after_the_interval_has_passed_does_not_wait(self, tmp_path):
        a, b, clock, slept = self.pair(tmp_path)
        a.wait()
        clock.now += 1.1
        assert b.wait() == 0.0 and slept == []

    def test_hold_off_pushes_the_next_slot_of_every_process_back(self, tmp_path):
        a, b, clock, slept = self.pair(tmp_path)
        a.hold_off(30.0)
        assert b.wait() == pytest.approx(30.0) and slept == [pytest.approx(30.0)]

    def test_hold_off_never_pulls_a_later_slot_forward(self, tmp_path):
        a, b, clock, slept = self.pair(tmp_path)
        a.hold_off(30.0)
        b.hold_off(5.0)
        assert a.wait() == pytest.approx(30.0)

    def test_a_slot_beyond_max_wait_is_cut_to_max_wait(self, tmp_path):
        a, b, clock, slept = self.pair(tmp_path, max_wait_s=60.0)
        a.hold_off(10_000.0)                                  # e.g. written before the wall clock jumped back
        assert b.wait() == pytest.approx(60.0)

    def test_a_negative_interval_is_refused(self, tmp_path):
        from bio_automation.inat import SharedRateLimiter

        with pytest.raises(ValueError, match=r"min_interval must be >= 0, got -1\.0"):
            SharedRateLimiter(tmp_path / "p.sqlite3", -1.0)

    def test_the_file_is_created_on_first_use_not_on_construction(self, tmp_path):
        from bio_automation.inat import SharedRateLimiter

        lim = SharedRateLimiter(tmp_path / "sub" / "p.sqlite3", 0.0)
        assert not (tmp_path / "sub").exists()
        lim.wait()
        assert (tmp_path / "sub" / "p.sqlite3").is_file()

    def test_the_default_limiter_uses_the_configured_file_and_pace(self, tmp_path):
        import os
        import subprocess
        import sys

        # a fresh interpreter sees the module-level default (this suite replaces it, see conftest); the file is opened lazily, so nothing is written
        code = ("from bio_automation import inat; l = inat._default_rate_limiter; "
                "print(type(l).__name__, l.path, l.min_interval)")
        env = dict(os.environ, BIO_INAT_RATE_DB=str(tmp_path / "pace.sqlite3"))
        env.pop("BIO_INAT_MIN_INTERVAL_S", None)
        out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True).stdout.strip()
        assert out == f"SharedRateLimiter {tmp_path / 'pace.sqlite3'} 1.1"
        assert not (tmp_path / "pace.sqlite3").exists()


class TestDefaultGetHoldOff:
    @pytest.fixture(autouse=True)
    def retry_settings(self, monkeypatch):
        from bio_automation import settings

        monkeypatch.setattr(settings, "INAT_RETRIES", 3)
        monkeypatch.setattr(settings, "INAT_BACKOFF_S", 2.0)
        monkeypatch.setattr(settings, "INAT_BACKOFF_MAX_S", 60.0)

    def test_a_429_holds_every_process_off_for_the_same_wait(self):
        import bio_automation.inat as inat_module

        held, sleeps = [], []
        opener = ScriptedUrlopen(http_error(429, "7"), http_error(429), {"results": []})
        assert inat_module._default_get(URL, "ua", urlopen=opener, sleep=sleeps.append, hold_off=held.append) == {"results": []}
        assert held == [7.0, 4.0] and sleeps == [7.0, 4.0]

    def test_a_server_error_is_retried_without_holding_anyone_off(self):
        import bio_automation.inat as inat_module

        held, sleeps = [], []
        opener = ScriptedUrlopen(http_error(503), {"results": []})
        assert inat_module._default_get(URL, "ua", urlopen=opener, sleep=sleeps.append, hold_off=held.append) == {"results": []}
        assert held == [] and sleeps == [2.0]

    def test_without_a_hold_off_a_429_goes_to_the_machine_wide_limiter(self, monkeypatch):
        import bio_automation.inat as inat_module

        held = []

        class Recording:
            def hold_off(self, s):
                held.append(s)

        monkeypatch.setattr(inat_module, "_default_rate_limiter", Recording())
        opener = ScriptedUrlopen(http_error(429), {"results": []})
        inat_module._default_get(URL, "ua", urlopen=opener, sleep=lambda s: None)
        assert held == [2.0]


class TestTaxonLineageCache:
    LINEAGE = [{"rank": "family", "name": "Fam", "common_name": "Fams"}]

    def test_an_entry_survives_into_another_instance(self, tmp_path):
        from bio_automation.inat import TaxonLineageCache

        TaxonLineageCache(tmp_path / "t.sqlite3", 100.0)[5] = self.LINEAGE
        other = TaxonLineageCache(tmp_path / "t.sqlite3", 100.0)
        assert 5 in other and other[5] == self.LINEAGE and other.get(5) == self.LINEAGE

    def test_an_entry_older_than_the_ttl_counts_as_absent(self, tmp_path):
        from bio_automation.inat import TaxonLineageCache

        clock = FakeClock()
        cache = TaxonLineageCache(tmp_path / "t.sqlite3", 100.0, time_fn=clock)
        cache[5] = self.LINEAGE
        clock.now += 99.0
        assert 5 in cache
        clock.now += 1.0
        assert 5 not in cache and cache.get(5, "none") == "none"
        with pytest.raises(KeyError, match="5"):
            cache[5]

    def test_an_empty_lineage_is_a_cached_value(self, tmp_path):
        from bio_automation.inat import TaxonLineageCache

        cache = TaxonLineageCache(tmp_path / "t.sqlite3", 100.0)
        cache[1] = []
        assert 1 in cache and cache[1] == []


def obs_payload(*ids, taxon=48484):
    return {"results": [dict(RAW_OBSERVATION, id=i, taxon=dict(RAW_OBSERVATION["taxon"], id=taxon)) for i in ids]}


class TestFetchInatObservations:
    def getter(self, observations, taxa):
        calls = []

        def get(url, ua):
            calls.append(url)
            return taxa if "/taxa/" in url else observations

        return get, calls

    def test_one_request_returns_each_observation_reshaped_as_the_single_fetch_does(self):
        from bio_automation.inat import fetch_inat_observations

        get, calls = self.getter(obs_payload(2, 1), None)
        out = fetch_inat_observations([2, 1, 2], get_fn=get, rate_limiter=RateLimiter(0))
        single = fetch_inat_observation(1, get_fn=RecordingGet(payload=obs_payload(1)), rate_limiter=RateLimiter(0))
        assert calls == ["https://api.inaturalist.org/v1/observations?id=1,2&per_page=2"]
        assert out[1] == single and out[2]["observation_id"] == 2

    def test_an_observation_the_api_does_not_return_is_a_not_found_error_for_that_id_only(self):
        from bio_automation.inat import fetch_inat_observations

        get, _ = self.getter(obs_payload(1), None)
        out = fetch_inat_observations([1, 9], get_fn=get, rate_limiter=RateLimiter(0))
        assert out[1]["observation_id"] == 1
        assert isinstance(out[9], InatNotFoundError) and str(out[9]) == "observation 9: no results in API response"

    def test_ids_are_split_into_requests_of_the_api_maximum(self, monkeypatch):
        from bio_automation import inat

        monkeypatch.setattr(inat, "OBSERVATIONS_PER_REQUEST", 2)
        get, calls = self.getter({"results": []}, None)
        inat.fetch_inat_observations([5, 1, 3], get_fn=get, rate_limiter=RateLimiter(0))
        assert calls == ["https://api.inaturalist.org/v1/observations?id=1,3&per_page=2",
                         "https://api.inaturalist.org/v1/observations?id=5&per_page=1"]

    def test_every_request_waits_for_its_slot(self):
        from bio_automation.inat import fetch_inat_observations

        waits = []

        class Counting:
            def wait(self):
                waits.append(1)

        get, calls = self.getter(obs_payload(1, 2), {"results": [{"id": 48484, "ancestors": []}]})
        fetch_inat_observations([1, 2], get_fn=get, rate_limiter=Counting(), with_lineage=True)
        assert len(calls) == 2 and len(waits) == 2

    def test_lineage_is_fetched_once_per_uncached_taxon_and_stored_in_the_cache(self):
        from bio_automation.inat import fetch_inat_observations

        taxa = {"results": [{"id": 48484, "ancestors": [{"rank": "family", "name": "Fam", "preferred_common_name": "Fams"}]}]}
        get, calls = self.getter(obs_payload(1, 2), taxa)
        cache = {}
        out = fetch_inat_observations([1, 2], get_fn=get, rate_limiter=RateLimiter(0), with_lineage=True, lineage_cache=cache)
        assert calls[1:] == ["https://api.inaturalist.org/v1/taxa/48484"]
        assert out[1]["taxon"]["lineage"] == out[2]["taxon"]["lineage"] == [{"rank": "family", "name": "Fam", "common_name": "Fams"}]
        assert cache == {48484: [{"rank": "family", "name": "Fam", "common_name": "Fams"}]}

    def test_a_cached_taxon_is_not_requested_again(self):
        from bio_automation.inat import fetch_inat_observations

        get, calls = self.getter(obs_payload(1), None)
        out = fetch_inat_observations([1], get_fn=get, rate_limiter=RateLimiter(0), with_lineage=True, lineage_cache={48484: []})
        assert calls == ["https://api.inaturalist.org/v1/observations?id=1&per_page=1"] and out[1]["taxon"]["lineage"] == []

    def test_a_taxon_the_api_does_not_return_refuses_its_observations(self):
        from bio_automation.inat import fetch_inat_observations

        get, _ = self.getter(obs_payload(1), {"results": []})
        out = fetch_inat_observations([1], get_fn=get, rate_limiter=RateLimiter(0), with_lineage=True)
        assert isinstance(out[1], InatNotFoundError) and str(out[1]) == "taxon 48484: no results in API response"

    def test_taxa_are_split_into_requests_of_thirty(self):
        from bio_automation.inat import fetch_taxon_lineages

        get, calls = self.getter(None, {"results": []})
        fetch_taxon_lineages(range(1, 32), get_fn=get, rate_limiter=RateLimiter(0))
        assert calls == [f"https://api.inaturalist.org/v1/taxa/{','.join(map(str, range(1, 31)))}", "https://api.inaturalist.org/v1/taxa/31"]

    def test_a_transient_error_propagates(self):
        from bio_automation.inat import fetch_inat_observations

        with pytest.raises(InatTransientError, match="throttled"):
            fetch_inat_observations([1], get_fn=RecordingGet(to_raise=InatTransientError("throttled")), rate_limiter=RateLimiter(0))
