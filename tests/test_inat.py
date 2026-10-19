"""Unit tests for bio_automation.inat (hermetic -- the urllib network
boundary is always stubbed via get_fn/monkeypatch, never hit for real)."""
from __future__ import annotations

import urllib.error

import pytest

from bio_automation.inat import (
    InatApiError,
    InatNotFoundError,
    RateLimiter,
    fetch_inat_observation,
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

        with pytest.raises(InatApiError, match="500"):
            inat_module._default_get("https://api.inaturalist.org/v1/observations/1", "ua/1.0")
