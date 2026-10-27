"""Unit tests for bio_automation.inat: this module is a thin, settings-bound wrapper around the standalone inat_client library
(https://github.com/rjmendez/inat-client). The client's own algorithm (retries, reshaping, rate limiting, the taxon-lineage
cache) is tested in that repo's own test suite, hermetically, and must not be re-tested here -- these tests cover only the
wiring: that bio-automation's settings values actually reach the library, and that an explicit caller-supplied override still
wins over them."""
from __future__ import annotations

import importlib

import pytest

from bio_automation import inat, settings
from bio_automation.inat import (
    INAT_API_BASE,
    InatApiError,
    InatNotFoundError,
    InatTransientError,
    RateLimiter,
    SharedRateLimiter,
    TaxonLineageCache,
    USER_AGENT_DEFAULT,
)
from inat_client import client as lib_client


class TestReExports:
    """These names must stay the exact objects inat_client defines -- a copy would silently drift."""

    def test_exception_classes_are_the_librarys_own(self):
        assert InatApiError is lib_client.InatApiError
        assert InatNotFoundError is lib_client.InatNotFoundError
        assert InatTransientError is lib_client.InatTransientError

    def test_rate_limiter_classes_are_the_librarys_own(self):
        assert RateLimiter is lib_client.RateLimiter
        assert SharedRateLimiter is lib_client.SharedRateLimiter
        assert TaxonLineageCache is lib_client.TaxonLineageCache

    def test_api_base_and_user_agent_are_the_librarys_own(self):
        assert INAT_API_BASE == lib_client.INAT_API_BASE == "https://api.inaturalist.org/v1"
        assert USER_AGENT_DEFAULT == lib_client.USER_AGENT_DEFAULT


class TestDefaultRateLimiter:
    def test_the_default_limiter_uses_settings_rate_db_and_min_interval(self, tmp_path, monkeypatch):
        monkeypatch.setattr(settings, "INAT_RATE_DB", tmp_path / "pace.sqlite3")
        monkeypatch.setattr(settings, "INAT_MIN_INTERVAL_S", 2.5)
        importlib.reload(inat)
        try:
            assert isinstance(inat._default_rate_limiter, SharedRateLimiter)
            assert inat._default_rate_limiter.path == tmp_path / "pace.sqlite3"
            assert inat._default_rate_limiter.min_interval == 2.5
        finally:
            importlib.reload(inat)   # rebuild the module-level singleton once settings are restored, for tests that run after this one


class TestDefaultGetSettingsBinding:
    def test_falls_back_to_settings_retries_backoff_and_the_default_limiters_hold_off(self, monkeypatch):
        calls = []
        monkeypatch.setattr(inat, "_lib_default_get",
                            lambda url, ua, **kw: calls.append(kw) or {"ok": True})
        monkeypatch.setattr(settings, "INAT_RETRIES", 9)
        monkeypatch.setattr(settings, "INAT_BACKOFF_S", 3.0)
        monkeypatch.setattr(settings, "INAT_BACKOFF_MAX_S", 40.0)

        result = inat._default_get("https://x", "ua/1.0")

        assert result == {"ok": True}
        assert calls == [{"urlopen": None, "sleep": None, "retries": 9, "backoff_s": 3.0, "backoff_max_s": 40.0,
                          "hold_off": inat._default_rate_limiter.hold_off}]

    def test_explicit_kwargs_win_over_settings(self, monkeypatch):
        calls = []
        monkeypatch.setattr(inat, "_lib_default_get", lambda url, ua, **kw: calls.append(kw) or {"ok": True})
        monkeypatch.setattr(settings, "INAT_RETRIES", 9)

        inat._default_get("https://x", "ua", retries=1, backoff_s=0.5, backoff_max_s=5.0)

        assert calls[0]["retries"] == 1 and calls[0]["backoff_s"] == 0.5 and calls[0]["backoff_max_s"] == 5.0


class TestFetchInatObservationDelegation:
    def test_uses_the_modules_own_default_get_and_rate_limiter_when_not_given(self, monkeypatch):
        captured = {}
        monkeypatch.setattr(inat, "_lib_fetch_inat_observation", lambda oid, **kw: captured.update(kw) or {"observation_id": oid})

        result = inat.fetch_inat_observation(42)

        assert result == {"observation_id": 42}
        assert captured["get_fn"] is inat._default_get
        assert captured["rate_limiter"] is inat._default_rate_limiter
        assert captured["with_lineage"] is False and captured["lineage_cache"] is None

    def test_an_explicit_get_fn_and_rate_limiter_win_over_the_module_defaults(self, monkeypatch):
        captured = {}
        monkeypatch.setattr(inat, "_lib_fetch_inat_observation", lambda oid, **kw: captured.update(kw) or {"observation_id": oid})
        sentinel_get, sentinel_limiter, sentinel_cache = object(), object(), {}

        inat.fetch_inat_observation(42, get_fn=sentinel_get, rate_limiter=sentinel_limiter, with_lineage=True, lineage_cache=sentinel_cache)

        assert captured["get_fn"] is sentinel_get
        assert captured["rate_limiter"] is sentinel_limiter
        assert captured["with_lineage"] is True and captured["lineage_cache"] is sentinel_cache


class TestFetchInatObservationsDelegation:
    def test_uses_the_modules_own_default_get_and_rate_limiter_when_not_given(self, monkeypatch):
        captured = {}
        monkeypatch.setattr(inat, "_lib_fetch_inat_observations", lambda ids, **kw: captured.update(kw) or {i: {} for i in ids})

        inat.fetch_inat_observations([1, 2])

        assert captured["get_fn"] is inat._default_get
        assert captured["rate_limiter"] is inat._default_rate_limiter

    def test_an_explicit_rate_limiter_wins_over_the_module_default(self, monkeypatch):
        captured = {}
        monkeypatch.setattr(inat, "_lib_fetch_inat_observations", lambda ids, **kw: captured.update(kw) or {})
        sentinel_limiter = object()

        inat.fetch_inat_observations([1], rate_limiter=sentinel_limiter)

        assert captured["rate_limiter"] is sentinel_limiter


class TestFetchTaxonLineageDelegation:
    def test_uses_the_modules_own_default_get_and_rate_limiter_when_not_given(self, monkeypatch):
        captured = {}
        monkeypatch.setattr(inat, "_lib_fetch_taxon_lineage", lambda tid, **kw: captured.update(kw) or [])

        inat.fetch_taxon_lineage(7)

        assert captured["get_fn"] is inat._default_get
        assert captured["rate_limiter"] is inat._default_rate_limiter

    def test_an_explicit_rate_limiter_wins_over_the_module_default(self, monkeypatch):
        captured = {}
        monkeypatch.setattr(inat, "_lib_fetch_taxon_lineage", lambda tid, **kw: captured.update(kw) or [])
        sentinel_limiter = object()

        inat.fetch_taxon_lineage(7, rate_limiter=sentinel_limiter)

        assert captured["rate_limiter"] is sentinel_limiter


class TestFetchTaxonLineagesDelegation:
    def test_uses_the_modules_own_default_get_and_rate_limiter_when_not_given(self, monkeypatch):
        captured = {}
        monkeypatch.setattr(inat, "_lib_fetch_taxon_lineages", lambda ids, **kw: captured.update(kw) or {})

        inat.fetch_taxon_lineages([7, 8])

        assert captured["get_fn"] is inat._default_get
        assert captured["rate_limiter"] is inat._default_rate_limiter

    def test_an_explicit_rate_limiter_wins_over_the_module_default(self, monkeypatch):
        captured = {}
        monkeypatch.setattr(inat, "_lib_fetch_taxon_lineages", lambda ids, **kw: captured.update(kw) or {})
        sentinel_limiter = object()

        inat.fetch_taxon_lineages([7], rate_limiter=sentinel_limiter)

        assert captured["rate_limiter"] is sentinel_limiter
