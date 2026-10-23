"""Suite-wide isolation: no test reaches the machine-wide iNat pace file (settings.INAT_RATE_DB on D:), which the running pipeline shares."""
from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def private_inat_pace(tmp_path, monkeypatch):
    from bio_automation import inat

    monkeypatch.setattr(inat, "_default_rate_limiter", inat.SharedRateLimiter(tmp_path / "inat_rate.sqlite3", 0.0))


@pytest.fixture(autouse=True)
def private_nas_io_gate(tmp_path, monkeypatch):
    """No test reaches the machine-wide NAS read gate file (settings.NAS_IO_DB on D:), which the running pipeline shares."""
    from bio_automation import nas_io, settings

    monkeypatch.setattr(settings, "NAS_IO_DB", tmp_path / "state" / "nas_io.sqlite3")
    monkeypatch.setattr(nas_io, "_defaults", {})


@pytest.fixture(autouse=True)
def private_prints_files(tmp_path, monkeypatch):
    """No test reads or writes the owner's real prints.csv, prints_manual.csv or Darkroom work folder (D:\\bio-results\\social, D:\\bio-cache)."""
    from bio_automation import settings

    monkeypatch.setattr(settings, "SOCIAL_PRINTS_FILE", tmp_path / "no-prints" / "prints.csv")
    monkeypatch.setattr(settings, "SOCIAL_PRINTS_MANUAL_FILE", tmp_path / "no-prints" / "prints_manual.csv")
    monkeypatch.setattr(settings, "DARKROOM_DIR", tmp_path / "no-darkroom")
    monkeypatch.setattr(settings, "SOCIAL_RULES_FILE", tmp_path / "no-prints" / "rules.toml")          # the owner's switches and decision log, too
    monkeypatch.setattr(settings, "SOCIAL_AUDIT_LOG", tmp_path / "no-prints" / "audit.jsonl")
    monkeypatch.setattr(settings, "SOCIAL_THEMES_FILE", tmp_path / "no-prints" / "themes.toml")
    monkeypatch.setattr(settings, "OBS_TAXONOMY_FILE", tmp_path / "no-prints" / "obs_taxonomy.json")
    monkeypatch.setattr(settings, "INAT_SNAPSHOT", tmp_path / "no-prints" / "inat_observations.json")      # nor the owner's real iNat snapshot and photos
    monkeypatch.setattr(settings, "INAT_PHOTO_CACHE", tmp_path / "no-prints" / "inat_photos")
    monkeypatch.setattr(settings, "SOCIAL_PAUSE_FILE", tmp_path / "no-prints" / "PAUSED")                   # nor the pause switch, alerts, digests, backups, engagement
    monkeypatch.setattr(settings, "SOCIAL_PAUSE_SYNC_FILE", None)                                            # the remote pause marker is off unless a test sets it
    monkeypatch.setattr(settings, "SOCIAL_ALERTS_LOG", tmp_path / "no-prints" / "alerts.jsonl")
    monkeypatch.setattr(settings, "SOCIAL_ALERT_COMMAND", "")
    monkeypatch.setattr(settings, "SOCIAL_DIGEST_DIR", tmp_path / "no-prints" / "digest")
    monkeypatch.setattr(settings, "SOCIAL_INTEGRITY_ACK_FILE", tmp_path / "no-prints" / "integrity_ack.json")
    monkeypatch.setattr(settings, "SOCIAL_BACKUP_DIR", tmp_path / "no-prints" / "backups")
    monkeypatch.setattr(settings, "SOCIAL_ENGAGEMENT_LOG", tmp_path / "no-prints" / "engagement.jsonl")
    monkeypatch.setattr(settings, "GALLERY_TITLE_CACHE", tmp_path / "no-prints" / "titles_cache.json")
    monkeypatch.setattr(settings, "GALLERY_OBS_INDEX", tmp_path / "no-prints" / "gallery_obs_index.json")
    monkeypatch.setattr(settings, "SOCIAL_LEDGER", tmp_path / "no-prints" / "ledger.jsonl")
    monkeypatch.setattr(settings, "SOCIAL_DIR", tmp_path / "no-prints" / "social")                          # nor the drafts, the gallery index, the Bluesky session
    monkeypatch.setattr(settings, "GALLERY_OBS_INDEX", tmp_path / "no-prints" / "gallery_obs_index.json")
    monkeypatch.setattr(settings, "INAT_BEST_PHOTO_DIR", tmp_path / "no-prints" / "inat_best")                # nor the downloaded best-size photo cache (photo_cache_cap evicts)
    monkeypatch.setattr(settings, "BSKY_OAUTH_DIR", tmp_path / "no-prints" / "oauth")
    for name in list(os.environ):                                                                       # no rule is switched by the machine's environment
        if name.startswith("BIO_SOCIAL_RULE_"):
            monkeypatch.delenv(name)


@pytest.fixture(autouse=True)
def no_hooks_between_tests():
    """The hook registry is process-wide: every test starts and ends with it empty (and no extension loaded)."""
    from bio_automation import social_hooks

    social_hooks.reset()
    yield
    social_hooks.reset()


@pytest.fixture(autouse=True)
def library_callers_do_not_load_the_real_gates(request, monkeypatch):
    """post_next, make_draft, draft_month and draft_themes load every social extension themselves (so a library caller cannot bypass the gates).
    Most tests exercise those functions with their own hooks or none, so the loading is switched off for them; a test that is about the loading
    (or about the real gates in the flow) is marked `real_gates` and gets the real thing."""
    if request.node.get_closest_marker("real_gates"):
        return
    from bio_automation import social

    monkeypatch.setattr(social, "_require_gates", lambda: None)


def pytest_configure(config):
    config.addinivalue_line("markers", "real_gates: the test runs the library entry points with the real social extensions loaded")
