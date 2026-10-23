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
    for name in list(os.environ):                                                                       # no rule is switched by the machine's environment
        if name.startswith("BIO_SOCIAL_RULE_"):
            monkeypatch.delenv(name)
