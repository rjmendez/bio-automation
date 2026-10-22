"""Suite-wide isolation: no test reaches the machine-wide iNat pace file (settings.INAT_RATE_DB on D:), which the running pipeline shares."""
from __future__ import annotations

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
