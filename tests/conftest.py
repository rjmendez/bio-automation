"""Suite-wide isolation: no test reaches the machine-wide iNat pace file (settings.INAT_RATE_DB on D:), which the running pipeline shares."""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def private_inat_pace(tmp_path, monkeypatch):
    from bio_automation import inat

    monkeypatch.setattr(inat, "_default_rate_limiter", inat.SharedRateLimiter(tmp_path / "inat_rate.sqlite3", 0.0))
