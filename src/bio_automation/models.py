from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Asset:
    file: str
    role: str
    status: str
    selected: bool | None = None
    # Non-destructive crop rect [x0, y0, x1, y1], normalized 0..1 against the
    # full decoded image, origin top-left (same convention LightCraft itself
    # uses for mask/crop coordinates). None means uncropped. The original raw
    # file is never touched -- this is only ever read back by render.py's
    # export functions and the review console's own preview, applied at
    # render/export time.
    crop: list[float] | None = None


@dataclass
class Observation:
    observation_id: int | None = None
    obsdate: str | None = None
    source: str = "local-ingest"
    taxon: dict[str, Any] = field(default_factory=dict)
    place_guess: str | None = None
    locality: str | None = None
    license: str | None = None
    quality_grade: str | None = None
    keywords_seed: list[str] = field(default_factory=list)

