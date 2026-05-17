from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Asset:
    file: str
    role: str
    status: str
    selected: bool | None = None


@dataclass
class Observation:
    observation_id: int | None = None
    obsdate: str | None = None
    source: str = "inat-readonly"
    taxon: dict[str, Any] = field(default_factory=dict)
    place_guess: str | None = None
    locality: str | None = None
    license: str | None = None
    quality_grade: str | None = None
    keywords_seed: list[str] = field(default_factory=list)

