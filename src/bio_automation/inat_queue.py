"""The iNat review queue: proposed-observation grouping and the keep/cull
pre-sort.

Ports the grouping algorithm from legacy pipeline/stages/build_inat_queue.py
(group by (src, day, camera); a new group starts whenever the previous
photo is more than *burst_gap_seconds* away in time, OR both photos have a
known location and those locations disagree by more than
*max_location_drift_m*; group ids are one running sequence number assigned
across the whole time-sorted queue, not restarted per bucket) as a pure
function over an already-loaded list of photo records, plus the "pre-sort a
keep list and a cull list, don't cull automatically" presentation that
legacy pipeline/stages/inat_console.py's approve/skip/hold review drove:
presort() only ever *reports* what the human (via inat_ledger's decisions
ledger) has already decided, bucketing anything with no decision yet (or an
explicit "pending") into "undecided" rather than guessing. It never marks
anything kept or culled on its own.

Deliberately not ported in this pass: location tiering itself (exif/
neighbor/none -- build_inat_queue.py's GPS-neighbour inheritance is real
I/O over a NAS/local inventory this repo does not have a fixture for),
resized upload-copy rendering (needs piexif + Pillow + the real source
trees), vision-based release screening, and taxon kNN suggestion
(inat_suggest.py/embed_index.py's Qdrant-backed voting). A caller supplies
this module with whatever lat/lon/loc_source it already resolved per photo.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class QueueItem:
    group: int
    day: str
    camera: str | None
    src: str | None
    photos: list[str] = field(default_factory=list)


def _haversine_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Great-circle distance in metres -- same formula and 6,371,000 m
    Earth radius as build_inat_queue.py's hav()."""
    radius = 6371000.0
    lat1, lat2 = math.radians(a[0]), math.radians(b[0])
    x = (math.sin((lat2 - lat1) / 2) ** 2
         + math.cos(lat1) * math.cos(lat2) * math.sin(math.radians(b[1] - a[1]) / 2) ** 2)
    return 2 * radius * math.asin(math.sqrt(x))


def group_photos(photos: list[dict], burst_gap_seconds: float = 120.0,
                  max_location_drift_m: float = 50.0) -> list[QueueItem]:
    """Group *photos* (each {"path", "day" ("YYYY-MM-DD"), "t" (unix
    seconds, may be None), "camera", "src", "lat", "lon"}) into proposed
    observations.

    Sorted by (src, day, camera, t, path) first, exactly like
    build_inat_queue.py, then walked once: a new group starts whenever the
    walk crosses into a different (src, day, camera) bucket, the time gap
    since the previous photo exceeds *burst_gap_seconds*, or both photos
    have a location and those locations are more than
    *max_location_drift_m* apart. Group numbers are one running sequence
    over the whole sorted list (group 1, 2, 3, ... in walk order), not
    restarted per bucket -- so re-running on the same photo set reproduces
    the same group numbers.
    """
    ordered = sorted(
        photos,
        key=lambda p: (p.get("src") or "", p["day"], p.get("camera") or "",
                       p.get("t") if p.get("t") is not None else -1, p["path"]),
    )

    groups: list[QueueItem] = []
    group_id = 0
    last_key = None
    last_t: float | None = None
    last_latlon: tuple[float, float] | None = None

    for photo in ordered:
        key = (photo.get("src"), photo["day"], photo.get("camera"))
        t = photo.get("t")
        lat, lon = photo.get("lat"), photo.get("lon")
        latlon = (lat, lon) if lat is not None and lon is not None else None

        same_group = False
        if last_key is not None and last_key == key and t is not None and last_t is not None \
                and t - last_t <= burst_gap_seconds:
            if latlon is None or last_latlon is None or _haversine_m(latlon, last_latlon) <= max_location_drift_m:
                same_group = True

        if not same_group:
            group_id += 1
            groups.append(QueueItem(group=group_id, day=photo["day"], camera=photo.get("camera"),
                                     src=photo.get("src"), photos=[]))
        groups[-1].photos.append(photo["path"])

        last_key, last_t, last_latlon = key, t, latlon

    return groups


STATUSES = ("approved", "skipped", "hold", "pending")
PRESORT_BUCKETS = ("keep", "cull", "hold", "undecided")
_STATUS_TO_BUCKET = {"approved": "keep", "skipped": "cull", "hold": "hold", "pending": "undecided"}


def presort(items: list[QueueItem], decisions: dict[int, dict]) -> dict[str, list[int]]:
    """Partition *items* (by group number) into keep/cull/hold/undecided
    using only decisions already on record (see
    inat_ledger.load_decisions) -- approved -> keep, skipped -> cull,
    hold -> hold, no decision yet (or an explicit "pending") ->
    undecided. Never infers a decision from any other field: that is
    exactly the "pre-sort them, don't cull yourself" contract this ports
    from the console's human-in-the-loop review.
    """
    buckets: dict[str, list[int]] = {name: [] for name in PRESORT_BUCKETS}
    for item in items:
        row = decisions.get(item.group)
        status = row["status"] if row else "pending"
        buckets[_STATUS_TO_BUCKET.get(status, "undecided")].append(item.group)
    return buckets
