"""Decision and posted ledgers for the iNaturalist review queue.

Ports the ledger half of legacy pipeline/stages/inat_console.py
(append-only inat_decisions{TAG}.jsonl, one row per approve/skip/hold a
human makes in the review console, keyed by integer group number -- see
inat_queue.py) and the idempotency half of legacy
pipeline/stages/inat_post.py (inat_posted{TAG}.jsonl, one row per group
actually attempted -- consulted so a re-run never double-posts). Status
vocabulary ("approved"/"skipped"/"hold"/"pending" for decisions,
"posted"/"partial"/"failed" for posted rows) and the approval field
validation (inat_console.py's do_POST /api/decide handler) match the
legacy scripts exactly.

Both ledgers are plain JSONL, append-only by convention, replayed
last-write-wins per group on load -- a group can appear more than once (the
human changed their mind in the console), and the latest row for that
group is the one that counts. Nothing here reads or writes any *{TAG}*
path by default; every path is passed in by the caller.
"""
from __future__ import annotations

import time
from typing import Callable

from .io import read_jsonl, write_jsonl

DECISION_STATUSES = ("approved", "skipped", "hold", "pending")
APPROVAL_REQUIRED_FIELDS = ("observed_on_string", "time_zone", "lat", "lon")
GEOPRIVACY_CHOICES = ("open", "obscured", "private")

# A group counts as "done" (never re-sent) once it has either fully posted
# or partially posted (an observation was created even if a photo upload
# after it failed) -- matching legacy inat_post.py's posted_groups(). Only
# "failed" (no observation ever created) is left retryable.
DONE_POST_STATUSES = ("posted", "partial")


def record_decision(path, group: int, status: str, *, fields: dict | None = None,
                     now_fn: Callable[[], float] = time.time) -> dict:
    """Append one decision row for *group*. *status* must be one of
    DECISION_STATUSES; anything else raises ValueError rather than
    silently being recorded as some other state (a typo must not become a
    cull). Approving a group additionally requires observed_on_string,
    time_zone, lat and lon in *fields*, and a valid geoprivacy -- the same
    guard inat_console.py's /api/decide applies, since an approved group
    with no location/time can never actually be posted.
    """
    if status not in DECISION_STATUSES:
        raise ValueError(f"status must be one of {DECISION_STATUSES}, got {status!r}")
    fields = dict(fields or {})
    if status == "approved":
        missing = [key for key in APPROVAL_REQUIRED_FIELDS if fields.get(key) in (None, "")]
        if missing:
            raise ValueError(f"approve needs: {', '.join(missing)}")
        if fields.get("geoprivacy") not in GEOPRIVACY_CHOICES:
            raise ValueError(f"geoprivacy must be one of {GEOPRIVACY_CHOICES}, "
                              f"got {fields.get('geoprivacy')!r}")

    row = {"group": group, "status": status, **fields, "ts": now_fn()}
    rows = read_jsonl(path)
    rows.append(row)
    write_jsonl(path, rows)
    return row


def load_decisions(path) -> dict[int, dict]:
    """Replay the decisions ledger into {group: latest decision row}."""
    latest: dict[int, dict] = {}
    for row in read_jsonl(path):
        latest[row["group"]] = row
    return latest


def load_posted(path) -> dict[int, dict]:
    """{group: latest posted row}. A group should only ever gain a
    "posted" or "partial" row once (that is the whole point of this
    ledger), but load like the decisions ledger regardless -- last entry
    wins if something upstream ever did append twice."""
    latest: dict[int, dict] = {}
    for row in read_jsonl(path):
        latest[row["group"]] = row
    return latest


def is_done(path, group: int) -> bool:
    """True if *group* already has a posted or partial row -- i.e. must
    never be sent again. A "failed" row (no observation was ever created)
    does not count as done; that group may still be retried."""
    row = load_posted(path).get(group)
    return row is not None and row.get("status") in DONE_POST_STATUSES


def record_posted(path, group: int, status: str, *, observation_id: int | None = None,
                   uuid: str | None = None, photos: int | None = None,
                   error: str | None = None, now_fn: Callable[[], float] = time.time) -> dict:
    if status not in ("posted", "partial", "failed"):
        raise ValueError(f"status must be one of ('posted', 'partial', 'failed'), got {status!r}")
    row = {"group": group, "status": status, "observation_id": observation_id, "uuid": uuid,
           "photos": photos, "error": error, "ts": now_fn()}
    rows = read_jsonl(path)
    rows.append(row)
    write_jsonl(path, rows)
    return row
