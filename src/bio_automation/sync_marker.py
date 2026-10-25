"""When did each Qdrant sync last write successfully? `bio qdrant-sync` and `bio qdrant-inat-sync` record a marker after a real (not --dry-run)
run; `social-alerts` and `social-selfcheck` call stale() and warn when a marker is older than settings.QDRANT_STALE_DAYS or was never written.
The marker is one small JSON file written in one go (temp file, then rename), so it is never half written."""
from __future__ import annotations

import contextlib
import datetime as dt
import json
import os
import time
from pathlib import Path

from . import settings

NAMES = ("qdrant-sync", "qdrant-inat-sync")
LOCK_WAIT_S = 30.0


def load(path: Path | None = None) -> dict[str, dict]:
    """{sync name: {"at": iso time, ...counts}}; a missing or unreadable file is empty."""
    try:
        data = json.loads(Path(path or settings.QDRANT_SYNC_MARKER).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {k: v for k, v in data.items() if isinstance(v, dict)} if isinstance(data, dict) else {}


def record(name: str, counts: dict | None = None, *, path: Path | None = None, now: dt.datetime | None = None) -> dict:
    """Read-modify-write of the shared marker under a byte-range lock (<marker>.lock), so the two syncs finishing together (or a manual run next to
    the task) cannot overwrite each other's entry. A lock still busy after LOCK_WAIT_S raises SocialError (the caller audits the failure)."""
    from .social import SocialError, byte_range_lock

    path = Path(path or settings.QDRANT_SYNC_MARKER)
    now = now or dt.datetime.now(settings.LOCAL_TZ)
    deadline = time.monotonic() + LOCK_WAIT_S
    with contextlib.ExitStack() as stack:
        while True:
            try:
                stack.enter_context(byte_range_lock(Path(str(path) + ".lock"), "the Qdrant sync marker is locked by another sync ({path})"))
                break
            except SocialError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.05)
        data = load(path)
        data[name] = {"at": now.isoformat(timespec="seconds"), **(counts or {})}
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp, path)
    return data[name]


def stale(now: dt.datetime, *, path: Path | None = None, max_days: float | None = None, names=NAMES) -> list[tuple[str, str]]:
    """[(sync name, message)] for each sync that never recorded a run or last ran more than max_days ago."""
    max_days = settings.QDRANT_STALE_DAYS if max_days is None else max_days
    marks, out = load(path), []
    for name in names:
        try:
            at = dt.datetime.fromisoformat(marks[name]["at"])
            if at.tzinfo is None:
                at = at.replace(tzinfo=now.tzinfo)
        except (KeyError, ValueError, TypeError):
            out.append((name, f"{name} has no recorded successful run (the Qdrant index may be empty or out of date)"))
            continue
        age = (now - at).total_seconds() / 86400
        if age > max_days:
            out.append((name, f"{name} last wrote {age:.1f} days ago ({marks[name]['at']}), more than {max_days:g}: the Qdrant index is stale"))
    return out
