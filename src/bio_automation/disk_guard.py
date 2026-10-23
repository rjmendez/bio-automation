"""Disk guard: a worker PAUSES while a drive is under its floor, then resumes by itself, instead of failing in the middle of a write.

One definition of "free space" for the whole package: social_selfcheck (its disk checks), social_alerts (its disk alerts), social_prep and the post run
all read the drives through default_drives()/free_bytes() here, and every threshold comes from settings (DISK_GUARD_*, SOCIAL_*_MIN_FREE_*).

wait_for_room(floors, ...) is the whole mechanism:
  - every floor met: returns True at once and writes nothing;
  - a floor missed: ONE disk_pause event (who, each short drive with its free and needed GB), then it polls every poll_s; it resumes only when every drive is
    at floor + resume_margin_gb (hysteresis: a drive hovering at the edge does not flap), writes ONE disk_resume event with the seconds waited, returns True;
  - max_wait_s > 0 and the wait ran out: a disk_timeout event and False, so a scheduled run can stop cleanly instead of waiting for ever. 0 waits as long as it takes.
Events go to the `audit` callable (Audit.log for the social runs; log_event() to settings.DISK_GUARD_LOG for the pipeline workers). social_alerts reads both.

A pipeline worker calls `guard_for_write(who)` before each write step (or `bio disk-guard --who nas_pipeline` from a script): pause is cheap, a half-written render is not.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Callable, NamedTuple

from . import settings

GB = 1024 ** 3


class Floor(NamedTuple):
    label: str          # "system" or "data"
    path: Path
    min_gb: float


def default_drives() -> dict[str, Path]:
    system = Path(os.environ.get("SystemDrive", "C:") + "\\") if os.name == "nt" else Path("/")
    return {"system": system, "data": Path(settings.CACHE_DIR)}


def free_bytes(path: Path) -> int:
    """Free bytes on the drive holding `path`, which may not exist yet (the nearest existing parent is measured)."""
    p = Path(path)
    while not p.exists() and p != p.parent:
        p = p.parent
    return shutil.disk_usage(p).free


def default_floors(*, system_gb: float | None = None, data_gb: float | None = None, extra: dict[str, Path] | None = None) -> list[Floor]:
    """The system drive and the data drive (the one under CACHE_DIR) with the settings floors; `extra` adds more watched paths at the data floor
    (e.g. the render output folder, which may be on another drive)."""
    sys_gb = settings.DISK_GUARD_SYSTEM_MIN_GB if system_gb is None else system_gb
    dat_gb = settings.DISK_GUARD_DATA_MIN_GB if data_gb is None else data_gb
    d = default_drives()
    out = [Floor("system", d["system"], sys_gb), Floor("data", d["data"], dat_gb)]
    out += [Floor(label, Path(p), dat_gb) for label, p in (extra or {}).items()]
    return out


def output_floors(out: str | Path, min_gb: float) -> list[Floor]:
    """ONLY the drive of a stage's output folder at that stage's own floor (its BIO_MIN_FREE_GB). The stages that use it (render_all, render_selected,
    topaz_enhance) write next to nothing to C:, so a low C: must not stop them (it did, for ever, under the old system-drive floor); C: is watched by
    the workers that do write there (guard_for_write) and by social-selfcheck."""
    return [Floor("output", Path(out), min_gb)]


def wait_for_output(out: str | Path, min_gb: float, *, who: str, say: Callable[[str], None] | None = None, **kw) -> bool:
    """The one call a pipeline stage makes before writing: wait (events to DISK_GUARD_LOG) for the output drive only, at most
    settings.DISK_GUARD_STAGE_MAX_WAIT_S (never for ever); False when that ran out, so the stage reports STOP and ends."""
    kw.setdefault("audit", log_event)
    kw.setdefault("max_wait_s", settings.DISK_GUARD_STAGE_MAX_WAIT_S)
    return wait_for_room(output_floors(out, min_gb), who=who, say=say, **kw)


def shortfalls(floors: list[Floor], free: Callable[[Path], int] = free_bytes, *, margin_gb: float = 0.0) -> list[dict]:
    """The floors not met (free < min + margin), each as {drive, path, free_gb, need_gb}."""
    out = []
    for f in floors:
        try:
            have = free(f.path) / GB
        except OSError as e:      # the drive is offline or the path went away: unknown space counts as short (fail closed) and the wait goes on
            out.append({"drive": f.label, "path": str(f.path), "free_gb": 0.0, "need_gb": f.min_gb + margin_gb, "error": f"{type(e).__name__}: {e}"[:200]})
            continue
        if have < f.min_gb + margin_gb:
            out.append({"drive": f.label, "path": str(f.path), "free_gb": round(have, 1), "need_gb": f.min_gb + margin_gb})
    return out


def describe(short: list[dict]) -> str:
    return "; ".join((f"{s['drive']} {s['path']} cannot be read ({s['error']}), needs {s['need_gb']:g}" if "error" in s
                      else f"{s['drive']} {s['path']} has {s['free_gb']:.1f} GB free, needs {s['need_gb']:g}") for s in short)


def log_event(event: str, **fields) -> dict:
    """Append {"at", "event", ...} to settings.DISK_GUARD_LOG (one write of one line)."""
    rec = {"at": dt.datetime.now(settings.LOCAL_TZ).isoformat(timespec="seconds"), "event": event, **fields}
    path = Path(settings.DISK_GUARD_LOG)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(rec, sort_keys=True, ensure_ascii=False) + "\n")
    return rec


def wait_for_room(floors: list[Floor], *, who: str, audit: Callable[..., object] | None = None,
                  free: Callable[[Path], int] = free_bytes, sleep: Callable[[float], None] = time.sleep,
                  clock: Callable[[], float] = time.monotonic, poll_s: float | None = None, max_wait_s: float | None = None,
                  margin_gb: float | None = None, say: Callable[[str], None] | None = None) -> bool:
    """True when every floor is met (after waiting, if it had to). False only when max_wait_s > 0 ran out. See the module docstring."""
    audit = audit or log_event
    poll = settings.DISK_GUARD_POLL_S if poll_s is None else poll_s
    limit = settings.DISK_GUARD_MAX_WAIT_S if max_wait_s is None else max_wait_s
    margin = settings.DISK_GUARD_RESUME_MARGIN_GB if margin_gb is None else margin_gb
    short = shortfalls(floors, free)
    if not short:
        return True
    say = say or (lambda m: print(m, file=sys.stderr, flush=True))
    t0 = clock()
    audit("disk_pause", who=who, short=short)
    say(f"{who}: paused, low disk space ({describe(short)}); resumes by itself at +{margin:g} GB")
    while True:
        waited = clock() - t0
        if limit > 0 and waited >= limit:
            audit("disk_timeout", who=who, waited_s=round(waited, 1), short=short)
            say(f"{who}: gave up after {waited:.0f}s, still low on disk space ({describe(short)})")
            return False
        sleep(poll if limit <= 0 else min(poll, max(limit - waited, 0.0)))
        short = shortfalls(floors, free, margin_gb=margin)
        if not short:
            waited = clock() - t0
            audit("disk_resume", who=who, waited_s=round(waited, 1))
            say(f"{who}: disk space is back, resuming after {waited:.0f}s")
            return True


def guard_for_write(who: str, *, extra: dict[str, Path] | None = None, **kw) -> bool:
    """wait_for_room on the default floors (settings), events to settings.DISK_GUARD_LOG: the one call a pipeline worker makes before a write step."""
    return wait_for_room(default_floors(extra=extra), who=who, **kw)


def add_parsers(sub) -> None:
    p = sub.add_parser("disk-guard", help="Exit 0 when C: and the data drive are above their floors (waiting for them with --wait); 3 when still short. "
                                          "For scripts that run pipeline steps: `bio disk-guard --wait --who nas_pipeline && <step>`.")
    p.add_argument("--who", default="disk-guard")
    p.add_argument("--wait", action="store_true", help="pause and poll until there is room (settings.DISK_GUARD_*), instead of failing")
    p.add_argument("--max-wait", type=float, default=None, help="seconds to wait at most (default settings.DISK_GUARD_MAX_WAIT_S; 0 = for ever)")
    p.add_argument("--path", action="append", default=[], help="also watch the drive of this path at the data floor (repeatable)")


def cmd_disk_guard(args: argparse.Namespace) -> int:
    floors = default_floors(extra={f"path{i}": Path(p) for i, p in enumerate(args.path, 1)})
    if args.wait:
        return 0 if wait_for_room(floors, who=args.who, max_wait_s=args.max_wait) else 3
    short = shortfalls(floors)
    if short:
        print(f"{args.who}: low disk space ({describe(short)})", file=sys.stderr)
        return 3
    print(f"{args.who}: disk space ok")
    return 0


COMMANDS = {"disk-guard": cmd_disk_guard}
