"""bio social-prep: the daily preparation of the Bluesky posts, split from the posting run.

Until 2026-10-23 the 08:00 post did the prep first and waited 16 minutes for social-index. Now a separate early task runs this command and the
post runs never wait for it: posting does not depend on prep (it posts what is already drafted), so a failed prep never blocks a post.

Steps, in order, each timed and audited as a prep_step event (step, seconds, ok):
  preflight     C: and D: have enough free space, otherwise stop with one clear line (nothing else runs)
  social-index  gallery file -> observation map (incremental: only new or changed files are read through exiftool)
  darkroom-sync Darkroom products -> prints.csv
  social-draft  top up the drafts (its maintenance hooks run at the end)
A failed step does not stop the later ones that can still run on what is already on disk; the exit code is 1 when any step failed."""
from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path
from typing import Callable

from . import settings

STEPS = ("social-index", "darkroom-sync", "social-draft")


def free_gb(path: str) -> float:
    return shutil.disk_usage(path).free / 1024 ** 3


def preflight(free: Callable[[str], float] = free_gb) -> str | None:
    """None when there is room to work, else the one-line reason to stop."""
    for drive, need in (("C:\\", settings.SOCIAL_PREP_MIN_FREE_C_GB), ("D:\\", settings.SOCIAL_PREP_MIN_FREE_D_GB)):
        have = free(drive)
        if have < need:
            return f"{drive[:2]} has {have:.1f} GB free, social-prep needs {need:g} GB"
    return None


def _split(value: str) -> list[str]:
    return [v for v in value.split(";") if v.strip()]


def default_steps(args: argparse.Namespace) -> dict[str, Callable[[], int]]:
    """The real step runners: the same command functions `bio social-index` etc. run, with their default arguments."""
    from . import cli

    def index() -> int:
        return cli.cmd_social_index(argparse.Namespace(gallery_dir=args.gallery_dir, audit=_split(settings.SOCIAL_INDEX_AUDIT),
                                                       candidates=_split(settings.SOCIAL_INDEX_CANDIDATES), out=args.index, full=False))

    def darkroom() -> int:
        return cli.cmd_darkroom_sync(argparse.Namespace(index=args.index, gallery_dir=args.gallery_dir, dry_run=False))

    def draft() -> int:
        return cli.cmd_social_draft(argparse.Namespace(keep_pending=None, date=None, index=args.index, gallery_dir=args.gallery_dir,
                                                       no_themes=False))

    return {"social-index": index, "darkroom-sync": darkroom, "social-draft": draft}


def run_prep(steps: dict[str, Callable[[], int]], *, audit, free: Callable[[str], float] = free_gb,
             clock: Callable[[], float] = time.monotonic, say: Callable[[str], None] = print) -> int:
    """Run preflight and then every step; return 0 when all succeeded, else 1. A step fails by returning non-zero or by raising."""
    t0 = clock()
    reason = preflight(free)
    audit.log("prep_step", step="preflight", seconds=round(clock() - t0, 1), ok=reason is None, **({"reason": reason} if reason else {}))
    if reason:
        say(f"bio social-prep: stopped, {reason}")
        return 1
    failed = []
    for name in STEPS:
        t0 = clock()
        error = None
        try:
            code = steps[name]()
            ok = code == 0
            if not ok:
                error = f"exit code {code}"
        except (Exception, SystemExit) as e:  # noqa: BLE001 -- a broken step must not stop the later ones; SystemExit is how the cli reports bad input
            ok, error = False, f"{type(e).__name__}: {e}"[:300]
        seconds = round(clock() - t0, 1)
        audit.log("prep_step", step=name, seconds=seconds, ok=ok, **({"error": error} if error else {}))
        say(f"bio social-prep: {name} {'ok' if ok else 'FAILED (' + str(error) + ')'} in {seconds}s")
        if not ok:
            failed.append(name)
    if failed:
        say(f"bio social-prep: failed steps: {', '.join(failed)}")
    return 1 if failed else 0


def add_parsers(sub) -> None:
    p = sub.add_parser("social-prep", help="Daily preparation, separate from posting: preflight disk check, social-index (incremental), darkroom-sync, "
                                           "social-draft, each timed and audited. Exits 1 when a step failed; posting never depends on it.")
    p.add_argument("--index", default=str(settings.GALLERY_OBS_INDEX))
    p.add_argument("--gallery-dir", default=str(settings.GALLERY_DIR))


def cmd_social_prep(args: argparse.Namespace) -> int:
    from .social_rules import Audit

    return run_prep(default_steps(args), audit=Audit(), free=free_gb)


COMMANDS = {"social-prep": cmd_social_prep}
